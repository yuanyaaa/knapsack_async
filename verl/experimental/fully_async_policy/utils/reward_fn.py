"""
Custom reward function for math tasks (AIME, MATH, deepscaler, etc.)
and code generation tasks (codecontests, apps, codeforces, taco, livecodebench).
Migrated from recipe/deepscaler/reward_fn.py and recipe/knapsack_rl/reward_fn.py
with fixed import paths and Python 3.12+ compatibility (pyext replaced).
"""
import base64
import json
import logging
import multiprocessing
import pickle
import zlib

from verl.utils.reward_score import math_dapo
from verl.experimental.fully_async_policy.utils import math_utils
from verl.experimental.fully_async_policy.utils import gpqa
from verl.experimental.fully_async_policy.utils import prime_code
from verl.experimental.fully_async_policy.utils.prime_code.testing_util import run_test

logger = logging.getLogger(__name__)


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    boxed_pred = math_dapo.last_boxed_only_string(solution)
    extracted_pred = math_dapo.remove_boxed(boxed_pred) if boxed_pred is not None else None

    return extracted_pred


def rllm_math_reward_fn(solution_str: str, ground_truth: str):
    """Reward function for math problems using RLLM's math utils.

    Copy from: https://github.com/agentica-project/rllm/blob/7b47687f6a9ef1bf5cbd56dd1af61fff08c4b0e4/rllm/rewards/math_reward.py
    """

    model_response = solution_str

    # Extract solution.
    if "</think>" in model_response:
        model_solution = model_response.split("</think>")[1]
    elif "\\boxed" in model_response:
        model_solution = model_response
    else:
        return 0.0, False, "[INVALID]"

    model_answer = math_utils.extract_answer(model_solution)
    if model_answer is None:
        return 0.0, False, "[INVALID]"

    # Process the ground truth(s)
    ground_truths = ground_truth
    if ground_truths is None:
        return 0.0, False, "[INVALID]"

    # Convert single answer to list for uniform processing
    if isinstance(ground_truths, (str, float, int)):
        ground_truths = [ground_truths]

    # Process each ground truth
    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = math_utils.extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)

    if not processed_ground_truths:
        return 0.0, False, "[INVALID]"

    # Check against all possible correct answers
    for ground_truth in processed_ground_truths:
        is_correct = math_utils.grade_answer_mathd(model_answer, ground_truth) or math_utils.grade_answer_sympy(model_answer, ground_truth)
        if is_correct:
            return 1.0, True, model_answer

    return 0.0, False, model_answer


def _lcb_run(in_outs, generation, debug, result, metadata_list, timeout):
    """Worker function for LiveCodeBench test execution in a subprocess."""
    r, m = run_test(in_outs, test=generation, debug=debug, timeout=timeout)
    result.append(r)
    metadata_list.append(m)


def _lcb_check(in_outs, generation, timeout, debug=True):
    """Run LiveCodeBench test cases with multiprocessing timeout protection."""
    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_lcb_run,
        args=(in_outs, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(timeout=(timeout + 1) * len(in_outs["inputs"]) + 5)
    if p.is_alive():
        p.kill()
    if not result:
        result = [[-1 for _ in range(len(in_outs["inputs"]))]]
    return result[0], metadata_list[0] if metadata_list else {}


def _compute_livecodebench_score(solution_str: str, ground_truth) -> dict:
    """Compute score for LiveCodeBench problems.

    Extracts python code from the solution, decodes test cases from ground_truth
    (supports both plain JSON and compressed pickle formats), and runs tests
    with multiprocessing timeout protection.
    """
    # Extract python code block from completion
    solution = solution_str.split("```python")[-1].split("```")[0]

    # Decode test cases
    try:
        in_outs = json.loads(ground_truth)
    except Exception:
        in_outs = json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(ground_truth.encode("utf-8"))))
        )

    success = False
    try:
        test_res, _ = _lcb_check(in_outs=in_outs, generation=solution, timeout=6, debug=False)
        success = all(map(lambda x: x is True, test_res))
    except Exception:
        logger.warning("LiveCodeBench test execution failed", exc_info=True)

    return {
        "score": 1.0 if success else 0.0,
        "acc": bool(success),
    }


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    prompt_str=None,
    return_pred=True,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    This custom reward function handles AIME, MATH, deepscaler and other math
    data sources that are not covered by the default verl reward function.

    Falls back to the default verl compute_score for known data sources.
    """
    if data_source in ["math_dapo", "math_dapo_reasoning"] or data_source.startswith("aime"):
        res = math_dapo.compute_score(solution_str, ground_truth)

    elif data_source in [
        "AIME", "AIME2025", "AMC", "MATH", "MINERVA",
        "OLYMPIAD_BENCH", "deepscaler",
        "DigitalLearningGmbH/MATH-lighteval",
    ]:
        score, is_correct, extracted_answer = rllm_math_reward_fn(solution_str, ground_truth)
        res = {
            "score": float(score),
            "acc": is_correct,
            "pred": extracted_answer if extracted_answer else str("[INVALID]"),
        }

    elif data_source in ["Idavidrein/gpqa", "gpqa-diamond", "gpqa"]:
        # GPQA multi-choice evaluation: first try pattern matching, then fallback to boxed answer
        res = gpqa.compute_score(solution_str, ground_truth)

        extracted_answer = extract_boxed_answer(solution_str)
        if res == 0:
            if extracted_answer == ground_truth:
                res = 1.0

        if return_pred:
            res = {
                "score": float(res),
                "acc": True if res == 1 else False,
                "pred": extracted_answer if extracted_answer else str("[INVALID]"),
            }

    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Code generation evaluation using prime_code (pyext-free, Python 3.12+ compatible)
        score, metadata = prime_code.compute_score(solution_str, ground_truth, continuous=True)
        res = {
            "score": float(score) if isinstance(score, (int, float)) else (1.0 if score else 0.0),
            "acc": bool(score == 1.0 if isinstance(score, float) else score is True),
        }

    elif "livecodebench" in data_source:
        # LiveCodeBench evaluation with multiprocessing timeout protection
        res = _compute_livecodebench_score(solution_str, ground_truth)

    else:
        # Fallback to default verl compute_score for other data sources
        from verl.utils.reward_score import default_compute_score
        res = default_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            sandbox_fusion_url=sandbox_fusion_url,
            concurrent_semaphore=concurrent_semaphore,
            memory_limit_mb=memory_limit_mb,
        )

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])

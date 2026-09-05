"""
Custom reward function for math tasks (AIME, MATH, deepscaler, etc.)
Migrated from recipe/deepscaler/reward_fn.py with fixed import paths.
"""
from verl.experimental.fully_async_policy.utils import math_utils, gpqa
from verl.utils.reward_score import math_dapo, prime_math

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
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "math_dapo" or data_source.startswith("aime"):
        res = math_dapo.compute_score(solution_str, ground_truth)

    elif data_source in ["AIME", "AIME2025", "AMC", "MATH", "MINERVA", "OLYMPIAD_BENCH", "deepscaler", "DigitalLearningGmbH/MATH-lighteval", "DAPO"]:
        score, is_correct, extracted_answer = rllm_math_reward_fn(solution_str, ground_truth)
        res = {
            "score": float(score),
            "acc": is_correct,
            "pred": extracted_answer if extracted_answer else str("[INVALID]"),
        }

    elif data_source in ["Idavidrein/gpqa"] :
        res = gpqa.compute_score(solution_str, ground_truth)  # this does not capture the \\boxed case

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

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])

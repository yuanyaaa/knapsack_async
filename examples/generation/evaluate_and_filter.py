#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Evaluate RL training parquet data and annotate/filter for priority sampling.

This script works in two modes:

Mode A (evaluate-only):
    Takes a *generated* parquet (output of main_generation_server.py, containing
    a `responses` column with N responses per prompt). Evaluates each response
    using a custom reward function via Ray remote workers, computes per-prompt
    metrics, and creates a *new* output parquet with the evaluation metrics
    written into the `extra_info` column. The original train parquet is never
    modified.

Mode B (generate + evaluate, end-to-end):
    First launches main_generation_server.py via verl's Ray-based distributed
    rollout to generate N samples per prompt, then evaluates and annotates.

The following fields are added to `extra_info`:
    - pass_rate:      fraction of responses with score > 0
    - raw_pass_rate:  same as pass_rate (before any EMA smoothing in training)
    - overlong_ratio: fraction of responses that hit the max_tokens limit
    - prompt_uid:     deterministic MD5-based unique ID for each prompt

Usage (Mode A - evaluate only):
    python evaluate_and_filter.py \
        --generated_parquet /path/to/generated.parquet \
        --input_parquet /path/to/train.parquet \
        --output_parquet /path/to/output.parquet \
        --reward_fn_path /path/to/reward_fn.py \
        --reward_fn_name compute_score \
        --max_response_length 4096 \
        --filter_pass_rate_min 0.0 \
        --filter_pass_rate_max 1.0 \
        --filter_overlong_ratio_max 0.05

Usage (Mode B - generate + evaluate):
    See run_evaluate_and_filter.sh for the full pipeline.
"""

import argparse
import hashlib
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

try:
    import orjson as json

    def json_loads(s):
        return json.loads(s)

    def json_dumps(obj):
        return json.dumps(obj).decode("utf-8")

except ImportError:
    import json

    def json_loads(s):
        return json.loads(s)

    def json_dumps(obj):
        return json.dumps(obj, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate and annotate RL training data for priority sampling"
    )

    # I/O
    parser.add_argument(
        "--generated_parquet", type=str, required=True,
        help="Path to generated parquet (output of main_generation_server.py, with 'responses' column)"
    )
    parser.add_argument(
        "--input_parquet", type=str, default=None,
        help="Path to original RL train parquet (for reference only, never modified). "
             "If set, its extra_info will be merged into the new output. "
             "If not set, uses generated_parquet as the base."
    )
    parser.add_argument(
        "--output_parquet", type=str, required=True,
        help="Path to output parquet (annotated, optionally filtered)"
    )

    # Reward function
    parser.add_argument("--reward_fn_path", type=str, required=True, help="Path to reward function file")
    parser.add_argument("--reward_fn_name", type=str, default="compute_score", help="Name of the reward function")

    # Data columns
    parser.add_argument("--prompt_key", type=str, default="prompt", help="Column name for prompts")
    parser.add_argument("--response_key", type=str, default="responses", help="Column name for responses")
    parser.add_argument("--data_source_key", type=str, default="data_source", help="Column name for data source")
    parser.add_argument(
        "--reward_model_key", type=str, default="reward_model",
        help="Column name for reward model data (containing ground_truth)"
    )

    # Generation config (for overlong detection)
    parser.add_argument(
        "--max_response_length", type=int, default=4096,
        help="Max response tokens used during generation (for overlong detection)"
    )
    parser.add_argument(
        "--truncation_length", type=int, default=None,
        help="Token length threshold for overlong detection. Responses with token count >= this "
             "value are considered overlong. Defaults to max_response_length if not set."
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to model for tokenizer-based overlong detection. "
             "If not set, overlong detection is skipped."
    )
    parser.add_argument(
        "--rollout_model_name", type=str, default=None,
        help="Name of the rollout model. Saved into extra_info for tracking. "
             "If the name contains many '/', only the last 2 segments are kept."
    )

    # Pass-rate clipping
    parser.add_argument(
        "--safe_low_bound", type=float, default=0.125,
        help="Lower bound for clipping pass_rate. raw_pass_rate is preserved before clipping."
    )
    parser.add_argument(
        "--safe_upper_bound", type=float, default=0.875,
        help="Upper bound for clipping pass_rate. raw_pass_rate is preserved before clipping."
    )

    # Filtering thresholds (set to None to disable)
    parser.add_argument(
        "--filter_pass_rate_min", type=float, default=None,
        help="Min pass_rate for filtering (exclusive). E.g., 0.0 means pass_rate > 0"
    )
    parser.add_argument(
        "--filter_pass_rate_max", type=float, default=None,
        help="Max pass_rate for filtering (exclusive). E.g., 1.0 means pass_rate < 1"
    )
    parser.add_argument(
        "--filter_overlong_ratio_max", type=float, default=None,
        help="Max overlong_ratio for filtering (inclusive). E.g., 0.05"
    )

    # Ray
    parser.add_argument("--num_cpus", type=int, default=None, help="Number of CPUs for Ray (None = all)")

    args = parser.parse_args()

    # Default truncation_length to max_response_length
    if args.truncation_length is None:
        args.truncation_length = args.max_response_length

    # Shorten rollout_model_name: keep only last 2 segments if too many '/'
    if args.rollout_model_name:
        parts = args.rollout_model_name.strip("/").split("/")
        if len(parts) > 2:
            args.rollout_model_name = "/".join(parts[-2:])

    return args


def generate_prompt_uid(prompt) -> str:
    """Generate a deterministic unique ID for a prompt based on its content.

    Uses stdlib json with canonical serialization (no extra spaces, sorted keys)
    to ensure consistent hashing regardless of whether orjson is installed.
    """
    import json as _json
    if isinstance(prompt, (list, np.ndarray)):
        prompt_str = _json.dumps(
            list(prompt) if isinstance(prompt, np.ndarray) else prompt,
            ensure_ascii=False, separators=(',', ':'), sort_keys=True,
        )
    elif isinstance(prompt, str):
        prompt_str = prompt
    else:
        prompt_str = str(prompt)
    return hashlib.md5(prompt_str.encode("utf-8")).hexdigest()[:16]


@ray.remote
def evaluate_single_item(
    reward_fn_path: str,
    reward_fn_name: str,
    data_source: str,
    response_lst: list,
    ground_truth,
    max_response_length: int,
):
    """Ray remote task: evaluate all responses for a single prompt.

    Mirrors the pattern in main_eval.py but computes richer per-prompt metrics
    (pass_rate, overlong_ratio) instead of just avg score.

    Args:
        reward_fn_path: Path to the reward function module.
        reward_fn_name: Name of the reward function in the module.
        data_source: Data source identifier for this prompt.
        response_lst: List of N response strings.
        ground_truth: Ground truth for reward evaluation.
        max_response_length: Max tokens used during generation (for overlong detection).

    Returns:
        dict with pass_rate, raw_pass_rate, overlong_ratio, avg_score, scores.
    """
    from verl.utils.import_utils import load_extern_object

    reward_fn = load_extern_object(module_path=reward_fn_path, object_name=reward_fn_name)

    scores = []
    overlong_count = 0

    for resp in response_lst:
        resp_text = str(resp) if not isinstance(resp, str) else resp

        # Heuristic overlong detection: if the response is very close to
        # max_response_length in token count, it likely hit the limit.
        # Since we only have text here, we use a character-based heuristic
        # or check for the finish_reason if available.
        # For text-only: a response that ends abruptly (no boxed answer, no
        # closing tag) after being very long is likely overlong.
        # A more reliable approach: check if the response was truncated by
        # looking at token count from the generation output.
        # For now, we'll use a simple heuristic based on the tokenizer later,
        # or the caller can pass token counts.
        # NOTE: The generation server outputs text only, so we rely on
        # character length as a rough proxy. A better approach would be to
        # save finish_reason in the generated parquet.

        # Compute reward score
        try:
            result = reward_fn(
                data_source=data_source,
                solution_str=resp_text,
                ground_truth=ground_truth,
            )
            if isinstance(result, dict):
                score = float(result.get("score", 0.0))
            elif isinstance(result, (int, float, bool)):
                score = float(result)
            else:
                score = float(result[0]) if result else 0.0
        except Exception as e:
            print(f"[WARN] Reward evaluation failed for data_source={data_source}: {e}")
            score = 0.0

        scores.append(score)

    n = len(response_lst)
    pass_count = sum(1 for s in scores if s > 0)
    pass_rate = pass_count / n if n > 0 else 0.0
    raw_pass_rate = pass_rate
    overlong_ratio = overlong_count / n if n > 0 else 0.0

    return {
        "data_source": data_source,
        "pass_rate": pass_rate,
        "raw_pass_rate": raw_pass_rate,
        "overlong_ratio": overlong_ratio,
        "avg_score": float(np.mean(scores)) if scores else 0.0,
        "scores": scores,
    }


@ray.remote
def evaluate_single_item_with_tokenizer(
    reward_fn_path: str,
    reward_fn_name: str,
    data_source: str,
    response_lst: list,
    ground_truth,
    max_response_length: int,
    model_path: str,
):
    """Ray remote task: evaluate responses with tokenizer-based overlong detection.

    Same as evaluate_single_item but also loads the tokenizer to count tokens
    for accurate overlong detection.
    """
    from verl.utils.import_utils import load_extern_object

    reward_fn = load_extern_object(module_path=reward_fn_path, object_name=reward_fn_name)

    # Load tokenizer for accurate token counting
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        tokenizer = None

    scores = []
    overlong_count = 0

    for resp in response_lst:
        resp_text = str(resp) if not isinstance(resp, str) else resp

        # Accurate overlong detection via tokenizer
        if tokenizer is not None:
            try:
                token_ids = tokenizer.encode(resp_text, add_special_tokens=False)
                if len(token_ids) >= max_response_length:
                    overlong_count += 1
            except Exception:
                pass  # fallback: don't count as overlong
        # If no tokenizer, overlong_count stays 0 (conservative)

        # Compute reward score
        try:
            result = reward_fn(
                data_source=data_source,
                solution_str=resp_text,
                ground_truth=ground_truth,
            )
            if isinstance(result, dict):
                score = float(result.get("score", 0.0))
            elif isinstance(result, (int, float, bool)):
                score = float(result)
            else:
                score = float(result[0]) if result else 0.0
        except Exception as e:
            print(f"[WARN] Reward evaluation failed for data_source={data_source}: {e}")
            score = 0.0

        scores.append(score)

    n = len(response_lst)
    pass_count = sum(1 for s in scores if s > 0)
    pass_rate = pass_count / n if n > 0 else 0.0
    raw_pass_rate = pass_rate
    overlong_ratio = overlong_count / n if n > 0 else 0.0

    return {
        "data_source": data_source,
        "pass_rate": pass_rate,
        "raw_pass_rate": raw_pass_rate,
        "overlong_ratio": overlong_ratio,
        "avg_score": float(np.mean(scores)) if scores else 0.0,
        "scores": scores,
    }


def print_difficulty_distribution(pass_rates: list[float], total: int, label: str = ""):
    """Print difficulty distribution based on raw pass rates."""
    extremely_hard = sum(1 for p in pass_rates if p == 0.0)
    hard = sum(1 for p in pass_rates if 0.0 < p <= 0.2)
    medium = sum(1 for p in pass_rates if 0.2 < p < 0.8)
    easy = sum(1 for p in pass_rates if 0.8 <= p < 1.0)
    extremely_easy = sum(1 for p in pass_rates if p == 1.0)

    prefix = f"  {label} " if label else "  "
    print(f"\n{prefix}Difficulty Distribution (N={total}):")
    print(f"    Extremely Hard (p=0):   {extremely_hard:>6} ({extremely_hard / total * 100:.1f}%)")
    print(f"    Hard (0<p<=0.2):        {hard:>6} ({hard / total * 100:.1f}%)")
    print(f"    Medium (0.2<p<0.8):     {medium:>6} ({medium / total * 100:.1f}%)")
    print(f"    Easy (0.8<=p<1):        {easy:>6} ({easy / total * 100:.1f}%)")
    print(f"    Extremely Easy (p=1):   {extremely_easy:>6} ({extremely_easy / total * 100:.1f}%)")


def print_summary(eval_results, dataset, data_source_key, has_data_source):
    """Print evaluation summary statistics."""
    total = len(eval_results)
    all_raw_pass_rates = [r["raw_pass_rate"] for r in eval_results]
    all_pass_rates = [r["pass_rate"] for r in eval_results]
    all_overlong_ratios = [r["overlong_ratio"] for r in eval_results]
    all_avg_scores = [r["avg_score"] for r in eval_results]

    # Per data-source statistics (with "overall" virtual data source)
    ds_metrics = defaultdict(lambda: {"pass_rates": [], "overlong_ratios": [], "avg_scores": []})
    for i in range(total):
        ds = dataset[data_source_key].iloc[i] if has_data_source else "default"
        ds_metrics[ds]["pass_rates"].append(eval_results[i]["pass_rate"])
        ds_metrics[ds]["overlong_ratios"].append(eval_results[i]["overlong_ratio"])
        ds_metrics[ds]["avg_scores"].append(eval_results[i]["avg_score"])

    ds_metrics["overall"]["pass_rates"] = all_pass_rates
    ds_metrics["overall"]["overlong_ratios"] = all_overlong_ratios
    ds_metrics["overall"]["avg_scores"] = all_avg_scores

    print("\n" + "=" * 80)
    print(f"{'Data Source':<30} {'Count':>6} {'Avg@N':>8} {'Pass Rate':>10} {'Overlong%':>10}")
    print("-" * 80)

    ds_names = sorted([k for k in ds_metrics if k != "overall"])
    for i, ds_name in enumerate(ds_names):
        m = ds_metrics[ds_name]
        if i < 10:
            print(
                f"{ds_name:<30} {len(m['pass_rates']):>6} "
                f"{np.mean(m['avg_scores']):>8.4f} "
                f"{np.mean(m['pass_rates']):>10.4f} "
                f"{np.mean(m['overlong_ratios']):>10.4f}"
            )
        elif i == 10:
            print(f"  ... ({len(ds_names) - 10} more data sources truncated)")

    m = ds_metrics["overall"]
    print("-" * 80)
    print(
        f"{'overall':<30} {len(m['pass_rates']):>6} "
        f"{np.mean(m['avg_scores']):>8.4f} "
        f"{np.mean(m['pass_rates']):>10.4f} "
        f"{np.mean(m['overlong_ratios']):>10.4f}"
    )
    print("=" * 80)

    print(f"\n  Overall Raw Pass Rate:   {np.mean(all_raw_pass_rates):.4f}")
    print(f"  Overall Pass Rate:       {np.mean(all_pass_rates):.4f}  (after clipping)")

    # Difficulty distribution based on raw_pass_rate
    print_difficulty_distribution(all_raw_pass_rates, total)


def main():
    args = parse_args()

    print("=" * 80)
    print("Evaluate & Annotate RL Training Data for Priority Sampling")
    print("=" * 80)
    print(f"  Generated parquet: {args.generated_parquet}")
    print(f"  Input parquet:     {args.input_parquet or '(same as generated)'}")
    print(f"  Output parquet:    {args.output_parquet}")
    print(f"  Reward fn:         {args.reward_fn_path}::{args.reward_fn_name}")
    print(f"  Max resp length:   {args.max_response_length}")
    print(f"  Truncation length: {args.truncation_length}")
    print(f"  Model path:        {args.model_path or '(none)'}")
    print(f"  Rollout model:     {args.rollout_model_name or '(none)'}")
    print(f"  Safe low bound:    {args.safe_low_bound}")
    print(f"  Safe upper bound:  {args.safe_upper_bound}")
    print()

    # ---- Step 1: Load generated parquet (with responses) ----
    print("[Step 1/4] Loading generated parquet...")
    t_step1 = time.time()
    gen_dataset = pd.read_parquet(args.generated_parquet)
    total = len(gen_dataset)
    print(f"  Loaded {total} samples from generated parquet")
    print(f"  Columns: {list(gen_dataset.columns)}")

    if args.response_key not in gen_dataset.columns:
        print(f"[ERROR] Response column '{args.response_key}' not found. "
              f"Available: {list(gen_dataset.columns)}")
        sys.exit(1)

    # Determine n_samples from the first row
    first_responses = gen_dataset[args.response_key].iloc[0]
    if isinstance(first_responses, (list, np.ndarray)):
        n_samples = len(first_responses)
    else:
        n_samples = 1
    print(f"  N samples per prompt: {n_samples}")

    # Check data columns
    has_data_source = args.data_source_key in gen_dataset.columns
    has_reward_model = args.reward_model_key in gen_dataset.columns

    if not has_data_source:
        print(f"  [WARN] Column '{args.data_source_key}' not found, using 'default' as data_source")
    if not has_reward_model:
        print(f"  [WARN] Column '{args.reward_model_key}' not found, ground_truth will be None")

    t_step1_elapsed = time.time() - t_step1
    print(f"  Step 1 completed in {t_step1_elapsed:.1f}s")

    # ---- Step 2: Initialize Ray and evaluate responses ----
    print("[Step 2/4] Initializing Ray and evaluating responses...")
    if not ray.is_initialized():
        ray_init_kwargs = {}
        if args.num_cpus is not None:
            ray_init_kwargs["num_cpus"] = args.num_cpus
        ray.init(**ray_init_kwargs)

    t0 = time.time()

    # Decide whether to use tokenizer-based overlong detection
    use_tokenizer = args.model_path is not None
    if use_tokenizer:
        print(f"  Using tokenizer-based overlong detection (model_path={args.model_path}, "
              f"truncation_length={args.truncation_length})")
    else:
        print(f"  Tokenizer-based overlong detection disabled (no --model_path)")

    # Create Ray remote tasks (following main_eval.py pattern)
    remote_tasks = []
    task_indices = []
    for i in range(total):
        data_source = gen_dataset[args.data_source_key].iloc[i] if has_data_source else "default"
        response_lst = gen_dataset[args.response_key].iloc[i]
        if isinstance(response_lst, np.ndarray):
            response_lst = response_lst.tolist()
        elif not isinstance(response_lst, list):
            response_lst = [response_lst]

        # Extract ground_truth from reward_model column
        ground_truth = None
        if has_reward_model:
            reward_data = gen_dataset[args.reward_model_key].iloc[i]
            if isinstance(reward_data, dict):
                ground_truth = reward_data.get("ground_truth", reward_data)
            else:
                ground_truth = reward_data

        if use_tokenizer:
            task = evaluate_single_item_with_tokenizer.remote(
                reward_fn_path=args.reward_fn_path,
                reward_fn_name=args.reward_fn_name,
                data_source=data_source,
                response_lst=response_lst,
                ground_truth=ground_truth,
                max_response_length=args.truncation_length,
                model_path=args.model_path,
            )
        else:
            task = evaluate_single_item.remote(
                reward_fn_path=args.reward_fn_path,
                reward_fn_name=args.reward_fn_name,
                data_source=data_source,
                response_lst=response_lst,
                ground_truth=ground_truth,
                max_response_length=args.max_response_length,
            )
        remote_tasks.append(task)
        task_indices.append(i)

    # Collect results using ray.wait (following main_eval.py pattern)
    eval_results = [None] * total
    pending_tasks = list(remote_tasks)
    pending_indices = list(task_indices)

    # Build a mapping from ObjectRef -> index for result collection
    ref_to_idx = {ref: idx for ref, idx in zip(pending_tasks, pending_indices)}

    with tqdm(total=total, desc="Evaluating") as pbar:
        while len(pending_tasks) > 0:
            done_ids, pending_tasks = ray.wait(pending_tasks)
            for result_id in done_ids:
                result = ray.get(result_id)
                idx = ref_to_idx[result_id]
                eval_results[idx] = result
                pbar.update(1)

    eval_time = time.time() - t0
    print(f"  Evaluation completed in {eval_time:.1f}s ({total} prompts, {total * n_samples} responses)")

    # ---- Step 2.5: Clip pass_rate using safe bounds ----
    clipped_count = 0
    for r in eval_results:
        raw_pr = r["raw_pass_rate"]
        clipped_pr = max(args.safe_low_bound, min(args.safe_upper_bound, raw_pr))
        if clipped_pr != raw_pr:
            clipped_count += 1
        r["pass_rate"] = clipped_pr
    print(f"  Clipped pass_rate to [{args.safe_low_bound}, {args.safe_upper_bound}]: "
          f"{clipped_count}/{total} prompts affected")

    # ---- Step 3: Print summary statistics ----
    print("[Step 3/4] Computing summary statistics...")
    t_step3 = time.time()
    print_summary(eval_results, gen_dataset, args.data_source_key, has_data_source)
    t_step3_elapsed = time.time() - t_step3
    print(f"  Step 3 completed in {t_step3_elapsed:.1f}s")

    # ---- Step 4: Create new annotated parquet and save ----
    print("\n[Step 4/4] Creating new annotated parquet...")
    t_step4 = time.time()

    # Always create a new output dataset from the generated parquet
    # (never modify the original train parquet)
    output_dataset = gen_dataset.copy()

    # If an original train parquet is provided, merge its existing extra_info
    original_extra_infos = [None] * total
    if args.input_parquet and args.input_parquet != args.generated_parquet:
        print(f"  Loading original train parquet for extra_info merge: {args.input_parquet}")
        original_dataset = pd.read_parquet(args.input_parquet)
        if len(original_dataset) != total:
            print(f"[WARN] Row count mismatch: generated={total}, original={len(original_dataset)}. "
                  f"Skipping extra_info merge from original.")
        elif "extra_info" in original_dataset.columns:
            for i in range(total):
                raw = original_dataset["extra_info"].iloc[i]
                if isinstance(raw, dict):
                    original_extra_infos[i] = raw
                elif isinstance(raw, str) and raw:
                    try:
                        original_extra_infos[i] = json_loads(raw)
                    except (ValueError, TypeError):
                        pass

    # Generate prompt_uid for each prompt
    prompts = gen_dataset[args.prompt_key].tolist()

    # Build extra_info column
    extra_info_list = []
    for i in range(total):
        # Start from existing extra_info (from original parquet or generated parquet)
        existing_extra_info = {}
        if original_extra_infos[i] is not None:
            existing_extra_info = original_extra_infos[i]
        elif "extra_info" in gen_dataset.columns:
            raw = gen_dataset["extra_info"].iloc[i]
            if isinstance(raw, dict):
                existing_extra_info = raw
            elif isinstance(raw, str) and raw:
                try:
                    existing_extra_info = json_loads(raw)
                except (ValueError, TypeError):
                    existing_extra_info = {}

        # Merge new metrics
        existing_extra_info["pass_rate"] = eval_results[i]["pass_rate"]
        existing_extra_info["raw_pass_rate"] = eval_results[i]["raw_pass_rate"]
        existing_extra_info["overlong_ratio"] = eval_results[i]["overlong_ratio"]
        existing_extra_info["prompt_uid"] = generate_prompt_uid(prompts[i])
        if args.truncation_length is not None:
            existing_extra_info["truncation_length"] = args.truncation_length
        if args.rollout_model_name is not None:
            existing_extra_info["rollout_model_name"] = args.rollout_model_name
        existing_extra_info["safe_low_bound"] = args.safe_low_bound
        existing_extra_info["safe_upper_bound"] = args.safe_upper_bound

        extra_info_list.append(existing_extra_info)

    output_dataset["extra_info"] = extra_info_list

    # Print distribution BEFORE filtering
    before_raw_pass_rates = [r["raw_pass_rate"] for r in eval_results]
    print_difficulty_distribution(before_raw_pass_rates, len(before_raw_pass_rates), label="[Before Filtering]")

    # Apply filtering if thresholds are specified
    filter_applied = False
    original_count = len(output_dataset)

    if (
        args.filter_pass_rate_min is not None
        or args.filter_pass_rate_max is not None
        or args.filter_overlong_ratio_max is not None
    ):
        mask = [True] * len(output_dataset)
        removed_by_pass_rate = 0
        removed_by_overlong = 0

        for i in range(len(output_dataset)):
            raw_pr = eval_results[i]["raw_pass_rate"]
            olr = eval_results[i]["overlong_ratio"]
            filtered_by_pr = False
            filtered_by_ol = False

            if args.filter_pass_rate_min is not None and raw_pr <= args.filter_pass_rate_min:
                filtered_by_pr = True
            if args.filter_pass_rate_max is not None and raw_pr >= args.filter_pass_rate_max:
                filtered_by_pr = True
            if args.filter_overlong_ratio_max is not None and olr > args.filter_overlong_ratio_max:
                filtered_by_ol = True

            if filtered_by_pr or filtered_by_ol:
                mask[i] = False
                if filtered_by_pr:
                    removed_by_pass_rate += 1
                if filtered_by_ol:
                    removed_by_overlong += 1

        output_dataset = output_dataset[mask].reset_index(drop=True)
        # Also filter eval_results for after-filtering distribution
        filtered_eval_results = [r for r, m in zip(eval_results, mask) if m]

        filter_desc = []
        if args.filter_pass_rate_min is not None:
            filter_desc.append(f"raw_pass_rate > {args.filter_pass_rate_min}")
        if args.filter_pass_rate_max is not None:
            filter_desc.append(f"raw_pass_rate < {args.filter_pass_rate_max}")
        if args.filter_overlong_ratio_max is not None:
            filter_desc.append(f"overlong_ratio <= {args.filter_overlong_ratio_max}")
        print(f"  Filter criteria: {' AND '.join(filter_desc)}")

        removed = original_count - len(output_dataset)
        print(f"  Filtered: {original_count} -> {len(output_dataset)} rows "
              f"({removed} removed, {removed / original_count * 100:.1f}%)")
        print(f"    - Removed by pass_rate:  {removed_by_pass_rate}")
        print(f"    - Removed by overlong:   {removed_by_overlong}")
        print(f"    (Note: a row can be removed by both reasons)")

        # Print distribution AFTER filtering
        after_raw_pass_rates = [r["raw_pass_rate"] for r in filtered_eval_results]
        if after_raw_pass_rates:
            print_difficulty_distribution(after_raw_pass_rates, len(after_raw_pass_rates), label="[After Filtering]")

        filter_applied = True

    # Remove the responses column if it exists (not needed in train parquet)
    if args.response_key in output_dataset.columns:
        output_dataset = output_dataset.drop(columns=[args.response_key])

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(args.output_parquet)), exist_ok=True)
    output_dataset.to_parquet(args.output_parquet, index=False)

    print(f"\n  Saved to: {args.output_parquet}")
    if not filter_applied:
        print(f"  Total: {len(output_dataset)} samples (no filtering applied)")

    t_step4_elapsed = time.time() - t_step4
    total_elapsed = time.time() - t_step1
    print(f"  Step 4 completed in {t_step4_elapsed:.1f}s")
    print(f"\n  Time summary:")
    print(f"    Step 1 (load):     {t_step1_elapsed:.1f}s")
    print(f"    Step 2 (evaluate): {eval_time:.1f}s")
    print(f"    Step 3 (summary):  {t_step3_elapsed:.1f}s")
    print(f"    Step 4 (save):     {t_step4_elapsed:.1f}s")
    print(f"    Total:             {total_elapsed:.1f}s")

    print("\nDone!")


if __name__ == "__main__":
    main()

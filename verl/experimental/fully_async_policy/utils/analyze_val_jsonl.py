#!/usr/bin/env python3
"""
Analyze validation JSONL dumps from fully_async val_only mode.

Reads per-sample JSONL files (produced by _dump_generations in _validate),
groups samples by prompt_uid, computes per-prompt metrics (pass_rate,
overlong_ratio, avg_score, difficulty distribution), prints a summary,
and optionally saves an annotated parquet with metrics in extra_info.

Usage:
    python analyze_val_jsonl.py \
        --val_data_dir /path/to/validation_data \
        --input_parquet /path/to/train.parquet \
        --output_parquet /path/to/output.parquet \
        --prompt_key prompt \
        --truncation_length 4096 \
        --filter_pass_rate_min 0.0 \
        --filter_pass_rate_max 1.0
"""

import argparse
import glob
import hashlib
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

try:
    import orjson

    def json_loads(s):
        return orjson.loads(s)

    def json_dumps(obj):
        return orjson.dumps(obj).decode("utf-8")

except ImportError:
    import json

    def json_loads(s):
        return json.loads(s)

    def json_dumps(obj):
        return json.dumps(obj, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze validation JSONL dumps and produce annotated parquet"
    )

    # I/O
    parser.add_argument(
        "--val_data_dir", type=str, default=None,
        help="Directory containing JSONL files from _dump_generations. "
             "Required unless --lightweight_json is provided."
    )
    parser.add_argument(
        "--lightweight_json", type=str, default=None,
        help="Path to a lightweight JSON file extracted from VAL_DATA_DIR. "
             "When provided, --val_data_dir is not needed. The file contains "
             "per-prompt rewards and response token lengths."
    )
    parser.add_argument(
        "--extract_only", action="store_true",
        help="Only extract a lightweight JSON from --val_data_dir and exit. "
             "Requires --val_data_dir and --model_path (for tokenization). "
             "Output path is controlled by --lightweight_json (auto-generated if not set)."
    )
    parser.add_argument(
        "--input_parquet", type=str, default=None,
        help="Path to original train parquet (for annotating extra_info). "
             "If not set, only prints summary without saving parquet."
    )
    parser.add_argument(
        "--output_parquet", type=str, default=None,
        help="Path to save annotated parquet. Required if --input_parquet is set."
    )

    # Data columns
    parser.add_argument("--prompt_key", type=str, default="prompt", help="Column name for prompts")

    # Overlong detection
    parser.add_argument(
        "--truncation_length", type=int, default=None,
        help="Token length threshold for overlong detection. Responses with token count >= this "
             "value are considered overlong. Also saved into extra_info for tracking. "
             "If not set, overlong detection is skipped."
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

    # Filtering thresholds
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
        help="Max overlong_ratio for filtering (inclusive). E.g., 0.05 means overlong_ratio <= 0.05"
    )

    args = parser.parse_args()

    # Validate: need either val_data_dir or lightweight_json (unless extract_only)
    if not args.extract_only and args.val_data_dir is None and args.lightweight_json is None:
        parser.error("Either --val_data_dir or --lightweight_json must be provided.")
    if args.extract_only and args.val_data_dir is None:
        parser.error("--extract_only requires --val_data_dir.")

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


def extract_lightweight_json(
    val_data_dir: str,
    output_path: str,
    tokenizer=None,
) -> str:
    """Extract a lightweight JSON file from VAL_DATA_DIR.

    The output file contains one JSON line per prompt_uid with:
        - prompt_uid: unique prompt identifier
        - rewards: list of reward scores for each sample
        - response_token_lengths: list of response token lengths (if tokenizer provided)

    Args:
        val_data_dir: Directory containing JSONL files.
        output_path: Path to save the lightweight JSON file.
        tokenizer: Optional tokenizer for computing token lengths.

    Returns:
        The output path.
    """
    samples = load_jsonl_files(val_data_dir)

    # Check if prompt_uid is available
    has_uid = any("prompt_uid" in s for s in samples[:10])
    if not has_uid:
        print("[WARN] No 'prompt_uid' field found. Generating from 'input' text hash.")
        for s in tqdm(samples, desc="Generating prompt_uid from input"):
            s["prompt_uid"] = generate_prompt_uid(s.get("input", ""))

    # Group by prompt_uid
    uid_groups: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        uid = sample.get("prompt_uid", "unknown")
        uid_groups[uid].append(sample)

    # Build lightweight records
    records = []
    for uid, group in tqdm(uid_groups.items(), desc="Building lightweight records"):
        rewards = [s.get("score", 0.0) for s in group]

        record = {
            "prompt_uid": uid,
            "rewards": rewards,
        }

        if tokenizer is not None:
            token_lengths = []
            for s in group:
                output_text = s.get("output", "")
                try:
                    token_ids = tokenizer.encode(output_text, add_special_tokens=False)
                    token_lengths.append(len(token_ids))
                except Exception:
                    token_lengths.append(len(output_text))
            record["response_token_lengths"] = token_lengths

        records.append(record)

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        for record in records:
            f.write(json_dumps(record) + "\n")

    # Print stats
    total_samples = sum(len(r["rewards"]) for r in records)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Extracted {len(records)} prompts ({total_samples} samples) -> {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    return output_path


def load_lightweight_json(filepath: str) -> list[dict]:
    """Load a lightweight JSON file and return list of per-prompt records.

    Each record has: prompt_uid, rewards,
    and optionally response_token_lengths.
    """
    records = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json_loads(line))
    print(f"  Loaded {len(records)} prompt records from {filepath}")
    return records


def compute_per_prompt_metrics_from_lightweight(
    records: list[dict],
    truncation_length: int | None = None,
) -> dict[str, dict]:
    """Compute per-prompt metrics from lightweight JSON records.

    Uses response_token_lengths for overlong detection if available.

    Args:
        records: List of lightweight records (prompt_uid, rewards, lengths).
        truncation_length: Token length threshold for overlong detection.

    Returns:
        Dict mapping prompt_uid -> metrics dict.
    """
    per_prompt = {}

    for record in tqdm(records, desc="Computing per-prompt metrics (lightweight)"):
        uid = record["prompt_uid"]
        rewards = record["rewards"]
        lengths = record.get("response_token_lengths", [])
        has_token_lengths = len(lengths) > 0

        scores = []
        overlong_count = 0
        for i, score in enumerate(rewards):
            if has_token_lengths and truncation_length is not None and i < len(lengths):
                if lengths[i] >= truncation_length:
                    score = 0.0
                    overlong_count += 1
            scores.append(score)

        n = len(scores)
        pass_count = sum(1 for s in scores if s > 0)
        pass_rate = pass_count / n if n > 0 else 0.0
        avg_score = float(np.mean(scores))
        overlong_ratio = overlong_count / n if n > 0 else 0.0
        avg_length = float(np.mean(lengths)) if lengths else 0.0

        per_prompt[uid] = {
            "pass_rate": pass_rate,
            "raw_pass_rate": pass_rate,
            "avg_score": avg_score,
            "n_samples": n,
            "overlong_ratio": overlong_ratio,
            "avg_length": avg_length,
        }

    return per_prompt


def load_jsonl_files(val_data_dir: str) -> list[dict]:
    """Load all JSONL files from the validation data directory."""
    jsonl_files = sorted(glob.glob(os.path.join(val_data_dir, "*.jsonl")))
    if not jsonl_files:
        print(f"[ERROR] No JSONL files found in {val_data_dir}")
        sys.exit(1)

    print(f"  Found {len(jsonl_files)} JSONL file(s): {[os.path.basename(f) for f in jsonl_files]}")

    all_samples = []
    for filepath in tqdm(jsonl_files, desc="Reading JSONL files"):
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_samples.append(json_loads(line))

    print(f"  Loaded {len(all_samples)} total samples")
    return all_samples


def compute_per_prompt_metrics(
    samples: list[dict],
    tokenizer=None,
    truncation_length: int | None = None,
) -> dict[str, dict]:
    """Group samples by prompt_uid and compute per-prompt metrics.

    Args:
        samples: List of JSONL records, each with at least 'score' and 'prompt_uid'.
        tokenizer: Optional tokenizer for overlong detection.
        truncation_length: Token length threshold for overlong detection.
            Responses with token count >= this value are considered overlong.
            If None, overlong detection is skipped even if tokenizer is provided.

    Returns:
        Dict mapping prompt_uid -> metrics dict with:
            pass_rate, avg_score, n_samples, overlong_ratio, data_source
    """
    uid_groups: dict[str, list[dict]] = defaultdict(list)

    for sample in tqdm(samples, desc="Grouping samples by prompt_uid"):
        uid = sample.get("prompt_uid", "unknown")
        uid_groups[uid].append(sample)

    per_prompt = {}
    
    # Function to process a single prompt group
    def process_single_group(uid, group):
        # First check for overlong responses and set their score to 0
        scores = []
        overlong_count = 0
        
        for s in group:
            score = s.get("score", 0.0)
            
            # Check if this sample is overlong (if tokenizer and truncation_length are provided)
            if tokenizer is not None and truncation_length is not None:
                output_text = s.get("output", "")
                try:
                    token_ids = tokenizer.encode(output_text, add_special_tokens=False)
                    if len(token_ids) >= truncation_length:
                        # Overlong response: set score to 0
                        score = 0.0
                        overlong_count += 1
                except Exception:
                    # If tokenization fails, treat as normal sample
                    pass
            
            scores.append(score)
        
        n = len(scores)
        pass_count = sum(1 for s in scores if s > 0)
        pass_rate = pass_count / n if n > 0 else 0.0
        avg_score = float(np.mean(scores))
        overlong_ratio = overlong_count / n if n > 0 else 0.0

        # Extract acc if available (from reward_extra_info)
        acc_values = [s.get("acc", None) for s in group]
        acc_values = [a for a in acc_values if a is not None]
        acc = float(np.mean(acc_values)) if acc_values else None

        # Compute average response length (token count if tokenizer available, else char count)
        if tokenizer is not None:
            response_lengths = []
            for s in group:
                output_text = s.get("output", "")
                try:
                    token_ids = tokenizer.encode(output_text, add_special_tokens=False)
                    response_lengths.append(len(token_ids))
                except Exception:
                    response_lengths.append(len(output_text))
        else:
            response_lengths = [len(s.get("output", "")) for s in group]
        avg_length = float(np.mean(response_lengths)) if response_lengths else 0.0

        result = {
            "pass_rate": pass_rate,
            "raw_pass_rate": pass_rate,  # will be preserved; pass_rate may be clipped later
            "avg_score": avg_score,
            "n_samples": n,
            "overlong_ratio": overlong_ratio,
            "avg_length": avg_length,
        }
        if acc is not None:
            result["acc"] = acc
        
        return uid, result

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) * 4)) as executor:
        # Submit all tasks
        future_to_uid = {
            executor.submit(process_single_group, uid, group): uid 
            for uid, group in uid_groups.items()
        }
        
        # Process results as they complete
        for future in tqdm(
            as_completed(future_to_uid), 
            desc="Computing per-prompt metrics", 
            total=len(uid_groups)
        ):
            uid, result = future.result()
            per_prompt[uid] = result

    return per_prompt


def print_difficulty_distribution(
    metrics_list: list[dict],
    total: int,
    label: str = "",
    pass_rate_key: str = "raw_pass_rate",
):
    """Print difficulty distribution based on raw pass rates, with avg length and overlong ratio per group.

    Args:
        metrics_list: List of per-prompt metric dicts (must contain pass_rate_key,
            and optionally 'avg_length' and 'overlong_ratio').
        total: Total number of prompts (used for percentage calculation).
        label: Optional label prefix for the output.
        pass_rate_key: Key to use for pass rate grouping (default: 'raw_pass_rate').
    """
    # Define difficulty buckets: (name, condition)
    buckets = [
        ("Extremely Hard (p=0)",   lambda p: p == 0.0),
        ("Hard (0<p<=0.2)",        lambda p: 0.0 < p <= 0.2),
        ("Medium (0.2<p<0.8)",     lambda p: 0.2 < p < 0.8),
        ("Easy (0.8<=p<1)",        lambda p: 0.8 <= p < 1.0),
        ("Extremely Easy (p=1)",   lambda p: p == 1.0),
    ]

    prefix = f"  {label} " if label else "  "
    print(f"\n{prefix}Difficulty Distribution (N={total}):")
    print(f"    {'Category':<25} {'Count':>6}  {'Pct':>6}  {'AvgLen':>8}  {'OverlongR':>10}")
    print(f"    {'-'*25} {'-'*6}  {'-'*6}  {'-'*8}  {'-'*10}")

    for name, cond in buckets:
        group = [m for m in metrics_list if cond(m.get(pass_rate_key, m.get("pass_rate", 0.0)))]
        count = len(group)
        pct = count / total * 100 if total > 0 else 0.0

        # Compute avg_length for this group
        lengths = [m["avg_length"] for m in group if "avg_length" in m]
        avg_len_str = f"{np.mean(lengths):.1f}" if lengths else "N/A"

        # Compute avg overlong_ratio for this group
        ol_ratios = [m["overlong_ratio"] for m in group if "overlong_ratio" in m]
        ol_str = f"{np.mean(ol_ratios):.4f}" if ol_ratios else "N/A"

        print(f"    {name:<25} {count:>6}  {pct:>5.1f}%  {avg_len_str:>8}  {ol_str:>10}")


def print_summary(per_prompt: dict[str, dict]):
    """Print evaluation summary statistics."""
    total = len(per_prompt)
    if total == 0:
        print("[WARN] No prompts to summarize")
        return

    all_metrics = list(per_prompt.values())
    all_raw_pass_rates = [m["raw_pass_rate"] for m in all_metrics]
    all_pass_rates = [m["pass_rate"] for m in all_metrics]
    all_avg_scores = [m["avg_score"] for m in all_metrics]
    all_overlong_ratios = [m["overlong_ratio"] for m in all_metrics]
    all_avg_lengths = [m["avg_length"] for m in all_metrics if "avg_length" in m]

    print("\n" + "=" * 70)
    print(f"  Evaluation Summary ({total} prompts)")
    print("-" * 70)
    print(f"  Overall Avg Score:       {np.mean(all_avg_scores):.4f}")
    print(f"  Overall Raw Pass Rate:   {np.mean(all_raw_pass_rates):.4f}")
    print(f"  Overall Pass Rate:       {np.mean(all_pass_rates):.4f}  (after clipping)")
    if all_avg_lengths:
        print(f"  Overall Avg Length:      {np.mean(all_avg_lengths):.1f}")
    if any(r > 0 for r in all_overlong_ratios):
        print(f"  Overall Overlong Rate:   {np.mean(all_overlong_ratios):.4f}")

    print_difficulty_distribution(all_metrics, total)
    print("=" * 70)


def save_annotated_parquet(
    per_prompt: dict[str, dict],
    input_parquet: str,
    output_parquet: str,
    prompt_key: str,
    filter_pass_rate_min: float | None = None,
    filter_pass_rate_max: float | None = None,
    filter_overlong_ratio_max: float | None = None,
    truncation_length: int | None = None,
    rollout_model_name: str | None = None,
    safe_low_bound: float = 0.125,
    safe_upper_bound: float = 0.875,
):
    """Save annotated parquet with per-prompt metrics in extra_info.

    Matches prompts from the input parquet to per_prompt metrics using
    prompt_uid generated from the prompt content.
    """
    import pandas as pd

    print(f"\n  Loading input parquet: {input_parquet}")
    
    # Check if fastparquet is available
    fastparquet_available = False
    try:
        import fastparquet
        fastparquet_available = True
    except ImportError:
        print("  [INFO] fastparquet not available, will use pyarrow")
    
    # Try different engines and options to handle nested data
    dataset = None
    attempts = [
        ('fastparquet', {}) if fastparquet_available else None,
        ('pyarrow', {'use_threads': False}),
        ('pyarrow', {'columns': ['prompt', 'extra_info']}),  # Only read necessary columns
        ('pyarrow', {}),  # Default pyarrow
    ]
    
    for attempt in attempts:
        if attempt is None:
            continue
        engine, kwargs = attempt
        try:
            print(f"  Trying engine={engine} with options: {kwargs}")
            dataset = pd.read_parquet(input_parquet, engine=engine, **kwargs)
            print(f"  Successfully loaded with {engine}")
            break
        except Exception as e:
            print(f"  [WARN] {engine} failed: {e}")
            continue
    
    if dataset is None:
        raise RuntimeError(f"Failed to read parquet file {input_parquet} with any engine")
    
    total = len(dataset)
    print(f"  Loaded {total} rows")

    annotated_count = 0
    extra_info_list = []

    for i in tqdm(range(total), desc="Annotating rows"):
        prompt = dataset[prompt_key].iloc[i]
        uid = generate_prompt_uid(prompt)

        # Load existing extra_info
        existing = {}
        if "extra_info" in dataset.columns:
            raw = dataset["extra_info"].iloc[i]
            if isinstance(raw, dict):
                existing = raw
            elif isinstance(raw, str) and raw:
                try:
                    existing = json_loads(raw)
                except (ValueError, TypeError):
                    existing = {}

        existing["prompt_uid"] = uid
        if truncation_length is not None:
            existing["truncation_length"] = truncation_length
        if rollout_model_name is not None:
            existing["rollout_model_name"] = rollout_model_name
        existing["safe_low_bound"] = safe_low_bound
        existing["safe_upper_bound"] = safe_upper_bound

        if uid in per_prompt:
            m = per_prompt[uid]
            existing["pass_rate"] = m["pass_rate"]
            existing["raw_pass_rate"] = m["raw_pass_rate"]
            existing["avg_score"] = m["avg_score"]
            existing["n_samples"] = m["n_samples"]
            existing["overlong_ratio"] = m["overlong_ratio"]
            if "avg_length" in m:
                existing["avg_length"] = m["avg_length"]
            if "acc" in m:
                existing["acc"] = m["acc"]
            annotated_count += 1

        extra_info_list.append(existing)

    dataset["extra_info"] = extra_info_list

    # Print distribution BEFORE filtering
    before_metrics = [ei for ei in extra_info_list if "pass_rate" in ei]
    if before_metrics:
        print_difficulty_distribution(before_metrics, len(before_metrics), label="[Before Filtering]")

    # Apply filtering
    original_count = len(dataset)
    has_filter = (
        filter_pass_rate_min is not None
        or filter_pass_rate_max is not None
        or filter_overlong_ratio_max is not None
    )
    if has_filter:
        mask = [True] * len(dataset)
        removed_by_pass_rate = 0
        removed_by_overlong = 0

        for i in tqdm(range(len(dataset)), desc="Filtering rows"):
            ei = extra_info_list[i]
            pr = ei.get("raw_pass_rate", ei.get("pass_rate", None))
            olr = ei.get("overlong_ratio", None)
            filtered_by_pr = False
            filtered_by_ol = False
            if pr is not None:
                if filter_pass_rate_min is not None and pr <= filter_pass_rate_min:
                    filtered_by_pr = True
                if filter_pass_rate_max is not None and pr >= filter_pass_rate_max:
                    filtered_by_pr = True
            if olr is not None:
                if filter_overlong_ratio_max is not None and olr > filter_overlong_ratio_max:
                    filtered_by_ol = True

            if filtered_by_pr or filtered_by_ol:
                mask[i] = False
                if filtered_by_pr:
                    removed_by_pass_rate += 1
                if filtered_by_ol:
                    removed_by_overlong += 1

        dataset = dataset[mask].reset_index(drop=True)
        # Also filter extra_info_list for the after-filtering distribution
        filtered_extra_info_list = [ei for ei, m in zip(extra_info_list, mask) if m]
        removed = original_count - len(dataset)
        filter_desc = []
        if filter_pass_rate_min is not None:
            filter_desc.append(f"raw_pass_rate > {filter_pass_rate_min}")
        if filter_pass_rate_max is not None:
            filter_desc.append(f"raw_pass_rate < {filter_pass_rate_max}")
        if filter_overlong_ratio_max is not None:
            filter_desc.append(f"overlong_ratio <= {filter_overlong_ratio_max}")
        print(f"  Filter criteria: {' AND '.join(filter_desc)}")
        print(f"  Filtered: {original_count} -> {len(dataset)} rows "
              f"({removed} removed, {removed / original_count * 100:.1f}%)")
        print(f"    - Removed by pass_rate:  {removed_by_pass_rate}")
        print(f"    - Removed by overlong:   {removed_by_overlong}")
        print(f"    (Note: a row can be removed by both reasons)")

        # Print distribution AFTER filtering
        after_metrics = [ei for ei in filtered_extra_info_list if "pass_rate" in ei]
        if after_metrics:
            print_difficulty_distribution(after_metrics, len(after_metrics), label="[After Filtering]")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_parquet)), exist_ok=True)
    dataset.to_parquet(output_parquet, index=False)
    print(f"  Saved annotated parquet: {output_parquet} "
          f"({len(dataset)} rows, {annotated_count} annotated)")


def main():
    args = parse_args()

    print("=" * 70)
    print("  Analyze Validation JSONL Dumps")
    print("=" * 70)
    print(f"  Val data dir:         {args.val_data_dir or '(none)'}")
    print(f"  Lightweight JSON:     {args.lightweight_json or '(none)'}")
    print(f"  Extract only:         {args.extract_only}")
    print(f"  Input parquet:        {args.input_parquet or '(none)'}")
    print(f"  Output parquet:       {args.output_parquet or '(none)'}")
    print(f"  Truncation length:    {args.truncation_length or '(none)'}")
    print(f"  Rollout model name:   {args.rollout_model_name or '(none)'}")
    print(f"  Safe low bound:       {args.safe_low_bound}")
    print(f"  Safe upper bound:     {args.safe_upper_bound}")
    print()

    # ---- Extract-only mode: dump lightweight JSON and exit ----
    if args.extract_only:
        # Load tokenizer for token length computation
        tokenizer = None
        if args.model_path:
            print(f"  Loading tokenizer from {args.model_path}...")
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
                print(f"  Tokenizer loaded successfully")
            except Exception as e:
                print(f"  [WARN] Failed to load tokenizer: {e}. Token lengths will not be computed.")

        # Auto-generate output path if not specified
        lw_output = args.lightweight_json
        if lw_output is None:
            lw_output = os.path.join(args.val_data_dir, "lightweight_metrics.jsonl")

        print(f"[Extract Only] Extracting lightweight JSON from {args.val_data_dir}...")
        extract_lightweight_json(
            val_data_dir=args.val_data_dir,
            output_path=lw_output,
            tokenizer=tokenizer,
        )
        print(f"\nDone! Lightweight JSON saved to: {lw_output}")
        return

    # ---- Normal analysis mode ----
    use_lightweight = args.lightweight_json is not None

    if use_lightweight:
        # Load from lightweight JSON (no tokenizer needed)
        print("[Step 1/3] Loading lightweight JSON...")
        lw_records = load_lightweight_json(args.lightweight_json)

        print("[Step 2/3] Computing per-prompt metrics (from lightweight JSON)...")
        per_prompt = compute_per_prompt_metrics_from_lightweight(
            lw_records,
            truncation_length=args.truncation_length,
        )
        print(f"  Computed metrics for {len(per_prompt)} unique prompts")
    else:
        # Original path: load from VAL_DATA_DIR
        print("[Step 1/3] Loading JSONL files...")
        samples = load_jsonl_files(args.val_data_dir)

        # Check if prompt_uid is available
        has_uid = any("prompt_uid" in s for s in samples[:10])
        if not has_uid:
            print("[WARN] No 'prompt_uid' field found in JSONL samples. "
                  "Grouping by 'input' text hash instead.")
            for s in tqdm(samples, desc="Generating prompt_uid from input"):
                s["prompt_uid"] = generate_prompt_uid(s.get("input", ""))

        # Load tokenizer for overlong detection (optional)
        tokenizer = None
        if args.model_path:
            print(f"  Loading tokenizer from {args.model_path} for overlong detection...")
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
                print(f"  Tokenizer loaded successfully")
            except Exception as e:
                print(f"  [WARN] Failed to load tokenizer: {e}. Overlong detection disabled.")

        print("[Step 2/3] Computing per-prompt metrics...")
        per_prompt = compute_per_prompt_metrics(
            samples,
            tokenizer=tokenizer,
            truncation_length=args.truncation_length,
        )
        print(f"  Computed metrics for {len(per_prompt)} unique prompts")

    # Clip pass_rate using safe bounds; raw_pass_rate is preserved
    clipped_count = 0
    for uid, m in per_prompt.items():
        raw_pr = m["raw_pass_rate"]
        clipped_pr = max(args.safe_low_bound, min(args.safe_upper_bound, raw_pr))
        if clipped_pr != raw_pr:
            clipped_count += 1
        m["pass_rate"] = clipped_pr
    print(f"  Clipped pass_rate to [{args.safe_low_bound}, {args.safe_upper_bound}]: "
          f"{clipped_count}/{len(per_prompt)} prompts affected")

    # Step 4: Print summary
    print_summary(per_prompt)

    # Step 5: Save annotated parquet (optional)
    if args.input_parquet and args.output_parquet:
        print("[Step 3/3] Saving annotated parquet...")
        save_annotated_parquet(
            per_prompt=per_prompt,
            input_parquet=args.input_parquet,
            output_parquet=args.output_parquet,
            prompt_key=args.prompt_key,
            filter_pass_rate_min=args.filter_pass_rate_min,
            filter_pass_rate_max=args.filter_pass_rate_max,
            filter_overlong_ratio_max=args.filter_overlong_ratio_max,
            truncation_length=args.truncation_length,
            rollout_model_name=args.rollout_model_name,
            safe_low_bound=args.safe_low_bound,
            safe_upper_bound=args.safe_upper_bound,
        )
    else:
        print("[Step 3/3] Skipping parquet save (no --input_parquet / --output_parquet)")

    print("\nDone!")


if __name__ == "__main__":
    main()

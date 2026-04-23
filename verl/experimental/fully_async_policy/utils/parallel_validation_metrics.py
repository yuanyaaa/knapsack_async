"""
Parallelized version of process_validation_metrics from metric_utils.py.

Uses ProcessPoolExecutor to parallelize bootstrap_metric computations across
(data_source, uid, var_name) combinations, which are completely independent.
This significantly speeds up validation metric processing when there are many
data sources and large K (samples per uid).
"""

import logging
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Any, Callable

import numpy as np

from verl.trainer.ppo.metric_utils import bootstrap_metric, calc_maj_val

logger = logging.getLogger(__name__)


def _compute_bootstrap_for_var(
    var_vals: list,
    pred_vals: list | None,
    ns: list[int],
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    """
    Compute all bootstrap metrics (best/worst/maj) for a single variable.

    This function is designed to be called in a worker process. It computes
    mean, std, best@N, worst@N, and maj@N metrics for one (uid, var_name) pair.

    Args:
        var_vals: List of variable values for this uid.
        pred_vals: List of prediction values (for majority voting), or None.
        ns: List of subset sizes for bootstrap sampling.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping metric names to values.
    """
    n_resps = len(var_vals)
    has_pred = pred_vals is not None

    metric = {f"mean@{n_resps}": float(np.mean(var_vals))}

    if n_resps <= 1:
        return metric

    metric[f"std@{n_resps}"] = float(np.std(var_vals))

    reduce_fns_best_worst = [np.max, np.min]

    for n in ns:
        # Compute best/worst metrics
        (bon_mean, bon_std), (won_mean, won_std) = bootstrap_metric(
            data=var_vals,
            subset_size=n,
            reduce_fns=reduce_fns_best_worst,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        metric[f"best@{n}/mean"] = bon_mean
        metric[f"best@{n}/std"] = bon_std
        metric[f"worst@{n}/mean"] = won_mean
        metric[f"worst@{n}/std"] = won_std

        # Compute maj metrics
        if has_pred:
            vote_data = [
                {"val": val, "pred": pred}
                for val, pred in zip(var_vals, pred_vals, strict=True)
            ]
            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                data=vote_data,
                subset_size=n,
                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            metric[f"maj@{n}/mean"] = maj_n_mean
            metric[f"maj@{n}/std"] = maj_n_std

    return metric


def _gen_ns(n_resps: int) -> list[int]:
    """Generate list of subset sizes for bootstrap sampling."""
    if n_resps <= 1:
        return []
    ns = []
    n = 2
    while n < n_resps:
        ns.append(n)
        n *= 2
    ns.append(n_resps)
    return ns


def _worker_task(args: tuple) -> tuple[str, str, str, dict[str, float]]:
    """
    Worker function for ProcessPoolExecutor.

    Args:
        args: Tuple of (data_source, uid, var_name, var_vals, pred_vals, ns, n_bootstrap, seed)

    Returns:
        Tuple of (data_source, uid, var_name, metric_dict)
    """
    data_source, uid, var_name, var_vals, pred_vals, ns, n_bootstrap, seed = args
    metric = _compute_bootstrap_for_var(var_vals, pred_vals, ns, n_bootstrap, seed)
    return (data_source, uid, var_name, metric)


def process_validation_metrics_parallel(
    data_sources: list[str],
    sample_uids: list[str],
    infos_dict: dict[str, list[Any]],
    seed: int = 42,
    max_workers: int | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Parallelized version of process_validation_metrics.

    Uses ProcessPoolExecutor to parallelize bootstrap computations across
    (data_source, uid, var_name) combinations. Falls back to sequential
    processing if the number of tasks is small.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_uids: List of sample uids corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.
        max_workers: Maximum number of worker processes. Defaults to None (auto).

    Returns:
        Same structure as process_validation_metrics:
        {data_source: {variable_name: {metric_name: value}}}
    """
    start_time = time.time()

    # Step 1: Group metrics by data source, prompt and variable (same as original)
    data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        uid = sample_uids[sample_idx]
        var2vals = data_src2uid2var2vals[data_source][uid]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    n_bootstrap = 1000

    # Step 2: Build task list for parallel execution
    ns_cache = {}
    tasks = []

    for data_source, uid2var2vals in data_src2uid2var2vals.items():
        for uid, var2vals in uid2var2vals.items():
            pred_vals = var2vals.get("pred")
            has_pred = pred_vals is not None

            for var_name, var_vals in var2vals.items():
                # Skip empty or string values
                if not var_vals or isinstance(var_vals[0], str):
                    continue

                n_resps = len(var_vals)
                if n_resps not in ns_cache:
                    ns_cache[n_resps] = _gen_ns(n_resps)
                ns = ns_cache[n_resps]

                tasks.append((
                    data_source,
                    uid,
                    var_name,
                    var_vals,
                    pred_vals if has_pred else None,
                    ns,
                    n_bootstrap,
                    seed,
                ))

    group_time = time.time() - start_time

    # Step 3: Execute tasks
    # Use parallel execution only when there are enough tasks to justify the overhead
    PARALLEL_THRESHOLD = 4
    data_src2uid2var2metric = {}

    if len(tasks) >= PARALLEL_THRESHOLD:
        # Determine worker count: use at most cpu_count/2 or number of tasks
        import os
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            max_workers = min(len(tasks), max(1, cpu_count // 2))

        logger.info(
            f"[parallel_validation_metrics] Running {len(tasks)} bootstrap tasks "
            f"with {max_workers} workers"
        )

        compute_start = time.time()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_worker_task, task): i for i, task in enumerate(tasks)}
            for future in as_completed(futures):
                data_source, uid, var_name, metric = future.result()
                uid_dict = data_src2uid2var2metric.setdefault(data_source, {})
                var_dict = uid_dict.setdefault(uid, {})
                var_dict[var_name] = metric
        compute_time = time.time() - compute_start
    else:
        # Sequential fallback for small task counts
        compute_start = time.time()
        for task in tasks:
            data_source, uid, var_name, metric = _worker_task(task)
            uid_dict = data_src2uid2var2metric.setdefault(data_source, {})
            var_dict = uid_dict.setdefault(uid, {})
            var_dict[var_name] = metric
        compute_time = time.time() - compute_start

    # Step 4: Aggregate metrics across uids (same as original)
    agg_start = time.time()
    data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, uid2var2metric in data_src2uid2var2metric.items():
        for uid, var2metric in uid2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
        for var_name, metric2uid_vals in var2metric2uid_vals.items():
            for metric_name, uid_vals in metric2uid_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(uid_vals)
    agg_time = time.time() - agg_start

    total_time = time.time() - start_time
    logger.info(
        f"[parallel_validation_metrics] Done in {total_time:.2f}s "
        f"(group={group_time:.2f}s, compute={compute_time:.2f}s, agg={agg_time:.2f}s, "
        f"tasks={len(tasks)}, parallel={'yes' if len(tasks) >= PARALLEL_THRESHOLD else 'no'})"
    )

    return data_src2var2metric2val


def enhanced_val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
    """Compute validation metrics with overall averaging and response length stats.

    Enhanced version of RayPPOTrainer._val_metrics_update that adds:
    - val-core/overall: average of core metrics across all data sources
    - Per-data-source response_length/mean, max, min, std, clip_ratio

    Shared by both FullyAsyncTrainer and FullyAsyncRollouter.

    Args:
        data_sources: numpy array of data source names per sample.
        sample_uids: list of sample uids.
        reward_extra_infos_dict: dict mapping info keys to per-sample value lists.
        sample_turns: list of numpy arrays of turn counts.

    Returns:
        dict mapping metric names to values.
    """
    data_src2var2metric2val = process_validation_metrics_parallel(
        data_sources, sample_uids, reward_extra_infos_dict
    )
    metric_dict = {}
    core_metric_values = defaultdict(list)

    for data_source, var2metric2val in data_src2var2metric2val.items():
        core_var = "acc" if "acc" in var2metric2val else "reward"
        for var_name, metric2val in var2metric2val.items():
            n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
            for metric_name, metric_val in metric2val.items():
                if (
                    (var_name == core_var)
                    and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                    and (f"@{n_max}" in metric_name)
                ):
                    metric_sec = "val-core"
                    core_metric_values[(var_name, metric_name)].append(metric_val)
                else:
                    metric_sec = "val-aux"
                pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                metric_dict[pfx] = metric_val

    # Compute overall average across data sources for val-core metrics
    if len(data_src2var2metric2val) > 1:
        for (var_name, metric_name), vals in core_metric_values.items():
            metric_dict[f"val-core/overall/{var_name}/{metric_name}"] = float(np.mean(vals))

    # Compute per-data-source response_length stats
    resp_lengths = reward_extra_infos_dict.get("response_length", [])
    max_resp_lengths = reward_extra_infos_dict.get("max_response_length", [])
    if resp_lengths and max_resp_lengths:
        resp_lengths_arr = np.array(resp_lengths, dtype=np.float64)
        max_resp_lengths_arr = np.array(max_resp_lengths, dtype=np.float64)
        for ds in data_src2var2metric2val.keys():
            ds_mask = (data_sources == ds)
            if ds_mask.sum() == 0:
                continue
            ds_resp_len = resp_lengths_arr[ds_mask]
            ds_max_resp_len = max_resp_lengths_arr[ds_mask]
            metric_dict[f"val-aux/{ds}/response_length/mean"] = float(ds_resp_len.mean())
            metric_dict[f"val-aux/{ds}/response_length/max"] = float(ds_resp_len.max())
            metric_dict[f"val-aux/{ds}/response_length/min"] = float(ds_resp_len.min())
            metric_dict[f"val-aux/{ds}/response_length/std"] = float(ds_resp_len.std())
            metric_dict[f"val-aux/{ds}/response_length/clip_ratio"] = float(
                (ds_resp_len == ds_max_resp_len).mean()
            )

    if len(sample_turns) > 0:
        sample_turns = np.concatenate(sample_turns)
        metric_dict["val-aux/num_turns/min"] = sample_turns.min()
        metric_dict["val-aux/num_turns/max"] = sample_turns.max()
        metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

    return metric_dict

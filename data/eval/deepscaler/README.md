# Evaluation datasets

These are the seven evaluation sets used by the Qwen3-4B six-arm training matrix.

- `aime24.parquet`: 30 examples
- `aime25.parquet`: 30 examples
- `amc_fixed.parquet`: 83 examples
- `math_100.parquet`: 100 examples
- `minerva_fixed_100.parquet`: 100 examples
- `olympiad_bench_fixed_100.parquet`: 100 examples
- `gpqa_diamond_100.parquet`: 100 examples

They were copied without modification from:

```text
/apdcephfs_zwfy8/share_303922427/pencywang/project/data_parquet/deepscaler
```

`scripts/submit_qwen3_4b_thinking_six_arms_zwfy8.sh` reads this directory by default.
Set `EVAL_DATA_ROOT` to override it.

# Qwen3-4B Thinking Six-Arm Matrix

## Data

All paths below are on the shared `zwfy8` filesystem.

- Original DAPO prompts (17,398 unique prompts):
  `/apdcephfs_zwfy8/share_303922427/pencywang/project/data_parquet/dapo/train.parquet`
- Qwen3-4B thinking inference:
  `/apdcephfs_zwfy8/share_303922427/pencywang/data/qwen3_4b_thinking/q3_4b_thinking_official_dapo17k_n8_32k_0904_1514_full.parquet`
- Per-prompt `n=8` rewards and response lengths:
  `/apdcephfs_zwfy8/share_303922427/pencywang/data/qwen3_4b_thinking/q3_4b_thinking_official_dapo17k_n8_32k_0904_1514_samples.jsonl`
- Training data (11,001 prompts):
  `/apdcephfs_zwfy8/share_303922427/pencywang/data/qwen3_4b_thinking/dapo17k_qwen3_4b_thinking_n8_gen32k_trunc16k_train_clip001_099_drop_allcorrect_overlong0125.parquet`

The training data was derived from the 32K thinking generations by recomputing rewards at
the 16K training boundary. Responses with at least 16,384 tokens count as incorrect.
All-correct prompts and prompts with more than one of eight responses reaching 16K are removed.
All-wrong prompts are retained, and the offline prior is clipped to `[0.01, 0.99]`.

## Arms

Each arm uses one 8-GPU rollout node and one 8-GPU trainer node.

1. `baseline`: shuffle-sequential sampling, fixed `n=8`
2. `haspos1`: stop after at least one positive, maximum `n=14`
3. `rej_fixedn8`: fixed `n=8` with rejection sampling
4. `adaptn14`: shuffle-sequential sampling, medium-focus adaptive `n=4..14`
5. `lowermid_adaptn14`: lower-mid sampling, medium-focus adaptive `n=4..14`
6. `rej_lowermid_adaptn14`: lower-mid adaptive `n=4..14` with rejection sampling

The six task entrypoints are:

- `scripts/run_qwen3_4b_thinking_baseline_pair_zwfy8.sh`
- `scripts/run_qwen3_4b_thinking_haspos1_pair_zwfy8.sh`
- `scripts/run_qwen3_4b_thinking_rej_fixedn8_pair_zwfy8.sh`
- `scripts/run_qwen3_4b_thinking_adaptn14_pair_zwfy8.sh`
- `scripts/run_qwen3_4b_thinking_lowermid_adaptn14_pair_zwfy8.sh`
- `scripts/run_qwen3_4b_thinking_rej_lowermid_adaptn14_pair_zwfy8.sh`

## Launch

Run one entrypoint on each task's head node:

```bash
cd /apdcephfs_zwfy8/share_303922427/pencywang/project/knapsack_async
read -rsp "Worker password: " PW; echo

export HEAD_IP=<8-GPU-head-IP>
export WORKER_IP=<8-GPU-worker-IP>
export SSH_PASSWORD="$PW"
export USER_SWANLAB_API_KEY=<SwanLab-key>

bash scripts/run_qwen3_4b_thinking_<arm>_pair_zwfy8.sh
```

Each entrypoint starts a fresh two-node Ray cluster after verifying that all 16 GPUs are idle.
Do not assign the same node to more than one arm.

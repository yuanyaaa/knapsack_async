#!/usr/bin/env bash
set -euo pipefail

# ==================== Evaluate & Filter via Fully Async val_only ====================
# This script evaluates an RL training parquet using the fully_async_main.py
# val_only mode. It leverages the async rollout manager for generation + reward
# evaluation, then dumps per-sample JSONL files to validation_data_dir.
#
# Key trick: set val_files=train_files so the existing _validate() pipeline
# runs generation + reward on the train data. Combined with val_only=True,
# the trainer exits after validation.
#
# After JSONL dump, run analyze_val_jsonl.py to compute per-prompt metrics
# (pass_rate, overlong_ratio, difficulty distribution) and optionally filter
# and save an annotated parquet.
# ====================================================================

# ==================== Paths ====================
MODEL_PATH=${MODEL_PATH:-"/apdcephfs/share_302503006/ziniuli/models/Qwen/Qwen2.5-Math-1.5B"}
MODEL_NAME=${MODEL_NAME:-"qwen2.5_math_7b"}  # Short name for the model (e.g. "Qwen-1.5B"), used in output filename
TRAIN_FILE=${TRAIN_FILE:-"/apdcephfs/share_302503006/ziniuli/processed_dataset/Eurus-2-RL-Data/code_train.parquet"}
REWARD_FN_PATH=${REWARD_FN_PATH:-"/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/utils/reward_fn.py"}
REWARD_FN_NAME=${REWARD_FN_NAME:-"compute_score"}

# ==================== GPU Config ====================
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
TP_SIZE=${TP_SIZE:-2}                       # Tensor parallel size
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}          # GPU memory utilization

# ==================== Sampling Config ====================
N_SAMPLES=${N_SAMPLES:-8}                                    # Number of samples per prompt (8 or 16)
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-$((1024 * 2))}          # Max prompt tokens
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-$((1024 * 2))}     # Max response tokens for generation
TRUNCATION_LENGTH=${TRUNCATION_LENGTH:-${MAX_RESPONSE_LENGTH}} # Token length threshold for overlong detection
TEMPERATURE=${TEMPERATURE:-1.0}                               # Sampling temperature
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}                                            # -1 for vLLM disabled
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-$((NNODES * NGPUS_PER_NODE / TP_SIZE * 512 / N_SAMPLES))}  # val batch size per iteration

# ==================== Rollout Engine ====================
rollout_mode="async"
rollout_name="vllm"  # "vllm" or "sglang"
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

# ==================== Data Columns ====================
PROMPT_KEY=${PROMPT_KEY:-"prompt"}

# ==================== Filtering (for analyze_val_jsonl.py) ====================
FILTER_PASS_RATE_MIN=${FILTER_PASS_RATE_MIN:-0}      # pass_rate > 0 (filter out all-wrong)
FILTER_PASS_RATE_MAX=${FILTER_PASS_RATE_MAX:-1}      # pass_rate < 1 (filter out all-correct)
FILTER_OVERLONG_MAX=${FILTER_OVERLONG_MAX:-0.05}     # overlong_ratio <= 0.05

# ==================== Lightweight JSON ====================
EXTRACT_LIGHTWEIGHT=${EXTRACT_LIGHTWEIGHT:-true}     # Extract lightweight JSON after JSONL dump
LIGHTWEIGHT_JSON=${LIGHTWEIGHT_JSON:-""}             # Path to lightweight JSON (auto-generated if empty)

# ==================== Logging ====================
LOG_DIR="/apdcephfs/share_302503006/ziniuli/project/verl-main/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
LOG_FILE="${LOG_DIR}/evaluate_and_filter_fully_async_${TIMESTAMP}.log"

# Redirect all stdout/stderr to both console and log file
exec > >(tee -a "${LOG_FILE}") 2>&1

# ==================== Auto-generate output paths ====================
VAL_DATA_DIR=${VAL_DATA_DIR:-""}
OUTPUT_PARQUET=${OUTPUT_PARQUET:-""}

# Build suffixes:
#   VAL_DATA_DIR uses MAX_RESPONSE_LENGTH (generation config)
#   OUTPUT_PARQUET uses TRUNCATION_LENGTH (filtering config)
VAL_SUFFIX="n${N_SAMPLES}_maxlen${MAX_RESPONSE_LENGTH}"
OUTPUT_SUFFIX="n${N_SAMPLES}_trunc${TRUNCATION_LENGTH}"
if [ -n "${MODEL_NAME}" ]; then
    VAL_SUFFIX="${MODEL_NAME}_${VAL_SUFFIX}"
    OUTPUT_SUFFIX="${MODEL_NAME}_${OUTPUT_SUFFIX}"
fi

if [ -z "${VAL_DATA_DIR}" ]; then
    INPUT_DIR=$(dirname "${TRAIN_FILE}")
    INPUT_BASENAME=$(basename "${TRAIN_FILE}" .parquet)
    VAL_DATA_DIR="${INPUT_DIR}/${INPUT_BASENAME}_evaluated_${VAL_SUFFIX}_validation_data"
fi
if [ -z "${OUTPUT_PARQUET}" ]; then
    INPUT_DIR=$(dirname "${TRAIN_FILE}")
    INPUT_BASENAME=$(basename "${TRAIN_FILE}" .parquet)
    OUTPUT_PARQUET="${INPUT_DIR}/${INPUT_BASENAME}_evaluated_${OUTPUT_SUFFIX}.parquet"
fi
if [ -z "${LIGHTWEIGHT_JSON}" ]; then
    INPUT_DIR=$(dirname "${TRAIN_FILE}")
    INPUT_BASENAME=$(basename "${TRAIN_FILE}" .parquet)
    LIGHTWEIGHT_JSON="${INPUT_DIR}/${INPUT_BASENAME}_evaluated_${VAL_SUFFIX}_lightweight.jsonl"
fi

echo "============================================================"
echo "  Evaluate & Filter (Fully Async val_only mode)"
echo "============================================================"
echo "  Train file:  ${TRAIN_FILE}"
echo "  Val data:    ${VAL_DATA_DIR}"
echo "  Output:      ${OUTPUT_PARQUET}"
echo "  Model:       ${MODEL_PATH}"
echo "  Reward:      ${REWARD_FN_PATH}::${REWARD_FN_NAME}"
echo "  Samples:     ${N_SAMPLES}"
echo "  Model name:  ${MODEL_NAME:-'(auto from MODEL_PATH)'}"
echo "  Trunc len:   ${TRUNCATION_LENGTH}"
echo "  Val batch:   ${VAL_BATCH_SIZE}"
echo "  Lightweight: ${LIGHTWEIGHT_JSON}"
echo "  Log file:    ${LOG_FILE}"
echo "============================================================"

# ==================== Step 1: Run fully_async val_only ====================
echo ""
echo "[Step 1/3] Running fully_async val_only (generation + reward)..."

python -m verl.experimental.fully_async_policy.fully_async_main \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TRAIN_FILE}" \
    data.prompt_key=${PROMPT_KEY} \
    data.truncation='left' \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.return_raw_chat=${return_raw_chat} \
    data.train_batch_size=0 \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 1)) \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.n=${N_SAMPLES} \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=${TOP_P} \
    actor_rollout_ref.rollout.top_k=${TOP_K} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.rollout.val_kwargs.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${TOP_K} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${N_SAMPLES} \
    reward.custom_reward_function.path="${REWARD_FN_PATH}" \
    reward.custom_reward_function.name=${REWARD_FN_NAME} \
    reward.reward_manager.name=naive \
    trainer.logger=['console'] \
    trainer.project_name=evaluate_and_filter \
    trainer.experiment_name="eval_n${N_SAMPLES}" \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.validation_data_dir="${VAL_DATA_DIR}" \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${NGPUS_PER_NODE}" \
    rollout.total_rollout_steps=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    algorithm.adv_estimator=grpo \
    critic.enable=False \
    async_training.use_trainer_do_validate=True \
    async_training.trigger_parameter_sync_step=1 \
    async_training.require_batches=1

echo "[Step 1/3] Validation JSONL dump complete: ${VAL_DATA_DIR}"

# ==================== Step 2: Extract lightweight JSON ====================
if [ "${EXTRACT_LIGHTWEIGHT}" = "true" ]; then
    echo ""
    echo "[Step 2/3] Extracting lightweight JSON from VAL_DATA_DIR..."

    EXTRACT_ARGS="--val_data_dir ${VAL_DATA_DIR} --lightweight_json ${LIGHTWEIGHT_JSON} --extract_only"
    if [ -n "${MODEL_PATH}" ]; then
        EXTRACT_ARGS="${EXTRACT_ARGS} --model_path ${MODEL_PATH}"
    fi

    python verl/experimental/fully_async_policy/utils/analyze_val_jsonl.py ${EXTRACT_ARGS}

    echo "[Step 2/3] Lightweight JSON saved: ${LIGHTWEIGHT_JSON}"
else
    echo ""
    echo "[Step 2/3] Skipping lightweight JSON extraction (EXTRACT_LIGHTWEIGHT=false)"
fi

# ==================== Step 3: Analyze and save annotated parquet ====================
echo ""
echo "[Step 3/3] Analyzing and saving annotated parquet..."

ANALYZE_ARGS=""
if [ -n "${FILTER_PASS_RATE_MIN}" ]; then
    ANALYZE_ARGS="${ANALYZE_ARGS} --filter_pass_rate_min ${FILTER_PASS_RATE_MIN}"
fi
if [ -n "${FILTER_PASS_RATE_MAX}" ]; then
    ANALYZE_ARGS="${ANALYZE_ARGS} --filter_pass_rate_max ${FILTER_PASS_RATE_MAX}"
fi
if [ -n "${FILTER_OVERLONG_MAX}" ]; then
    ANALYZE_ARGS="${ANALYZE_ARGS} --filter_overlong_ratio_max ${FILTER_OVERLONG_MAX}"
fi

# Determine rollout_model_name: use MODEL_NAME if set, otherwise fall back to MODEL_PATH
ROLLOUT_MODEL_NAME=${MODEL_NAME:-${MODEL_PATH}}

# Use lightweight JSON if available, otherwise fall back to val_data_dir
if [ "${EXTRACT_LIGHTWEIGHT}" = "true" ] && [ -f "${LIGHTWEIGHT_JSON}" ]; then
    echo "  Using lightweight JSON: ${LIGHTWEIGHT_JSON}"
    python verl/experimental/fully_async_policy/utils/analyze_val_jsonl.py \
        --lightweight_json "${LIGHTWEIGHT_JSON}" \
        --input_parquet "${TRAIN_FILE}" \
        --output_parquet "${OUTPUT_PARQUET}" \
        --prompt_key "${PROMPT_KEY}" \
        --truncation_length ${TRUNCATION_LENGTH} \
        --rollout_model_name "${ROLLOUT_MODEL_NAME}" \
        ${ANALYZE_ARGS}
else
    echo "  Using val_data_dir: ${VAL_DATA_DIR}"
    python verl/experimental/fully_async_policy/utils/analyze_val_jsonl.py \
        --val_data_dir "${VAL_DATA_DIR}" \
        --input_parquet "${TRAIN_FILE}" \
        --output_parquet "${OUTPUT_PARQUET}" \
        --prompt_key "${PROMPT_KEY}" \
        --truncation_length ${TRUNCATION_LENGTH} \
        --model_path "${MODEL_PATH}" \
        --rollout_model_name "${ROLLOUT_MODEL_NAME}" \
        ${ANALYZE_ARGS}
fi

echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "  Validation data:   ${VAL_DATA_DIR}"
if [ "${EXTRACT_LIGHTWEIGHT}" = "true" ] && [ -f "${LIGHTWEIGHT_JSON}" ]; then
echo "  Lightweight JSON:  ${LIGHTWEIGHT_JSON}"
fi
echo "  Annotated output:  ${OUTPUT_PARQUET}"
echo "  Log file:          ${LOG_FILE}"
echo "============================================================"

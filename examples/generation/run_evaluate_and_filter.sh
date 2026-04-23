#!/usr/bin/env bash
set -euo pipefail

# ==================== Evaluate & Filter RL Training Data ====================
# This script evaluates an RL training parquet with multi-sampling (avg@N)
# using the standard two-step pipeline:
#   Step 1: Generate N responses per prompt via main_generation_server.py
#   Step 2: Evaluate responses and annotate parquet via evaluate_and_filter.py
#
# The annotated output parquet has per-prompt metrics (pass_rate, overlong_ratio,
# prompt_uid) written into the extra_info column, ready for priority sampling.
# ====================================================================

# ==================== Paths ====================
MODEL_PATH=${MODEL_PATH:-"/apdcephfs/share_302503006/ziniuli/models/Qwen/Qwen2.5-Math-1.5B"}
MODEL_NAME=${MODEL_NAME:-"qwen2.5_1.5b"}  # Short name for the model, used in output filename
TRAIN_FILE=${TRAIN_FILE:-"/apdcephfs/share_302503006/ziniuli/processed_dataset/lighteval-MATH/train.parquet"}
REWARD_FN_PATH=${REWARD_FN_PATH:-"/apdcephfs/share_302503006/ziniuli/project/verl-main/verl/experimental/fully_async_policy/utils/reward_fn.py"}
REWARD_FN_NAME=${REWARD_FN_NAME:-"compute_score"}

# ==================== Sampling Config ====================
N_SAMPLES=${N_SAMPLES:-8}                                    # Number of samples per prompt (8 or 16)
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-$((1024 * 2))}     # Max response tokens
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-$((1024 * 2))}         # Max prompt tokens
TRUNCATION_LENGTH=${TRUNCATION_LENGTH:-${MAX_RESPONSE_LENGTH}} # Token length threshold for overlong detection
TEMPERATURE=${TEMPERATURE:-1.0}                               # Sampling temperature
TOP_P=${TOP_P:-1.0}

# ==================== GPU Config ====================
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
TP_SIZE=${TP_SIZE:-2}                       # Tensor parallel size
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}          # GPU memory utilization
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-$((NNODES * NGPUS_PER_NODE / TP_SIZE * 512 / N_SAMPLES))}  # prompts per batch for generation

# ==================== Data Columns ====================
PROMPT_KEY=${PROMPT_KEY:-"prompt"}

# ==================== Filtering ====================
FILTER_PASS_RATE_MIN=${FILTER_PASS_RATE_MIN:-0}      # pass_rate > 0 (filter out all-wrong)
FILTER_PASS_RATE_MAX=${FILTER_PASS_RATE_MAX:-1}      # pass_rate < 1 (filter out all-correct)
FILTER_OVERLONG_MAX=${FILTER_OVERLONG_MAX:-0.05}     # overlong_ratio <= 0.05

# ==================== Logging ====================
LOG_DIR="/apdcephfs/share_302503006/ziniuli/project/verl-main/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
LOG_FILE="${LOG_DIR}/evaluate_and_filter_${TIMESTAMP}.log"

# Redirect all stdout/stderr to both console and log file
exec > >(tee -a "${LOG_FILE}") 2>&1

# ==================== Auto-generate output paths ====================
INPUT_DIR=$(dirname "${TRAIN_FILE}")
INPUT_BASENAME=$(basename "${TRAIN_FILE}" .parquet)

# Build suffixes:
#   GENERATED_PARQUET uses MAX_RESPONSE_LENGTH (generation config)
#   OUTPUT_PARQUET uses TRUNCATION_LENGTH (filtering config)
GEN_SUFFIX="n${N_SAMPLES}_maxlen${MAX_RESPONSE_LENGTH}"
OUTPUT_SUFFIX="n${N_SAMPLES}_trunc${TRUNCATION_LENGTH}"
if [ -n "${MODEL_NAME}" ]; then
    GEN_SUFFIX="${MODEL_NAME}_${GEN_SUFFIX}"
    OUTPUT_SUFFIX="${MODEL_NAME}_${OUTPUT_SUFFIX}"
fi

GENERATED_PARQUET=${GENERATED_PARQUET:-"${INPUT_DIR}/${INPUT_BASENAME}_generated_${GEN_SUFFIX}.parquet"}
OUTPUT_PARQUET=${OUTPUT_PARQUET:-"${INPUT_DIR}/${INPUT_BASENAME}_evaluated_${OUTPUT_SUFFIX}.parquet"}

echo "============================================================"
echo "  Evaluate & Filter RL Training Data"
echo "============================================================"
echo "  Train file:       ${TRAIN_FILE}"
echo "  Generated output: ${GENERATED_PARQUET}"
echo "  Final output:     ${OUTPUT_PARQUET}"
echo "  Model:            ${MODEL_PATH}"
echo "  Model name:       ${MODEL_NAME:-'(auto from MODEL_PATH)'}"
echo "  Reward:           ${REWARD_FN_PATH}::${REWARD_FN_NAME}"
echo "  Samples:          ${N_SAMPLES}"
echo "  Trunc len:        ${TRUNCATION_LENGTH}"
echo "  Val batch size:   ${VAL_BATCH_SIZE}"
echo "  Log file:         ${LOG_FILE}"
echo "============================================================"

# ==================== Step 1: Generate responses ====================
echo ""
echo "[Step 1/2] Generating ${N_SAMPLES} responses per prompt..."
STEP1_START=${SECONDS}

export VLLM_USE_V1=1

python -m verl.trainer.main_generation_server \
    data.train_files="${TRAIN_FILE}" \
    data.prompt_key=${PROMPT_KEY} \
    +data.output_path="${GENERATED_PARQUET}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${N_SAMPLES} \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=${TOP_P} \
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    data.val_batch_size=${VAL_BATCH_SIZE}

STEP1_ELAPSED=$((SECONDS - STEP1_START))
echo "[Step 1/2] Generation complete: ${GENERATED_PARQUET} (elapsed: ${STEP1_ELAPSED}s)"

# ==================== Step 2: Evaluate and annotate ====================
echo ""
echo "[Step 2/2] Evaluating responses and annotating parquet..."
STEP2_START=${SECONDS}

FILTER_ARGS=""
if [ -n "${FILTER_PASS_RATE_MIN}" ]; then
    FILTER_ARGS="${FILTER_ARGS} --filter_pass_rate_min ${FILTER_PASS_RATE_MIN}"
fi
if [ -n "${FILTER_PASS_RATE_MAX}" ]; then
    FILTER_ARGS="${FILTER_ARGS} --filter_pass_rate_max ${FILTER_PASS_RATE_MAX}"
fi
if [ -n "${FILTER_OVERLONG_MAX}" ]; then
    FILTER_ARGS="${FILTER_ARGS} --filter_overlong_ratio_max ${FILTER_OVERLONG_MAX}"
fi

# Determine rollout_model_name: use MODEL_NAME if set, otherwise fall back to MODEL_PATH
ROLLOUT_MODEL_NAME=${MODEL_NAME:-${MODEL_PATH}}

python examples/generation/evaluate_and_filter.py \
    --generated_parquet "${GENERATED_PARQUET}" \
    --input_parquet "${TRAIN_FILE}" \
    --output_parquet "${OUTPUT_PARQUET}" \
    --reward_fn_path "${REWARD_FN_PATH}" \
    --reward_fn_name "${REWARD_FN_NAME}" \
    --max_response_length ${MAX_RESPONSE_LENGTH} \
    --truncation_length ${TRUNCATION_LENGTH} \
    --model_path "${MODEL_PATH}" \
    --rollout_model_name "${ROLLOUT_MODEL_NAME}" \
    --prompt_key "${PROMPT_KEY}" \
    ${FILTER_ARGS}

STEP2_ELAPSED=$((SECONDS - STEP2_START))
TOTAL_ELAPSED=${SECONDS}

echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "  Generated parquet: ${GENERATED_PARQUET}"
echo "  Annotated output:  ${OUTPUT_PARQUET}"
echo "  Log file:          ${LOG_FILE}"
echo "  Step 1 (generate): ${STEP1_ELAPSED}s"
echo "  Step 2 (evaluate): ${STEP2_ELAPSED}s"
echo "  Total elapsed:     ${TOTAL_ELAPSED}s"
echo "============================================================"

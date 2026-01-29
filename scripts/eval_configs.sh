#!/usr/bin/env bash

set -e

# ===== Environment Setup =====
export PYTHONPATH=.:$PYTHONPATH
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn

# ===== Default Arguments =====
NUM_GPU=1
LIMIT=""
OUTPUT_DIR="./eval_results_configs"
LLADA_MODEL="GSAI-ML/LLaDA-8B-Instruct"
DREAM_MODEL="Dream-org/Dream-v0-Instruct-7B"

# ===== Parse Arguments =====
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num_gpu)
      NUM_GPU="$2"; shift 2 ;;
    --limit)
      LIMIT="$2"; shift 2 ;;
    --output_dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --llada_model)
      LLADA_MODEL="$2"; shift 2 ;;
    --dream_model)
      DREAM_MODEL="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --num_gpu N         Number of GPUs to use (default: 1)"
      echo "  --limit N           Limit number of samples (default: full dataset)"
      echo "  --output_dir DIR    Output directory (default: ./eval_results_configs)"
      echo "  --llada_model PATH  LLaDA model path"
      echo "  --dream_model PATH  Dream model path"
      echo ""
      echo "Edit CONFIGS array in this script to define parameter combinations."
      exit 0
      ;;
    *)
      echo "Error: Unknown argument: $1"
      exit 1 ;;
  esac
done


CONFIGS=(
  "dream|humaneval|90.1|1|0.5|8"
  "dream|mmlu|90.1|1|0.5|8"
  "dream|hellaswag|90.1|1|0.5|8"
  "dream|gsm8k|90.1|1|0.5|8"
)
# ============================================================

mkdir -p "${OUTPUT_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

LIMIT_ARG=""
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARG="--limit ${LIMIT}"
  echo ">>> Running with --limit ${LIMIT}"
fi

echo ""
echo "============================================================"
echo "Custom Configuration Evaluation"
echo "============================================================"
echo "Timestamp: ${TIMESTAMP}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Number of GPUs: ${NUM_GPU}"
echo "Configurations to run: ${#CONFIGS[@]}"
echo "============================================================"
echo ""

# ===== Benchmark configs =====
get_llada_config() {
  local bench="$1"
  case "$bench" in
    gsm8k)
      TASK="gsm8k_cot_llada"; MAX_NEW_TOKENS=256; STEPS=256; BLOCK_SIZE=32; NUM_FEWSHOT=0
      EXTRA_ARGS=""; SUPPRESS_TOKENS="" ;;
    mmlu)
      TASK="mmlu_generative"; MAX_NEW_TOKENS=3; STEPS=3; BLOCK_SIZE=3; NUM_FEWSHOT=0
      EXTRA_ARGS=""; SUPPRESS_TOKENS="" ;;
    humaneval)
      TASK="humaneval_instruct"; MAX_NEW_TOKENS=512; STEPS=512; BLOCK_SIZE=32; NUM_FEWSHOT=0
      EXTRA_ARGS="--confirm_run_unsafe_code"; SUPPRESS_TOKENS="[126081]" ;;
    hellaswag)
      TASK="hellaswag_gen"; MAX_NEW_TOKENS=5; STEPS=5; BLOCK_SIZE=5; NUM_FEWSHOT=0
      EXTRA_ARGS=""; SUPPRESS_TOKENS="" ;;
  esac
}

get_dream_config() {
  local bench="$1"
  case "$bench" in
    gsm8k)
      TASK="gsm8k_cot"; MAX_NEW_TOKENS=256; STEPS=256; NUM_FEWSHOT=0
      ALG="entropy"; TEMPERATURE=0.1; TOP_P=0.9; EXTRA_ARGS="" ;;
    mmlu)
      TASK="mmlu_generative"; MAX_NEW_TOKENS=3; STEPS=3; NUM_FEWSHOT=0
      ALG="entropy"; TEMPERATURE=0.1; TOP_P=0.9; EXTRA_ARGS="" ;;
    humaneval)
      TASK="humaneval_instruct_dream"; MAX_NEW_TOKENS=512; STEPS=512; NUM_FEWSHOT=0
      ALG="entropy"; TEMPERATURE=0.1; TOP_P=0.9; EXTRA_ARGS="--confirm_run_unsafe_code" ;;
    hellaswag)
      TASK="hellaswag_gen"; MAX_NEW_TOKENS=5; STEPS=5; NUM_FEWSHOT=0
      ALG="entropy"; TEMPERATURE=0.1; TOP_P=0.9; EXTRA_ARGS="" ;;
  esac
}

# ===== Run functions =====
run_llada() {
  local bench="$1" max_t="$2" min_t="$3" geo="$4" dist="$5"
  get_llada_config "$bench"
  
  local tag="maxT${max_t}_minT${min_t}_geo${geo}_dist${dist}"
  tag="${tag//./p}"
  
  local eff_path="${OUTPUT_DIR}/llada_${bench}_efficiency_${tag}_${TIMESTAMP}.json"
  local eval_path="${OUTPUT_DIR}/llada_${bench}_lm_eval_${tag}_${TIMESTAMP}"
  
  echo ">>> [LLaDA] ${tag}"
  
  local suppress_arg=""
  [[ -n "${SUPPRESS_TOKENS}" ]] && suppress_arg=",suppress_tokens=${SUPPRESS_TOKENS}"
  
  accelerate launch --num_processes "${NUM_GPU}" \
    dllm/pipelines/llada/eval.py \
    --tasks "${TASK}" \
    --model llada \
    --apply_chat_template \
    --num_fewshot "${NUM_FEWSHOT}" \
    --model_args "pretrained=${LLADA_MODEL},max_new_tokens=${MAX_NEW_TOKENS},steps=${STEPS},block_size=${BLOCK_SIZE},cfg=0.0,dtype=bfloat16,max_threshold=${max_t},min_threshold=${min_t},geo_decay=${geo},max_distance=${dist},efficiency_output_path=${eff_path}${suppress_arg}" \
    --output_path "${eval_path}" \
    --log_samples \
    ${EXTRA_ARGS} \
    ${LIMIT_ARG}
  
  echo "    Done: ${eff_path}"
}

run_dream() {
  local bench="$1" max_t="$2" min_t="$3" geo="$4" dist="$5"
  get_dream_config "$bench"
  
  local tag="maxT${max_t}_minT${min_t}_geo${geo}_dist${dist}"
  tag="${tag//./p}"
  
  local eff_path="${OUTPUT_DIR}/dream_${bench}_efficiency_${tag}_${TIMESTAMP}.json"
  local eval_path="${OUTPUT_DIR}/dream_${bench}_lm_eval_${tag}_${TIMESTAMP}"
  
  echo ">>> [Dream] ${bench} ${tag}"
  
  accelerate launch --num_processes "${NUM_GPU}" \
    dllm/pipelines/dream/eval.py \
    --tasks "${TASK}" \
    --model dream \
    --apply_chat_template \
    --num_fewshot "${NUM_FEWSHOT}" \
    --model_args "pretrained=${DREAM_MODEL},max_new_tokens=${MAX_NEW_TOKENS},steps=${STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},alg=${ALG},dtype=bfloat16,add_bos_token=False,escape_until=False,max_threshold=${max_t},min_threshold=${min_t},geo_decay=${geo},max_distance=${dist},efficiency_output_path=${eff_path}" \
    --output_path "${eval_path}" \
    --log_samples \
    ${EXTRA_ARGS} \
    ${LIMIT_ARG}
  
  echo "    Done: ${eff_path}"
}

# ===== Main loop =====
total=${#CONFIGS[@]}
current=0

for config in "${CONFIGS[@]}"; do
  current=$((current + 1))
  echo ""
  echo "============================================================"
  echo "Running configuration ${current}/${total}: ${config}"
  echo "============================================================"
  
  IFS='|' read -r model bench max_t min_t geo dist <<< "$config"
  
  if [[ "$model" == "llada" ]]; then
    run_llada "$bench" "$max_t" "$min_t" "$geo" "$dist"
  elif [[ "$model" == "dream" ]]; then
    run_dream "$bench" "$max_t" "$min_t" "$geo" "$dist"
  else
    echo "Error: Unknown model '$model' in config: $config"
    exit 1
  fi
done

# ===== Summary =====
echo ""
echo "============================================================"
echo "All configurations complete!"
echo "============================================================"
echo ""
echo "Results saved to: ${OUTPUT_DIR}/"
echo ""

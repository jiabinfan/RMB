#!/bin/bash
#SBATCH --job-name=rmb_stage1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORK_ROOT="${RMB_WORK_ROOT:-${SCRATCH:-$REPO_ROOT/.artifacts}/rmb}"
VENV="${RMB_VENV:-$WORK_ROOT/venv}"
OUTPUT_ROOT="${RMB_OUTPUT_ROOT:-$WORK_ROOT/checkpoints}"

if [[ -n "${RMB_MODULES:-}" ]]; then
  read -r -a modules <<< "$RMB_MODULES"
  module --force purge
  module load "${modules[@]}"
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Missing RMB environment: $VENV" >&2
  echo "Submit scripts/setup_env.slurm first." >&2
  exit 1
fi
source "$VENV/bin/activate"

export HF_HOME="${HF_HOME:-$WORK_ROOT/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORK_ROOT/cache}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$OUTPUT_ROOT"

cd "$REPO_ROOT/reward_models"
python hsic_rms_train.py \
  --base_model "${BASE_MODEL:-google/gemma-2b-it}" \
  --dataset "${DATASET:-llm-blender/Unified-Feedback}" \
  --dataset_mode "${DATASET_MODE:-40K}" \
  --log_dir "$OUTPUT_ROOT" \
  --wandb_name "${WANDB_NAME:-rmb_hsic}" \
  --output_tag "${OUTPUT_TAG:-paper}" \
  --num_adapters "${NUM_ADAPTERS:-3}" \
  --diversity_type hsic \
  --diversity_lambda "${HSIC_LAMBDA:-0.1}" \
  --lora_r "${LORA_R:-32}" \
  --lora_alpha "${LORA_ALPHA:-64}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --per_device_train_batch_size "${MICRO_BATCH_SIZE:-2}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-2}" \
  --max_length "${MAX_LENGTH:-1024}" \
  --eval_tasks unified \
  --eval_size "${EVAL_SIZE:-1000}" \
  --evaluation_strategy no \
  --save_strategy no \
  --report_to "${REPORT_TO:-none}" \
  --random_seed "${SEED:-32}"

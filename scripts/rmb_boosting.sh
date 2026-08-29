#!/bin/bash
#SBATCH --job-name=rmb_stage2
#SBATCH --account=aip-lilimou
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/%x_%j.out

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
VENV="${RMB_VENV:-$SCRATCH/venvs/rmb}"
RESULT_ROOT="${RMB_RESULT_ROOT:-$SCRATCH/rmb/results}"
ADAPTER_CHECKPOINT="${ADAPTER_CHECKPOINT:-}"

module --force purge
module load StdEnv/2023 python/3.11.5 arrow/18.1.0

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Missing RMB environment: $VENV" >&2
  echo "Submit scripts/setup_env.slurm first." >&2
  exit 1
fi
if [[ -z "$ADAPTER_CHECKPOINT" ]]; then
  echo "Set ADAPTER_CHECKPOINT to a Stage-1 best_step_* directory." >&2
  exit 1
fi
source "$VENV/bin/activate"

export HF_HOME="${HF_HOME:-$SCRATCH/rmb/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH/rmb/cache}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$RESULT_ROOT"

cd "$REPO_ROOT/reward_models"
python run_booster_rmb.py \
  --base_model "${BASE_MODEL:-google/gemma-2b-it}" \
  --checkpoint_dir "$ADAPTER_CHECKPOINT" \
  --dataset "${DATASET:-llm-blender/Unified-Feedback}" \
  --eval_dataset "${EVAL_DATASET:-llm-blender/Unified-Feedback}" \
  --dataset_mode "${DATASET_MODE:-40K-heldout}" \
  --batch_size "${BATCH_SIZE:-8}" \
  --max_length "${MAX_LENGTH:-1024}" \
  --attn_implementation sdpa \
  --tree_method gpu_hist \
  --num_round "${NUM_ROUND:-256}" \
  --early_stopping "${EARLY_STOPPING:-30}" \
  --xgb_max_depth "${XGB_MAX_DEPTH:-5}" \
  --xgb_reg_lambda "${XGB_REG_LAMBDA:-0.1}" \
  --booster_out "${BOOSTER_OUT:-$RESULT_ROOT/rmb_booster.json}"

#!/bin/bash
#SBATCH --job-name=rmb_env
#SBATCH --account=aip-lilimou
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/%x_%j.out

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
VENV="${RMB_VENV:-$SCRATCH/venvs/rmb}"

module --force purge
module load StdEnv/2023 python/3.11.5 arrow/18.1.0

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH/rmb/cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH/rmb/pip-cache}"
mkdir -p "$(dirname "$VENV")" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"
if [[ ! -x "$VENV/bin/python" ]]; then
  virtualenv --no-download "$VENV"
fi
source "$VENV/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "$REPO_ROOT/requirements-hpc.txt"

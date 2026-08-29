#!/bin/bash
#SBATCH --job-name=rmb_env
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
WORK_ROOT="${RMB_WORK_ROOT:-${SCRATCH:-$REPO_ROOT/.artifacts}/rmb}"
VENV="${RMB_VENV:-$WORK_ROOT/venv}"

if [[ -n "${RMB_MODULES:-}" ]]; then
  read -r -a modules <<< "$RMB_MODULES"
  module --force purge
  module load "${modules[@]}"
fi

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORK_ROOT/cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORK_ROOT/pip-cache}"
mkdir -p "$(dirname "$VENV")" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"
if [[ ! -x "$VENV/bin/python" ]]; then
  python -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$REPO_ROOT/requirements-hpc.txt"

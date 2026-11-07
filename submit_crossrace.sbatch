#!/bin/bash
#SBATCH --account=compute2-ACCOUNT
#SBATCH --qos=compute2-ACCOUNT
#SBATCH --partition=condo-artsci
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
# Measured at ~12 minutes per transfer, so 20 takes about 4 hours. Eight hours
# leaves headroom; the script skips transfers that are already complete.
#SBATCH --time=08:00:00
#SBATCH --job-name=crossrace

# Cross-race transfer as a batch job, so it survives a dropped session.
#
#   ./run_crossrace.sh              all 20 transfers
#   ./run_crossrace.sh phcpsmw      just one
#
# --output and --error are set by the wrapper, not here: Slurm cannot parse a
# path containing a space in an #SBATCH directive.

set -euo pipefail

RQ="/path/to/project"
VENV="$RQ/env"

echo "=================================================="
echo "job        ${SLURM_JOB_ID:-interactive} on $(hostname)"
echo "args       ${*:---all}"
echo "started    $(date)"
echo "=================================================="

nvidia-smi || { echo "no GPU attached to this job" >&2; exit 1; }

export HF_HOME="$RQ/.hf-home"
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false

source "$VENV/bin/activate"
cd "$RQ/HSI_analysis"
python cross_race_predict.py "${@:---all}"

echo "=================================================="
echo "finished   $(date)"
echo "=================================================="

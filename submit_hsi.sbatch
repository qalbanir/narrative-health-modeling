#!/bin/bash
#SBATCH --account=compute2-ACCOUNT
#SBATCH --qos=compute2-ACCOUNT
#SBATCH --partition=condo-artsci
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
# A real run takes ~95 minutes (phcpsmc, 5 folds, H100). Four hours is generous
# headroom. Keep this tight: Slurm's backfill scheduler slots short jobs into
# gaps between long ones, so an inflated limit pushes the job far down the queue.
#SBATCH --time=04:00:00
#SBATCH --job-name=hsi

# Batch submission for one HSI fine-tuning script on RIS compute2 (Slurm).
#
# Submit through the wrapper, which handles the space in "project":
#
#   ./run_hsi.sh phcpsmc.py
#
# --output and --error are deliberately NOT #SBATCH directives: Slurm cannot
# parse a path containing a space in a directive line. The wrapper passes them
# as quoted command-line arguments instead.

set -euo pipefail

RQ="/path/to/project"
SCRIPTS="$RQ/HSI_analysis/roberta_scripts"
VENV="$RQ/env"

if [ $# -lt 1 ]; then
    echo "Usage: sbatch submit_hsi.sbatch <script.py>" >&2
    echo "Available:" >&2
    ls -1 "$SCRIPTS"/*.py 2>/dev/null | xargs -n1 basename >&2
    exit 1
fi

TARGET="$SCRIPTS/$1"
if [ ! -f "$TARGET" ]; then
    echo "No such script: $TARGET" >&2
    exit 1
fi

mkdir -p "$RQ/logs"

echo "=================================================="
echo "job        $SLURM_JOB_ID on $(hostname)"
echo "script     $1"
echo "started    $(date)"
echo "=================================================="

nvidia-smi || { echo "nvidia-smi failed -- no GPU attached to this job" >&2; exit 1; }

# Keep the HuggingFace cache inside project so roberta-large is downloaded once
# and reused, rather than re-fetched into a home directory with a quota.
export HF_HOME="$RQ/.hf-home"
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HUB_CACHE"

source "$VENV/bin/activate"
echo "python     $(which python)"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "$SCRIPTS"
python "$TARGET"

echo "=================================================="
echo "finished   $(date)"
echo "=================================================="

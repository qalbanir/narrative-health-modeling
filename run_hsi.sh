#!/bin/bash
# Submit an HSI fine-tuning job on RIS compute2.
#
#   ./run_hsi.sh phcpsmc.py            one job
#   ./run_hsi.sh phcpsmc.py mhcpsmc.py several
#   ./run_hsi.sh --list                what is available
#
# Exists because "project" contains a space, which Slurm cannot handle inside
# an #SBATCH directive. Here the paths are shell-quoted and passed to sbatch as
# arguments, which works.

set -euo pipefail

RQ="/path/to/project"
SCRIPTS="$RQ/HSI_analysis/roberta_scripts"
LOGS="$RQ/logs"
SBATCH_FILE="$RQ/HSI_analysis/submit_hsi.sbatch"

# Slurm emails when a job finishes or fails, so there is no need to watch the
# queue. Change this address, or set NOTIFY="" to turn notifications off.
NOTIFY="${HSI_NOTIFY_EMAIL:-[EMAIL]}"

if [ $# -lt 1 ] || [ "$1" = "--list" ]; then
    echo "Usage: ./run_hsi.sh <script.py> [more.py ...]"
    echo
    echo "Available scripts:"
    ls -1 "$SCRIPTS"/*.py 2>/dev/null | xargs -n1 basename | sed 's/^/  /'
    exit 0
fi

mkdir -p "$LOGS"

for script in "$@"; do
    if [ ! -f "$SCRIPTS/$script" ]; then
        echo "No such script: $script" >&2
        echo "Run './run_hsi.sh --list' to see the options." >&2
        exit 1
    fi

    name="${script%.py}"
    mail_args=()
    if [ -n "$NOTIFY" ]; then
        mail_args=(--mail-type=END,FAIL --mail-user="$NOTIFY")
    fi

    jobid=$(sbatch --parsable \
        --job-name="$name" \
        --output="$LOGS/%x_%j.out" \
        --error="$LOGS/%x_%j.err" \
        "${mail_args[@]}" \
        "$SBATCH_FILE" "$script")

    echo "submitted $script  ->  job $jobid"
    echo "  log: $LOGS/${name}_${jobid}.out"
done

echo
echo "Watch with:  squeue -u \$USER"

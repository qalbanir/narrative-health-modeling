#!/bin/bash
# Submit the cross-race transfer as a batch job.
#
#   ./run_crossrace.sh              all 20 transfers (~1 hour)
#   ./run_crossrace.sh phcpsmw      a single run
#
# Wraps sbatch so the log paths, which contain a space, are shell-quoted.

set -euo pipefail

RQ="/path/to/project"
LOGS="$RQ/logs"
SBATCH_FILE="$RQ/HSI_analysis/submit_crossrace.sbatch"
NOTIFY="${HSI_NOTIFY_EMAIL:-[EMAIL]}"

mkdir -p "$LOGS"

mail_args=()
if [ -n "$NOTIFY" ]; then
    mail_args=(--mail-type=END,FAIL --mail-user="$NOTIFY")
fi

jobid=$(sbatch --parsable \
    --job-name=crossrace \
    --output="$LOGS/%x_%j.out" \
    --error="$LOGS/%x_%j.err" \
    "${mail_args[@]}" \
    "$SBATCH_FILE" "$@")

echo "submitted cross-race transfer  ->  job $jobid"
echo "  args: ${*:---all}"
echo "  log:  $LOGS/crossrace_${jobid}.out"
echo
echo "Watch with:  squeue -u \$USER"

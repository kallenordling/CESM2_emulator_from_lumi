#!/bin/bash
# watch_eval_triggers.sh
#
# Polls eval_triggers/ for JSON files written by the trainer (inside container)
# and submits the corresponding sbatch eval job.
#
# Run outside the container, e.g. in a screen/tmux on a login node:
#   bash watch_eval_triggers.sh
#
# Or as a minimal SLURM job (no GPU needed):
#   sbatch --job-name=eval_watcher --partition=small --time=2-00:00:00 \
#          --ntasks=1 --cpus-per-task=1 --mem=256M \
#          --output=logs/eval_watcher_%j.out watch_eval_triggers.sh

# When run as a SLURM job, $0 points to /var/spool/slurmd/jobXXX/ so we
# use SLURM_SUBMIT_DIR instead. Fall back to dirname $0 for manual runs.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$(realpath "$0")")}"
TRIGGER_DIR="${PROJECT_DIR}/eval_triggers"
DONE_DIR="${TRIGGER_DIR}/done"
POLL_INTERVAL=60   # seconds between checks

mkdir -p "$TRIGGER_DIR" "$DONE_DIR"
echo "[watcher] Started. Watching ${TRIGGER_DIR} every ${POLL_INTERVAL}s"

while true; do
    for trigger in "${TRIGGER_DIR}"/eval_request_*.json; do
        # glob returns the pattern itself if no match
        [[ -f "$trigger" ]] || continue

        echo "[watcher] Found trigger: $trigger"

        # Parse fields with python (available on login node via module or system)
        CHECKPOINT=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['checkpoint'])" "$trigger" 2>/dev/null)
        OUTPUT_DIR=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['output_dir'])" "$trigger" 2>/dev/null)
        SBATCH_SCRIPT=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['sbatch_script'])" "$trigger" 2>/dev/null)
        LOG_DIR=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['log_dir'])" "$trigger" 2>/dev/null)
        EPOCH=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['epoch'])" "$trigger" 2>/dev/null)

        if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
            echo "[watcher] WARNING: checkpoint not found: $CHECKPOINT — skipping"
            mv "$trigger" "${DONE_DIR}/$(basename $trigger).skipped"
            continue
        fi

        mkdir -p "$OUTPUT_DIR"

        JOB_INFO=$(sbatch \
            --job-name="eval_ep$(printf '%04d' $EPOCH)" \
            --output="${LOG_DIR}/eval_aero_ep$(printf '%04d' $EPOCH)_%j.out" \
            --export="ALL,CHECKPOINT=${CHECKPOINT},OUTPUT_DIR=${OUTPUT_DIR}" \
            "$SBATCH_SCRIPT" 2>&1)

        echo "[watcher] sbatch result: $JOB_INFO"

        # Move trigger to done/ so we don't resubmit
        mv "$trigger" "${DONE_DIR}/$(basename $trigger).submitted"
    done

    sleep "$POLL_INTERVAL"
done

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

# ── Startup purge: drop stale non-production triggers ────────────────────────
# A restart (e.g. after a walltime kill) would otherwise drain whatever piled
# up while the watcher was down — including triggers from one-off A/B forks
# (run_gmeantest, run_tcrefulltest, …). Only auto-dispatch evals for the
# production run(s); move everything else to done/ so it can't fire.
#
# PROD_RUN may be a SPACE-SEPARATED list of run names — a trigger is kept if it
# matches ANY of them. This lets one shared watcher serve concurrent production
# runs cleanly (e.g. PROD_RUN="run_sensfix run_sensfix_b12") so a restart's
# startup-purge doesn't drop the other run's pending triggers. Override when a
# production save_name changes or when running an A/B alongside production.
PROD_RUN="${PROD_RUN:-run_slope-tcre}"
for trigger in "${TRIGGER_DIR}"/eval_request_*.json; do
    [[ -f "$trigger" ]] || continue
    keep=""
    for name in ${PROD_RUN//,/ }; do      # accept comma- or space-separated list
        grep -q "$name" "$trigger" && { keep=1; break; }
    done
    if [[ -z "$keep" ]]; then
        echo "[watcher] startup-purge: non-production trigger $(basename "$trigger") → done/"
        mv "$trigger" "${DONE_DIR}/$(basename "$trigger").purged"
    fi
done

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

        # Explicitly pass SINGULARITYENV_PYTHONPATH so that --export=ALL cannot
        # propagate a stale value from the training job's container environment.
        _EVAL_VENV=/projappl/project_462001328/venvs/diffesm_laif/lib/python3.12/site-packages
        JOB_INFO=$(sbatch \
            --job-name="eval_ep$(printf '%04d' $EPOCH)" \
            --output="${LOG_DIR}/eval_aero_ep$(printf '%04d' $EPOCH)_%j.out" \
            --export="ALL,CHECKPOINT=${CHECKPOINT},OUTPUT_DIR=${OUTPUT_DIR},SINGULARITYENV_PYTHONPATH=${_EVAL_VENV}" \
            "$SBATCH_SCRIPT" 2>&1)

        echo "[watcher] sbatch result: $JOB_INFO"

        # Move trigger to done/ so we don't resubmit
        mv "$trigger" "${DONE_DIR}/$(basename $trigger).submitted"
    done

    sleep "$POLL_INTERVAL"
done

#!/bin/bash
# Poll LUMI for the training chains and emit ONE LINE per event, nothing when quiet.
#
#   STATE     the queue changed (a job appeared, started, or left)
#   TERMINAL  a job left the queue, with its final state, exit code and runtime
#   NOCHAIN   a job left and the queue is now empty -> the chain is dead
#   EPOCH     a checkpoint at a round epoch (every 50) was written
#
# Failure paths are matched explicitly, so silence means running. Meant to run
# under a persistent Monitor; it lives in the repo, not a session scratchpad,
# because the scratchpad copy vanished with the session that made it.
#
# Usage:  bash scripts/watch_lumi_runs.sh [runs_dir]
HOST=nordlin1@lumi.csc.fi
RUNS_DIR=${1:-/scratch/project_462001112/runs_asinh99}
prev_state=""
prev_ckpt=""
while true; do
    snap=$(timeout 90 ssh -o BatchMode=yes "$HOST" '
        squeue -u $USER -h -O "JobID:.11,Name:.22,State:.12" 2>/dev/null
        # TERMINAL states only: sacct also lists RUNNING jobs, and reporting
        # those as finished made an earlier version cry wolf on every start.
        sacct -X -S $(date -d "12 hours ago" +%Y-%m-%dT%H:%M) \
              -o JobID,JobName%22,State,ExitCode,Elapsed -n -P 2>/dev/null \
          | grep -E "asinh|minmax|filmattn|mseyb|eval" \
          | grep -E "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED" \
          | tail -8
        echo "CKPT $(ls -t '"$RUNS_DIR"'/*.pt 2>/dev/null | head -1)"
    ' 2>/dev/null) || { sleep 300; continue; }

    # The checkpoint line carries the run name in its path, so keep it out of
    # the queue snapshot or every checkpoint fires a spurious STATE event.
    state=$(grep -E "asinh|minmax|filmattn|eval" <<<"$snap" | grep -vE "\||^CKPT" \
            | tr -s ' ' | sed 's/^ *//' | sort)   # squeue order varies poll to poll
    hist=$(grep "|" <<<"$snap")
    ckpt=$(sed -n 's/^CKPT //p' <<<"$snap")

    if [[ "$state" != "$prev_state" && -n "$prev_state" ]]; then
        [[ -n "$state" ]] && echo "STATE $(date +%H:%M) ${state//$'\n'/ ; }"
        while IFS='|' read -r id name st code el; do
            [[ -z "$id" ]] && continue
            grep -q "$id" <<<"$prev_state" || continue   # was queued last poll
            grep -q "$id" <<<"$state" && continue        # ...and is gone now
            echo "TERMINAL $id $name $st exit=$code elapsed=$el"
            if [[ -z "${state//[[:space:]]/}" ]]; then
                echo "NOCHAIN $id ended and the queue is now empty — chain is dead"
            fi
        done <<<"$hist"
    fi
    prev_state="$state"

    if [[ "$ckpt" != "$prev_ckpt" && -n "$prev_ckpt" ]]; then
        ep=$(sed -n 's/.*_\([0-9][0-9]*\)\.pt$/\1/p' <<<"$ckpt")
        [[ -n "$ep" && $((ep % 50)) -eq 0 ]] && echo "EPOCH $ckpt"
    fi
    prev_ckpt="$ckpt"
    sleep 300
done

#!/bin/bash
# Submit a LUMI job with the account taken from LUMI_PROJECT.
#
#   bash lsubmit.sh run2_aero.sh
#   bash lsubmit.sh run_eval_aero.sh --time=02:00:00
#   LUMI_PROJECT=462001112 bash lsubmit.sh run_debug_aero.sh
#
# WHY THIS EXISTS
# ---------------
# SLURM does not expand variables inside #SBATCH lines, so the account cannot be
# parameterised the way the paths are. The launchers therefore carry no
# `#SBATCH --account=` directive at all, and the account arrives one of two ways:
#
#   1. here, as `sbatch --account=...` on the command line (highest precedence)
#   2. via the SBATCH_ACCOUNT environment variable that lumi_env.sh exports
#
# Either is fine; this wrapper just makes (1) the default so a bare submission
# from a shell that never sourced lumi_env.sh cannot land on the wrong project.
# As a backstop, every launcher calls assert_account() once running, which kills
# the job in its first seconds if SLURM_JOB_ACCOUNT disagrees with LUMI_PROJECT.
set -euo pipefail

# Locate the repo. Under sbatch, SLURM COPIES this script to
# /var/spool/slurmd/job<id>/, so dirname "$0" is the spool directory and a
# relative source fails with "No such file or directory" — which is exactly
# what happened on Roihu job 660362. SLURM_SUBMIT_DIR is where sbatch was
# invoked, so try that first, then the script's own directory (correct when
# run directly), then an already-exported LUMI_REPO.
_find_repo() {
    local d
    for d in "${SLURM_SUBMIT_DIR:-}" \
             "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" \
             "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)" \
             "${LUMI_REPO:-}"; do
        [ -n "$d" ] && [ -f "$d/lumi_env.sh" ] && { echo "$d"; return 0; }
    done
    echo "ERROR: cannot locate lumi_env.sh. Submit from the repo directory, or" >&2
    echo "       export LUMI_REPO=/path/to/CESM2_emulator_from_lumi first." >&2
    return 1
}
_REPO_DIR="$(_find_repo)" || exit 1
source "${_REPO_DIR}/lumi_env.sh"

SCRIPT="${1:?usage: bash lsubmit.sh <script.sh> [extra sbatch args...]}"
shift || true

if [ ! -f "${SCRIPT}" ]; then
    echo "[lsubmit] ERROR: no such script: ${SCRIPT}" >&2
    exit 1
fi

lumi_env_banner
echo "[lsubmit] sbatch --account=${LUMI_ACCOUNT} $* ${SCRIPT}"
exec sbatch --account="${LUMI_ACCOUNT}" "$@" "${SCRIPT}"

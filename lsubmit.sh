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

source "$(dirname "${BASH_SOURCE[0]}")/lumi_env.sh"

SCRIPT="${1:?usage: bash lsubmit.sh <script.sh> [extra sbatch args...]}"
shift || true

if [ ! -f "${SCRIPT}" ]; then
    echo "[lsubmit] ERROR: no such script: ${SCRIPT}" >&2
    exit 1
fi

lumi_env_banner
echo "[lsubmit] sbatch --account=${LUMI_ACCOUNT} $* ${SCRIPT}"
exec sbatch --account="${LUMI_ACCOUNT}" "$@" "${SCRIPT}"

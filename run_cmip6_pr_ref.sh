#!/bin/bash

# Single source of truth for the LUMI project id and its paths.
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
# CPU-only: download CESM2 CMIP6 PRECIPITATION for the out-of-training scenarios
# from ESGF and aggregate it into the annual reference file the paper figures
# read. Runs inside the LAIF singularity container (project venv injected).
#
# WHY
# ---
# ssp126/ssp245 are the scenarios the emulator never trained on. Their CESM2
# TEMPERATURE reference (${LUMI_DATA}/cmip6/ssp126.nc, ssp245.nc) did not come
# from this repo at all — its attributes read
# "status: ...created; by gcs.cmip6.ldeo@gmail.com", i.e. the Pangeo
# Google-Cloud CMIP6 archive. Nothing here could produce the same thing for
# precipitation, so cmip6/ has no `pr` and the unseen-scenario precip panels
# could only ever be emulator-only (no difference map, no bias panel, no test).
#
# This script closes that gap from ESGF, which does publish CESM2 `pr` for
# r4/r10/r11 in both scenarios (verified 2026-08-13, 2 time chunks each).
#
# WHAT IT PRODUCES
#   ${LUMI_DATA}/cmip6/<experiment>/pr/<member>/pr_Amon_*.nc   (raw monthly)
#   ${LUMI_DATA}/cmip6/<experiment>_pr.nc                      (annual, 3 members)
#
# The second file is what scripts/paper_fig_maps.py and
# scripts/paper_fig_timeseries.py look for (CMIP6_REFS). Once it exists they
# stop printing "no CESM2 PRECT reference" and produce real emulator-minus-CESM2
# panels — drop --emulator-only from the maps call.
#
# EXPECT LOW POWER, NOT A LIT-UP MAP: n=3 CESM2 members, and precipitation's
# internal variability is ~32% of its forced signal (vs 3.3% for temperature).
# Pattern correlation and RMSE become computable; significant area will stay
# near zero. That is a data limit, not an emulator failure.
#
# COST: ~1.5 GB over HTTPS, a few minutes. Run in an allocation, NOT on the
# login node:
#   srun --account=${LUMI_ACCOUNT} --partition=small --time=01:00:00 \
#        --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G \
#        bash run_cmip6_pr_ref.sh
#
# Usage (from the repo dir, after `git pull`):
#   bash run_cmip6_pr_ref.sh                    # ssp126 + ssp245, 3 passes
#   bash run_cmip6_pr_ref.sh ssp126             # one scenario
#   PASSES=5 bash run_cmip6_pr_ref.sh
#   DISCOVER_ONLY=1 bash run_cmip6_pr_ref.sh    # dry run, downloads nothing
set -euo pipefail

CMIP6_DIR="${LUMI_DATA}/cmip6"
SCENARIOS=("${@:-ssp126 ssp245}")
read -r -a SCENARIOS <<< "${SCENARIOS[*]}"
MEMBERS="${MEMBERS:-r4i1p1f1 r10i1p1f1 r11i1p1f1}"
PASSES="${PASSES:-3}"

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif

# Inject the project venv's site-packages into the container
# (matches run_download_input4mips.sh).
_VENV_SITE=$(realpath ${LUMI_VENV} 2>/dev/null \
             || echo ${LUMI_VENV})/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
export PYTHONNOUSERSITE=1

echo "[pr-ref] scenarios = ${SCENARIOS[*]}"
echo "[pr-ref] members   = ${MEMBERS}"
echo "[pr-ref] cmip6 dir = ${CMIP6_DIR}"
echo "[pr-ref] SIF       = ${SIF}"

for EXP in "${SCENARIOS[@]}"; do
    echo
    echo "[pr-ref] ================= ${EXP} ================="

    if [[ -n "${DISCOVER_ONLY:-}" ]]; then
        singularity exec "${SIF}" python download_cmip6_cesm2.py \
            --outdir "${CMIP6_DIR}" --experiment "${EXP}" --variables pr \
            --members ${MEMBERS} --discover-only
        continue
    fi

    # The downloader skips files that already exist, so re-running only fetches
    # what is missing; the retry loop IS the recovery for a flaky mirror.
    rc=1
    for pass in $(seq 1 "${PASSES}"); do
        echo "[pr-ref] ${EXP} download pass ${pass}/${PASSES}"
        if singularity exec "${SIF}" python download_cmip6_cesm2.py \
               --outdir "${CMIP6_DIR}" --experiment "${EXP}" --variables pr \
               --members ${MEMBERS}; then
            rc=0
            break
        fi
        echo "[pr-ref] pass ${pass} had failures; retrying …"
        sleep $((pass * 10))
    done
    if [[ "${rc}" -ne 0 ]]; then
        echo "[pr-ref] DOWNLOAD FAILED for ${EXP} after ${PASSES} passes" >&2
        exit 1
    fi

    # Aggregate to the annual (year, member, lat, lon) file the figures read.
    # Default weighting reproduces the shipped tas reference exactly — do not
    # change it for one variable only (see the script's docstring).
    echo "[pr-ref] aggregating ${EXP} -> ${CMIP6_DIR}/${EXP}_pr.nc"
    singularity exec "${SIF}" python scripts/build_cmip6_annual_ref.py \
        --cmip6-dir "${CMIP6_DIR}" --experiment "${EXP}" --variable pr
done

if [[ -n "${DISCOVER_ONLY:-}" ]]; then
    echo "[pr-ref] discover-only: nothing downloaded"
    exit 0
fi

echo
echo "[pr-ref] done. Built:"
for EXP in "${SCENARIOS[@]}"; do
    ls -l "${CMIP6_DIR}/${EXP}_pr.nc" 2>/dev/null || echo "  MISSING ${EXP}_pr.nc"
done
echo
echo "[pr-ref] next — the unseen precip panels no longer need --emulator-only:"
echo "  python scripts/paper_fig_maps.py --eval-dir <eval> --n-ref-members 0 \\"
echo "         --scenarios ssp126 ssp245 --vars PRECT \\"
echo "         --out plots/paper_fig_maps_unseen_PRECT.png"
echo "  python scripts/paper_fig_timeseries.py --var PRECT --eval-dir <eval> \\"
echo "         --n-ref-members 0 --scenarios ssp126 ssp245 \\"
echo "         --out plots/paper_fig_timeseries_unseen_PRECT.png"

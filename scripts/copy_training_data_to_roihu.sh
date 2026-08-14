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
# Copy all conditioning (emission) files + training data from the LUMI scratch
# mount to the Roihu scratch mount, for porting the emulator to Roihu.
#
# Sources (matches run2_*.sh staging list + configs/config_data.yaml):
#   - cond files: emissions_{hist,ssp370,aaer,ghg}_only_timefixed_bc.nc  (training)
#                 emissions_ssp126_only_timefixed_co2fix_bc.nc,
#                 emissions_ssp245_only_timefixed_bc.nc                  (eval-only)
#   - training trees: training_data/{TREFHT,PRECT}/{hist,ssp370,AAER,GHG}
#     (held-out validation member LE2-1231.001 lives inside the same trees)
#
# COPIES, does not move: the gainfix training chain on LUMI stages these same
# files at every link start — deleting them from LUMI scratch would kill it.
#
# This version transfers everything in a SINGLE rsync call via --files-from,
# instead of looping and re-invoking rsync per file/directory. rsync handles
# the whole list as one job (one connection, one progress stream, resumable
# as a unit).
#
# Usage:  ./copy_training_data_to_roihu.sh [--dry-run]

set -euo pipefail

SRC_MOUNT=/home/nordling/mnt/lumi_sc2
DST_MOUNT=/home/nordling/mnt/roihu_sc
SRC_ROOT=${SRC_MOUNT}/emulator_data
DST_ROOT=${DST_MOUNT}/emulator_data

DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY="--dry-run"

# ── Mount liveness checks (sshfs mounts silently go stale) ───────────────────
if ! mountpoint -q "${SRC_MOUNT}" || ! ls "${SRC_ROOT}" >/dev/null 2>&1; then
    echo "ERROR: LUMI scratch mount ${SRC_MOUNT} is not mounted (or stale)." >&2
    echo "       Remount: sshfs nordlin1@lumi.csc.fi:${LUMI_SCRATCH}/ ${SRC_MOUNT}" >&2
    exit 1
fi
if ! mountpoint -q "${DST_MOUNT}" || ! ls "${DST_ROOT}" >/dev/null 2>&1; then
    echo "ERROR: Roihu scratch mount ${DST_MOUNT} is not mounted (or stale)." >&2
    echo "       Remount: sshfs nordlin1@roihu-cpu.csc.fi:/scratch/project_2019839/ ${DST_MOUNT}" >&2
    exit 1
fi

COND_FILES=(
    emissions_hist_only_timefixed_bc.nc
    emissions_ssp370_only_timefixed_bc.nc
    emissions_aaer_only_timefixed_bc.nc
    emissions_ghg_only_timefixed_bc.nc
    emissions_ssp126_only_timefixed_co2fix_bc.nc
    emissions_ssp245_only_timefixed_bc.nc
)
TRAIN_VARS=(TREFHT PRECT)
TRAIN_SCENARIOS=(hist ssp370 AAER GHG)

# Fail loud if anything expected is missing on the source before starting.
missing=0
for f in "${COND_FILES[@]}"; do
    [ -f "${SRC_ROOT}/${f}" ] || { echo "MISSING cond file: ${SRC_ROOT}/${f}" >&2; missing=1; }
done
for var in "${TRAIN_VARS[@]}"; do
    for scen in "${TRAIN_SCENARIOS[@]}"; do
        [ -d "${SRC_ROOT}/training_data/${var}/${scen}" ] \
            || { echo "MISSING training tree: ${SRC_ROOT}/training_data/${var}/${scen}" >&2; missing=1; }
    done
done
[ "${missing}" -eq 0 ] || { echo "Aborting: source incomplete (see above)." >&2; exit 1; }

mkdir -p "${DST_ROOT}"
for var in "${TRAIN_VARS[@]}"; do
    mkdir -p "${DST_ROOT}/training_data/${var}"
done

# ── Build the file list (paths relative to SRC_ROOT) for a single rsync run ──
FILE_LIST=$(mktemp)
trap 'rm -f "${FILE_LIST}"' EXIT

for f in "${COND_FILES[@]}"; do
    echo "${f}" >> "${FILE_LIST}"
done
for var in "${TRAIN_VARS[@]}"; do
    for scen in "${TRAIN_SCENARIOS[@]}"; do
        echo "training_data/${var}/${scen}" >> "${FILE_LIST}"
    done
done

echo "── Copying cond files + training trees in one rsync pass ───────────────"
rsync -ahr --partial --info=progress2 \
    --files-from="${FILE_LIST}" \
    ${DRY} \
    "${SRC_ROOT}/" "${DST_ROOT}/"

[ -n "${DRY}" ] && { echo "Dry run complete — nothing copied."; exit 0; }

echo "── Verifying ──────────────────────────────────────────────────────────"
fail=0
for f in "${COND_FILES[@]}"; do
    s=$(stat -c%s "${SRC_ROOT}/${f}"); d=$(stat -c%s "${DST_ROOT}/${f}" 2>/dev/null || echo 0)
    [ "${s}" = "${d}" ] || { echo "SIZE MISMATCH: ${f} (src ${s} vs dst ${d})" >&2; fail=1; }
done
for var in "${TRAIN_VARS[@]}"; do
    for scen in "${TRAIN_SCENARIOS[@]}"; do
        s=$(find "${SRC_ROOT}/training_data/${var}/${scen}" -type f | wc -l)
        d=$(find "${DST_ROOT}/training_data/${var}/${scen}" -type f 2>/dev/null | wc -l)
        [ "${s}" = "${d}" ] || { echo "FILE-COUNT MISMATCH: ${var}/${scen} (src ${s} vs dst ${d})" >&2; fail=1; }
    done
done
if [ "${fail}" -eq 0 ]; then
    echo "OK — all cond files byte-identical in size, all tree file counts match."
    echo "Total on Roihu: $(du -sh "${DST_ROOT}" | awk '{print $1}')"
else
    echo "Verification FAILED — re-run the script (rsync resumes partial copies)." >&2
    exit 1
fi

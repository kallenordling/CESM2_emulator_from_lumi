#!/bin/bash
# Copy the data an EVAL needs from one LUMI project's scratch to another's.
#
# WHY
# ---
# `export LUMI_PROJECT=<id>` repoints the account AND every data path at once
# (lumi_env.sh), so switching projects to escape a full /scratch quota also
# needs the data to exist under the new project. This copies exactly what
# eval_aero.py opens, and nothing else — the source scratch holds ~100 files of
# superseded cond-file generations that are not worth moving.
#
# WHAT IT COPIES (~5 GB total, measured 2026-08-13)
#   training_data/PRECT          ~3.7 GB  95 members  (TREFHT is usually already there)
#   emissions_*_bc.nc            ~1.1 GB  the 7 cond files EXPERIMENTS references
#   cmip6/*.nc                   ~0.2 GB  the annual CESM2 references
#   runs/<ckpt>.pt               ~0.7 GB  optional, --checkpoint
#
# It does NOT build the venv. The target project needs venvs/diffesm_laif
# (LUMI_VENV); build it with `bash setup_venv_laif.sh` after this.
#
# Usage (ON LUMI):
#   bash scripts/migrate_data_to_project.sh 462001112                  # dry run
#   bash scripts/migrate_data_to_project.sh 462001112 --apply
#   bash scripts/migrate_data_to_project.sh 462001112 --apply \
#        --from 462001328 --checkpoint runs/run_mseyb_BCprect_490.pt
#
# rsync is resumable and skips files already present with the same size+mtime,
# so re-running after an interruption costs only the diff.
set -euo pipefail

DST=""; SRC="462001328"; APPLY=0; CKPT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --apply)      APPLY=1 ;;
        --from)       SRC="$2"; shift ;;
        --checkpoint) CKPT="$2"; shift ;;
        -*)           echo "unknown flag $1" >&2; exit 1 ;;
        *)            DST="$1" ;;
    esac
    shift
done
[ -n "${DST}" ] || { echo "usage: $0 <new_project_id> [--from <old>] [--apply] [--checkpoint runs/x.pt]" >&2; exit 1; }
[ "${DST}" = "${SRC}" ] && { echo "source and destination are the same project" >&2; exit 1; }

S="/scratch/project_${SRC}/emulator_data"
D="/scratch/project_${DST}/emulator_data"
[ -d "${S}" ] || { echo "ERROR: source ${S} not found (are you on LUMI, and in project ${SRC}?)" >&2; exit 1; }

RSYNC=(rsync -a --info=progress2 --human-readable)
[ "${APPLY}" = "1" ] || RSYNC+=(--dry-run)

echo "[migrate] ${S}  ->  ${D}"
[ "${APPLY}" = "1" ] || echo "[migrate] DRY RUN — add --apply to copy"

# Free space check: the destination quota is the whole reason for this move, so
# failing here beats failing 4 GB into a copy.
if command -v lfs >/dev/null 2>&1; then
    echo "[migrate] destination quota:"
    lfs quota -hp "$(stat -c %g "/scratch/project_${DST}" 2>/dev/null)" \
        "/scratch/project_${DST}" 2>/dev/null || echo "  (could not read quota)"
fi

mkdir -p "${D}/cmip6" "${D}/training_data" "/scratch/project_${DST}/eval_output"

# ── training trees ───────────────────────────────────────────────────────────
# PRECT is the bulk. TREFHT is usually already present on the older project;
# rsync verifies rather than assuming, and copies only what differs.
for v in TREFHT PRECT; do
    if [ -d "${S}/training_data/${v}" ]; then
        echo "[migrate] training_data/${v}"
        "${RSYNC[@]}" "${S}/training_data/${v}/" "${D}/training_data/${v}/"
    fi
done

# ── conditioning files ───────────────────────────────────────────────────────
# Exactly the ones eval_aero.py's EXPERIMENTS list opens. Add CMIP7 files here
# if you evaluate those scenarios.
COND=(
    emissions_hist_only_timefixed_bc.nc
    emissions_ssp370_only_timefixed_bc.nc
    emissions_ssp126_only_timefixed_co2fix_bc.nc
    emissions_ssp245_only_timefixed_bc.nc
    emissions_aaer_only_timefixed_bc.nc
    emissions_ghg_only_timefixed_bc.nc
    emissions_ssp370co2_ssp126aer_bc_2015-2079.nc
    emissions_ssp370co2_ssp126aer_bc.nc
)
echo "[migrate] conditioning files"
for f in "${COND[@]}"; do
    [ -f "${S}/${f}" ] && "${RSYNC[@]}" "${S}/${f}" "${D}/${f}" \
        || echo "  [skip] ${f} not in source"
done

# ── CESM2 annual references ──────────────────────────────────────────────────
echo "[migrate] cmip6 references"
"${RSYNC[@]}" --include='*.nc' --exclude='*' "${S}/cmip6/" "${D}/cmip6/"

# ── checkpoint ───────────────────────────────────────────────────────────────
if [ -n "${CKPT}" ]; then
    SR="/projappl/project_${SRC}/CESM2_emulator_from_lumi/${CKPT}"
    DR="/projappl/project_${DST}/CESM2_emulator_from_lumi/${CKPT}"
    if [ -f "${SR}" ]; then
        echo "[migrate] checkpoint ${CKPT}"
        [ "${APPLY}" = "1" ] && mkdir -p "$(dirname "${DR}")"
        "${RSYNC[@]}" "${SR}" "${DR}"
    else
        echo "  [skip] ${SR} not found"
    fi
fi

echo
if [ "${APPLY}" = "1" ]; then
    echo "[migrate] done. Remaining steps on the target project:"
else
    echo "[migrate] dry run complete. To execute, re-run with --apply, then:"
fi
cat <<EOF
  1. export LUMI_PROJECT=${DST}          # add to ~/.bashrc on LUMI
  2. cd /projappl/project_${DST}/CESM2_emulator_from_lumi && git pull
  3. bash setup_venv_laif.sh             # target has no venvs/diffesm_laif yet
  4. bash submit_eval_ens25.sh runs/<ckpt>.pt "ssp370-126aer ssp370" ramip_ens25
EOF

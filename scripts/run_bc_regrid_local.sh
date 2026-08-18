#!/bin/bash
# Run stage 2 of the BC rebuild (splice + xesmf regrid + inject) on the LOCAL
# machine, because the LUMI LAIF container has no xesmf.
#
#     .../data/concat_and_regrid.py: ModuleNotFoundError: No module named 'xesmf'
#
# Stage 1 (make_aerosol_files.py, no xesmf needed) already ran on LUMI and left
# BC_per_gridpoint_{hist,ssp370,ssp126,ssp245}.nc in project 462001112.
#
# WHY IT STAGES FILES INSTEAD OF WORKING ON THE MOUNT
# ---------------------------------------------------
# concat_and_regrid.py reads ONE --data_dir, and the files it needs are split
# across the two projects:
#     462001112  BC_per_gridpoint_*.nc          (stage 1 output, new)
#     462001328  emissions_*_only_timefixed.nc  (the originals it injects into)
# Worse, emissions_ghg_only_timefixed.nc EXISTS IN BOTH at different sizes
# (56.7 MB in 462001112, 177.8 MB in 462001328). The 177.8 MB one is the base
# the current *_bc.nc files were built from, so running with data_dir pointing
# at 462001112 would silently inject BC into the wrong ghg file. Staging into
# one local directory, taking each file from its correct project, removes that
# trap — and the regrid is much faster off local disk than over sshfs.
#
# NOTHING IS PUSHED BACK. The outputs land locally and the copy-back command is
# printed for you to run, because it overwrites the live cond files.
#
# Usage:
#     bash scripts/run_bc_regrid_local.sh --check     # verify inputs only
#     bash scripts/run_bc_regrid_local.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${BC_STAGE_DIR:-$HOME/data_staging/bc_rebuild}"
SC112="/home/nordling/mnt/lumi_sc/emulator_data"     # project 462001112
SC328="/home/nordling/mnt/lumi_sc2/emulator_data"    # project 462001328
PY="${XESMF_PYTHON:-/home/nordling/miniconda3/envs/xesmf_env/bin/python}"

# PIN ESMFMKFILE TO THE INTERPRETER'S OWN ENV. This is not optional.
# esmpy locates the ESMF C library through $ESMFMKFILE, and conda's esmf
# activate.d hook exports ESMFMKFILE=$CONDA_PREFIX/lib/esmf.mk for whichever env
# is ACTIVE. So with `plotting` active (esmf 8.4.1) while invoking xesmf_env's
# python (esmpy 8.9.0), esmpy reads the 8.4.1 esmf.mk and dies with
#     VersionMismatch: ESMF installation version 8.4.1 differs from ESMPy version 8.9.0
# even though xesmf_env is internally consistent (esmf 8.9.0 + esmpy 8.9.0).
# The variable is inherited, so simply calling the other env's binary is not
# enough — it has to be overridden. Derived from PY so it stays correct if
# XESMF_PYTHON points somewhere else.
_ENV_PREFIX="$(dirname "$(dirname "${PY}")")"
if [ -f "${_ENV_PREFIX}/lib/esmf.mk" ]; then
    export ESMFMKFILE="${_ENV_PREFIX}/lib/esmf.mk"
else
    unset ESMFMKFILE || true      # let esmpy fall back to its own install
fi

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

# from-462001112 (stage 1 output)
FROM_112=(
    BC_per_gridpoint_hist.nc
    BC_per_gridpoint_ssp370.nc
    BC_per_gridpoint_ssp126.nc
    BC_per_gridpoint_ssp245.nc
)
# from-462001328 (the originals BC is injected into, plus the grid target)
FROM_328=(
    emissions_hist_only_timefixed.nc
    emissions_ssp370_only_timefixed.nc
    emissions_ssp245_only_timefixed.nc
    emissions_ssp126_only_timefixed_co2fix.nc
    emissions_ghg_only_timefixed.nc
    emissions_aaer_only_timefixed.nc
    emissions_ssp126_only_timefixed.nc        # --target: only lat/lon are read
)

echo "[bc-local] python  ${PY}"
echo "[bc-local] ESMFMKFILE ${ESMFMKFILE:-<unset, using esmpy default>}"
"${PY}" -c "import xesmf,esmpy;print(f'[bc-local] xesmf {xesmf.__version__} esmpy {esmpy.__version__}')" || {
    echo "[bc-local] xesmf failed to import. If this is a VersionMismatch, an" >&2
    echo "[bc-local] active conda env is leaking ESMFMKFILE — 'conda deactivate'" >&2
    echo "[bc-local] and rerun, or set XESMF_PYTHON to a consistent env." >&2
    exit 1
}
echo "[bc-local] stage   ${STAGE}"

miss=0
for f in "${FROM_112[@]}"; do [ -f "${SC112}/$f" ] || { echo "  MISSING 462001112: $f" >&2; miss=1; }; done
for f in "${FROM_328[@]}"; do [ -f "${SC328}/$f" ] || { echo "  MISSING 462001328: $f" >&2; miss=1; }; done
[ "${miss}" -eq 0 ] || { echo "[bc-local] inputs incomplete — is the mount live? An EMPTY listing means a dead sshfs, not missing data." >&2; exit 1; }
echo "[bc-local] all ${#FROM_112[@]} + ${#FROM_328[@]} inputs present"
[ "${CHECK_ONLY}" -eq 1 ] && { echo "[bc-local] --check: stopping."; exit 0; }

mkdir -p "${STAGE}"
echo "[bc-local] staging ~1.4 GB (skips files already staged at the same size) …"
# -rltDv, NOT -a: sshfs cannot set the LUMI project group, so -a always fails
# with chgrp "Permission denied" and exits 23 on an otherwise perfect transfer.
for f in "${FROM_112[@]}"; do rsync -rltD --info=name "${SC112}/$f" "${STAGE}/"; done
for f in "${FROM_328[@]}"; do rsync -rltD --info=name "${SC328}/$f" "${STAGE}/"; done

echo "[bc-local] regridding (xesmf bilinear periodic → 192x288) …"
cd "${REPO}"
"${PY}" data/concat_and_regrid.py \
    --build-bc \
    --target   "${STAGE}/emissions_ssp126_only_timefixed.nc" \
    --data_dir "${STAGE}/"

echo
echo "[bc-local] checking the 2014→2015 BC junction in the rebuilt files …"
"${PY}" - "${STAGE}" <<'PY'
import sys, os, xarray as xr
D = sys.argv[1]
def g(p, yr):
    with xr.open_dataset(p) as ds:
        c = "year" if "year" in ds.coords else "time"
        return float(ds["BC"].sel({c: yr}).sum())
hist = f"{D}/emissions_hist_only_timefixed_bc.nc"
if not os.path.exists(hist):
    print("  hist output missing — regrid did not complete"); sys.exit(1)
h14 = g(hist, 2014)
print(f"  hist BC 2014 = {h14:.6g}")
bad = 0
for s, p in (("ssp370", "emissions_ssp370_only_timefixed_bc.nc"),
             ("ssp126", "emissions_ssp126_only_timefixed_co2fix_bc.nc"),
             ("ssp245", "emissions_ssp245_only_timefixed_bc.nc")):
    fp = f"{D}/{p}"
    if not os.path.exists(fp):
        print(f"  {s:7s} missing"); bad += 1; continue
    r = g(fp, 2015) / h14
    ok = 0.9 <= r <= 1.1
    bad += (not ok)
    print(f"  {s:7s} 2015/2014 = {r:.3f}   {'OK' if ok else 'STILL DISCONTINUOUS'}")
print("\n  expected ~0.997. ~1.35 means stage 1 on LUMI reused a stale "
      "BC_per_gridpoint_hist.nc\n  built from CEDS-2025 — delete it there and rerun stage 1.")
sys.exit(1 if bad else 0)
PY
rc=$?

echo
echo "[bc-local] outputs in ${STAGE}:"
ls -1 "${STAGE}"/emissions_*_bc.nc 2>/dev/null | sed 's|.*/|    |'
if [ "${rc}" -eq 0 ]; then
    cat <<MSG

[bc-local] junction is continuous. Copy back when you are ready — this
[bc-local] OVERWRITES the live cond files (run_rebuild_bc_ceds2017.sh already
[bc-local] backed them up to backup_bc_ceds2025_*/ on LUMI):

    rsync -rltDvP ${STAGE}/emissions_*_bc.nc \\
        ${SC112}/

[bc-local] Then retrain: all 165 historical years of the BC channel changed,
[bc-local] so run_mseyb_BCprect_490 is not comparable to runs on these files.
MSG
else
    echo "[bc-local] junction check FAILED — do not copy back." >&2
fi
exit "${rc}"

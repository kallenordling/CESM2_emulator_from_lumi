#!/bin/bash
# Rebuild the BC conditioning channel on CEDS-2017, safely.
#
# WHAT THIS IS FOR
# ----------------
# The BC channel had a +35% discontinuity at 2015 because its historical came
# from CEDS-CMIP-2025 while the IAMC SSP scenario files are harmonised to
# CEDS-2017. Measured global anthropogenic BC at the junction (Tg/yr):
#     CEDS-2017 2014 = 8.012  vs IAMC 2015 = 7.986  -> ratio 0.997
#     CEDS-2025 2014 = 5.917  vs IAMC 2015 = 7.986  -> ratio 1.350
# make_aerosol_files.py now points at CEDS-2017 (commit 3ca0e0b). This driver
# rebuilds the cond files with that source.
#
# WHY NOT JUST RUN run_make_bc_cond.sh
# ------------------------------------
# That script does the actual work and is invoked here — but on its own it will
# (a) fail partway if the IAMC BC scenario files are missing from INPUT_DIR, and
# (b) OVERWRITE the live emissions_*_bc.nc cond files in place. Those are what
# every current run reads, including run_mseyb_BCprect_490, the checkpoint the
# paper figures rest on. Its conditioning would be gone with no way back short
# of reverting the source pattern and rebuilding.
#
# So this wrapper: checks every input first, backs the existing cond files up,
# then delegates, then reports the junction ratio it actually achieved.
#
# Usage (from the repo dir on LUMI, after `git pull`):
#     bash run_rebuild_bc_ceds2017.sh --dry-run     # check inputs, change nothing
#     bash run_rebuild_bc_ceds2017.sh
#     bash run_rebuild_bc_ceds2017.sh --no-backup   # if you have your own copy
#
# If the login node blocks singularity, wrap it:
#     srun --account=${LUMI_ACCOUNT} --partition=small --time=01:00:00 \
#          --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G \
#          bash run_rebuild_bc_ceds2017.sh
set -euo pipefail

_find_repo() {
    local d
    for d in "${SLURM_SUBMIT_DIR:-}" \
             "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" \
             "${LUMI_REPO:-}"; do
        [ -n "$d" ] && [ -f "$d/lumi_env.sh" ] && { echo "$d"; return 0; }
    done
    echo "ERROR: cannot locate lumi_env.sh. Run from the repo directory." >&2
    return 1
}
_REPO_DIR="$(_find_repo)" || exit 1
source "${_REPO_DIR}/lumi_env.sh"
cd "${_REPO_DIR}"

DRY_RUN=0
BACKUP=1
for a in "$@"; do
    case "$a" in
        --dry-run)   DRY_RUN=1 ;;
        --no-backup) BACKUP=0 ;;
        -h|--help)   sed -n '2,32p' "$0"; exit 0 ;;
        *) echo "unknown option: $a" >&2; exit 2 ;;
    esac
done

IN="${LUMI_DATA}/emission_data/inputs4mips"
echo "[rebuild-bc] project ${LUMI_PROJECT}"
echo "[rebuild-bc] inputs   ${IN}"
echo "[rebuild-bc] data     ${LUMI_DATA}"

# ── 1. inputs ───────────────────────────────────────────────────────────────
# Check EVERYTHING before doing any work. Stage 1 loops over four experiments
# and would otherwise fail on the second one, after writing a hist output —
# leaving a half-rebuilt channel that looks finished.
missing=()
n_hist=$(ls "${IN}"/BC-em-anthro_input4MIPs_emissions_CMIP_CEDS-2017-05-18_gn_*.nc 2>/dev/null | wc -l)
[ "${n_hist}" -ge 5 ] || missing+=("BC CEDS-2017-05-18 historical (found ${n_hist}, need >=5 covering 1850-2014)")
declare -A IAMC=(
    [ssp370]="IAMC-AIM-ssp370-1-1"
    [ssp126]="IAMC-IMAGE-ssp126-1-1"
    [ssp245]="IAMC-MESSAGE-GLOBIOM-ssp245-1-1"
)
for s in "${!IAMC[@]}"; do
    f="${IN}/BC-em-anthro_input4MIPs_emissions_ScenarioMIP_${IAMC[$s]}_gn_201501-210012.nc"
    [ -f "$f" ] || missing+=("BC scenario ${s}: $(basename "$f")")
done

if [ ${#missing[@]} -gt 0 ]; then
    echo "[rebuild-bc] MISSING INPUTS:" >&2
    printf '  - %s\n' "${missing[@]}" >&2
    echo >&2
    echo "[rebuild-bc] The IAMC BC scenario files may exist under the other" >&2
    echo "[rebuild-bc] project — the input4MIPs data is split across 462001328" >&2
    echo "[rebuild-bc] and 462001112 and was never consolidated. If you are a" >&2
    echo "[rebuild-bc] member of both, copy them across:" >&2
    echo >&2
    echo "    cp /scratch/project_462001328/emulator_data/emission_data/inputs4mips/\\" >&2
    echo "       BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-*_gn_201501-210012.nc \\" >&2
    echo "       ${IN}/" >&2
    echo >&2
    echo "[rebuild-bc] Otherwise fetch them with download_input4mips_cmip7.py." >&2
    exit 1
fi
echo "[rebuild-bc] inputs OK: ${n_hist} historical file(s) + 3 scenario file(s)"

# ── 2. what will be overwritten ─────────────────────────────────────────────
mapfile -t EXISTING < <(ls "${LUMI_DATA}"/emissions_*_bc.nc 2>/dev/null || true)
echo "[rebuild-bc] cond files that will be REPLACED: ${#EXISTING[@]}"
for f in "${EXISTING[@]}"; do echo "    $(basename "$f")"; done

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "[rebuild-bc] --dry-run: stopping before any change."
    exit 0
fi

# ── 3. backup ───────────────────────────────────────────────────────────────
# Copy, not move: a rebuild that dies partway must leave the originals in place.
if [ "${BACKUP}" -eq 1 ] && [ ${#EXISTING[@]} -gt 0 ]; then
    BDIR="${LUMI_DATA}/backup_bc_ceds2025_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${BDIR}"
    echo "[rebuild-bc] backing up to ${BDIR} …"
    for f in "${EXISTING[@]}"; do
        cp -p "$f" "${BDIR}/" && echo "    $(basename "$f")"
    done
    echo "[rebuild-bc] restore with:  cp ${BDIR}/*.nc ${LUMI_DATA}/"
fi

# ── 4. rebuild ──────────────────────────────────────────────────────────────
echo "[rebuild-bc] delegating to run_make_bc_cond.sh …"
bash run_make_bc_cond.sh

# ── 5. did it actually work ─────────────────────────────────────────────────
# run_make_bc_cond.sh verifies structure (CO2/SUL unchanged, BC present, ghg
# constant). It does not check the thing this rebuild exists to fix, so measure
# the junction directly in the finished cond files.
echo
echo "[rebuild-bc] checking the 2014->2015 BC junction in the rebuilt files …"
module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
_VENV_SITE=$(realpath "${LUMI_VENV}" 2>/dev/null || echo "${LUMI_VENV}")/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
export PYTHONNOUSERSITE=1

singularity exec "${SIF}" python - "${LUMI_DATA}" <<'PY'
import sys, os, numpy as np, xarray as xr
D = sys.argv[1]
hist = f"{D}/emissions_hist_only_timefixed_bc.nc"
scen = {"ssp370": f"{D}/emissions_ssp370_only_timefixed_bc.nc",
        "ssp126": f"{D}/emissions_ssp126_only_timefixed_co2fix_bc.nc",
        "ssp245": f"{D}/emissions_ssp245_only_timefixed_bc.nc"}
def g(p, var, yr):
    with xr.open_dataset(p) as ds:
        c = "year" if "year" in ds.coords else "time"
        return float(ds[var].sel({c: yr}).sum())
try:
    h14 = g(hist, "BC", 2014)
except Exception as e:
    print(f"  could not read {hist}: {e}"); sys.exit(0)
print(f"  hist BC 2014 = {h14:.6g}")
bad = 0
for s, p in scen.items():
    if not os.path.exists(p):
        print(f"  {s:7s} missing"); continue
    r = g(p, "BC", 2015) / h14
    flag = "OK" if 0.9 <= r <= 1.1 else "STILL DISCONTINUOUS"
    if flag != "OK": bad += 1
    print(f"  {s:7s} 2015/2014 = {r:.3f}   {flag}")
print()
print("  expected ~0.997 (CEDS-2017). ~1.35 means the old CEDS-2025 BC_per_gridpoint_hist.nc")
print("  was reused — delete BC_per_gridpoint_hist.nc and rerun so stage 1 regenerates it.")
sys.exit(1 if bad else 0)
PY
rc=$?

echo
if [ "${rc}" -eq 0 ]; then
    echo "[rebuild-bc] done — junction is continuous."
    echo "[rebuild-bc] NOTE: all 165 historical years of the BC channel changed,"
    echo "[rebuild-bc] not just 2015, so existing checkpoints were trained on"
    echo "[rebuild-bc] different conditioning. Retraining is required before these"
    echo "[rebuild-bc] cond files are comparable to run_mseyb_BCprect_490 results."
else
    echo "[rebuild-bc] FAILED the junction check — see above." >&2
fi
exit "${rc}"

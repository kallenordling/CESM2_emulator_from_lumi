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
# LOCAL (workstation) runner for the CMIP7 emissions download + cond-file build.
# No singularity, no SLURM — plain python from whatever env is active. Points at
# the sshfs-mounted LUMI directories by default.
#
# Mount layout (from `mount`):
#   /home/nordling/mnt/lumi_sc2  ->  ${LUMI_SCRATCH}/     (data)
#   /home/nordling/mnt/lumi2     ->  ${LUMI_PROJAPPL}/
# so LUMI's  ${LUMI_DATA}/...  is reachable locally
# as  ${LUMI_MOUNT}/emulator_data/...
#
# Usage:
#   bash run_cmip7_local.sh check       # mount liveness + inputs + deps, no work
#   bash run_cmip7_local.sh download    # fetch the input4MIPs emissions (~17.5 GB)
#   bash run_cmip7_local.sh cond        # build the CMIP7 cond files
#   bash run_cmip7_local.sh all         # download then cond
#
# Override any path:
#   LUMI_MOUNT=/other/mount bash run_cmip7_local.sh check
#   INPUT_DIR=~/cmip7_emissions OUTPUT_DIR=~/cmip7_cond bash run_cmip7_local.sh all
#
# NOTE ON SPEED: reading/writing ~17.5 GB over sshfs is slow, and the cond build
# streams every historical file twice. If it drags, point INPUT_DIR/OUTPUT_DIR at
# local disk instead and copy the results to the mount afterwards.
set -euo pipefail

LUMI_MOUNT="${LUMI_MOUNT:-/home/nordling/mnt/lumi_sc2}"
SCRATCH="${SCRATCH:-${LUMI_MOUNT}/emulator_data}"

INPUT_DIR="${INPUT_DIR:-${SCRATCH}/emission_data/inputs4mips}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}}"
# Grid template: an existing cond file, already on the 192x288 CESM2 grid.
TARGET="${TARGET:-${SCRATCH}/emissions_hist_only_timefixed_bc.nc}"

PY="${PY:-python3}"
CMD="${1:-check}"

cd "$(dirname "$0")"

echo "[cmip7-local] LUMI_MOUNT = ${LUMI_MOUNT}"
echo "[cmip7-local] INPUT_DIR  = ${INPUT_DIR}"
echo "[cmip7-local] OUTPUT_DIR = ${OUTPUT_DIR}"
echo "[cmip7-local] TARGET     = ${TARGET}"
echo

# ── mount liveness ───────────────────────────────────────────────────────────
# A dropped sshfs mount still appears in `mount` but lists EMPTY, which is
# indistinguishable from "the data is gone" unless checked explicitly. Only
# enforced when the paths actually live under the mount.
check_mount() {
    case "${INPUT_DIR}${OUTPUT_DIR}${TARGET}" in
        *"${LUMI_MOUNT}"*) ;;
        *) echo "[cmip7-local] paths are off-mount; skipping mount check"; return 0 ;;
    esac

    if [[ ! -d "${LUMI_MOUNT}" ]]; then
        echo "[cmip7-local] ERROR: ${LUMI_MOUNT} does not exist."
        return 1
    fi
    if ! mount | grep -q " ${LUMI_MOUNT} "; then
        echo "[cmip7-local] ERROR: ${LUMI_MOUNT} is not a mount point."
        echo "[cmip7-local]   Mount it, or pass local paths:"
        echo "[cmip7-local]   INPUT_DIR=~/cmip7_emissions OUTPUT_DIR=~/cmip7_cond bash $0 ${CMD}"
        return 1
    fi
    if [[ -z "$(timeout 30 ls -A "${LUMI_MOUNT}" 2>/dev/null)" ]]; then
        echo "[cmip7-local] ERROR: ${LUMI_MOUNT} is mounted but lists EMPTY —"
        echo "[cmip7-local]   the sshfs connection has dropped (this is NOT missing data)."
        echo "[cmip7-local]   Remount, e.g.:"
        echo "[cmip7-local]     fusermount -u ${LUMI_MOUNT}"
        echo "[cmip7-local]     sshfs nordlin1@lumi.csc.fi:${LUMI_SCRATCH}/ ${LUMI_MOUNT}"
        echo "[cmip7-local]   …then re-run. Or work off local disk:"
        echo "[cmip7-local]     INPUT_DIR=~/cmip7_emissions OUTPUT_DIR=~/cmip7_cond bash $0 ${CMD}"
        return 1
    fi
    echo "[cmip7-local] mount OK: ${LUMI_MOUNT} is live"
    return 0
}

# ── dependency check ─────────────────────────────────────────────────────────
check_deps() {
    local missing=0
    "${PY}" - <<'EOF' || missing=1
import importlib, sys
need = {"xarray": "conda install -c conda-forge xarray",
        "numpy": "conda install numpy",
        "requests": "pip install requests",
        "tqdm": "pip install tqdm",
        "netCDF4": "conda install -c conda-forge netcdf4",
        "xesmf": "conda install -c conda-forge xesmf esmpy   # regridding; conda only, pip will not work"}
bad = []
for m, how in need.items():
    try:
        importlib.import_module(m)
    except ImportError:
        bad.append((m, how))
for m, how in bad:
    print(f"  MISSING {m:10s} -> {how}")
if bad:
    print(f"\n{len(bad)} dependency/ies missing.")
    sys.exit(1)
print("  all python deps present (xarray, numpy, requests, tqdm, netCDF4, xesmf)")
EOF
    return $missing
}

case "${CMD}" in

check)
    ok=0
    check_mount || ok=1
    echo
    echo "[cmip7-local] python deps:"
    check_deps || ok=1
    echo
    if [[ -f "${TARGET}" ]]; then
        echo "[cmip7-local] grid template OK: ${TARGET}"
    else
        echo "[cmip7-local] MISSING grid template: ${TARGET}"
        echo "[cmip7-local]   Any file on the 192x288 CESM2 grid works. From LUMI:"
        echo "[cmip7-local]   scp nordlin1@lumi.csc.fi:${LUMI_DATA}/emissions_hist_only_timefixed_bc.nc ~/"
        echo "[cmip7-local]   then re-run with TARGET=~/emissions_hist_only_timefixed_bc.nc"
        echo "[cmip7-local]   Do NOT synthesize this grid — a fractional offset would"
        echo "[cmip7-local]   silently misalign every cond field against the trained model."
        ok=1
    fi
    echo
    if [[ -d "${INPUT_DIR}" ]]; then
        echo "[cmip7-local] input4MIPs files present in INPUT_DIR:"
        "${PY}" data/make_cmip7_cond.py --target "${TARGET}" \
            --input-dir "${INPUT_DIR}" --dry-run || ok=1
    else
        echo "[cmip7-local] INPUT_DIR does not exist yet: ${INPUT_DIR}"
        echo "[cmip7-local]   run:  bash $0 download"
        ok=1
    fi
    echo
    [[ ${ok} -eq 0 ]] && echo "[cmip7-local] READY — run: bash $0 cond" \
                      || echo "[cmip7-local] not ready; fix the items above"
    exit ${ok}
    ;;

download)
    check_mount || exit 1
    mkdir -p "${INPUT_DIR}"
    echo "[cmip7-local] downloading input4MIPs CMIP7 emissions (~17.5 GB) …"
    echo "[cmip7-local] resumable: interrupting is safe, re-run to continue."
    exec "${PY}" download_input4mips_cmip7.py \
        --layout flat \
        --outdir "${INPUT_DIR}" \
        "${@:2}"
    ;;

cond)
    check_mount || exit 1
    check_deps  || exit 1
    if [[ ! -f "${TARGET}" ]]; then
        echo "[cmip7-local] ERROR: grid template not found: ${TARGET}"
        echo "[cmip7-local]   see:  bash $0 check"
        exit 1
    fi
    mkdir -p "${OUTPUT_DIR}"
    exec "${PY}" data/make_cmip7_cond.py \
        --target     "${TARGET}" \
        --input-dir  "${INPUT_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        "${@:2}"
    ;;

all)
    bash "$0" download
    bash "$0" cond
    ;;

*)
    echo "usage: bash $0 {check|download|cond|all} [extra args passed through]"
    exit 2
    ;;
esac

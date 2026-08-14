#!/bin/bash
#SBATCH --job-name=dl_input4mips
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

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
assert_account
lumi_env_banner
#
# Download CMIP7 gridded emissions (BC, SO2, CO2) from ESGF input4MIPs.
# CPU-only, single task — this is network-bound, not compute-bound, so more
# cores/memory buys nothing. 6h covers ~12 GB with plenty of margin for slow
# ESGF nodes; the job exits as soon as it finishes.
#
# Submit from the repo dir on LUMI (after `git pull`):
#     sbatch run_download_input4mips_slurm.sh
#
# Useful variants (env overrides, see below):
#     LAYOUT=nested sbatch run_download_input4mips_slurm.sh          # keep tree structure
#     SPECIES="SO2 BC" sbatch run_download_input4mips_slurm.sh       # subset
#     SOURCES="CEDS-CMIP-2025-04-18" sbatch run_download_input4mips_slurm.sh
#     PASSES=5 sbatch run_download_input4mips_slurm.sh               # more retries
#
# Before submitting, it is worth running the (fast, network-only) discovery on a
# login node to see the plan and what ESGF currently publishes:
#     bash run_download_input4mips.sh --list-sources
#     bash run_download_input4mips.sh --discover-only
#
# DEFAULT OUTPUT = the FLAT directory the cond-building pipeline already globs:
#     ${LUMI_DATA}/emission_data/inputs4mips/
# data/make_aerosol_files.py and data/make_co2_files.py match input4MIPs files by
# FILENAME in one flat INPUT_DIR, so a nested tree would be invisible to them.
# Filenames encode source_id + date range and are globally unique, so dropping
# them all in one directory cannot collide. Existing files are skipped, so this
# is safe to re-run alongside the CMIP6-era files already in that directory
# (note BC/hist there is ALREADY CEDS-CMIP-2025-04-18, i.e. a CMIP7 source).
#
# Default set (~12 GB, 24 files, verified on ESGF 2026-08-06):
#     historical  CEDS-CMIP-2025-04-18    1750-2023, 6 files/species
#     scenarios   IIASA-IAMC-h-1-1-0      2022-2100, 1 file/species  (high)
#                 IIASA-IAMC-vl-1-1-0     2022-2100, 1 file/species  (very low)
# CMIP7 ScenarioMIP names scenarios by warming level (vl/l/ml/m/h/hl/ln), not
# RCP numbers; only h and vl had gridded emissions as of 2026-08-06.
set -euo pipefail

DATA_ROOT=${LUMI_DATA}
OUTDIR="${OUTDIR:-${DATA_ROOT}/emission_data/inputs4mips}"
LAYOUT="${LAYOUT:-flat}"
PASSES="${PASSES:-3}"
SPECIES="${SPECIES:-BC SO2 CO2}"
SOURCES="${SOURCES:-CEDS-CMIP-2025-04-18 IIASA-IAMC-h-1-1-0 IIASA-IAMC-vl-1-1-0}"
GRID="${GRID:-gn}"

mkdir -p logs

# --- co-location check -------------------------------------------------------
# OUTDIR must be the SAME flat directory the cond builders read, or the new CMIP7
# files land somewhere data/make_{aerosol,co2}_files.py will never glob. Those
# scripts default to EMUL_INPUT_DIR, falling back to the path below; if you have
# EMUL_INPUT_DIR set in your environment, it wins there, so honour it here too.
PIPELINE_DIR="${EMUL_INPUT_DIR:-${LUMI_DATA}/emission_data/inputs4mips}"
PIPELINE_DIR="${PIPELINE_DIR%/}"

if [[ "${OUTDIR%/}" != "${PIPELINE_DIR}" ]]; then
    echo "[input4mips] WARNING: OUTDIR does not match the cond-builder input dir."
    echo "[input4mips]   downloading to : ${OUTDIR}"
    echo "[input4mips]   pipeline reads : ${PIPELINE_DIR}"
    echo "[input4mips]   (fine if deliberate — e.g. LAYOUT=nested for archival)"
fi

if [[ -d "${OUTDIR}" ]]; then
    _n_existing=$(find "${OUTDIR}" -maxdepth 1 -name "*input4MIPs*.nc" 2>/dev/null | wc -l)
    echo "[input4mips] OUTDIR exists, already holds ${_n_existing} input4MIPs .nc file(s):"
    find "${OUTDIR}" -maxdepth 1 -name "*input4MIPs*.nc" -printf "  %f\n" 2>/dev/null \
        | sort | head -20
    [[ "${_n_existing}" -gt 20 ]] && echo "  … and $((_n_existing - 20)) more"
    if [[ "${_n_existing}" -eq 0 ]]; then
        echo "[input4mips] WARNING: no existing input4MIPs files here. If your older"
        echo "[input4mips]   emission files live elsewhere, this is the WRONG directory —"
        echo "[input4mips]   cancel and re-submit with OUTDIR=/path/to/them."
    fi
else
    echo "[input4mips] WARNING: ${OUTDIR} does not exist yet — creating it."
    echo "[input4mips]   If your existing emission files are in a DIFFERENT directory,"
    echo "[input4mips]   cancel now (scancel ${SLURM_JOB_ID:-<jobid>}) and re-submit with"
    echo "[input4mips]   OUTDIR=/path/to/them, or the pipeline will not see this data."
fi
mkdir -p "${OUTDIR}"
echo

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif

# Inject the project venv's site-packages into the container (matches run_download_data.sh).
_VENV_SITE=$(realpath ${LUMI_VENV} 2>/dev/null \
             || echo ${LUMI_VENV})/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
export PYTHONNOUSERSITE=1

echo "[input4mips] job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "[input4mips] OUTDIR  = ${OUTDIR}  (layout=${LAYOUT})"
echo "[input4mips] SPECIES = ${SPECIES}"
echo "[input4mips] SOURCES = ${SOURCES}"
echo "[input4mips] PASSES  = ${PASSES}   GRID = ${GRID}"
echo "[input4mips] SIF     = ${SIF}"
echo

run_downloader() {
    singularity exec "${SIF}" python download_input4mips_cmip7.py \
        --outdir     "${OUTDIR}" \
        --layout     "${LAYOUT}" \
        --grid-label "${GRID}" \
        --species    ${SPECIES} \
        --sources    ${SOURCES} \
        "$@"
}

# Retry passes: the downloader resumes .part files and skips completed ones, so
# each pass only picks up what the last one missed. It exits non-zero if ANY
# file failed (deliberately unlike download_lens2.py, whose silent partial
# failures once left 268 PRECT files missing).
rc=1
for pass in $(seq 1 "${PASSES}"); do
    echo "[input4mips] === pass ${pass}/${PASSES} ==="
    if run_downloader; then
        rc=0
        echo "[input4mips] pass ${pass} completed with no failures"
        break
    fi
    rc=1
    echo "[input4mips] pass ${pass} had failures; retrying after backoff …"
    sleep $((pass * 15))
done

if [[ "${rc}" -ne 0 ]]; then
    echo "[input4mips] FAILED after ${PASSES} passes — files still missing."
    echo "[input4mips] Re-submit to resume (completed files are skipped)."
    exit 1
fi

# Verify by re-querying ESGF and confirming every planned file is on disk.
echo
echo "[input4mips] verifying downloaded tree against ESGF …"
run_downloader --discover-only | tee "${TMPDIR:-/tmp}/i4m_verify_${SLURM_JOB_ID:-0}.txt"

awk '
    /^  GET /  { missing++; print "  MISSING: " $0 }
    /^  HAVE / { have++ }
    END {
        printf "[input4mips] VERIFY: %d present, %d missing\n", have+0, missing+0
        if (missing+0 > 0) exit 1
    }' "${TMPDIR:-/tmp}/i4m_verify_${SLURM_JOB_ID:-0}.txt"

rm -f "${TMPDIR:-/tmp}/i4m_verify_${SLURM_JOB_ID:-0}.txt"

echo
echo "[input4mips] done — CMIP7 BC/SO2/CO2 emissions complete under ${OUTDIR}"
echo "[input4mips] NOTE: these are CMIP7 sources; data/make_aerosol_files.py and"
echo "[input4mips]       data/make_co2_files.py still glob CMIP6-era filenames for"
echo "[input4mips]       most (species, exp) pairs — update _ANTHRO_PATTERNS there"
echo "[input4mips]       before building cond files from this data."

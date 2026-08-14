#!/bin/bash

# Single source of truth for the LUMI project id and its paths.
source "$(dirname "${BASH_SOURCE[0]}")/lumi_env.sh"
# Build the annual multi-member CESM2 references from the downloaded RAMIP
# files, in the (year, member, lat, lon) layout eval_aero.py and the paper
# figures read. Runs inside the LAIF singularity container (project venv
# injected), same pattern as run_cmip6_pr_ref.sh.
#
# INPUT   <ramip-dir>/<experiment>/<variable>/*.nc   (scripts/download_ramip_ceda.sh)
# OUTPUT  ${LUMI_DATA}/cmip6/ramip_<experiment>[_pr].nc
#
# WHY THE ramip_ PREFIX
# ---------------------
# It deliberately does NOT overwrite the files already there:
#   cmip6/ssp370-126aer.nc   the ONE-member file that predates the CEDA download
#   cmip6/ssp370.nc          the 3-member CMIP6 (not RAMIP) ssp370 ensemble
# Both remain valid references and you will want to compare the old
# single-member answer against the new 10-member one.
#
# WHY 2015-2079 BY DEFAULT
# ------------------------
# ssp370-126aer stops in 2079 while RAMIP's ssp370 control runs to 2100. The
# aerosol-removal signal is a DIFFERENCE between them, so both are capped to the
# shared window; a longer control would silently contribute years with no
# counterpart.
#
# TABLE IS APmon, NOT Amon: CMIP6Plus renamed the monthly atmosphere table, and
# the filenames carry it (tas_APmon_CESM2_...), so the aggregator's glob needs it.
#
# Usage (on LUMI, from the repo dir):
#   bash run_build_ramip_refs.sh
#   bash run_build_ramip_refs.sh --ramip-dir /some/other/ramip
#   bash run_build_ramip_refs.sh --experiments "ssp370-sas126aer ssp370"
#   bash run_build_ramip_refs.sh --end 2100 --experiments ssp370   # control alone
set -euo pipefail

EXPERIMENTS="ssp370-126aer ssp370"
VARIABLES="tas pr"
TABLE="APmon"
START=2015
END=2079
OUTDIR="${LUMI_DATA}/cmip6"
RAMIP_DIR=""

while [ $# -gt 0 ]; do
    case "$1" in
        --experiments) EXPERIMENTS="$2"; shift ;;
        --variables)   VARIABLES="$2"; shift ;;
        --table)       TABLE="$2"; shift ;;
        --start)       START="$2"; shift ;;
        --end)         END="$2"; shift ;;
        --ramip-dir)   RAMIP_DIR="$2"; shift ;;
        --outdir)      OUTDIR="$2"; shift ;;
        -h|--help)     sed -n '2,40p' "$0"; exit 0 ;;
        *)             echo "unknown argument: $1" >&2; exit 1 ;;
    esac
    shift
done

# Locate the downloaded tree. The downloader writes to ${LUMI_DATA}/ramip, but
# falls back to ./ramip when LUMI_DATA is unset in the calling shell — which is
# easy to do and lands 6 GB inside the git repo on /projappl. Accept either, and
# say so, rather than failing with "no files match".
if [ -z "${RAMIP_DIR}" ]; then
    if   [ -d "${LUMI_DATA}/ramip" ];              then RAMIP_DIR="${LUMI_DATA}/ramip"
    elif [ -d "$(dirname "$0")/ramip" ];           then RAMIP_DIR="$(cd "$(dirname "$0")/ramip" && pwd)"
        echo "[ramip-ref] NOTE: using ${RAMIP_DIR} (inside the repo on /projappl)."
        echo "[ramip-ref]       That is the LUMI_DATA-unset fallback of the downloader."
        echo "[ramip-ref]       Move it to ${LUMI_DATA}/ramip when convenient:"
        echo "[ramip-ref]         mv ${RAMIP_DIR} ${LUMI_DATA}/"
    else
        echo "[ramip-ref] ERROR: no ramip/ tree found." >&2
        echo "[ramip-ref]        Looked in ${LUMI_DATA}/ramip and $(dirname "$0")/ramip." >&2
        echo "[ramip-ref]        Download it first: scripts/download_ramip_ceda.sh --apply" >&2
        exit 1
    fi
fi

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif
_VENV_SITE=$(realpath ${LUMI_VENV} 2>/dev/null || echo ${LUMI_VENV})/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
export PYTHONNOUSERSITE=1

echo "[ramip-ref] input      ${RAMIP_DIR}"
echo "[ramip-ref] output     ${OUTDIR}"
echo "[ramip-ref] window     ${START}-${END}   table ${TABLE}"
echo "[ramip-ref] experiments ${EXPERIMENTS}   variables ${VARIABLES}"
mkdir -p "${OUTDIR}"

built=(); missing=()
for exp in ${EXPERIMENTS}; do
    for var in ${VARIABLES}; do
        src="${RAMIP_DIR}/${exp}/${var}"
        if [ ! -d "${src}" ]; then
            echo "[ramip-ref] [skip] ${exp}/${var}: ${src} not present"
            missing+=("${exp}/${var}")
            continue
        fi
        # tas keeps the bare name, other variables get a suffix — the same
        # convention build_cmip6_annual_ref.py and CMIP6_REFS already use.
        if [ "${var}" = "tas" ]; then out="${OUTDIR}/ramip_${exp}.nc"
        else                          out="${OUTDIR}/ramip_${exp}_${var}.nc"; fi

        echo
        echo "[ramip-ref] === ${exp} / ${var} -> $(basename "${out}") ==="
        singularity exec "${SIF}" python scripts/build_cmip6_annual_ref.py \
            --cmip6-dir "${src}" \
            --experiment "${exp}" --variable "${var}" --table "${TABLE}" \
            --start "${START}" --end "${END}" --out "${out}"
        built+=("$(basename "${out}")")
    done
done

echo
echo "[ramip-ref] built ${#built[@]} reference file(s):"
for b in "${built[@]}"; do
    printf '   %-40s %s\n' "${b}" "$(du -h "${OUTDIR}/${b}" 2>/dev/null | cut -f1)"
done
if [ "${#missing[@]}" -gt 0 ]; then
    echo "[ramip-ref] MISSING inputs: ${missing[*]}"
    exit 1
fi

cat <<EOF

[ramip-ref] next — point the eval at the 10-member reference instead of the
            single-member one, in eval_aero.py's ssp370-126aer entry:

    data_dir = os.path.join(SCRATCH, "cmip6", "ramip_ssp370-126aer.nc")

            and note RAMIP now also gives you its OWN ssp370 control
            (ramip_ssp370.nc), which is the right baseline for the
            aerosol-removal difference — same ensemble, same model config.
EOF

#!/bin/bash
# Download CESM2 RAMIP data from the CEDA archive (CMIP6Plus/RAMIP/NCAR/CESM2).
#
# WHY CEDA AND NOT ESGF
# ---------------------
# RAMIP is not on ESGF. Verified 2026-08-13 across the ORNL, LLNL and CEDA index
# nodes: `project=RAMIP`, `activity_id=RAMIP`, `experiment_id=ssp370-126aer` and
# a free-text 126aer search restricted to CESM2 all return ZERO datasets. The
# CEDA archive publishes it directly instead, under CMIP6Plus.
#
# AUTHENTICATION IS REQUIRED — this is the part that bites.
# The directory listings are public but the FILES are not: an unauthenticated
# GET returns an HTML LOGIN PAGE with HTTP 200, which naive downloaders happily
# write to <name>.nc. This script verifies the NetCDF magic bytes of every file
# and deletes anything that is secretly HTML, so a missing token fails loudly
# instead of poisoning the data tree.
#
#   1. Register / log in:  https://services.ceda.ac.uk
#   2. Generate an access token (a Bearer token, limited lifespan):
#      https://help.ceda.ac.uk/article/5100-archive-access-tokens
#   3. export CEDA_TOKEN=<the token>      # do NOT commit it or paste it anywhere
#
# WHAT IT DOWNLOADS BY DEFAULT (~5.4 GB)
#   ssp370-126aer  10 members  tas + pr   2015-2079   the GLOBAL aerosol-cleanup
#   ssp370         10 members  tas + pr   2015-2079   RAMIP's own control
#
# Take the control from RAMIP, not from CMIP6 ssp370: the aerosol-removal signal
# is a DIFFERENCE, and differencing against the same ensemble under the same
# model configuration cancels shared drift and model-version differences. Using
# the 3-member CMIP6 ssp370 instead leaves both in the answer.
#
# The regional experiments live here too and are the ones matching the project's
# regional-forcing goal (each applies ssp126 aerosols over ONE region only):
#   --experiments "ssp370-sas126aer ssp370"     South Asia
#   --experiments "ssp370-eas126aer ssp370"     East Asia
#   --experiments "ssp370-afr126aer ssp370"     Africa
#   --experiments "ssp370-nae126aer ssp370"     North America + Europe
# NOTE they are NOT comparable to emissions_ssp370co2_ssp126aer_bc.nc, which
# changes aerosols GLOBALLY. A regional comparison needs its own cond file.
#
# Usage (anywhere with network; on LUMI writes straight to scratch):
#   export CEDA_TOKEN=...
#   bash scripts/download_ramip_ceda.sh                      # dry run
#   bash scripts/download_ramip_ceda.sh --apply
#   bash scripts/download_ramip_ceda.sh --apply --members "r1i1p1f1 r2i1p1f1"
#
# Resumable: existing, verified files are skipped; partial ones resume (curl -C).
set -uo pipefail

# TWO HOSTS, and using the wrong one is the whole difficulty here:
#   data.ceda.ac.uk  browse UI + ?json listings. IGNORES the Bearer token and
#                    serves an HTML login page with HTTP 200 for file GETs.
#   dap.ceda.ac.uk   the actual data endpoint. Honours the token, returns the
#                    NetCDF (verified: HTTP 206, magic bytes \x89HDF).
# Same path on both; only the host differs.
LIST_BASE="https://data.ceda.ac.uk/badc/cmip6/data/CMIP6Plus/RAMIP/NCAR/CESM2"
DAP_BASE="https://dap.ceda.ac.uk/badc/cmip6/data/CMIP6Plus/RAMIP/NCAR/CESM2"
EXPERIMENTS="ssp370-126aer ssp370"
MEMBERS=""                      # empty = discover all from the archive listing
VARIABLES="tas pr"
TABLE="APmon"                   # CMIP6Plus table holding tas/pr (NOT Amon)
GRID="gn"
OUTDIR="${LUMI_DATA:-.}/ramip"
APPLY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --apply)       APPLY=1 ;;
        --experiments) EXPERIMENTS="$2"; shift ;;
        --members)     MEMBERS="$2"; shift ;;
        --variables)   VARIABLES="$2"; shift ;;
        --table)       TABLE="$2"; shift ;;
        --outdir)      OUTDIR="$2"; shift ;;
        -h|--help)     sed -n '2,50p' "$0"; exit 0 ;;
        *)             echo "unknown argument: $1" >&2; exit 1 ;;
    esac
    shift
done

if [ -z "${CEDA_TOKEN:-}" ]; then
    echo "ERROR: CEDA_TOKEN is not set — CEDA returns an HTML login page instead" >&2
    echo "       of the data, with HTTP 200, so this would silently write garbage." >&2
    echo "       Generate one at https://services.ceda.ac.uk (see --help)." >&2
    exit 1
fi

AUTH=(-H "Authorization: Bearer ${CEDA_TOKEN}")
# Listing endpoints are public; only file GETs need the token.
_ls() { curl -sS -L --max-time 120 "$1?json" 2>/dev/null; }
_names() {   # stdin: CEDA json -> names of entries, optionally filtered by type
    python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit(0)
want=sys.argv[1] if len(sys.argv)>1 else None
for i in d.get('items',[]):
    if want is None or i.get('type')==want: print(i.get('name',''))
" "${1:-}"
}

echo "[ramip] listings  ${LIST_BASE}"
echo "[ramip] downloads ${DAP_BASE}"
echo "[ramip] outdir    ${OUTDIR}"
echo "[ramip] variables ${VARIABLES}   table ${TABLE}"
[ "${APPLY}" = "1" ] || echo "[ramip] DRY RUN — add --apply to download"

total=0; got=0; skipped=0; failed=0
for exp in ${EXPERIMENTS}; do
    mems="${MEMBERS}"
    if [ -z "${mems}" ]; then
        mems=$(_ls "${LIST_BASE}/${exp}" | _names dir | tr '\n' ' ')
    fi
    if [ -z "${mems}" ]; then
        echo "[ramip] ${exp}: no members found (is the experiment name right?)" >&2
        continue
    fi
    echo "[ramip] ${exp}: $(echo ${mems} | wc -w) members"

    for mem in ${mems}; do
        for var in ${VARIABLES}; do
            vdir="${LIST_BASE}/${exp}/${mem}/${TABLE}/${var}/${GRID}"
            ddir="${DAP_BASE}/${exp}/${mem}/${TABLE}/${var}/${GRID}"
            # The version directory (vYYYYMMDD) is not fixed across experiments,
            # so it is discovered rather than hardcoded.
            ver=$(_ls "${vdir}" | _names dir | sort | tail -1)
            [ -n "${ver}" ] || { echo "  [miss] ${exp}/${mem}/${var}: no version dir"; failed=$((failed+1)); continue; }
            files=$(_ls "${vdir}/${ver}" | _names file)
            [ -n "${files}" ] || { echo "  [miss] ${exp}/${mem}/${var}: no files"; failed=$((failed+1)); continue; }

            for fn in ${files}; do
                case "${fn}" in *.nc) ;; *) continue ;; esac
                total=$((total+1))
                dest="${OUTDIR}/${exp}/${var}/${fn}"
                if [ -s "${dest}" ] && head -c4 "${dest}" | grep -qa -e 'CDF' -e 'HDF'; then
                    skipped=$((skipped+1)); continue
                fi
                if [ "${APPLY}" != "1" ]; then
                    echo "  [get ] ${exp}/${var}/${fn}"
                    continue
                fi
                mkdir -p "$(dirname "${dest}")"
                echo "  [get ] ${fn}"
                curl -sS -L --fail --retry 3 --retry-delay 5 -C - \
                     "${AUTH[@]}" -o "${dest}" "${ddir}/${ver}/${fn}"
                rc=$?
                # HTTP 200 + HTML login page is the failure mode this guards.
                if [ "${rc}" -ne 0 ] || ! head -c4 "${dest}" 2>/dev/null | grep -qa -e 'CDF' -e 'HDF'; then
                    echo "  [FAIL] ${fn} is not NetCDF — token missing, expired, or no access" >&2
                    rm -f "${dest}"
                    failed=$((failed+1))
                else
                    got=$((got+1))
                fi
            done
        done
    done
done

echo
echo "[ramip] files: ${total} seen, ${got} downloaded, ${skipped} already present, ${failed} failed"
[ "${failed}" -gt 0 ] && exit 1
if [ "${APPLY}" = "1" ]; then
    echo "[ramip] next — build the annual multi-member references:"
    for exp in ${EXPERIMENTS}; do
        echo "  python scripts/build_cmip6_annual_ref.py --cmip6-dir ${OUTDIR}/${exp} \\"
        echo "         --experiment ${exp} --variable tas --table ${TABLE} \\"
        echo "         --start 2015 --end 2079 --out \${LUMI_DATA}/cmip6/${exp}.nc"
    done
fi
exit 0

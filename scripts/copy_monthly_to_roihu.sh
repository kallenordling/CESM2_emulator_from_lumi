#!/bin/bash
# Copy the RAW MONTHLY CESM2 files (sf/ + lens2/) from the LUMI scratch mount to
# the Roihu one, for the monthly-resolution training work.
#
# WHAT AND WHY
# ------------
# These are the files the annual training trees were staged FROM — cam.h0
# monthly timeseries, all four training scenarios, both variables:
#
#   sf/AAER    TREFHT + PRECT   20 members   1850-2050
#   sf/GHG     TREFHT + PRECT   15 members   1850-2050
#   lens2/     TREFHT + PRECT   30 members   BHIST + BSSP370 (cmip6 & smbb)
#
# ~50 GB. Monthly resolution was always present here; it is the staging step
# that collapsed it with groupby('time.year').mean(), which is why
# training_data/ carries integer `time` with units=year. Nothing needs
# re-downloading — PRECT is even archived directly in these, rather than needing
# the PRECC+PRECL derivation the AWS catalog forces.
#
# COPIES, does not move: LUMI still stages from these.
#
# ROUTE: this goes LUMI-mount -> local machine -> Roihu-mount, i.e. every byte
# crosses sshfs twice. It is simple and resumable but not fast. If LUMI and
# Roihu can reach each other directly, an rsync between them will be far
# quicker; this exists because the two sshfs mounts are what is definitely
# available.
#
# Usage:
#   bash scripts/copy_monthly_to_roihu.sh --dry-run
#   bash scripts/copy_monthly_to_roihu.sh
#   bash scripts/copy_monthly_to_roihu.sh --only sf        # one tree at a time
set -euo pipefail

SRC_MOUNT=/home/nordling/mnt/lumi_sc2
DST_MOUNT=/home/nordling/mnt/roihu_sc
SRC_ROOT="${SRC_MOUNT}/emulator_data"
DST_ROOT="${DST_MOUNT}/emulator_data"

DRY=""; ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY="--dry-run" ;;
        --only)    ONLY="$2"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
    shift
done

TREES=(sf lens2)
[ -n "${ONLY}" ] && TREES=("${ONLY}")

# ── mount liveness: sshfs goes stale silently, and an empty listing is
#    indistinguishable from missing data ───────────────────────────────────────
for m in "${SRC_MOUNT}" "${DST_MOUNT}"; do
    if ! mountpoint -q "$m" || ! ls "$m" >/dev/null 2>&1; then
        echo "ERROR: ${m} is not mounted (or is stale)." >&2
        exit 1
    fi
done

for t in "${TREES[@]}"; do
    [ -d "${SRC_ROOT}/${t}" ] || { echo "ERROR: missing source ${SRC_ROOT}/${t}" >&2; exit 1; }
done

echo "── source ────────────────────────────────────────────────────────────"
need=0
for t in "${TREES[@]}"; do
    kb=$(du -sk "${SRC_ROOT}/${t}" | cut -f1); need=$((need + kb))
    printf "  %-8s %6s  %5d files\n" "$t" \
        "$(du -sh "${SRC_ROOT}/${t}" | cut -f1)" \
        "$(find "${SRC_ROOT}/${t}" -name '*.nc' | wc -l)"
done
avail=$(df -Pk "${DST_MOUNT}" | tail -1 | awk '{print $4}')
# awk formats and we print as strings: printf %f rejects awk's dot-decimal
# output under a comma-decimal locale (LC_NUMERIC).
echo "  need ~$(awk -v k=$need 'BEGIN{printf "%.1f", k/1048576}') GB," \
     "destination has $(awk -v k=$avail 'BEGIN{printf "%.1f", k/1048576}') GB free"
if [ "${need}" -gt "${avail}" ]; then
    echo "ERROR: not enough space on ${DST_MOUNT}" >&2; exit 1
fi

echo "── copying ───────────────────────────────────────────────────────────"
[ -n "${DRY}" ] && echo "  (dry run)"
mkdir -p "${DST_ROOT}"
for t in "${TREES[@]}"; do
    echo "  ${t} →  ${DST_ROOT}/${t}/"
    # --partial keeps interrupted files for resume; --size-only avoids
    # re-copying on sshfs timestamp jitter between two different filesystems.
    rsync -ahr --partial --size-only --info=progress2 ${DRY} \
        "${SRC_ROOT}/${t}/" "${DST_ROOT}/${t}/"
done

if [ -z "${DRY}" ]; then
    echo "── verify ────────────────────────────────────────────────────────────"
    for t in "${TREES[@]}"; do
        s=$(find "${SRC_ROOT}/${t}" -name '*.nc' | wc -l)
        d=$(find "${DST_ROOT}/${t}" -name '*.nc' 2>/dev/null | wc -l)
        printf "  %-8s source %4d files, destination %4d  %s\n" \
            "$t" "$s" "$d" "$([ "$s" = "$d" ] && echo OK || echo MISMATCH)"
    done
fi

cat <<EOF

Next: these are RAW files, not a training tree. ClimateDataset wants
  training_data_monthly/<VAR>/<scenario>/<member>/chunk_*.nc
so they still need staging — the same step that produced the annual trees, but
WITHOUT the groupby('time.year').mean().
EOF

#!/bin/bash
# What is filling up $HOME on LUMI, and what is safe to delete.
#
# LUMI home is small (20 GB by default) and is NOT where data or venvs belong —
# but caches land there silently and fill it. Symptoms seen in this project's
# job logs before anyone noticed the quota was gone:
#
#     Could not save font_manager cache [Errno 122] Disk quota exceeded:
#         '/users/<user>/.cache/matplotlib/fontlist-v3.11.0.json'
#
# A full home does not usually kill a job outright; it makes pip, conda,
# matplotlib and MIOpen fail in confusing ways, so it is worth keeping clear.
#
# READ-ONLY BY DEFAULT. --clean removes only the caches listed in SAFE below,
# and prints each one first. Nothing else is ever touched.
#
# Usage (ON LUMI):
#   bash scripts/home_usage_report.sh              # report only
#   bash scripts/home_usage_report.sh --clean      # remove the safe caches
set -uo pipefail

# Every du/find below uses -x / -xdev so the walk never leaves the home
# filesystem. Without it the scan descends into any network mount under $HOME
# (sshfs, bind mounts) and takes minutes to hours to produce a wrong answer.

CLEAN=0
[ "${1:-}" = "--clean" ] && CLEAN=1

echo "=============================================================="
echo " HOME USAGE  ${HOME}"
echo "=============================================================="

# ── quota ────────────────────────────────────────────────────────────────────
if command -v lfs >/dev/null 2>&1; then
    echo "[quota]"
    lfs quota -h "${HOME}" 2>/dev/null | sed 's/^/  /' || echo "  (lfs quota unavailable)"
elif command -v quota >/dev/null 2>&1; then
    quota -s 2>/dev/null | sed 's/^/  /'
fi
echo
echo "[total]"
du -shx "${HOME}" 2>/dev/null | sed 's/^/  /'

# ── biggest top-level entries ────────────────────────────────────────────────
echo
echo "[top-level, largest first]"
du -shx "${HOME}"/* "${HOME}"/.[!.]* 2>/dev/null | sort -rh | head -25 | sed 's/^/  /'

# ── the usual suspects ───────────────────────────────────────────────────────
# Caches: regenerated automatically, safe to delete, often the whole problem.
SAFE=(
    "${HOME}/.cache/pip"                # pip wheel/http cache
    "${HOME}/.cache/matplotlib"         # font cache (the error above)
    "${HOME}/.cache/miopen"             # ROCm kernel cache — can reach many GB
    "${HOME}/.cache/comgr"              # ROCm LLVM compile cache. THE INODE KILLER:
                                        # measured 87,848 files / 2.3 GB on one
                                        # account, 87% of the 100K file quota, with
                                        # disk use still only a third of its limit.
                                        # Watch the FILE count, not just the bytes.
    "${HOME}/.cache/torch"              # torch hub downloads
    "${HOME}/.cache/huggingface"        # model/dataset downloads
    "${HOME}/.cache/conda"
    "${HOME}/.conda/pkgs"               # conda package tarballs
    "${HOME}/.singularity/cache"
    "${HOME}/.apptainer/cache"          # container layer cache — often huge
    "${HOME}/.nv/ComputeCache"
    "${HOME}/.triton/cache"
    "${HOME}/.local/share/Trash"
)
echo
echo "[caches — safe to delete, regenerated on demand]"
total_kb=0
for d in "${SAFE[@]}"; do
    if [ -e "$d" ]; then
        kb=$(du -skx "$d" 2>/dev/null | cut -f1)
        total_kb=$((total_kb + kb))
        printf "  %8s  %s\n" "$(du -shx "$d" 2>/dev/null | cut -f1)" "$d"
    fi
done
printf "  --------  reclaimable: %s\n" \
    "$(awk -v k="${total_kb}" 'BEGIN{printf "%.1f GB", k/1048576}')"

# ── things that should NOT be in home at all ─────────────────────────────────
echo
echo "[misplaced — these belong on /scratch or /projappl, not \$HOME]"
find "${HOME}" -xdev -maxdepth 3 \( -name "*.nc" -o -name "*.pt" -o -name "*.ckpt" \
     -o -name "*.tar" -o -name "*.tar.gz" -o -name "*.sif" \) \
     -size +100M -printf "  %10s  %p\n" 2>/dev/null | head -20 \
     || echo "  (none over 100 MB)"

echo
echo "[python envs in home — venvs belong in /projappl/project_*/venvs]"
for d in "${HOME}"/.local/lib/python*  "${HOME}"/venv* "${HOME}"/*env*; do
    [ -d "$d" ] && printf "  %8s  %s\n" "$(du -shx "$d" 2>/dev/null | cut -f1)" "$d"
done

echo
echo "[oldest big directories — candidates for archival]"
du -shx "${HOME}"/* 2>/dev/null | sort -rh | head -10 | while read -r sz p; do
    printf "  %8s  %s   (last modified %s)\n" "$sz" "$p" \
        "$(date -r "$p" +%Y-%m-%d 2>/dev/null || echo '?')"
done

# ── optional cleanup ─────────────────────────────────────────────────────────
if [ "${CLEAN}" = "1" ]; then
    echo
    echo "=============================================================="
    echo " REMOVING CACHES (only the paths listed above as safe)"
    echo "=============================================================="
    for d in "${SAFE[@]}"; do
        if [ -e "$d" ]; then
            echo "  rm -rf $d"
            rm -rf "$d"
        fi
    done
    echo
    echo "[after]"
    du -shx "${HOME}" 2>/dev/null | sed 's/^/  /'
else
    echo
    echo "Re-run with --clean to delete the cache paths listed above."
    echo "Nothing has been removed."
fi

#!/bin/bash
# Build (or verify) the python environment for THIS host's architecture, then
# tell you exactly how to run the download from here.
#
#   bash setup_env.sh                 # create/verify the venv, print next step
#   bash setup_env.sh --download      # …and run get_data.py right away
#   bash setup_env.sh --download --variable TREFHT --monthly
#
# WHY THIS EXISTS
# ---------------
# Roihu is effectively two machines and the software is on only one of them:
#
#   GPU partitions + login  aarch64 (Grace)  AIDA modules EXIST
#   CPU partitions          x86_64           NO python module at all
#
# Both python-pytorch and python-data resolve to
# /appl/soft/manual/aida/aarch64/…/container.sif, which is architecture-pinned,
# so on x86_64 they fail with
#   the image's architecture (arm64) could not run on the host's (amd64)
# and a venv built on the ARM login node contains aarch64 wheels no matter what
# directory you put it in. Four separate failures came out of that; this script
# picks the right strategy from uname -m instead.
#
#   aarch64 : module load python-data/3.12, venv --system-site-packages on top
#   x86_64  : NO module, standalone venv from /usr/bin/python3
#
# The venv path carries the architecture, so the two never collide and a job
# only ever finds the one built for the node it is on.
set -euo pipefail

DOWNLOAD=0
PASSTHRU=()
while [ $# -gt 0 ]; do
    case "$1" in
        --download) DOWNLOAD=1 ;;
        -h|--help)  sed -n '2,30p' "$0"; exit 0 ;;
        *)          PASSTHRU+=("$1") ;;
    esac
    shift
done

export SITE="${SITE:-roihu}"
_find_repo() {
    local d
    for d in "${SLURM_SUBMIT_DIR:-}" \
             "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" \
             "${LUMI_REPO:-}"; do
        [ -n "$d" ] && [ -f "$d/lumi_env.sh" ] && { echo "$d"; return 0; }
    done
    echo "ERROR: cannot locate lumi_env.sh — run this from the repo." >&2
    return 1
}
REPO="$(_find_repo)" || exit 1
source "${REPO}/lumi_env.sh"

ARCH="$(uname -m)"
VENV="${LUMI_PROJAPPL}/venvs/diffesm-${ARCH}"
PKGS=(xarray netCDF4 dask pandas joblib intake intake-esm s3fs)

echo "=============================================================="
echo " host $(hostname -s)   arch ${ARCH}   site ${SITE}"
echo " venv ${VENV}"
echo "=============================================================="

# ── module strategy, by architecture ────────────────────────────────────────
USE_MODULE=1
if [ "${SITE}" = "roihu" ] && [ "${ARCH}" != "aarch64" ]; then
    USE_MODULE=0
    echo "[setup] ${ARCH}: skipping modules — Roihu's AIDA stack is aarch64-only"
fi

module purge 2>/dev/null || true
if [ "${USE_MODULE}" = "1" ]; then
    CMD="${SITE_MODULE_CMD_CPU-${SITE_MODULE_CMD}}"
    echo "[setup] ${CMD}"
    if ! eval "${CMD}"; then
        echo "[setup] module load failed — falling back to a standalone venv" >&2
        USE_MODULE=0
    fi
fi

# ── create the venv if missing ──────────────────────────────────────────────
if [ ! -f "${VENV}/bin/activate" ]; then
    mkdir -p "$(dirname "${VENV}")"
    if [ "${USE_MODULE}" = "1" ]; then
        # Layer on the module so xarray/netCDF4/dask come from it and only the
        # few extras are installed here.
        echo "[setup] creating venv (--system-site-packages, on top of the module)"
        python3 -m venv --system-site-packages "${VENV}"
    else
        # Nothing to inherit: build it all from the system interpreter.
        echo "[setup] creating STANDALONE venv from /usr/bin/python3"
        /usr/bin/python3 -m venv "${VENV}"
    fi
else
    echo "[setup] venv already exists — verifying"
fi
source "${VENV}/bin/activate"
echo "[setup] python $(python3 -V 2>&1) at $(command -v python3)"

# ── the check that would have caught the aarch64-in-x86_64 venv ─────────────
python3 - "${VENV}" <<'ARCHCHK' || exit 1
import glob, os, platform, struct, sys
venv = sys.argv[1]
sos = glob.glob(os.path.join(venv, "lib", "python*", "site-packages", "**", "*.so"),
                recursive=True)
if not sos:
    sys.exit(0)                      # nothing compiled yet; nothing to check
E = {0x3E: "x86_64", 0xB7: "aarch64"}
with open(sos[0], "rb") as fh:
    mach = E.get(struct.unpack_from("<H", fh.read(20), 18)[0], "unknown")
host = platform.machine()
if mach != "unknown" and mach != host:
    print(f"[setup] ERROR: venv holds {mach} binaries but this host is {host}.",
          file=sys.stderr)
    print(f"[setup]        Delete it and re-run this script ON a {host} host:",
          file=sys.stderr)
    print(f"[setup]          rm -rf {venv}", file=sys.stderr)
    sys.exit(1)
print(f"[setup] venv binaries are {mach}, host is {host} — match")
ARCHCHK

# ── install only what is actually missing ───────────────────────────────────
missing=()
for m in xarray netCDF4 dask pandas joblib intake intake_esm s3fs; do
    python3 -c "import ${m}" 2>/dev/null || missing+=("${m}")
done
if [ "${#missing[@]}" -gt 0 ]; then
    echo "[setup] installing: ${missing[*]}"
    python3 -m pip install --quiet --upgrade pip
    python3 -m pip install "${PKGS[@]}"
else
    echo "[setup] all required packages already importable"
fi

echo "[setup] verifying imports …"
python3 - <<'PY'
import importlib, sys
bad = []
for m in ("xarray", "netCDF4", "dask", "pandas", "joblib",
          "intake", "intake_esm", "s3fs"):
    try:
        mod = importlib.import_module(m)
        print(f"    {m:12s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        bad.append(f"{m} ({type(e).__name__})")
if bad:
    print("  FAILED: " + ", ".join(bad), file=sys.stderr); sys.exit(1)
PY

echo
if [ "${DOWNLOAD}" = "1" ]; then
    OUT="${LUMI_DATA}/training_data"
    case " ${PASSTHRU[*]-} " in *" --monthly "*) OUT="${LUMI_DATA}/training_data_monthly" ;; esac
    echo "[setup] running the download now → ${OUT}"
    cd "${REPO}"
    exec python3 get_data.py --skip-existing --output-dir "${OUT}" "${PASSTHRU[@]-}"
fi

cat <<EOF
[setup] environment ready. To download from HERE (${ARCH}):

    source ${VENV}/bin/activate
    cd ${REPO}
    python3 get_data.py --monthly --skip-existing \\
        --output-dir ${LUMI_DATA}/training_data_monthly

  or in one step:   bash setup_env.sh --download --monthly

  Long downloads on a login node should run under tmux:
    tmux new -s dl   …   detach with Ctrl-b d, reattach: tmux attach -t dl

  Via SLURM instead (only where this architecture has compute):
    $( [ "${ARCH}" = "aarch64" ] \
        && echo "sbatch --account=${LUMI_ACCOUNT} --partition=gpuinteractive run_get_data.sh --monthly" \
        || echo "SITE_MODULE_CMD_CPU='' sbatch --account=${LUMI_ACCOUNT} run_get_data.sh --monthly" )
EOF

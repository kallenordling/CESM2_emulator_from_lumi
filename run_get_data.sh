#!/bin/bash
#SBATCH --job-name=get_data
#SBATCH --output=logs/get_data_%j.out
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#
# Download CESM2-LE training data (AWS S3 via the NCAR intake catalog) under
# SLURM. CPU-only: this is network- and I/O-bound, so it belongs on a CPU
# partition, NOT on a GPU node.
#
#     sbatch --account=project_2019839 run_get_data.sh --monthly
#     sbatch --account=project_2019839 run_get_data.sh            # annual
#     sbatch --account=project_2019839 run_get_data.sh --monthly --variable TREFHT
#
# THE FAILURE THIS GUARDS AGAINST
# -------------------------------
# CSC compute nodes frequently have NO direct internet. get_data.py streams
# from s3://ncar-cesm2-lens, so on such a node it fails — often only after the
# catalog fetch, i.e. minutes in and with a confusing traceback. This script
# therefore TESTS CONNECTIVITY FIRST and exits immediately with instructions if
# the node is offline, rather than burning a queue slot to fail later.
#
# If compute nodes are offline, the download must run somewhere with network:
#   * an interactive session on a node that does have it, or
#   * the login node under tmux/screen (long but simple), or
#   * set the site's HTTP(S) proxy, if one is documented, e.g.
#       export https_proxy=... http_proxy=...
#     and resubmit.
#
# RESUMABILITY: --skip-existing is always passed. Members whose output
# directory already holds chunk files are skipped, so hitting the 3-day limit
# is not a disaster — resubmit and it continues. An empty directory from a
# killed member is retried rather than skipped.
#
# SIZE: monthly is ~12x annual — ~80 MB per member-year per variable against
# ~200 kB. On Roihu, /scratch is 250 G with a 180-day cleanup, so check
# headroom before pulling the full member set, and consider --members-style
# subsetting by editing get_data.py's common_members if you only need a few.
set -euo pipefail

export SITE="${SITE:-roihu}"
source "$(dirname "$0")/lumi_env.sh"
assert_account
lumi_env_banner

cd "${LUMI_REPO}"
mkdir -p logs

# ── environment ─────────────────────────────────────────────────────────────
module purge
eval "${SITE_MODULE_CMD}"
echo "[get_data] python $(command -v python3) — $(python3 -V 2>&1)"

# get_data.py needs intake-esm / s3fs, which the PyTorch module may not carry.
missing=""
for m in xarray intake intake_esm s3fs joblib; do
    python3 -c "import ${m}" 2>/dev/null || missing="${missing} ${m}"
done
if [ -n "${missing}" ]; then
    echo "[get_data] ERROR: missing python packages:${missing}" >&2
    echo "[get_data]        Install into a venv on \$LUMI_PROJAPPL (15 G on Roihu):" >&2
    echo "[get_data]          python3 -m venv --system-site-packages ${LUMI_VENV}" >&2
    echo "[get_data]          source ${LUMI_VENV}/bin/activate" >&2
    echo "[get_data]          pip install intake intake-esm s3fs" >&2
    echo "[get_data]        then re-submit with SITE_MODULE_CMD extended to activate it," >&2
    echo "[get_data]        or build the env with tykky/0.5.2." >&2
    exit 1
fi

# ── connectivity pre-check (see header) ─────────────────────────────────────
echo "[get_data] testing outbound network from $(hostname) …"
if ! python3 - <<'PY'
import sys, urllib.request
# The catalog itself lives on GitHub; the data on AWS S3. Both must be
# reachable, and they are different hosts, so test both.
for url in ("https://raw.githubusercontent.com/NCAR/cesm2-le-aws/main/"
            "intake-catalogs/aws-cesm2-le.json",
            "https://ncar-cesm2-lens.s3.amazonaws.com/"):
    try:
        urllib.request.urlopen(url, timeout=25)
        print(f"  reachable: {url.split('/')[2]}")
    except Exception as e:
        print(f"  UNREACHABLE: {url.split('/')[2]} -> {type(e).__name__}: {e}")
        sys.exit(1)
PY
then
    echo "[get_data] ERROR: this compute node has no route to the data." >&2
    echo "[get_data]        Compute nodes at CSC often have no direct internet." >&2
    echo "[get_data]        Run the download from a login node under tmux, from an" >&2
    echo "[get_data]        interactive node with network, or export the site proxy" >&2
    echo "[get_data]        (https_proxy/http_proxy) and resubmit." >&2
    exit 1
fi

# ── run ─────────────────────────────────────────────────────────────────────
OUTDIR_DEFAULT="${LUMI_DATA}/training_data"
case " $* " in
    *" --monthly "*) OUTDIR_DEFAULT="${LUMI_DATA}/training_data_monthly" ;;
esac
# Monthly must NOT share a directory with annual data: both write
# <outdir>/<var>/<member>/chunk_N.nc, and open_mfdataset would then combine
# annual and monthly chunks into one incoherent series.
case " $* " in
    *" --output-dir "*) OUTDIR_ARG="" ;;                     # caller chose one
    *)                  OUTDIR_ARG="--output-dir ${OUTDIR_DEFAULT}" ;;
esac

echo "[get_data] outdir : ${OUTDIR_DEFAULT}"
echo "[get_data] n-jobs : ${SLURM_CPUS_PER_TASK:-4}"
echo "[get_data] args   : $*"
df -h "${LUMI_SCRATCH}" 2>/dev/null | tail -1 | sed 's/^/[get_data] scratch: /'

srun --unbuffered python3 get_data.py \
    --n-jobs "${SLURM_CPUS_PER_TASK:-4}" \
    --skip-existing \
    ${OUTDIR_ARG} \
    "$@"

echo "[get_data] done"
du -sh "${OUTDIR_DEFAULT}" 2>/dev/null | sed 's/^/[get_data] wrote: /'

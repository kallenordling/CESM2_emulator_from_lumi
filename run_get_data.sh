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

cd "${LUMI_REPO}"
mkdir -p logs

# ── environment ─────────────────────────────────────────────────────────────
# CPU module, NOT the PyTorch one. On Roihu the GPU nodes are Grace/ARM64 and
# python-pytorch/2.10 is an aarch64 container, which dies on the x86_64 CPU
# partitions with "the image's architecture (arm64) could not run on the host's
# (amd64)". This job is a download: it needs xarray/intake/s3fs, not torch.
# Roihu's AIDA stack (python-data, python-pytorch) is built ONLY for aarch64:
#   /appl/soft/manual/aida/aarch64/python-data/.../container.sif
# so on the x86_64 CPU partitions there is no python module to load at all and
# `module load python-data` dies with an architecture error. Set
# SITE_MODULE_CMD_CPU="" (explicitly empty) to skip the module entirely and rely
# on a self-contained venv built from /usr/bin/python3. Note ${VAR-default},
# not ${VAR:-default}: an empty value must mean "no module", not "use the
# default".
module purge
_MODCMD="${SITE_MODULE_CMD_CPU-${SITE_MODULE_CMD}}"
if [ -n "${_MODCMD}" ]; then
    eval "${_MODCMD}" || {
        echo "[get_data] module load failed on $(uname -m). Roihu's python" >&2
        echo "[get_data] modules are aarch64-only, so on an x86_64 partition" >&2
        echo "[get_data] use a self-contained venv instead:" >&2
        echo "[get_data]   SITE_MODULE_CMD_CPU='' sbatch ... run_get_data.sh" >&2
        echo "[get_data] having built it with /usr/bin/python3 on an x86_64 host." >&2
        exit 1
    }
else
    echo "[get_data] no module (SITE_MODULE_CMD_CPU empty) — using the venv alone"
fi
if [ -n "${LUMI_VENV_CPU:-}" ] && [ -f "${LUMI_VENV_CPU}/bin/activate" ]; then
    echo "[get_data] activating ${LUMI_VENV_CPU}"
    source "${LUMI_VENV_CPU}/bin/activate"
    # VERIFY, don't trust the directory name. The venv path is derived from
    # uname -m, but nothing stops a venv being BUILT on one architecture and
    # written to the other's path — which is exactly what happens if you pip
    # install on roihu-gpu-login2 (ARM) for a job that runs on `small` (x86).
    # Read the ELF header of a compiled extension: e_machine at offset 18 is
    # 0x3E for x86-64 and 0xB7 for aarch64.
    python3 - "${LUMI_VENV_CPU}" <<'ARCHCHK' || exit 1
import glob, os, platform, struct, sys
venv = sys.argv[1]
sos = glob.glob(os.path.join(venv, "lib", "python*", "site-packages", "**", "*.so"),
                recursive=True)
if not sos:
    print("[get_data] venv has no compiled extensions — nothing to verify")
    sys.exit(0)
E_MACHINE = {0x3E: "x86_64", 0xB7: "aarch64", 0x28: "arm", 0x14: "ppc64"}
with open(sos[0], "rb") as fh:
    hdr = fh.read(20)
mach = E_MACHINE.get(struct.unpack_from("<H", hdr, 18)[0], "unknown")
host = platform.machine()
print(f"[get_data] venv built for {mach}, host is {host}  ({os.path.basename(sos[0])})")
if mach != "unknown" and mach != host:
    print(f"[get_data] ERROR: venv architecture {mach} != host {host}.", file=sys.stderr)
    print(f"[get_data]        Rebuild it ON an {host} host — for Roihu that is",
          file=sys.stderr)
    print(f"[get_data]        roihu-cpu.csc.fi, not roihu-gpu-login2 (ARM).",
          file=sys.stderr)
    sys.exit(1)
ARCHCHK
elif [ -n "${LUMI_VENV_CPU:-}" ]; then
    echo "[get_data] no venv at ${LUMI_VENV_CPU} (arch $(uname -m)) — using the module alone"
    # A venv built on a DIFFERENT architecture is not usable here. The path is
    # arch-derived precisely so the wrong one is never picked up silently.
    for other in "${LUMI_PROJAPPL}"/venvs/diffesm-*; do
        [ -d "${other}" ] && [ "${other}" != "${LUMI_VENV_CPU}" ] \
            && echo "[get_data]   note: ${other} exists but is for another architecture"
    done
fi
echo "[get_data] arch   $(uname -m)"
echo "[get_data] python $(command -v python3) — $(python3 -V 2>&1)"

# get_data.py needs intake-esm / s3fs, which the PyTorch module may not carry.
missing=""
for m in xarray intake intake_esm s3fs joblib; do
    python3 -c "import ${m}" 2>/dev/null || missing="${missing} ${m}"
done
if [ -n "${missing}" ]; then
    echo "[get_data] ERROR: missing python packages:${missing}" >&2
    echo "[get_data]        This host is $(uname -m). Build the venv ON A HOST OF" >&2
    echo "[get_data]        THIS ARCHITECTURE — roihu-gpu-login2 is ARM64 and will" >&2
    echo "[get_data]        silently produce aarch64 wheels no matter what the" >&2
    echo "[get_data]        directory is called:" >&2
    echo "[get_data]          ssh roihu-cpu.csc.fi        # x86_64 login node" >&2
    echo "[get_data]          ${SITE_MODULE_CMD_CPU:-${SITE_MODULE_CMD}}" >&2
    echo "[get_data]          python3 -m venv --system-site-packages ${LUMI_VENV_CPU:-${LUMI_VENV}}" >&2
    echo "[get_data]          source ${LUMI_VENV_CPU:-${LUMI_VENV}}/bin/activate" >&2
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

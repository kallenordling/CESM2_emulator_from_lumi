#!/bin/bash
# Single source of truth for the LUMI project id and every path derived from it.
#
# Set the project ONCE, here or in your environment:
#
#     export LUMI_PROJECT=462001112        # in ~/.bashrc on LUMI, or
#     LUMI_PROJECT=462001112 sbatch ...    # per submission
#
# and every launcher picks it up. Sourced by the run_*.sh scripts; also exports
# the variables so the python side (lumi_paths.py) and the YAML configs
# (${oc.env:LUMI_SCRATCH,...}) resolve to the same place.
#
# THE ACCOUNT IS SPECIAL. SLURM does NOT expand variables inside #SBATCH lines,
# so `#SBATCH --account=${LUMI_PROJECT}` silently does the wrong thing. The
# account therefore comes from, in order of precedence:
#   1. `sbatch --account=...` on the command line  (what lsubmit.sh does)
#   2. the SBATCH_ACCOUNT environment variable      (exported below)
# and assert_account() re-checks it INSIDE the job, so a mismatch fails in the
# first seconds instead of after the data paths turn out to be unwritable.

# shellcheck disable=SC2155

# ── SITE ─────────────────────────────────────────────────────────────────────
# The variables below keep their LUMI_* names on every site. That is a
# historical name, not a claim about the machine: ~85 files source this file
# and read LUMI_REPO / LUMI_DATA / LUMI_ACCOUNT, and renaming them would be a
# large diff for no behavioural gain. Only the VALUES change per site.
#
#   export SITE=roihu           # then everything below points at Roihu
#
# Roihu differs structurally from LUMI: there is no /projappl, so the repo and
# venv live under the project's scratch instead. GPUs are NVIDIA/CUDA rather
# than AMD/ROCm, which matters for the container and for the ROCm-specific
# workaround in models/video_net.py.
SITE="${SITE:-lumi}"

case "${SITE}" in
  lumi)
    LUMI_PROJECT="${LUMI_PROJECT:-462001328}"
    export LUMI_ACCOUNT="project_${LUMI_PROJECT}"
    export LUMI_SCRATCH="/scratch/project_${LUMI_PROJECT}"
    export LUMI_PROJAPPL="/projappl/project_${LUMI_PROJECT}"
    export LUMI_REPO="${LUMI_PROJAPPL}/CESM2_emulator_from_lumi"
    export LUMI_VENV="${LUMI_PROJAPPL}/venvs/diffesm_laif"
    export SITE_GPU_VENDOR="amd"
    export SITE_MODULE_CMD="${SITE_MODULE_CMD:-module use /appl/local/laifs/modules; module load lumi-aif-singularity-bindings}"
    export SITE_CONTAINER="${SITE_CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
    export SITE_GPU_PARTITION="${SITE_GPU_PARTITION:-standard-g}"
    export SITE_GPUS_PER_NODE="${SITE_GPUS_PER_NODE:-8}"
    export SITE_GRES="${SITE_GRES:-gpu:8}"
    export SITE_MODULE_CMD_CPU="${SITE_MODULE_CMD_CPU:-${SITE_MODULE_CMD}}"
    export LUMI_VENV_CPU="${LUMI_VENV_CPU:-${LUMI_VENV}}"
    ;;
  roihu)
    # Verified from scripts/probe_site.sh on roihu-gpu-login2, 2026-08-14.
    LUMI_PROJECT="${LUMI_PROJECT:-2019839}"
    export LUMI_ACCOUNT="project_${LUMI_PROJECT}"
    export LUMI_SCRATCH="/scratch/project_${LUMI_PROJECT}"   # 250 G, 180-day cleanup
    export LUMI_PROJAPPL="/projappl/project_${LUMI_PROJECT}" # 15 G — code only
    export LUMI_REPO="${LUMI_PROJAPPL}/CESM2_emulator_from_lumi"
    export LUMI_VENV="${LUMI_PROJAPPL}/venvs/diffesm"
    export SITE_GPU_VENDOR="nvidia"
    # NO CONTAINER NEEDED. Roihu ships PyTorch as a module and has no site .sif
    # images, so the LUMI pattern (singularity exec + injected venv) has nothing
    # to mirror. Load the module instead; use tykky (0.5.2) only if extra
    # packages are required beyond what the module provides.
    # ROIHU IS HETEROGENEOUS — this bit costs a job if you miss it.
    #   GPU partitions (gh200)          : Grace CPU, ARM64  (aarch64)
    #   CPU partitions (small/medium/…) : x86_64            (amd64)
    # python-pytorch/2.10 is an aarch64 container and CANNOT run on the CPU
    # partitions:
    #   FATAL: the image's architecture (arm64) could not run on the host's (amd64)
    # So GPU work and CPU work need DIFFERENT modules, and any venv with
    # compiled wheels is architecture-specific too — one venv cannot serve both.
    export SITE_MODULE_CMD="${SITE_MODULE_CMD:-module load python-pytorch/2.10}"
    # CPU-only jobs (downloads, regridding, figures) — no torch needed.
    export SITE_MODULE_CMD_CPU="${SITE_MODULE_CMD_CPU:-module load python-data/3.12}"
    export SITE_CONTAINER=""
    # Venvs are per-architecture, and the name is DERIVED FROM uname -m rather
    # than hardcoded. Building on roihu-gpu-login2 (ARM) produces aarch64
    # wheels; a venv called "-x86" full of aarch64 binaries then fails on the
    # x86_64 CPU partitions with an ELF error. Deriving the name means a job
    # only ever finds the venv built for the node it is running on, and a
    # wrong-arch build simply lands somewhere else instead of poisoning the
    # right path. Build the CPU venv on an x86_64 host (roihu-cpu.csc.fi).
    export LUMI_VENV_CPU="${LUMI_VENV_CPU:-${LUMI_PROJAPPL}/venvs/diffesm-$(uname -m)}"
    # GH200, 4 per node. Partitions available to project_2019839:
    #   gputest        15 min   (smoke tests)
    #   gpuinteractive 12 h
    #   gpumedium      1-12:00  (the workhorse)
    #   gpularge       1-12:00
    export SITE_GPU_PARTITION="${SITE_GPU_PARTITION:-gpumedium}"
    export SITE_GPUS_PER_NODE="${SITE_GPUS_PER_NODE:-4}"
    export SITE_GRES="${SITE_GRES:-gpu:gh200:4}"
    ;;
  *)
    echo "[lumi_env] ERROR: unknown SITE='${SITE}' (expected lumi or roihu)" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac
export SITE
export LUMI_PROJECT
export LUMI_PKGS="${LUMI_SCRATCH}/python_packages"
export LUMI_DATA="${LUMI_SCRATCH}/emulator_data"
export LUMI_EVAL_OUT="${LUMI_SCRATCH}/eval_output"

# Container-internal view of projappl. Some launchers need this exact prefix
# because the bind mount inside the singularity image resolves differently.
if [ "${SITE}" = "lumi" ]; then
    export LUMI_REPO_PFS="/pfs/lustrep1/projappl/project_${LUMI_PROJECT}/CESM2_emulator_from_lumi"
else
    export LUMI_REPO_PFS="${LUMI_REPO}"   # no /pfs bind-mount view off LUMI
fi

# Make `sbatch` default to the right account even when called bare.
export SBATCH_ACCOUNT="${LUMI_ACCOUNT}"
export SALLOC_ACCOUNT="${LUMI_ACCOUNT}"

# Prefer the container-internal repo path when it exists (see run_eval_aero.sh).
if [ -d "${LUMI_REPO_PFS}" ]; then
    export LUMI_WORK_DIR="${LUMI_REPO_PFS}"
else
    export LUMI_WORK_DIR="${LUMI_REPO}"
fi

# Abort a running job whose account does not match LUMI_PROJECT. Catches the
# case where the launcher was submitted with a stale SBATCH_ACCOUNT or an
# explicit --account, which would bill (and write) to the wrong project.
assert_account() {
    if [ -n "${SLURM_JOB_ACCOUNT:-}" ] && [ "${SLURM_JOB_ACCOUNT}" != "${LUMI_ACCOUNT}" ]; then
        echo "[lumi_env] ERROR: job account '${SLURM_JOB_ACCOUNT}' != LUMI_ACCOUNT" \
             "'${LUMI_ACCOUNT}' (LUMI_PROJECT=${LUMI_PROJECT})." >&2
        echo "[lumi_env]        Submit with: bash lsubmit.sh <script.sh>" >&2
        echo "[lumi_env]        or export LUMI_PROJECT to match the account." >&2
        exit 1
    fi
    return 0
}

# One-line banner so every job log opens with the resolved configuration; a
# misfire is then visible in the first lines instead of after hours.
lumi_env_banner() {
    echo "[lumi_env] SITE=${SITE} LUMI_PROJECT=${LUMI_PROJECT} account=${LUMI_ACCOUNT}" \
         "scratch=${LUMI_SCRATCH} repo=${LUMI_WORK_DIR}"
}

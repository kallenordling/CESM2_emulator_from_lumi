#!/bin/bash
#SBATCH --job-name=diffesm
#SBATCH --output=logs/roihu_%x_%j.out
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gh200:4
#SBATCH --time=12:00:00
#
# Generic Roihu launcher. SBATCH lines cannot expand variables, so partition,
# gres and account are literal here — change them in the header, not by export.
# Submit with the account on the command line:
#
#     sbatch --account=project_2019839 run_roihu.sh main_aero.py
#     sbatch --account=project_2019839 --partition=gputest --time=00:15:00 \
#            run_roihu.sh eval_aero.py --experiments ssp370-126aer
#
# WHY THIS IS NOT A COPY OF run_eval_aero.sh
# ------------------------------------------
# LUMI's launchers do `module use /appl/local/laifs/modules` then
# `singularity exec lumi-multitorch-latest.sif python ...` with the project venv
# injected via SINGULARITYENV_PYTHONPATH. None of that applies here. Verified on
# roihu-gpu-login2 (scripts/probe_site.sh, 2026-08-14): Roihu ships PyTorch as a
# MODULE (python-pytorch/2.10) and has no site .sif images, so there is nothing
# to mirror and no container layer at all. Use tykky/0.5.2 only if you need
# packages the module lacks.
#
# HARDWARE DIFFERENCE THAT MATTERS
# --------------------------------
# GH200 (NVIDIA Grace Hopper), 4 per node, versus LUMI's MI250X. Two
# consequences for this codebase:
#   * CUDA, not ROCm. The torch.compiler.disable workaround at
#     models/video_net.py:49 exists for an MI250X signal-11 under gradient
#     checkpointing and is unnecessary here — it only costs the compiled
#     recomputation path. It is NOT yet made conditional; see SITE_GPU_VENDOR.
#   * Far more HBM per GPU than an MI250X GCD, which is what makes seq_len: 12
#     plausible without dropping batch_size to 1. Confirm the actual figure with
#     nvidia-smi on a compute node — the login node has none, so probe_site.sh
#     could not report it.
#
# 4 GPUs/node here vs 8 on LUMI: to keep the effective batch at 256, halve
# gradient_accumulation_steps or double batch_size. See memory
# node_count_lr_tradeoff — adjust accumulation BEFORE touching the LR.
set -euo pipefail

export SITE=roihu
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

SCRIPT="${1:?usage: sbatch --account=project_2019839 run_roihu.sh <script.py> [args...]}"
shift || true

# `module` is a SHELL FUNCTION, not a binary, and sbatch only inherits it when
# the submitting shell happened to have it defined. Submitting over a
# non-interactive ssh (`ssh host 'sbatch ...'`) does not, and the job then dies
# on `module: command not found` two seconds in — job 743114. Define it here
# instead of depending on the submitter's environment. zz-csc-env.sh also sets
# the arch-correct MODULEPATH, which matters because the login node is x86_64
# and the GH200 nodes are aarch64.
if ! declare -F module >/dev/null 2>&1 && ! command -v module >/dev/null 2>&1; then
    # `set -u` must come off while sourcing: zz-csc-env.sh reads $PS1, which is
    # unset in a batch shell, and would abort the job on "PS1: unbound variable".
    # It also RETURNS IMMEDIATELY in a non-interactive shell unless
    # CSC_ENV_INIT_NON_INTERACTIVE=yes, which is why sourcing it alone left
    # `module` undefined (job 743157). Source both, in order: the CSC script for
    # the arch-correct MODULEPATH, then lmod's own init for the function itself.
    set +u
    export CSC_ENV_INIT_NON_INTERACTIVE=yes
    for _init in /etc/profile.d/zz-csc-env.sh /usr/share/lmod/lmod/init/bash; do
        [ -r "${_init}" ] && source "${_init}"
        declare -F module >/dev/null 2>&1 && break
    done
    set -u
fi
if ! declare -F module >/dev/null 2>&1 && ! command -v module >/dev/null 2>&1; then
    echo "[roihu] ERROR: no 'module' command and none of the init scripts" >&2
    echo "[roihu]        defined one. Cannot load ${SITE_MODULE_CMD}." >&2
    exit 1
fi

module purge
eval "${SITE_MODULE_CMD}"

# The module provides torch, numpy, xarray and friends but NOT this repo's
# extras — einops-exts, ema-pytorch, beartype, diffusers, accelerate, hydra.
# Job 695994 died on `ModuleNotFoundError: einops_exts` after successfully
# reporting all four GH200s, i.e. the environment was fine and only the
# packages were missing.
#
# Install them to LUMI_PKGS with `pip install --target` and inject via
# PYTHONPATH rather than building a venv. This mirrors what the LUMI launchers
# already do (SINGULARITYENV_PYTHONPATH) and, more importantly, it does not
# depend on `python3 -m venv` behaving sensibly inside a containerised module —
# python-pytorch/2.10 is a singularity image, and a venv created against a
# container python breaks as soon as the module version changes underneath it.
# A --target directory is just files on a path: no interpreter symlink to go
# stale. See scripts/setup_roihu_pkgs.sh.
if [ -d "${LUMI_PKGS}" ]; then
    export PYTHONPATH="${LUMI_PKGS}${PYTHONPATH:+:${PYTHONPATH}}"
    echo "[roihu] PYTHONPATH += ${LUMI_PKGS}"
fi

# Fail HERE, on the login-visible banner, rather than four ranks deep inside
# srun where the traceback is printed once per task and the real message is
# buried in "CANCELLED AT ... DUE TO TASK FAILURE".
missing=""
for m in einops einops_exts ema_pytorch beartype diffusers accelerate hydra omegaconf xarray; do
    python3 -c "import ${m}" 2>/dev/null || missing="${missing} ${m}"
done
if [ -n "${missing}" ]; then
    echo "[roihu] ERROR: missing python packages:${missing}" >&2
    echo "[roihu]        Install them once, ON THE GPU LOGIN NODE (aarch64 —" >&2
    echo "[roihu]        wheels built on an x86 login node will not load here):" >&2
    echo "[roihu]          bash scripts/setup_roihu_pkgs.sh" >&2
    exit 1
fi

echo "[roihu] python  $(command -v python3)"
python3 -c "import torch; print(f'[roihu] torch {torch.__version__} cuda={torch.cuda.is_available()} n_gpu={torch.cuda.device_count()}')"
python3 -c "import torch; [print(f'[roihu]   gpu{i}: {torch.cuda.get_device_name(i)} '
            f'{torch.cuda.get_device_properties(i).total_memory/2**30:.0f} GiB')
            for i in range(torch.cuda.device_count())]" 2>/dev/null || true

cd "${LUMI_REPO}"
echo "[roihu] cwd=$(pwd)  data=${LUMI_DATA}"
echo "[roihu] running: ${SCRIPT} $*"

# One task per GPU, matching --ntasks-per-node to --gres.
srun --unbuffered python3 "${SCRIPT}" "$@"

#!/bin/bash
# Report everything needed to write correct launchers for a new cluster.
#
# WHY: the run_*.sh scripts hardcode LUMI's AI stack —
#   module use /appl/local/laifs/modules
#   singularity exec /appl/local/laifs/containers/lumi-multitorch-latest.sif
# None of that exists elsewhere. Rather than guess at Roihu's equivalents and
# produce launchers that look right and fail at submit time, run this there and
# paste the output back.
#
# READ-ONLY. Runs nothing, installs nothing, submits nothing.
#
# Usage (on the target cluster):
#   bash scripts/probe_site.sh
#   bash scripts/probe_site.sh > site_probe.txt 2>&1
set -uo pipefail

hr() { printf '\n== %s %s\n' "$1" "$(printf '=%.0s' $(seq 1 $((60 - ${#1}))))"; }

hr "IDENTITY"
echo "  host      $(hostname -f 2>/dev/null || hostname)"
echo "  user      $(whoami)"
echo "  groups    $(groups 2>/dev/null | tr ' ' '\n' | grep -i project | tr '\n' ' ')"
[ -n "${CSC_PROJECT:-}" ] && echo "  CSC_PROJECT  ${CSC_PROJECT}"

hr "FILESYSTEMS"
for d in /scratch /projappl /project /users "${HOME}"; do
    [ -d "$d" ] && echo "  exists: $d" || echo "  absent: $d"
done
echo "  --- project dirs visible ---"
ls -d /scratch/project_* /projappl/project_* 2>/dev/null | head -10 | sed 's/^/    /'
echo "  --- quota ---"
command -v lfs >/dev/null 2>&1 && lfs quota -h "${HOME}" 2>/dev/null | head -4 | sed 's/^/    /'
command -v csc-workspaces >/dev/null 2>&1 && csc-workspaces 2>/dev/null | head -20 | sed 's/^/    /'

hr "SCHEDULER"
command -v sbatch >/dev/null 2>&1 && echo "  sbatch: $(command -v sbatch)" || echo "  NO sbatch"
echo "  --- partitions ---"
sinfo -o "%20P %10a %10l %10D %15G" 2>/dev/null | head -20 | sed 's/^/    /'
echo "  --- your accounts ---"
sacctmgr -nP show assoc user="$(whoami)" format=Account,Partition 2>/dev/null | head -10 | sed 's/^/    /'

hr "GPU"
command -v nvidia-smi >/dev/null 2>&1 \
    && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null | sed 's/^/    /' \
    || echo "    no nvidia-smi on this node (login nodes often have none — check a compute node)"
command -v rocm-smi >/dev/null 2>&1 && echo "    rocm-smi present (AMD)"

hr "MODULES — pytorch / python / container runtimes"
if command -v module >/dev/null 2>&1 || [ -n "${MODULESHOME:-}" ]; then
    for pat in pytorch torch python cuda singularity apptainer tykky container ai; do
        out=$(module -t avail "$pat" 2>&1 | grep -v '^/' | grep -v '^$' | head -6)
        [ -n "$out" ] && { echo "  [$pat]"; echo "$out" | sed 's/^/    /'; }
    done
else
    echo "  no module system found"
fi

hr "CONTAINER RUNTIME"
for c in singularity apptainer; do
    command -v $c >/dev/null 2>&1 && echo "  $c: $($c --version 2>&1 | head -1)"
done
echo "  --- site container images, if any ---"
for d in /appl/local/*/containers /appl/containers /appl/soft/ai /projappl/*/containers; do
    ls -1 $d/*.sif 2>/dev/null | head -5 | sed 's/^/    /'
done

hr "EXISTING PYTHON"
command -v python3 >/dev/null 2>&1 && echo "  python3: $(python3 -V 2>&1) at $(command -v python3)"
python3 -c "import torch;print('  torch',torch.__version__,'cuda',torch.cuda.is_available())" 2>/dev/null \
    || echo "  torch not importable in the default environment"

hr "WHAT TO SEND BACK"
cat <<'EOF'
  The launchers need, specifically:
    1. the module line(s) that make PyTorch+CUDA available
    2. the container image path, if the site provides one
    3. the partition name for GPU jobs, and the account string
    4. GPUs per node and their memory
  Paste this whole output; that covers all four.
EOF

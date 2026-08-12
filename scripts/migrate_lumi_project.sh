#!/bin/bash
# Rewrite every hardcoded LUMI project id in the repo, old -> new.
#
# YOU PROBABLY DO NOT NEED THIS ANY MORE. The project is a parameter now:
#
#     export LUMI_PROJECT=462001112
#
# and lumi_env.sh (shell) + lumi_paths.py (python) derive every path from it,
# including the ones inside the YAML configs (resolved at load). This script
# remains for the one case that variable cannot cover: permanently changing the
# RECORDED DEFAULT baked into the configs and the remaining .json/.md files, so
# an environment that never sets LUMI_PROJECT lands on the new project.
#
# The project id is not configurable anywhere: it is baked into ~400 string
# literals across SBATCH --account lines, /scratch data paths, /projappl repo and
# venv paths, and the /pfs/lustrep1 container-internal variant. This script does
# the mechanical part; it does NOT move data or build the venv (see the checklist
# it prints at the end, and README notes below).
#
# Usage:
#   bash scripts/migrate_lumi_project.sh <new_id> [old_id]      # dry run
#   bash scripts/migrate_lumi_project.sh <new_id> [old_id] --apply
#
#   <new_id>  digits only, e.g. 462001112 (NOT "project_462001112")
#   [old_id]  defaults to 462001328
#
# Always dry-run first and read the summary. Commit before --apply so the whole
# rewrite is one revertible commit.
set -euo pipefail

NEW="${1:?usage: bash scripts/migrate_lumi_project.sh <new_id> [old_id] [--apply]}"
OLD="${2:-462001328}"
APPLY=0
for a in "$@"; do [ "$a" = "--apply" ] && APPLY=1; done
# allow `... <new> --apply` with old_id omitted
[ "${OLD}" = "--apply" ] && OLD=462001328

if ! [[ "${NEW}" =~ ^[0-9]+$ ]]; then
    echo "[migrate] ERROR: new id must be digits only, got '${NEW}'" >&2
    echo "[migrate]        pass 462001112, not project_462001112" >&2
    exit 1
fi
# LUMI ids are 9 digits (project_462xxxxxx / project_465xxxxxx). An 8-digit id is
# almost always a dropped character, and a wrong --account silently fails every
# job at submit time while wrong paths fail hours later.
if [ "${#NEW}" -ne 9 ]; then
    echo "[migrate] WARNING: '${NEW}' has ${#NEW} digits; LUMI project ids have 9" >&2
    echo "[migrate]          (current id ${OLD} has ${#OLD}). Verify with:" >&2
    echo "[migrate]            lumi-workspaces      # or: groups" >&2
    if [ "${APPLY}" = "1" ]; then
        echo "[migrate]          Refusing to --apply a suspicious id." >&2
        echo "[migrate]          Re-run with MIGRATE_FORCE=1 if it really is right." >&2
        [ "${MIGRATE_FORCE:-0}" = "1" ] || exit 1
    fi
fi

cd "$(dirname "$0")/.."

# Skip .git, caches, and anything binary — the id also appears inside .nc/.pt
# blobs' metadata in principle, and sed would corrupt them.
# Exclude THIS script: it carries the old id as its default and throughout the
# checklist below, so rewriting it would clobber the default and leave no way to
# migrate again (or back). Migrations happen more than once.
SELF="./scripts/$(basename "$0")"
mapfile -t FILES < <(grep -rl "${OLD}" \
    --exclude-dir=.git --exclude-dir=__pycache__ --exclude-dir=.venv \
    --binary-files=without-match . | grep -vx -- "${SELF}" | sort)

if [ "${#FILES[@]}" -eq 0 ]; then
    echo "[migrate] no occurrences of ${OLD} — nothing to do"
    exit 0
fi

# Count over exactly the files that will be rewritten, so the number reported
# matches the number changed (the self-exclusion above must not be double-counted).
TOTAL=$(grep -o "${OLD}" "${FILES[@]}" | wc -l)
echo "[migrate] ${OLD} -> ${NEW}"
echo "[migrate] ${TOTAL} occurrences in ${#FILES[@]} files"
echo

if [ "${APPLY}" = "1" ]; then
    if [ -n "$(git status --porcelain)" ]; then
        echo "[migrate] WARNING: working tree is dirty. The rewrite will be mixed" >&2
        echo "[migrate]          in with your other changes. Ctrl-C to stop." >&2
        sleep 5
    fi
    sed -i "s/${OLD}/${NEW}/g" "${FILES[@]}"
    echo "[migrate] rewritten. Verify:"
    echo "    git diff --stat"
    echo "    grep -rn '${OLD}' --exclude-dir=.git . | grep -v __pycache__   # expect empty"
else
    echo "[migrate] DRY RUN — files that would change:"
    printf '    %s\n' "${FILES[@]}"
    echo
    echo "[migrate] re-run with --apply to write."
fi

cat <<EOF

────────────────────────────────────────────────────────────────────────────
NOT done by this script — the repo rewrite is the small half of the migration.
Run these ON LUMI, as the new project, in this order:

1. QUOTA FIRST. The data is the long pole; check the new project can hold it.
     lumi-workspaces
     du -sh /scratch/project_${OLD}/emulator_data          # training data
     du -sh /projappl/project_${OLD}/CESM2_emulator_from_lumi/runs  # checkpoints

2. TRAINING DATA — /scratch/project_${OLD}/emulator_data
   Copy with a JOB, not on a login node (it is TB-scale and login nodes are
   rate-limited / will kill it):
     mkdir -p /scratch/project_${NEW}/emulator_data
     sbatch --account=project_${NEW} --partition=small --time=24:00:00 \\
            --wrap="rsync -aH --info=progress2 \\
              /scratch/project_${OLD}/emulator_data/ \\
              /scratch/project_${NEW}/emulator_data/"
   NOTE: /scratch is PURGED (90 days on LUMI). If the old project is expiring,
   this copy is the only thing standing between you and re-downloading.

3. CHECKPOINTS — runs/*.pt under the projappl checkout. Small enough for rsync
   on a login node, but they are the irreplaceable artefact; copy them FIRST if
   the old allocation is close to ending.
     rsync -aH /projappl/project_${OLD}/CESM2_emulator_from_lumi/runs/ \\
               /projappl/project_${NEW}/CESM2_emulator_from_lumi/runs/

4. EVAL OUTPUT — /scratch/project_${OLD}/eval_output (~10 GB per eval dir, more
   for large-ensemble runs). Optional: regenerable from checkpoints, but only if
   the training data survives.

5. REPO CHECKOUT under the new projappl:
     cd /projappl/project_${NEW}
     git clone <remote> CESM2_emulator_from_lumi
     cd CESM2_emulator_from_lumi && git checkout <branch> && git pull

6. VENV — 68 of the rewritten references point at
   /projappl/project_${NEW}/venvs/diffesm_laif, which will not exist yet:
     bash setup_venv_laif.sh          # after the rewrite, so it targets the new path
   Plus the extra packages dir /scratch/project_${OLD}/python_packages
   (SINGULARITYENV_PYTHONPATH in run_eval_aero.sh) — rsync it too.

7. LOCAL SSHFS MOUNT — outside the repo, on your workstation. The mounts under
   ~/mnt/ point at the OLD project's scratch/projappl. Repoint them or every
   local analysis script silently reads stale data. A mount that has gone stale
   lists EMPTY rather than erroring.

8. SMOKE TEST before trusting anything long:
     sbatch run_debug_aero.sh        # bounded dev-g run
   Check the log's first lines for the resolved account and paths.
────────────────────────────────────────────────────────────────────────────
EOF

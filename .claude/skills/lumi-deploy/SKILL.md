---
name: lumi-deploy
description: Commit and push local changes, then print the exact LUMI pull + sbatch commands to deploy a change for training/eval. Use when the user wants to ship a change to LUMI.
disable-model-invocation: true
---

# lumi-deploy

Ship local repo changes to LUMI. The local repo is the source of truth; the
mount at `/mnt/lumi2/CESM2_emulator_from_lumi` (and the LUMI projappl checkout)
lag until pulled. **Never edit the mount directly.**

## Steps

1. **Stage & commit.** Show `git status` + `git diff --stat`. If on `main`,
   branch first. Commit with a clear message ending in:
   `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
   (Only commit/push when the user has asked — this skill implies that.)

2. **Push.** `git push origin <branch>`. Report the ref range.

3. **Print LUMI deploy commands** for the user to run on a LUMI login node
   (they run via `! <cmd>` in this session or in their own shell). Fill in the
   current branch and the relevant launcher:

   ```bash
   cd /pfs/lustrep1/projappl/project_462001328/CESM2_emulator_from_lumi
   git pull origin <branch>
   sbatch run_debug_aero.sh      # or run2_aero.sh (prod) / run_eval_aero.sh
   ```

4. **Remind** to verify the commit landed (`git log -1`) before `sbatch`, since a
   running job uses whatever is checked out at launch time.

## Notes
- Common launchers: `run2_aero.sh` (production training), `run_debug_aero.sh`
  (dev-g, bounded), `run_eval_aero.sh` (small-g eval), `watch_eval_triggers.sh`.
- Don't `sbatch` from this session — these run on LUMI, not locally.

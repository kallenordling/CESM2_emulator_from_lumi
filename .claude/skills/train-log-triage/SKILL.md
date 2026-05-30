---
name: train-log-triage
description: Summarize a SLURM training or eval log on the LUMI mount — epoch progress, latest loss line, first real error (filtering MIOpen/NCCL/libfabric noise), and loss/bias trend. Use when the user asks how training/eval is going or why a job crashed/stalled.
disable-model-invocation: true
---

# train-log-triage

Triage a LUMI SLURM log and report status concisely. Logs live at:

```
/mnt/lumi2/CESM2_emulator_from_lumi/logs/
  diffusion_aero_<jobid>.out   # training
  eval_aero_ep<NNNN>_<jobid>.out  # eval
  eval_watcher_<jobid>.out     # eval dispatcher
```

## Steps

1. **Pick the log.** If the user gave a job id or epoch, use it. Otherwise grab
   the newest matching file: `ls -t .../logs/*.out | head`. State which file.

2. **Progress.** For training: last `[EPOCH N] duration:` line + last
   `{'Training/Loss': ...}` line (report Loss, MSE, COND/TCRE/EBM, and Epoch).
   For eval: which experiments finished (`→ saved ...`, `[DONE]`), and any
   `[SHARD]` aggregation / `tcre_summary.json` write.

3. **Errors.** Grep for real failures and show the first one:
   `grep -niE "error|traceback|unbound|abort|exitcode|root cause|failed|RuntimeError" LOG`
   then **filter out the known-benign noise** — do NOT report these as problems:
   - `MIOpen(HIP): Error [Init] Not found` (kernel-cache misses)
   - `libfabric.so.1 ... cannot be preloaded`
   - `NCCL WARN NET/OFI ... RC: -38` (transport fallback)
   - `expandable_segments not supported`, xarray FutureWarning/SerializationWarning

4. **Verdict.** One line: healthy / crashed (with root cause + file:line) /
   stalled / completed. If crashed, quote the actual Python traceback.

## Notes
- All eval ranks share one log file (`--unbuffered`, 4 tasks interleaved), so
  progress bars and `[DONE]` lines from different ranks are interwoven.
- A job that "resumed from epoch N" with `max_epochs=N` exits immediately with an
  empty training loop — that's not a crash.
- Cross-check active commits with the mount before blaming code (the mount lags
  the local repo).

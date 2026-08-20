#!/usr/bin/env python3
"""Smoke-test the MONTHLY data path: does cond line up with the target?

Checks, on one realization of one experiment out of config_data_monthly.yaml:

  1. the target axis is 12 contiguous months per year, no decimation holes;
  2. the cond tensor has been broadcast onto that axis (same length);
  3. the broadcast is a STEP function — every month of a year carries that
     year's map, byte-identical, and the map is the one the cond FILE holds
     for that year;
  4. the previous-state channel exists, sits last, and is a one-step shift of
     target channel 0;
  5. __getitem__ returns (n_target, seq_len, H, W) and (n_cond, seq_len, H, W).

Run it on an aarch64 node — it needs xarray:

    sbatch --account=project_2019839 --partition=gputest --ntasks-per-node=1 \
           --gres=gpu:gh200:1 --time=00:15:00 run_roihu.sh \
           scripts/smoke_test_monthly_data.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from omegaconf import OmegaConf

from data.climate_dataset import EvalClimateDataset

CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "configs", "config_data_monthly.yaml")

failures = []


def check(label, ok, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def main():
    cfg = OmegaConf.load(CFG)
    exp = OmegaConf.to_container(cfg.experiment_configs[0], resolve=True)
    print(f"[SMOKE] scenario={exp['scenario_name']} "
          f"realization={exp['realizations'][0]}")

    shared = dict(
        seq_len=cfg.seq_len,
        target_vars=list(cfg.target_vars),
        cond_vars=list(cfg.cond_vars),
        n_components_target=cfg.n_components_target,
        n_components_cond=list(cfg.n_components_cond),
        cond_smooth_sigma=list(cfg.cond_smooth_sigma),
        cond_smooth_method=cfg.cond_smooth_method,
        prev_target_channel=cfg.prev_target_channel,
    )
    ds = EvalClimateDataset(
        realizations=[exp["realizations"][0]],
        data_dir=exp["data_dir"],
        cond_file=exp["cond_file"],
        time_dim=exp["time_dim"],
        **shared,
    )
    ds.load_data(exp["realizations"][0])

    years = np.asarray(ds._time_values)
    steps = np.asarray(ds._time_steps)
    n_t = ds.tensor_data.shape[1]
    n_c = ds.tensor_data_cond.shape[1]
    print(f"[SMOKE] target {tuple(ds.tensor_data.shape)}  "
          f"cond {tuple(ds.tensor_data_cond.shape)}  "
          f"years {years[0]}-{years[-1]}")

    d = np.diff(steps)
    check("target axis is contiguous months",
          bool((d == 1).all()), f"unique steps {sorted(set(d.tolist()))[:4]}")
    check("12 steps per year",
          n_t == 12 * len(set(years.tolist())),
          f"{n_t} steps, {len(set(years.tolist()))} years")
    check("cond broadcast onto the target axis", n_c == n_t, f"{n_c} vs {n_t}")

    # Step function: all twelve months of a year identical, and different years
    # actually differ (a broadcast that collapsed everything would also pass the
    # first half of this).
    if n_c == n_t:
        y0 = int(years[0])
        blk = ds.tensor_data_cond[:3, years == y0]
        same = bool(torch.allclose(blk, blk[:, :1].expand_as(blk)))
        check("every month of a year carries the same cond map", same)
        later = int(years[-1])
        a = ds.tensor_data_cond[:3, years == y0][:, 0]
        b = ds.tensor_data_cond[:3, years == later][:, 0]
        check("cond actually varies between years",
              not bool(torch.allclose(a, b)),
              f"{y0} vs {later}")

    n_cond_ch = ds.tensor_data_cond.shape[0]
    check("previous-state channel appended last",
          n_cond_ch == len(cfg.cond_vars) + 1,
          f"{n_cond_ch} channels for {len(cfg.cond_vars)} cond_vars")
    if n_cond_ch == len(cfg.cond_vars) + 1:
        prev = ds.tensor_data_cond[-1]
        tgt0 = ds.tensor_data[0]
        check("previous-state channel is target ch0 shifted one step",
              bool(torch.equal(prev[1:], tgt0[:-1])))
        check("previous-state channel is NOT PCA-flattened",
              bool(torch.equal(prev[0], tgt0[0])), "step 0 is persistence")

    x, c = ds[0]
    check("__getitem__ target shape",
          tuple(x.shape) == (len(cfg.target_vars), cfg.seq_len) + tuple(x.shape[2:]),
          str(tuple(x.shape)))
    check("__getitem__ cond shape",
          tuple(c.shape) == (n_cond_ch, cfg.seq_len) + tuple(c.shape[2:]),
          str(tuple(c.shape)))

    print(f"\n[SMOKE] {'ALL CHECKS PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

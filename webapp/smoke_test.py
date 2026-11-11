#!/usr/bin/env python3
"""Smoke test of the assembled Space on CPU.

    bash webapp/build_space.sh webapp/_build/space --with-bundle
    python webapp/smoke_test.py webapp/_build/space [--steps 50] [--members 2]

Samples the UNMODIFIED scenario and compares its global mean with the
25-member reference from the eval the bundle was exported from. Sampling
noise for a few members is ~0.1 K, so agreement to ~0.2 K is a pass; a
broken pipeline (wrong anchors, sigma, denorm) misses by degrees.
"""
import argparse
import os
import sys
import time

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("space_dir")
p.add_argument("--scenario", default="ssp245")
p.add_argument("--year", type=int, default=2050)
p.add_argument("--steps", type=int, default=50)
p.add_argument("--members", type=int, default=2)
a = p.parse_args()

sys.path.insert(0, os.path.abspath(a.space_dir))
os.chdir(a.space_dir)
from emulator import Edit, Emulator  # noqa: E402

t0 = time.time()
emu = Emulator("bundle", device="cpu")
print(f"load {time.time() - t0:.1f}s  scenarios={emu.scenarios}")

raw = emu.raw_cond(a.scenario, a.year)
c0 = emu.normalise(raw)
c1 = emu.normalise(emu.apply_edit(raw, Edit(sul=0.0, bc=0.0, region="East Asia")))
print("range unmodified:", {k: round(v["max"], 3) for k, v in emu.range_report(c0).items()})
print("East Asia aerosol-off changes SUL/BC cells:",
      int((np.abs(c1 - c0)[1:] > 1e-6).sum()))

t0 = time.time()
mod, base = emu.sample([c1, c0], a.members, seed=0, steps=a.steps)
dt = time.time() - t0
print(f"sampled 2 x {a.members} members x {a.steps} steps in {dt:.0f}s "
      f"({dt / (2 * a.members):.0f}s per member)")

iT = emu.target_vars.index("TREFHT")
blT = emu.baseline["TREFHT"]
g_base = [emu.gmean(m[iT] - blT) for m in base]
g_mod = [emu.gmean(m[iT] - blT) for m in mod]
ys, ref = emu.ref[("TREFHT", a.scenario)]
g_ref = emu.gmean(ref[list(ys).index(a.year)] - blT)
print(f"unmodified dT  members {np.round(g_base, 3)}  vs 25-member ref {g_ref:+.3f}")
print(f"East Asia aerosols off: dT  members {np.round(g_mod, 3)}  "
      f"paired effect {np.mean(np.array(g_mod) - np.array(g_base)):+.3f} K")
ok = abs(np.mean(g_base) - g_ref) < 0.25
print("PASS" if ok else "FAIL: unmodified global mean far from the reference")
sys.exit(0 if ok else 1)

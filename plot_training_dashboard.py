"""Training-progress dashboard for one run: loss terms, EBM coefficients, and
the comparison to CESM2 (TCRE slope ratio + per-scenario global-mean ΔT bias).

Pulls two sources:
  • training logs  logs/diffusion_aero_*.out  (chained → merged by epoch),
    filtered to the run by its checkpoint/trigger names.
  • eval summaries eval_output/<run>/best_ep*/{tcre_summary.json,
    global_mean_anomaly.csv}  — the model-vs-CESM2 numbers.

Usage:
    python plot_training_dashboard.py [--run run_sensfix]
        [--log-dir /mnt/lumi2/CESM2_emulator_from_lumi/logs]
        [--eval-dir /mnt/lumi_sc2/eval_output] [--out <png>]
"""
import argparse, csv, glob, json, os, re, subprocess
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOSS_KEYS  = ["MSE LOSS", "COND LOSS", "TCRE LOSS", "EBM LOSS", "INTER LOSS"]
EBM_KEYS   = ["EBM/alpha_ghg", "EBM/alpha_aero", "EBM/lambda"]
SCALE_KEYS = ["COND SCALE", "TCRE SCALE", "EBM SCALE"]
TRAIN_KEYS = ["Training/Loss"] + LOSS_KEYS + EBM_KEYS + SCALE_KEYS
SCENARIOS  = ["hist", "ssp370", "ssp126", "aaer", "ghg"]
SCEN_COL   = {"hist": "#1f77b4", "ssp370": "#d62728", "ssp126": "#9467bd",
              "aaer": "#2ca02c", "ghg": "#ff7f0e"}
NUM = r"(-?[0-9]+\.?[0-9]*(?:[eE][+\-]?[0-9]+)?)"


def find_run_logs(log_dir, run):
    """Logs that reference this run's checkpoints/triggers. The `_(ep|best|[0-9])`
    suffix disambiguates run_sensfix from run_sensfix_b12."""
    pat = rf"{re.escape(run)}_(ep|best|[0-9])"
    hits = []
    for f in sorted(glob.glob(os.path.join(log_dir, "diffusion_aero_*.out"))):
        if subprocess.run(["grep", "-qE", pat, f]).returncode == 0:
            hits.append(f)
    return hits


def parse_training(files):
    per_epoch = defaultdict(lambda: defaultdict(list))   # ep -> key -> [vals]
    val = {}                                             # ep -> {VAL/*: v}
    dur = {}                                             # ep -> seconds/epoch
    for f in files:
        for line in open(f, errors="ignore"):
            # [EPOCH 46] duration: 2.6 min  (154s)  steps: 40
            dm = re.search(r"\[EPOCH (\d+)\] duration:\s*[\d.]+\s*min\s*\((\d+)\s*s\)", line)
            if dm:
                dur.setdefault(int(dm.group(1)), float(dm.group(2)))
                continue
            if "'Epoch'" not in line:
                continue
            em = re.search(r"'Epoch':\s*(\d+)", line)
            if not em:
                continue
            ep = int(em.group(1))
            if "VAL/" in line:
                for k in ("VAL/MSE", "VAL/Skill"):
                    m = re.search(rf"'{re.escape(k)}':\s*(?:np\.float64\()?{NUM}", line)
                    if m:
                        val.setdefault(ep, {})[k] = float(m.group(1))
                continue
            if "Training/Loss" not in line:
                continue
            for k in TRAIN_KEYS:
                m = re.search(rf"'{re.escape(k)}':\s*{NUM}", line)
                if m:
                    per_epoch[ep][k].append(float(m.group(1)))
    train = {ep: {k: float(np.mean(v)) for k, v in d.items()}
             for ep, d in per_epoch.items()}
    return train, val, dur


def parse_evals(eval_dir, run):
    base = os.path.join(eval_dir, run)
    out = {}   # ep -> {"ratio": {scen: r}, "bias": {scen: b}}
    for d in sorted(glob.glob(os.path.join(base, "best_ep*"))):
        m = re.search(r"best_ep(\d+)", d)
        if not m:
            continue
        ep = int(m.group(1))
        rec = {"ratio": {}, "bias": {}}
        ts = os.path.join(d, "tcre_summary.json")
        if os.path.exists(ts):
            try:
                j = json.load(open(ts))
                for s, v in j.get("per_scenario", {}).items():
                    if v.get("ratio") is not None:
                        rec["ratio"][s] = float(v["ratio"])
            except Exception:
                pass
        cs = os.path.join(d, "global_mean_anomaly.csv")
        if os.path.exists(cs):
            rows = list(csv.DictReader(open(cs)))
            for s in SCENARIOS:
                a = [(float(r["model_anom_degC"]), float(r["cesm_anom_degC"]))
                     for r in rows if r["experiment"] == s]
                if a:
                    rec["bias"][s] = float(np.mean([m - c for m, c in a]))
        if rec["ratio"] or rec["bias"]:
            out[ep] = rec
    return out


def _series(d, key):
    eps = sorted(e for e in d if key in d[e])
    return np.array(eps), np.array([d[e][key] for e in eps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run_sensfix")
    ap.add_argument("--log-dir", default="/mnt/lumi2/CESM2_emulator_from_lumi/logs")
    ap.add_argument("--eval-dir", default="/mnt/lumi_sc2/eval_output")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    logs = find_run_logs(args.log_dir, args.run)
    print(f"[{args.run}] {len(logs)} training log(s)")
    train, val, dur = parse_training(logs)
    evals = parse_evals(args.eval_dir, args.run)
    if not train:
        raise SystemExit(f"No training records found for {args.run} in {args.log_dir}")
    ep_max = max(train)
    print(f"  epochs {min(train)}–{ep_max}; {len(evals)} eval checkpoints; "
          f"{len(dur)} epoch durations")

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    ax = axes.flatten()

    # 1 ── loss components (log y) ───────────────────────────────────────────
    for k in ["Training/Loss"] + LOSS_KEYS:
        e, v = _series(train, k)
        if len(e):
            ax[0].plot(e, np.clip(v, 1e-6, None), lw=1.1,
                       label=k.replace(" LOSS", "").replace("Training/Loss", "TOTAL"))
    ax[0].set_yscale("log"); ax[0].set_title("Loss components (per-epoch mean)")
    ax[0].set_xlabel("epoch"); ax[0].legend(fontsize=8, ncol=2); ax[0].grid(alpha=.3)

    # 2 ── EBM learned coefficients (no direct CESM2 ref — normalized units) ──
    for k in EBM_KEYS:
        e, v = _series(train, k)
        if len(e):
            ax[1].plot(e, v, lw=1.2, label=k.split("/")[1])
    ax[1].axhline(0, color="grey", lw=.5)
    ax[1].set_title("EBM coefficients (learned: F=αco2·CO2+αaero·SUL+λ·ΔT)")
    ax[1].set_xlabel("epoch"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    # 3 ── adaptive loss scales ───────────────────────────────────────────────
    for k in SCALE_KEYS:
        e, v = _series(train, k)
        if len(e):
            ax[2].plot(e, v, lw=1.2, label=k)
    ax[2].set_title("Adaptive loss scales"); ax[2].set_xlabel("epoch")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

    # 4 ── epoch duration (throughput / Lustre stalls / low-t overhead) ───────
    if dur:
        de = np.array(sorted(dur)); dv = np.array([dur[e] for e in de]) / 60.0  # min
        ax[3].plot(de, dv, lw=1.0, color="darkorange")
        med = float(np.median(dv))
        ax[3].axhline(med, color="grey", ls="--", lw=.8, label=f"median {med:.1f} min")
        ax[3].set_ylim(0, max(dv.max() * 1.1, med * 1.5))
        ax[3].legend(fontsize=8)
    ax[3].set_title("Epoch duration (min)"); ax[3].set_xlabel("epoch"); ax[3].grid(alpha=.3)

    # 5 ── TCRE ratio model/CESM2 (=1 perfect) — the CESM2 sensitivity compare ─
    for s in SCENARIOS:
        eps = sorted(e for e in evals if s in evals[e]["ratio"])
        if eps:
            ax[4].plot(eps, [evals[e]["ratio"][s] for e in eps], "o-", ms=3,
                       color=SCEN_COL[s], label=s)
    ax[4].axhline(1.0, color="k", ls="--", lw=1, label="CESM2 (=1)")
    ax[4].set_ylim(0.8, 2.2); ax[4].set_title("TCRE slope ratio  model / CESM2")
    ax[4].set_xlabel("eval epoch"); ax[4].legend(fontsize=8); ax[4].grid(alpha=.3)

    # 6 ── per-scenario global-mean ΔT bias model−CESM2 (=0 perfect) ──────────
    for s in SCENARIOS:
        eps = sorted(e for e in evals if s in evals[e]["bias"])
        if eps:
            ax[5].plot(eps, [evals[e]["bias"][s] for e in eps], "o-", ms=3,
                       color=SCEN_COL[s], label=s)
    ax[5].axhline(0.0, color="k", ls="--", lw=1, label="CESM2 (=0)")
    ax[5].set_title("Global-mean ΔT bias  model − CESM2  (°C)")
    ax[5].set_xlabel("eval epoch"); ax[5].legend(fontsize=8); ax[5].grid(alpha=.3)

    # 7 ── held-out validation ───────────────────────────────────────────────
    e, v = _series(val, "VAL/Skill")
    if len(e):
        ax[6].plot(e, v, lw=1.1, color="purple", label="VAL/Skill")
    e2, v2 = _series(val, "VAL/MSE")
    if len(e2):
        axb = ax[6].twinx(); axb.plot(e2, v2, lw=1.0, color="teal", alpha=.6); axb.set_ylabel("VAL/MSE", color="teal")
    ax[6].set_title("Held-out validation"); ax[6].set_xlabel("epoch")
    ax[6].legend(fontsize=8, loc="upper left"); ax[6].grid(alpha=.3)

    ax[7].set_visible(False)   # 2×4 grid, 7 panels used

    fig.suptitle(f"Training dashboard — {args.run}  |  epoch {min(train)}–{ep_max}  "
                 f"|  {len(evals)} evals", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out or f"training_dashboard_{args.run}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

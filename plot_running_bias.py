"""Plot global-mean bias (model - CESM2) for the currently-running A/B arms.

Reads each run's latest ``global_mean_anomaly.csv`` (columns
``experiment,year,model_anom_degC,cesm_anom_degC,bias_degC``) and produces:

  * bias_curves.png  — one panel per scenario, bias(year) overlaid per run
  * bias_summary.png — grouped bar of the time-mean |bias| per scenario per run
  * a printed table of time-mean bias per scenario per run

By default it auto-discovers the newest CSV under each ``run_*`` directory in
EVAL_ROOT, restricted to ``--runs`` if given. Point EVAL_ROOT at the scratch
mount (default /mnt/lumi_sc/eval_output).

    python plot_running_bias.py --runs sensfix intssp mseyb ybias
    EVAL_ROOT=/some/other/eval_output python plot_running_bias.py
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVAL_ROOT = os.environ.get("EVAL_ROOT", "/mnt/lumi_sc/eval_output")
SCENARIOS = ["hist", "ssp370", "ssp126", "ghg", "aaer"]


def newest_csv(run_dir: str):
    """Newest global_mean_anomaly.csv under run_dir/best_ep*/ (by mtime)."""
    cands = glob.glob(os.path.join(run_dir, "best_ep*", "global_mean_anomaly.csv"))
    cands += glob.glob(os.path.join(run_dir, "manual_ep*", "global_mean_anomaly.csv"))
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def discover_runs(wanted):
    """Map run-name -> newest CSV path. ``wanted`` is a list of bare names or None."""
    out = {}
    for run_dir in sorted(glob.glob(os.path.join(EVAL_ROOT, "run_*"))):
        name = os.path.basename(run_dir)[len("run_"):]
        if wanted and name not in wanted:
            continue
        csv = newest_csv(run_dir)
        if csv:
            out[name] = csv
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["sensfix", "intssp", "mseyb", "ybias"],
                    help="run names (without the run_ prefix). Default = the 4 live arms.")
    ap.add_argument("--out-prefix", default="bias")
    args = ap.parse_args()

    runs = discover_runs(args.runs)
    if not runs:
        raise SystemExit(f"No CSVs found under {EVAL_ROOT} for runs={args.runs}")

    data = {}      # run -> DataFrame
    epoch = {}     # run -> epoch label
    for name, csv in runs.items():
        data[name] = pd.read_csv(csv)
        epoch[name] = os.path.basename(os.path.dirname(csv))
        print(f"[load] {name:10s} {epoch[name]:14s} {csv}")

    colors = dict(zip(runs, plt.cm.tab10(np.linspace(0, 1, max(len(runs), 3)))))

    # ── Figure 1: bias(year) per scenario, line per run ──────────────────────
    ncol = 3
    nrow = int(np.ceil(len(SCENARIOS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4 * nrow), squeeze=False)
    for si, sc in enumerate(SCENARIOS):
        ax = axes[si // ncol][si % ncol]
        for name, df in data.items():
            d = df[df["experiment"] == sc]
            if d.empty:
                continue
            ax.plot(d["year"], d["bias_degC"], lw=1.6,
                    color=colors[name], label=f"{name} ({epoch[name]})")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_title(f"{sc}  —  bias (model − CESM2)")
        ax.set_xlabel("year"); ax.set_ylabel("bias [°C]")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    for k in range(len(SCENARIOS), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Global-mean bias per scenario — running arms", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out1 = f"{args.out_prefix}_curves.png"
    fig.savefig(out1, dpi=130); plt.close(fig)
    print(f"[plot] {out1}")

    # ── time-mean bias table + Figure 2: grouped bars ────────────────────────
    print(f"\n{'scenario':>8} | " + " | ".join(f"{n:>10}" for n in runs))
    meanbias = {sc: {} for sc in SCENARIOS}
    for sc in SCENARIOS:
        row = []
        for name, df in data.items():
            d = df[df["experiment"] == sc]
            mb = float(d["bias_degC"].mean()) if not d.empty else np.nan
            meanbias[sc][name] = mb
            row.append(f"{mb:+10.3f}")
        print(f"{sc:>8} | " + " | ".join(row))

    fig, ax = plt.subplots(figsize=(1.6 * len(SCENARIOS) + 3, 5))
    x = np.arange(len(SCENARIOS))
    w = 0.8 / max(len(runs), 1)
    for ri, name in enumerate(runs):
        vals = [meanbias[sc].get(name, np.nan) for sc in SCENARIOS]
        ax.bar(x + ri * w - 0.4 + w / 2, vals, w,
               color=colors[name], label=f"{name} ({epoch[name]})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(SCENARIOS)
    ax.set_ylabel("time-mean bias [°C]")
    ax.set_title("Time-mean global bias per scenario — running arms")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=9)
    plt.tight_layout()
    out2 = f"{args.out_prefix}_summary.png"
    fig.savefig(out2, dpi=130); plt.close(fig)
    print(f"[plot] {out2}")


if __name__ == "__main__":
    main()

"""Parse and plot training metrics from diffusion_aero_16* SLURM log files."""
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = Path("/mnt/lumi2/CESM2_emulator_from_lumi/logs")
PATTERN = "diffusion_aero_*.out"

METRICS = [
    ("VAL/MSE",  "Val MSE"),
    ("VAL/COND", "Val Cond"),
    ("VAL/DISC", "Val Disc"),
]

# Match lines like: {'VAL/MSE': np.float64(0.001), ...} {'Epoch': 177, ...}
METRIC_RE  = re.compile(r"'(VAL/[^']+)':\s*(?:np\.float64\()?([0-9eE+\-.]+)\)?")
EPOCH_RE   = re.compile(r"'Epoch':\s*(\d+)")


def parse_log(path: Path):
    """Return {epoch: metrics_dict} from a log file."""
    records = {}
    with open(path) as f:
        for line in f:
            if "'Epoch'" not in line or "VAL/" not in line:
                continue
            em = EPOCH_RE.search(line)
            if not em:
                continue
            epoch = int(em.group(1))
            if epoch in records:
                continue   # keep first occurrence (deduplicate GPU ranks)
            metrics = {k: float(v) for k, v in METRIC_RE.findall(line)}
            if metrics:
                records[epoch] = metrics
    return records


def main():
    log_files = sorted(glob.glob(str(LOG_DIR / PATTERN)))
    print(f"Found {len(log_files)} log files")

    all_records: dict[int, dict] = {}
    file_info = []
    for path in log_files:
        records = parse_log(Path(path))
        if not records:
            print(f"  {Path(path).name}: no training data")
            continue
        epochs = sorted(records)
        new = sum(1 for e in epochs if e not in all_records)
        for e, m in records.items():
            if e not in all_records:
                all_records[e] = m
        print(f"  {Path(path).name}: epochs {min(epochs)}–{max(epochs)}, {new} new")
        file_info.append((Path(path).name, min(epochs), max(epochs)))

    if not all_records:
        print("No data found.")
        return

    epochs = np.array(sorted(all_records.keys()))
    print(f"\nTotal unique epochs: {len(epochs)} (range {epochs[0]}–{epochs[-1]})")

    # Identify best epoch by VAL/MSE (lower is better)
    mses = np.array([all_records[e].get("VAL/MSE", np.nan) for e in epochs])
    best_idx = int(np.nanargmin(mses))
    best_ep  = epochs[best_idx]
    best_mse = mses[best_idx]
    print(f"Best VAL/MSE = {best_mse:.6f} at epoch {best_ep}")

    # Job restart boundaries
    restarts = sorted({info[1] for info in file_info[1:]})

    ncols = 2
    nrows = (len(METRICS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows))
    axes = axes.flatten()

    for ax_i, (key, label) in enumerate(METRICS):
        ax = axes[ax_i]
        vals = np.array([all_records[e].get(key, np.nan) for e in epochs])
        ax.plot(epochs, vals, lw=1.2, color="steelblue", label=label)
        ax.axvline(best_ep, color="crimson", lw=1.0, linestyle="--", alpha=0.7,
                   label=f"Best (ep {best_ep})")
        ax.scatter([best_ep], [all_records[best_ep].get(key, np.nan)],
                   color="crimson", s=40, zorder=5)
        for b in restarts:
            ax.axvline(b, color="gray", lw=0.8, linestyle=":", alpha=0.6)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[len(METRICS):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Training — slope-tcre  |  {len(epochs)} epochs  |  "
        f"Best MSE {best_mse:.6f} @ ep {best_ep}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out_path = Path("/home/nordling/PycharmProjects/CESM2_emulator_from_lumi/training_progress.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()

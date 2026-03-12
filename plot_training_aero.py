"""Parse and plot training metrics from diffusion_aero_16* SLURM log files."""
import ast
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

LOG_DIR = Path("/mnt/lumi/CESM2_emulator_from_lumi/logs")
PATTERN = "diffusion_aero_*.out"

# Metrics to plot and their display names
METRICS = [
    ("Training/Loss", "Total Loss"),
    ("MSE LOSS", "MSE Loss"),
    ("COND LOSS", "Cond Loss"),
    ("ANOM ERROR", "Anom Error"),
    ("ANOM SIGNAL", "Anom Signal"),
    ("ANOM SKILL", "Anom Skill"),
    ("SENS", "Sensitivity"),
    ("SCEN DISC", "Scenario Disc."),
    ("COND SCALE", "Cond Scale"),
]

LINE_RE = re.compile(r"(\{.*?\})\s+(\{'Epoch'.*?\})")


def parse_log(path: Path):
    """Return list of (epoch, metrics_dict) from a log file."""
    records = []
    seen_epochs = set()
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            # Skip lines with batch info (second occurrence per epoch)
            if "batch/" in line:
                continue
            try:
                metrics = ast.literal_eval(m.group(1))
                epoch_d = ast.literal_eval(m.group(2))
                epoch = epoch_d["Epoch"]
                if epoch not in seen_epochs:
                    seen_epochs.add(epoch)
                    records.append((epoch, metrics))
            except Exception:
                continue
    return records


def main():
    log_files = sorted(glob.glob(str(LOG_DIR / PATTERN)))
    print(f"Found {len(log_files)} log files")

    # Parse all files; deduplicate by epoch (first occurrence wins)
    all_records: dict[int, dict] = {}
    file_info = []
    for path in log_files:
        records = parse_log(Path(path))
        if not records:
            print(f"  {Path(path).name}: no training data")
            continue
        epochs = [r[0] for r in records]
        new = sum(1 for e in epochs if e not in all_records)
        for epoch, metrics in records:
            if epoch not in all_records:
                all_records[epoch] = metrics
        print(f"  {Path(path).name}: epochs {min(epochs)}-{max(epochs)}, {new} new")
        file_info.append((Path(path).name, min(epochs), max(epochs)))

    if not all_records:
        print("No data found.")
        return

    epochs = sorted(all_records.keys())
    print(f"\nTotal unique epochs: {len(epochs)} (range {min(epochs)}-{max(epochs)})")

    # Build arrays
    data = {key: [] for key, _ in METRICS}
    ep_arr = []
    for e in epochs:
        ep_arr.append(e)
        for key, _ in METRICS:
            data[key].append(all_records[e].get(key, np.nan))
    ep_arr = np.array(ep_arr)

    # --- Plot ---
    n_metrics = len(METRICS)
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    # Color vertical lines at job boundaries
    boundaries = sorted({info[1] for info in file_info} | {info[2] for info in file_info})

    for ax, (key, label) in zip(axes, METRICS):
        vals = np.array(data[key])
        ax.plot(ep_arr, vals, lw=1.2, color="steelblue")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        # Mark job restarts
        for b in boundaries:
            if b in ep_arr and b != ep_arr[0]:
                ax.axvline(b, color="gray", lw=0.8, linestyle="--", alpha=0.6)

    # Hide unused axes
    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    # Add legend for restart lines
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color="gray", lw=0.8, linestyle="--", alpha=0.6, label="Job restart")]
    fig.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.suptitle(
        f"Training metrics — diffusion_aero_16* ({len(epochs)} epochs, {len(file_info)} jobs)",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    out_path = Path("/home/nordling/PycharmProjects/CESM2_emulator_from_lumi/training_aero_16.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()

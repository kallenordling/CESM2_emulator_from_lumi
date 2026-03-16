"""
plot_training_bias.py

Plots how the model's global-mean bias evolves across validation checkpoints.

  X-axis : epoch
  Y-axis : year
  Color  : bias = model_gmean_mean_anom − cesm_gmean_mean_anom  [K]

One subplot per experiment (hist, ssp370, aaer, ghg).

Usage:
    python plot_training_bias.py [eval_output_dir] [--out figure.png]

Defaults to /mnt/lumi/CESM2_emulator_from_lumi/eval_output
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr

EXPERIMENTS = ["hist", "ssp370", "aaer", "ghg"]


def parse_epoch(dirname: str) -> int | None:
    m = re.search(r"ep(\d+)", dirname)
    return int(m.group(1)) if m else None


def load_bias(nc_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (years, bias) arrays from one NC file."""
    ds = xr.open_dataset(nc_path)
    model = ds["TREFHT_model_gmean_mean_anom"].values
    cesm  = ds["TREFHT_cesm_gmean_mean_anom"].values
    years = ds["year"].values
    ds.close()
    return years, model - cesm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "eval_dir",
        nargs="?",
        default="/mnt/lumi/CESM2_emulator_from_lumi/eval_output",
    )
    parser.add_argument("--out", default="training_bias.png")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        raise SystemExit(f"Directory not found: {eval_dir}")

    # Collect epoch dirs
    epoch_dirs = sorted(
        [(parse_epoch(d.name), d) for d in eval_dir.iterdir() if d.is_dir()],
        key=lambda x: x[0],
    )
    epoch_dirs = [(ep, d) for ep, d in epoch_dirs if ep is not None]
    if not epoch_dirs:
        raise SystemExit("No best_ep* directories found.")

    epochs = np.array([ep for ep, _ in epoch_dirs])

    # Build bias matrices  {exp: (n_epochs, n_years)}
    # Each experiment may have a different year axis, so we track per-experiment years.
    bias_matrices = {}
    years_per_exp = {}

    for exp in EXPERIMENTS:
        rows = []
        years_ref = None
        for ep, d in epoch_dirs:
            nc = d / f"TREFHT_{exp}.nc"
            if not nc.exists():
                rows.append(None)
                continue
            years, bias = load_bias(nc)
            if years_ref is None:
                years_ref = years
            rows.append(bias)

        if years_ref is None:
            continue

        years_per_exp[exp] = years_ref
        n_years = len(years_ref)
        matrix = np.full((len(epochs), n_years), np.nan)
        for i, row in enumerate(rows):
            if row is not None:
                matrix[i] = row[:n_years]
        bias_matrices[exp] = matrix

    if not bias_matrices:
        raise SystemExit("No valid NC files found.")

    # Plot
    fig, axes = plt.subplots(
        1, len(EXPERIMENTS), figsize=(5 * len(EXPERIMENTS), 6), sharey=True
    )

    # Symmetric color limits across all experiments
    all_vals = np.concatenate([m[~np.isnan(m)] for m in bias_matrices.values()])
    clim = np.nanpercentile(np.abs(all_vals), 95)
    norm = mcolors.TwoSlopeNorm(vmin=-clim, vcenter=0, vmax=clim)

    for ax, exp in zip(axes, EXPERIMENTS):
        if exp not in bias_matrices:
            ax.set_visible(False)
            continue
        mat = bias_matrices[exp]  # (n_epochs, n_years)
        years_ref = years_per_exp[exp]

        # pcolormesh needs edges
        ep_edges = np.concatenate([[epochs[0] - 1], (epochs[:-1] + epochs[1:]) / 2, [epochs[-1] + 1]])
        yr_edges = np.concatenate([[years_ref[0] - 0.5], years_ref[:-1] + 0.5, [years_ref[-1] + 0.5]])

        pcm = ax.pcolormesh(
            ep_edges,
            yr_edges,
            mat.T,  # shape (n_years, n_epochs)
            norm=norm,
            cmap="RdBu_r",
            shading="flat",
        )

        ax.set_title(exp, fontsize=12)
        ax.set_xlabel("Epoch")
        if ax is axes[0]:
            ax.set_ylabel("Year")

        ax.tick_params(axis="x", rotation=45)

    # Shared colorbar
    cbar = fig.colorbar(pcm, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Bias  [model − CESM,  K]")

    fig.suptitle("Global-mean temperature anomaly bias vs. training epoch", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()

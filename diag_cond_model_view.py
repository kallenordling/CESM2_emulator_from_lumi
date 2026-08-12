f"""Show the conditioning EXACTLY as the model sees it — all three pipeline stages.

Unlike plot_cond_input.py (which stops at smoothing), this runs the REAL
ClimateDataset.load_data() three times with progressively more of the pipeline
enabled, so every stage tensor is produced by the genuine training path — no
reimplementation, zero divergence risk:

    A  normalized                         (cond_smooth=off, PCA=off)
    B  normalized + spatial smoothing     (cond_smooth=on,  PCA=off)
    C  normalized + smoothing + PCA       (cond_smooth=on,  PCA=on)  ← MODEL INPUT

All knobs (cond_vars, n_components_cond, cond_smooth_sigma, cond_smooth_method)
are read from configs/config_data.yaml, so the figure always reflects what the
trainer is doing RIGHT NOW. Edit the config, rerun, see the difference.

For each cond channel it writes a figure with rows:
    A normalized | B +smoothed | C +PCA (model input) | A−C removed total
× columns = N evenly-spaced years (default 8, --n-cols), with the hist→ssp
junction years force-included; or pass explicit --years.

It also prints, per channel:
  - PCA components kept + explained variance,
  - the spatial pattern correlation across the hist→ssp inventory junction at
    stage A vs stage C (how much of the NA/EU→Asia regional reshuffle survives),
  - global-mean range per stage.

Run ON LUMI (needs the /scratch cond files + data.climate_dataset), e.g.:
    singularity exec <SIF> bash -c 'cd <repo> && python diag_cond_model_view.py --scenario aaer'

Or from a local mount with the cond files staged:
    PATH_REMAP={L.SCRATCH}:/mnt/lumi_sc2 \
        python diag_cond_model_view.py --scenario aaer
"""
import lumi_paths as L
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from data.climate_dataset import ClimateDataset

CONFIG     = os.environ.get("CONFIG", "configs/config_data.yaml")
PATH_REMAP = os.environ.get("PATH_REMAP", "/mnt/lumi_sc2")


def remap(path: str) -> str:
    if not PATH_REMAP or ":" not in PATH_REMAP or not isinstance(path, str):
        return path
    src, dst = PATH_REMAP.split(":", 1)
    return path.replace(src, dst, 1) if path.startswith(src) else path


# Mirror load_data's year selection so our column labels line up with the
# T-axis of tensor_data_cond (every 5th hist year + every 2nd future year).
HIST_YEARS   = list(range(1850, 2015, 5))
FUTURE_YEARS = list(range(2015, 2101, 2))
SELECTED     = set(HIST_YEARS + FUTURE_YEARS)


def selected_years_for(cond_file: str, time_dim: str) -> np.ndarray:
    """Return, in file order, the years that survive load_data's selection."""
    import xarray as xr
    ds = xr.open_dataset(cond_file)
    if time_dim not in ds.dims and "year" in ds.dims:
        ds = ds.rename({"year": time_dim})
    vals = ds[time_dim].values
    if hasattr(vals[0], "year") or not np.issubdtype(np.asarray(vals).dtype, np.number):
        years = np.array([int(str(v)[:4]) for v in vals])
        keep = np.array([y in SELECTED for y in years])
    else:
        years = vals.astype(int)
        keep = np.array([int(y) in SELECTED for y in years])
    ds.close()
    return years[keep]


def build(cond_file, data_dir, time_dim, target_vars, cond_vars,
          smooth_sigma, smooth_method, n_comp_cond, realization):
    """Instantiate ClimateDataset (cond_only) and run the real load_data."""
    ds = ClimateDataset(
        seq_len=1,
        realizations=[realization],
        data_dir=data_dir,
        target_vars=target_vars,
        cond_file=cond_file,
        cond_vars=cond_vars,
        n_components_target=None,
        n_components_cond=n_comp_cond,
        cond_smooth_sigma=smooth_sigma,
        cond_smooth_method=smooth_method,
        cond_only=True,
        time_dim=time_dim,
    )
    # Suppress the side-effect diagnostic PNGs written into the data tree.
    ds._save_cond_diagnostics = lambda *a, **k: None
    ds.load_data(realization)
    return ds


def pattern_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="aaer",
                    help="experiment name in config_data.yaml (hist/ssp370/aaer/ghg)")
    ap.add_argument("--cond-file", default=None, help="explicit cond .nc (overrides --scenario)")
    ap.add_argument("--years", type=int, nargs="+", default=None,
                    help="explicit selected years to show as columns (overrides --n-cols)")
    ap.add_argument("--n-cols", type=int, default=8,
                    help="number of evenly-spaced year columns when --years not given (default 8)")
    ap.add_argument("--decades", action="store_true",
                    help="one map column per decade (nearest available year); overrides --n-cols")
    ap.add_argument("--out-prefix", default=None,
                    help="output PNG prefix (default cond_model_view_<scenario>)")
    args = ap.parse_args()

    dc = OmegaConf.load(CONFIG)
    cond_vars     = OmegaConf.to_container(dc.cond_vars, resolve=True)
    target_vars   = OmegaConf.to_container(dc.get("target_vars", dc.get("vars", ["TREFHT"])),
                                           resolve=True)
    n_comp_cond   = dc.get("n_components_cond", None)
    n_comp_cond   = OmegaConf.to_container(n_comp_cond, resolve=True) \
        if n_comp_cond is not None else None
    smooth_sigma  = dc.get("cond_smooth_sigma", None)
    smooth_sigma  = OmegaConf.to_container(smooth_sigma, resolve=True) \
        if smooth_sigma is not None else None
    smooth_method = dc.get("cond_smooth_method", "gaussian")

    # Resolve the cond file + a realization + data_dir/time_dim from the config.
    cond_file, data_dir, time_dim, realization = args.cond_file, ".", "time", "r1"
    for ec in dc.get("experiment_configs", []):
        if ec.get("scenario_name") == args.scenario:
            cond_file = cond_file or ec.get("cond_file")
            data_dir  = remap(ec.get("data_dir", "."))
            time_dim  = ec.get("time_dim", "time")
            reals     = OmegaConf.to_container(ec.get("realizations", ["r1"]), resolve=True)
            realization = reals[0] if reals else "r1"
            break
    if cond_file is None:
        raise SystemExit(f"scenario '{args.scenario}' not found and no --cond-file given")
    cond_file = remap(cond_file)
    prefix = args.out_prefix or f"cond_model_view_{args.scenario}"

    print(f"[cfg] cond_file={cond_file}")
    print(f"[cfg] cond_vars={cond_vars}  n_components_cond={n_comp_cond}")
    print(f"[cfg] cond_smooth_sigma={smooth_sigma}  method={smooth_method}")

    years = selected_years_for(cond_file, time_dim)
    print(f"[cfg] {len(years)} selected years, {years.min()}-{years.max()}")

    # Stage tensors via the REAL pipeline with later stages disabled.
    print("[run] stage A: normalized only ...")
    A = build(cond_file, data_dir, time_dim, target_vars, cond_vars,
              None, smooth_method, None, realization).tensor_data_cond.numpy()
    print("[run] stage B: + smoothing ...")
    B = build(cond_file, data_dir, time_dim, target_vars, cond_vars,
              smooth_sigma, smooth_method, None, realization).tensor_data_cond.numpy()
    print("[run] stage C: + PCA  (== MODEL INPUT) ...")
    dsC = build(cond_file, data_dir, time_dim, target_vars, cond_vars,
                smooth_sigma, smooth_method, n_comp_cond, realization)
    C = dsC.tensor_data_cond.numpy()

    assert A.shape == B.shape == C.shape, (A.shape, B.shape, C.shape)
    assert A.shape[1] == len(years), (A.shape, len(years))

    # Columns: explicit --years, else N evenly-spaced selected years across the
    # available range, with the hist→ssp junction years (last <2015, first ≥2015)
    # always force-included so the regional reshuffle stays visible.
    if args.years:
        col_years = sorted({int(y) for y in args.years if y in set(years)})
    elif args.decades:
        lo = int(np.ceil(years.min() / 10.0) * 10)
        hi = int(np.floor(years.max() / 10.0) * 10)
        targets = list(range(lo, hi + 1, 10))           # one column per decade
        col_years = sorted({int(years[np.argmin(np.abs(years - t))]) for t in targets})
    else:
        targets = list(np.linspace(years.min(), years.max(), args.n_cols))
        pre_j, post_j = years[years < 2015], years[years >= 2015]
        if len(pre_j) and len(post_j):
            targets += [pre_j.max(), post_j.min()]      # straddle the junction
        col_years = sorted({int(years[np.argmin(np.abs(years - t))]) for t in targets})
    col_idx = [int(np.where(years == y)[0][0]) for y in col_years]
    print(f"[cfg] {len(col_years)} columns = {col_years}")

    # Index of the hist→ssp junction in selected space (last <2015 → first ≥2015).
    pre = np.where(years < 2015)[0]
    post = np.where(years >= 2015)[0]
    junction = (int(pre[-1]), int(post[0])) if len(pre) and len(post) else None

    stages = [("A normalized", A), ("B +smoothed", B),
              ("C +PCA  (MODEL INPUT)", C), ("A-C removed (total)", A - C)]

    for v_idx, var in enumerate(cond_vars):
        # ── text report ──────────────────────────────────────────────────────
        pca = dsC._pca_cond[v_idx] if dsC._pca_cond is not None else None
        print(f"\n[{var}]")
        if pca is not None:
            print(f"  PCA: {pca.n_components_} comps, "
                  f"{pca.explained_variance_ratio_.sum()*100:.1f}% variance kept")
        for label, arr in stages[:3]:
            ch = arr[v_idx]
            print(f"  {label:24s} gmean range [{ch.mean(axis=(1,2)).min():+.3f}, "
                  f"{ch.mean(axis=(1,2)).max():+.3f}]  overall [{ch.min():+.3f}, {ch.max():+.3f}]")
        if junction is not None:
            i0, i1 = junction
            cA = pattern_corr(A[v_idx, i0], A[v_idx, i1])
            cC = pattern_corr(C[v_idx, i0], C[v_idx, i1])
            print(f"  junction {years[i0]}->{years[i1]} spatial pattern corr: "
                  f"A(normalized)={cA:.4f}   C(model input)={cC:.4f}")
            print(f"    (lower corr = more regional reshuffle; PCA/smoothing raising it"
                  f" = regional signal being erased)")

        # ── figure ───────────────────────────────────────────────────────────
        nrow, ncol = len(stages), len(col_years)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow),
                                 squeeze=False)
        for r, (label, arr) in enumerate(stages):
            ch = arr[v_idx]
            is_diff = label.startswith("A-C")
            vmax = max(1e-6, np.abs(ch[col_idx]).max()) if is_diff else 1.0
            vmin = -vmax if is_diff else -1.0
            cmap = "PuOr_r" if is_diff else "RdBu_r"
            for c, (yr, ti) in enumerate(zip(col_years, col_idx)):
                ax = axes[r][c]
                im = ax.imshow(ch[ti], origin="lower", aspect="auto",
                               cmap=cmap, vmin=vmin, vmax=vmax)
                if r == 0:
                    ax.set_title(f"year {yr}", fontsize=11)
                if c == 0:
                    ax.set_ylabel(label, fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                plt.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle(
            f"{var} — how the model sees '{args.scenario}'  "
            f"(smooth={smooth_method} σ={smooth_sigma}, PCA={n_comp_cond})",
            fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out = f"{prefix}_{var}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  [plot] {out}")

    # ── per-channel global-mean time series across all stages ─────────────────
    fig, axes = plt.subplots(len(cond_vars), 1, figsize=(11, 4 * len(cond_vars)),
                             squeeze=False)
    for v_idx, var in enumerate(cond_vars):
        ax = axes[v_idx][0]
        for label, arr in stages[:3]:
            ax.plot(years, arr[v_idx].mean(axis=(1, 2)), lw=1.8, label=label)
        if junction is not None:
            ax.axvline(2015, color="grey", ls=":", lw=1)
        ax.set_title(f"{var} — spatial-mean of cond, per pipeline stage")
        ax.set_xlabel("year"); ax.set_ylabel("normalized cond")
        ax.grid(alpha=.3); ax.legend(fontsize=9)
    plt.tight_layout()
    out = f"{prefix}_timeseries.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()

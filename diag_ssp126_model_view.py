"""Show how the MODEL sees ssp126 emissions — and isolate the aaer-basis artifact.

ssp126 is a model-only (never-trained) scenario, so eval has no PCA basis fit on
it.  eval_aero.py therefore routes ssp126 through the REFERENCE basis, which is
the *aaer* scenario's basis (MultiExperimentDataset.get_pca_state picks the aaer
child as ``ref_scenario``; the per-scenario lookup misses 'ssp126' and falls back
to ``pca_cond`` = aaer).  See eval_aero.py:1907-1922 and commit 851b840
("ssp126 OOD -> aaer basis, logged").

This reconstructs ssp126 cond through the REAL ClimateDataset pipeline four ways
and overlays them so the distortion is visible:

    norm            normalized + smoothed, NO PCA          (raw signal)
    own-PCA         ssp126 projected on ITS OWN 5-EOF basis (faithful denoise)
    aaer-PCA        ssp126 projected on the AAER basis      = WHAT THE MODEL SEES
    aaer-ref        aaer projected on the aaer basis        (where ssp126 is dragged)

The aaer basis is fit on aaer's aerosol spatial patterns; applying it to ssp126's
high-then-declining SUL field can inflate/mistime the masking, which is the
suspected cause of ssp126's cold aaer-like start + mid-century unmasking hump.

Outputs (per cond channel, default both CO2 + SUL):
  * ssp126_model_view_timeseries.png — global-mean of each stage vs year
  * ssp126_model_view_<VAR>.png      — maps at key years (2015/2050/2100) for
    norm | own-PCA | aaer-PCA(model) | (aaer-PCA − own-PCA) difference
and a printed per-stage global-mean table at 2015/2050/2100.

Run ON LUMI (needs torch + /scratch cond files), e.g.:
    singularity exec <SIF> bash -c 'cd <repo> && python diag_ssp126_model_view.py'
Or from a local mount with the cond files staged + torch available:
    PATH_REMAP=/scratch/project_462001328:/mnt/lumi_sc2 python diag_ssp126_model_view.py
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from data.climate_dataset import ClimateDataset, pca_denoise_dataset

CONFIG     = os.environ.get("CONFIG", "configs/config_data.yaml")
PATH_REMAP = os.environ.get("PATH_REMAP", "/mnt/lumi_sc2")
KEY_YEARS  = [2015, 2050, 2100]


def remap(path: str) -> str:
    if not PATH_REMAP or ":" not in PATH_REMAP or not isinstance(path, str):
        return path
    src, dst = PATH_REMAP.split(":", 1)
    return path.replace(src, dst, 1) if path.startswith(src) else path


def scenario_cfg(dc, name, fallback_from=None, cond_file_override=None):
    """Resolve (cond_file, data_dir, time_dim, realization) for a scenario.

    ssp126 is eval-only (defined in eval_aero.py, NOT config_data.yaml), so it
    won't be in experiment_configs. For such scenarios pass ``fallback_from`` (a
    scenario that IS in the config, e.g. aaer): we reuse its data_dir/time_dim/
    realization (irrelevant under cond_only) and derive the cond file as
    ``emissions_<name>_only_timefixed.nc`` in the same directory — matching
    eval_aero.py's EMIS_DIR layout. ``cond_file_override`` wins if given.
    """
    for ec in dc.get("experiment_configs", []):
        if ec.get("scenario_name") == name:
            reals = OmegaConf.to_container(ec.get("realizations", ["r1"]), resolve=True)
            cond = cond_file_override or ec.get("cond_file")
            return (remap(cond), remap(ec.get("data_dir", ".")),
                    ec.get("time_dim", "time"), reals[0] if reals else "r1")
    if fallback_from is not None:
        f_cond, f_data, f_time, f_real = scenario_cfg(dc, fallback_from)
        cond = cond_file_override or os.path.join(
            os.path.dirname(f_cond), f"emissions_{name}_only_timefixed.nc")
        print(f"[cfg] '{name}' not in {CONFIG}; using cond {cond} "
              f"(data_dir/time/real borrowed from '{fallback_from}', unused under cond_only)")
        return remap(cond), f_data, f_time, f_real
    raise SystemExit(f"scenario '{name}' not found in {CONFIG} and no fallback given")


def build(cond_file, data_dir, time_dim, target_vars, cond_vars,
          smooth_sigma, smooth_method, n_comp_cond, realization):
    """Run the real load_data; return the ClimateDataset (tensor + fitted PCA)."""
    ds = ClimateDataset(
        seq_len=1, realizations=[realization], data_dir=data_dir,
        target_vars=target_vars, cond_file=cond_file, cond_vars=cond_vars,
        n_components_target=None, n_components_cond=n_comp_cond,
        cond_smooth_sigma=smooth_sigma, cond_smooth_method=smooth_method,
        cond_only=True, time_dim=time_dim,
    )
    ds._save_cond_diagnostics = lambda *a, **k: None   # suppress side-effect PNGs
    ds.load_data(realization)
    return ds


def years_of(cond_file, time_dim):
    import xarray as xr
    HIST = set(range(1850, 2015, 5)); FUT = set(range(2015, 2101, 2))
    ds = xr.open_dataset(cond_file)
    if time_dim not in ds.dims and "year" in ds.dims:
        ds = ds.rename({"year": time_dim})
    vals = ds[time_dim].values
    if hasattr(vals[0], "year") or not np.issubdtype(np.asarray(vals).dtype, np.number):
        yrs = np.array([int(str(v)[:4]) for v in vals])
    else:
        yrs = vals.astype(int)
    ds.close()
    keep = np.array([y in (HIST | FUT) for y in yrs])
    return yrs[keep]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-scenario", default="aaer",
                    help="scenario whose PCA basis ssp126 is routed through (eval uses aaer)")
    ap.add_argument("--ssp126-cond-file", default=None,
                    help="explicit ssp126 cond .nc (default: derive from --ref-scenario's dir)")
    ap.add_argument("--out-prefix", default="ssp126_model_view")
    args = ap.parse_args()

    dc = OmegaConf.load(CONFIG)
    cond_vars   = OmegaConf.to_container(dc.cond_vars, resolve=True)
    target_vars = OmegaConf.to_container(dc.get("target_vars", dc.get("vars", ["TREFHT"])),
                                         resolve=True)
    n_comp_cond = dc.get("n_components_cond", None)
    n_comp_cond = OmegaConf.to_container(n_comp_cond, resolve=True) if n_comp_cond is not None else None
    smooth_sigma = dc.get("cond_smooth_sigma", None)
    smooth_sigma = OmegaConf.to_container(smooth_sigma, resolve=True) if smooth_sigma is not None else None
    smooth_method = dc.get("cond_smooth_method", "gaussian")

    s_cond, s_data, s_time, s_real = scenario_cfg(
        dc, "ssp126", fallback_from=args.ref_scenario,
        cond_file_override=args.ssp126_cond_file)
    r_cond, r_data, r_time, r_real = scenario_cfg(dc, args.ref_scenario)
    print(f"[cfg] cond_vars={cond_vars}  n_components_cond={n_comp_cond}"
          f"  smooth σ={smooth_sigma} ({smooth_method})")
    print(f"[cfg] ssp126 cond={s_cond}")
    print(f"[cfg] {args.ref_scenario}  cond={r_cond}")

    s_years = years_of(s_cond, s_time)

    # ── fit the reference (aaer) basis exactly as training did ───────────────
    print(f"[run] fit '{args.ref_scenario}' PCA basis (the basis eval feeds ssp126) ...")
    ref_ds = build(r_cond, r_data, r_time, target_vars, cond_vars,
                   smooth_sigma, smooth_method, n_comp_cond, r_real)
    ref_pca = ref_ds._pca_cond                      # list[PCA] per cond var
    aaer_ref = ref_ds.tensor_data_cond.numpy()      # aaer on aaer basis (model sees this for aaer)
    r_years = years_of(r_cond, r_time)

    # ── ssp126: normalized+smoothed (no PCA), and own-basis PCA ──────────────
    print("[run] ssp126 stage: normalized + smoothed (no PCA) ...")
    s_norm_ds = build(s_cond, s_data, s_time, target_vars, cond_vars,
                      smooth_sigma, smooth_method, None, s_real)
    s_norm = s_norm_ds.tensor_data_cond             # (V, T, H, W) torch
    print("[run] ssp126 stage: own-basis PCA ...")
    s_own = build(s_cond, s_data, s_time, target_vars, cond_vars,
                  smooth_sigma, smooth_method, n_comp_cond, s_real).tensor_data_cond.numpy()

    # ── ssp126 through the AAER basis == what the model actually sees ────────
    print(f"[run] ssp126 projected on '{args.ref_scenario}' basis == MODEL INPUT ...")
    s_aaer, _ = pca_denoise_dataset(s_norm.clone(), n_comp_cond, cond_vars, pca_objects=ref_pca)
    s_aaer = s_aaer.numpy()
    s_norm = s_norm.numpy()

    stages = [("norm (no PCA)", s_norm), ("own-PCA", s_own),
              ("aaer-PCA (MODEL)", s_aaer)]

    # ── per-stage global-mean table at key years ─────────────────────────────
    for v_idx, var in enumerate(cond_vars):
        print(f"\n[{var}] global-mean of cond as the model sees it:")
        hdr = "  year  " + "".join(f"{lbl:>18}" for lbl, _ in stages) + f"{'aaer-ref':>18}"
        print(hdr)
        for ky in KEY_YEARS:
            ti = int(np.argmin(np.abs(s_years - ky)))
            ra = int(np.argmin(np.abs(r_years - ky))) if (r_years.min() <= ky <= r_years.max()) else None
            row = f"  {s_years[ti]:<6d}"
            for _, arr in stages:
                row += f"{arr[v_idx, ti].mean():+18.3f}"
            row += f"{aaer_ref[v_idx, ra].mean():+18.3f}" if ra is not None else f"{'--':>18}"
            print(row)

    # ── Figure 1: global-mean time series per stage, per channel ─────────────
    fig, axes = plt.subplots(len(cond_vars), 1, figsize=(11, 4 * len(cond_vars)), squeeze=False)
    for v_idx, var in enumerate(cond_vars):
        ax = axes[v_idx][0]
        for lbl, arr in stages:
            ax.plot(s_years, arr[v_idx].mean(axis=(1, 2)), lw=2, label=f"ssp126 {lbl}")
        ax.plot(r_years, aaer_ref[v_idx].mean(axis=(1, 2)), lw=1.4, ls=":",
                color="k", label=f"{args.ref_scenario}-ref (aaer basis)")
        ax.axvline(2015, color="grey", ls=":", lw=1)
        ax.set_title(f"{var} — how the model sees ssp126 (global mean of cond)")
        ax.set_xlabel("year"); ax.set_ylabel("normalized cond"); ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    plt.tight_layout()
    out = f"{args.out_prefix}_timeseries.png"
    fig.savefig(out, dpi=130); plt.close(fig); print(f"\n[plot] {out}")

    # ── Figure 2: maps at key years (per channel) ────────────────────────────
    map_rows = [("norm", s_norm), ("own-PCA", s_own),
                ("aaer-PCA (MODEL)", s_aaer), ("aaer-PCA − own", s_aaer - s_own)]
    for v_idx, var in enumerate(cond_vars):
        ncol = len(KEY_YEARS); nrow = len(map_rows)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False)
        for r, (lbl, arr) in enumerate(map_rows):
            ch = arr[v_idx]
            is_diff = lbl.startswith("aaer-PCA −")
            for c, ky in enumerate(KEY_YEARS):
                ti = int(np.argmin(np.abs(s_years - ky)))
                ax = axes[r][c]
                vmax = max(1e-6, np.abs(ch[ti]).max()) if is_diff else 1.0
                vmin, cmap = (-vmax, "PuOr_r") if is_diff else (-1.0, "RdBu_r")
                im = ax.imshow(ch[ti], origin="lower", aspect="auto", cmap=cmap,
                               vmin=vmin, vmax=vmax)
                if r == 0:
                    ax.set_title(f"year {s_years[ti]}", fontsize=11)
                if c == 0:
                    ax.set_ylabel(lbl, fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                plt.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle(f"{var} — ssp126 through own vs aaer PCA basis "
                     f"(n_comp={n_comp_cond})", fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out = f"{args.out_prefix}_{var}.png"
        fig.savefig(out, dpi=130); plt.close(fig); print(f"[plot] {out}")


if __name__ == "__main__":
    main()

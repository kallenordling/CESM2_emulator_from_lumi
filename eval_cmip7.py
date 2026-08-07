#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on the CMIP7 (h / vl) scenarios — MODEL-ONLY.

Unlike eval_aero.py, there is NO reference data to score against: CESM2 has not
been run under CMIP7 forcing, so h and vl produce emulator projections with no
truth. Everything here is therefore a projection diagnostic — anomaly time
series, spatial maps, cumulative-CO2 relationships — and never a skill score.
No bias/patcorr/TCRE-vs-truth numbers are produced, because none can be.

Reuses eval_aero.py's model/sampling machinery verbatim (load_model,
build_cond_tensor, generate_timeseries, plotting, NetCDF writer) so sampling,
normalisation, COND_NORM handling and denormalisation are IDENTICAL to the
production eval. Only the experiment list and the missing-reference handling
differ.

Inputs — the cond files from data/make_cmip7_cond.py:
    emissions_hist_cmip7_only_timefixed_bc.nc   1850-2023  (baseline + context)
    emissions_h_cmip7_only_timefixed_bc.nc      2024-2100
    emissions_vl_cmip7_only_timefixed_bc.nc     2024-2100

The 1850-1900 baseline is taken from the model's OWN generated CMIP7-hist run
(eval_aero.py falls back to this too when CESM2 hist is unavailable), so all
anomalies are internally consistent and directly comparable to the CMIP6 eval's
model-baseline anomalies.

PCA: h and vl are OOD scenarios with no persisted per-scenario basis, so a fresh
[30,5,5]-EOF basis is FIT on each of them — the same "fit" sentinel path
eval_aero.py uses for ssp126 (eval_aero.py:385). CMIP7-hist reuses the trained
'hist' basis by default since it is the same scenario character; override with
--pca-basis.

Usage:
    python eval_cmip7.py --checkpoint runs/run_gainfix_1055.pt \
        --output-dir eval_output/cmip7_ep1055

    # fewer members / faster smoke test
    python eval_cmip7.py --checkpoint ... --members 1 --sample-steps 25

    # channel-count variants (e.g. a noBCprect checkpoint)
    python eval_cmip7.py --checkpoint ... \
        --model-config configs/config_aero_noBCprect.yaml \
        --data-config  configs/config_data_noBCprect.yaml
"""

import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

# eval_aero imports cartopy/matplotlib and defines the sampling stack; importing
# it does NOT run its main() (it is __main__-guarded).
import eval_aero as EA
from data.climate_dataset import DENORM_FN
from custom_diffusers.continuous_ddpm import ContinuousDDPM


# ── CMIP7 experiment definitions ─────────────────────────────────────────────
# No data_dir / realizations: there is no CESM2 output under CMIP7 forcing.
def build_experiments(cond_dir: str, scenarios: list) -> list:
    exps = [dict(
        name      = "hist_cmip7",
        cond_file = os.path.join(cond_dir, "emissions_hist_cmip7_only_timefixed_bc.nc"),
        pca_key   = "hist",      # same scenario character as the trained hist basis
        map_years = [1900, 2000, 2023],
        color     = "#1f77b4",
    )]
    meta = {
        "h":  dict(color="#d62728", label="CMIP7 high"),
        "vl": dict(color="#2ca02c", label="CMIP7 very low"),
    }
    for sc in scenarios:
        exps.append(dict(
            name      = sc,
            cond_file = os.path.join(cond_dir, f"emissions_{sc}_cmip7_only_timefixed_bc.nc"),
            pca_key   = None,    # OOD -> fit a fresh basis
            map_years = [2024, 2060, 2100],
            color     = meta.get(sc, {}).get("color", "#7f7f7f"),
        ))
    return exps


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Model-only evaluation of a checkpoint on the CMIP7 h/vl scenarios",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--checkpoint", default=None,
                    help="Checkpoint .pt (default: newest in --runs-dir)")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--model-config", default="configs/config_aero.yaml")
    ap.add_argument("--data-config", default="configs/config_data.yaml")
    ap.add_argument("--cond-dir", default=EA.EMIS_DIR,
                    help="Directory holding the emissions_*_cmip7_*.nc files")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scenarios", nargs="+", default=["h", "vl"])
    ap.add_argument("--members", type=int, default=5,
                    help="Ensemble members (different sampling seeds)")
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--guidance-co2", type=float, default=1.0)
    ap.add_argument("--guidance-sul", type=float, default=1.0)
    ap.add_argument("--guidance-bc", type=float, default=1.0)
    ap.add_argument("--force-cfg", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="Disable bf16 autocast")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--target-var", default="TREFHT", choices=["TREFHT", "PRECT"])
    ap.add_argument("--no-cache", action="store_true",
                    help="Ignore cached samples (_samples_*.npy in --output-dir) "
                         "and always re-run the diffusion sampling")
    ap.add_argument("--pca-basis", default="auto",
                    choices=["auto", "fit", "none"],
                    help="auto = persisted basis when the checkpoint has one for "
                         "the experiment's pca_key, else fit fresh; "
                         "fit = always fit fresh; none = skip PCA")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_path = args.checkpoint or EA.find_latest_checkpoint(args.runs_dir)
    device = torch.device(args.device)
    dtype = torch.float32
    autocast_dt = None if args.fp32 else torch.bfloat16

    print(f"[CMIP7-EVAL] checkpoint : {ckpt_path}")
    print(f"[CMIP7-EVAL] model cfg  : {args.model_config}")
    print(f"[CMIP7-EVAL] data cfg   : {args.data_config}")
    print(f"[CMIP7-EVAL] cond dir   : {args.cond_dir}")
    print(f"[CMIP7-EVAL] device     : {device}\n")
    print("[CMIP7-EVAL] MODEL-ONLY: CESM2 has no CMIP7 runs, so these are")
    print("[CMIP7-EVAL] projections with NO reference — no skill scores are computed.\n")

    # ── configs ─────────────────────────────────────────────────────────────
    model_cfg = OmegaConf.load(args.model_config)
    data_cfg = OmegaConf.load(args.data_config)

    cond_vars = list(data_cfg.get("cond_vars", ["CO2", "SUL"]))
    n_comp_cond = data_cfg.get("n_components_cond", None)
    if n_comp_cond is not None and not isinstance(n_comp_cond, int):
        n_comp_cond = list(n_comp_cond)
    smooth_sigma = data_cfg.get("cond_smooth_sigma", None)
    if smooth_sigma is not None and not isinstance(smooth_sigma, (int, float)):
        smooth_sigma = list(smooth_sigma)
    smooth_method = data_cfg.get("cond_smooth_method", "gaussian")

    target_vars = list(data_cfg.get("target_vars", ["TREFHT"]))
    out_channels = int(model_cfg.model.get("out_channels", 1))
    if args.target_var not in target_vars:
        print(f"[CMIP7-EVAL] ERROR: --target-var {args.target_var} not in "
              f"target_vars={target_vars} of {args.data_config}")
        return 1
    target_channel = target_vars.index(args.target_var)
    denorm_fn = DENORM_FN[args.target_var]
    units = "°C" if args.target_var == "TREFHT" else "mm/day"
    unit_tag = "degC" if args.target_var == "TREFHT" else "mmday"

    print(f"[COND] cond_vars={cond_vars}  n_components_cond={n_comp_cond}")
    print(f"[COND] smooth sigma={smooth_sigma} method={smooth_method}")
    print(f"[TARGET] {args.target_var} (channel {target_channel} of {out_channels})\n")

    # ── cond files: validate BEFORE loading the model, so a missing file fails
    #    in a second rather than after a multi-minute checkpoint load ────────
    experiments = build_experiments(args.cond_dir, args.scenarios)
    missing = [e["cond_file"] for e in experiments if not os.path.exists(e["cond_file"])]
    if missing:
        print("[CMIP7-EVAL] ERROR: cond file(s) not found:")
        for m in missing:
            print(f"  {m}")
        print("\nBuild them first:  bash run_make_cmip7_cond.sh"
              "   (local: bash run_cmip7_local.sh cond)")
        return 1

    # ── model ───────────────────────────────────────────────────────────────
    model, pca_state = EA.load_model(ckpt_path, args.model_config, device)
    scheduler = ContinuousDDPM()
    pca_map = (pca_state or {}).get("per_scenario")
    if pca_map:
        print(f"[PCA] checkpoint carries per-scenario bases: {sorted(pca_map)}")
    else:
        print("[PCA] checkpoint has no per-scenario bases")

    # ── generate ────────────────────────────────────────────────────────────
    results: dict = {}
    baseline_map = None
    LAT = LON = None

    for exp in experiments:
        name = exp["name"]
        print(f"\n{'='*70}\n{name}\n{'='*70}")

        # PCA basis selection: persisted where the checkpoint has one for this
        # scenario, else fit fresh on this cond (OOD path, eval_aero.py:385).
        if args.pca_basis == "none" or n_comp_cond is None:
            pca_objects = None
            print("  [PCA] skipped")
        elif args.pca_basis == "fit":
            pca_objects = "fit"
            print("  [PCA] fitting a fresh basis (forced)")
        else:
            entry = (pca_map or {}).get(exp["pca_key"]) if exp["pca_key"] else None
            # A per-scenario entry is {"cond": [...], "target": [...]} (see
            # MultiExperimentDataset.get_pca_state); pca_denoise_dataset wants
            # the per-CHANNEL LIST, so unwrap "cond" exactly as eval_aero.py:2306
            # does. Passing the dict through raises KeyError: 0 downstream.
            cond_basis = entry.get("cond") if isinstance(entry, dict) else entry
            if cond_basis is not None:
                # pca_denoise_dataset indexes [0..n_cond-1]; a shorter list would
                # IndexError deep in the PCA loop with no useful message.
                if len(cond_basis) != len(cond_vars):
                    print(f"  [PCA] ERROR: persisted '{exp['pca_key']}' basis has "
                          f"{len(cond_basis)} channel bases but cond_vars has "
                          f"{len(cond_vars)} ({cond_vars}).")
                    print("  [PCA] The checkpoint was trained with a different cond "
                          "channel set than --data-config declares. Use the matching "
                          "data config, or pass --pca-basis fit to fit fresh.")
                    return 1
                pca_objects = cond_basis
                print(f"  [PCA] using persisted '{exp['pca_key']}' basis "
                      f"({len(cond_basis)} channel bases)")
            else:
                pca_objects = "fit"
                print("  [PCA] no persisted basis — fitting a fresh per-scenario basis")

        cond_tensor, years, lat, lon = EA.build_cond_tensor(
            exp["cond_file"], cond_vars, "time",
            pca_objects, n_comp_cond,
            cond_smooth_sigma=smooth_sigma,
            cond_smooth_method=smooth_method,
        )
        if LAT is None:
            LAT, LON = lat, lon
            # save_netcdf() and plot_anomaly_maps() read eval_aero's MODULE-LEVEL
            # LAT/LON (eval_aero.py:224-225), which are None until eval_aero's own
            # main() sets them — and that never runs when we import the module.
            # Publish them here or those calls blow up with
            # "NoneType has no attribute deg2rad" AFTER all the sampling is done.
            EA.LAT, EA.LON = lat, lon
        print(f"  cond {years[0]}–{years[-1]}  shape={tuple(cond_tensor.shape)}")

        # Sampling costs ~15 min per member, so a failure in the cheap
        # write/plot stage downstream would otherwise discard an hour-plus of GPU
        # time. Cache the denormalised ensemble per experiment and reuse it.
        cache = os.path.join(args.output_dir,
                             f"_samples_{args.target_var}_{name}.npy")
        if os.path.exists(cache) and not args.no_cache:
            gen_ensemble = np.load(cache)
            if (gen_ensemble.shape[0] == args.members
                    and gen_ensemble.shape[1] == len(years)):
                print(f"  reusing cached samples {gen_ensemble.shape} from {cache}")
            else:
                print(f"  cached samples {gen_ensemble.shape} do not match "
                      f"({args.members}, {len(years)}, …) — regenerating")
                gen_ensemble = None
        else:
            gen_ensemble = None

        if gen_ensemble is None:
            members = []
            for m in range(args.members):
                print(f"  member {m+1}/{args.members} …")
                gen_norm = EA.generate_timeseries(
                    model, scheduler, cond_tensor, device, dtype,
                    args.sample_steps, args.batch_size, seed=m,
                    guidance_co2=args.guidance_co2,
                    guidance_sul=args.guidance_sul,
                    guidance_bc=args.guidance_bc,
                    force_cfg=args.force_cfg,
                    autocast_dtype=autocast_dt,
                    out_channels=out_channels,
                    target_channel=target_channel,
                )
                members.append(denorm_fn(gen_norm))
            gen_ensemble = np.stack(members, axis=0)      # (N, T, H, W)
            np.save(cache, gen_ensemble)
            print(f"  cached samples -> {cache}")
        gen_mean = gen_ensemble.mean(axis=0)          # (T, H, W)

        # Baseline from the model's own CMIP7-hist 1850-1900 window.
        if name == "hist_cmip7":
            mask = (years >= EA.BASELINE_START) & (years <= EA.BASELINE_END)
            if not mask.any():
                print("  ERROR: hist cond file has no 1850-1900 window for the baseline")
                return 1
            baseline_map = gen_mean[mask].mean(axis=0)
            print(f"  [BASELINE] model 1850-1900 mean={baseline_map.mean():.2f}{units}")

        if baseline_map is None:
            print("  ERROR: no baseline — hist_cmip7 must be evaluated first")
            return 1

        anom_ens = gen_ensemble - baseline_map[None, None]
        gmean_ens = np.stack([EA.area_weighted_gmean(a, LAT) for a in anom_ens])

        results[name] = dict(
            gen_anom      = gmean_ens.mean(axis=0),
            gen_anom_ens  = gmean_ens,
            gen_years     = years,
            cesm_anom     = None,      # no CMIP7 reference exists
            cesm_anom_ens = None,
            cesm_years    = None,
            color         = exp["color"],
        )
        print(f"  global-mean anomaly {years[0]}: {gmean_ens.mean(axis=0)[0]:+.3f}{units}"
              f"   {years[-1]}: {gmean_ens.mean(axis=0)[-1]:+.3f}{units}")

        # ── per-experiment outputs ──────────────────────────────────────────
        nc_out = os.path.join(args.output_dir, f"{args.target_var}_{name}.nc")
        EA.save_netcdf(
            name=name, gen_ensemble=gen_ensemble, gen_years=years,
            baseline_map=baseline_map, cesm_ensemble=None, cesm_years=None,
            out_path=nc_out, ckpt_path=ckpt_path,
            gen_baseline_map=baseline_map,
            var=args.target_var, units=unit_tag,
        )
        print(f"  wrote {nc_out}")

        map_out = os.path.join(args.output_dir, f"anomaly_maps_{name}.png")
        EA.plot_anomaly_maps(
            name, gen_mean, years, baseline_map, exp["map_years"],
            cesm_data=None, cesm_years=None, out_path=map_out,
            gen_ensemble=gen_ensemble, cesm_ensemble=None,
            var=args.target_var, units=units,
            do_norm_bias=False,   # needs a reference; none exists
        )
        print(f"  wrote {map_out}")

    # ── combined outputs ────────────────────────────────────────────────────
    ts_out = os.path.join(args.output_dir, f"timeseries_{args.target_var}_cmip7.png")
    EA.plot_timeseries(results, ts_out, var=args.target_var, units=units,
                       title_word="temperature" if args.target_var == "TREFHT"
                       else "precipitation",
                       include_mmm=False)   # MMM files are CMIP6-scenario keyed
    print(f"\nwrote {ts_out}")

    csv_out = os.path.join(args.output_dir, f"global_mean_anomaly_cmip7_{unit_tag}.csv")
    EA.save_csv(results, csv_out, unit_tag=unit_tag)
    print(f"wrote {csv_out}")

    # ── summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}\nCMIP7 projections ({args.target_var}, vs model 1850-1900 baseline)\n{'='*70}")
    for name, d in results.items():
        a, y = d["gen_anom"], d["gen_years"]
        tail = a[-10:].mean() if len(a) >= 10 else a[-1]
        print(f"  {name:12s} {y[0]}-{y[-1]}   end-decade mean {tail:+.3f}{units}"
              f"   spread {d['gen_anom_ens'][:, -10:].mean(axis=1).std():.3f}")
    print("\nNo bias/skill numbers: CMIP7 has no CESM2 reference run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

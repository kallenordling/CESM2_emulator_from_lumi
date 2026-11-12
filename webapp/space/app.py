"""Gradio front end: edit emissions, generate CESM2-emulated climate maps.

Runs on a Hugging Face Space (ZeroGPU or CPU) or locally:

    EMULATOR_BUNDLE_DIR=../bundle python app.py
"""
from __future__ import annotations

import os
import tempfile
from collections import OrderedDict

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

from emulator import REGIONS, SCENARIO_LABELS, Edit, Emulator, locate_bundle

try:                                   # ZeroGPU on Hugging Face; no-op elsewhere
    import spaces
    gpu = spaces.GPU(duration=int(os.environ.get("GPU_DURATION", "120")))
    ZEROGPU = True
except ImportError:
    def gpu(fn):
        return fn
    ZEROGPU = False

EMU = Emulator(locate_bundle())
ON_GPU = EMU.device.type == "cuda"
MAX_MEMBERS = 8 if ON_GPU else 3
# rough CPU cost of one denoising step for one member on a 2-vCPU Space
SEC_PER_STEP = float(os.environ.get("SEC_PER_STEP", "4.5"))
DEFAULT_STEPS = 50 if ON_GPU else 25
LABEL_TO_SCEN = {SCENARIO_LABELS[s]: s for s in EMU.scenarios}
COND_NAMES = {"CO2": "cumulative CO$_2$", "SUL": "SO$_2$", "BC": "black carbon"}


# ── plotting ──────────────────────────────────────────────────────────────
def _sym_lim(*fields, q=99.0, floor=0.1):
    v = np.nanpercentile(np.abs(np.concatenate([f.ravel() for f in fields])), q)
    return max(float(v), floor)


# Every difference shown in the app is edited minus unmodified; say so on the plot.
DIFF_TITLE = "Difference: your scenario − unmodified"


def _map_panels(fields, titles, cmap, units, lims, suptitle):
    n = len(fields)
    fig = plt.figure(figsize=(5.2 * n, 4.0), layout="constrained")
    for k, (f, title, lim) in enumerate(zip(fields, titles, lims)):
        ax = fig.add_subplot(1, n, k + 1, projection=ccrs.Robinson())
        fc, lonc = add_cyclic_point(f, coord=EMU.lon)
        im = ax.pcolormesh(lonc, EMU.lat, fc, transform=ccrs.PlateCarree(),
                           cmap=cmap, vmin=-lim, vmax=lim, shading="auto",
                           rasterized=True)
        ax.coastlines(linewidth=0.5)
        ax.set_global()
        ax.set_title(title, fontsize=10)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.8,
                          extend="both")
        cb.set_label(units, fontsize=9)
    fig.suptitle(suptitle, fontsize=11)
    return fig


def _cond_fig(cond_mod, cond_base, edited):
    rows = 2 if edited else 1
    nc = len(EMU.cond_vars)
    fig = plt.figure(figsize=(5.2 * nc, 3.9 * rows), layout="constrained")
    for c, var in enumerate(EMU.cond_vars):
        ax = fig.add_subplot(rows, nc, c + 1, projection=ccrs.Robinson())
        fc, lonc = add_cyclic_point(cond_mod[c], coord=EMU.lon)
        im = ax.pcolormesh(lonc, EMU.lat, fc, transform=ccrs.PlateCarree(),
                           cmap="magma", vmin=-1, vmax=1, shading="auto",
                           rasterized=True)
        if (cond_mod[c] > 1).any():
            ax.contour(lonc, EMU.lat, fc, levels=[1.0], colors="cyan",
                       linewidths=0.8, transform=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.4, color="0.6")
        ax.set_global()
        ax.set_title(f"{COND_NAMES.get(var, var)} — model input", fontsize=10)
        fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.8,
                     extend="max").set_label("normalised (training range −1 to 1)",
                                             fontsize=8)
        if edited:
            d = cond_mod[c] - cond_base[c]
            ax = fig.add_subplot(rows, nc, nc + c + 1, projection=ccrs.Robinson())
            if not np.any(np.abs(d) > 1e-6):
                ax.coastlines(linewidth=0.4, color="0.6")
                ax.set_global()
                ax.set_title(f"{COND_NAMES.get(var, var)} — unchanged", fontsize=10)
                continue
            fc, lonc = add_cyclic_point(d, coord=EMU.lon)
            lim = max(float(np.abs(d).max()), 1e-3)
            im = ax.pcolormesh(lonc, EMU.lat, fc, transform=ccrs.PlateCarree(),
                               cmap="PuOr_r", vmin=-lim, vmax=lim, shading="auto",
                               rasterized=True)
            ax.coastlines(linewidth=0.4, color="0.5")
            ax.set_global()
            ax.set_title(f"{COND_NAMES.get(var, var)} — edited − unmodified", fontsize=10)
            fig.colorbar(im, ax=ax, orientation="horizontal",
                         shrink=0.8).set_label("normalised units", fontsize=8)
    fig.suptitle("What the model sees (after smoothing and normalisation). "
                 "Cyan contour = beyond the training range.", fontsize=10)
    return fig


# ── generation ────────────────────────────────────────────────────────────
@gpu
def _run_sampler(conds, n_members, seed, steps, progress=None):
    return EMU.sample(conds, n_members, seed, steps, progress=progress)


# Sampling is deterministic given (cond, members, seed, steps): the noise comes
# from a generator seeded per call. So a cached unmodified run is still EXACTLY
# paired with an edited run made later with the same seed, and on CPU the
# "compare with unmodified" option costs nothing after its first use.
_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_CACHE_MAX = 128          # ~0.45 MB per member and entry


def _cached_sample(key, cond, run, progress, label):
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    n_members, seed, steps = run
    # A ZeroGPU call runs in another process, where a progress callback cannot
    # reach the browser; report per-step progress only when running in-process.
    prog = None if ZEROGPU else (
        lambda frac, desc="": progress(frac, desc=f"{label}: {desc}"))
    (out,) = _run_sampler([cond], n_members, seed, steps, progress=prog)
    _CACHE[key] = out
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return out


def _stats(members, var_idx):
    """Global means per member -> (mean, standard error)."""
    g = np.array([EMU.gmean(m[var_idx]) for m in members])
    se = g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else float("nan")
    return float(g.mean()), float(se)


def _fmt(m, se, unit, sign=True):
    s = f"{m:+.2f}" if sign else f"{m:.2f}"
    return s + (f" ± {se:.2f}" if np.isfinite(se) else "") + f" {unit}"


def _write_netcdf(scen, year, edit, mod, base):
    import xarray as xr
    data = {}
    for k, var in enumerate(EMU.target_vars):
        units = EMU.meta["target_norm"][var]["units"]
        data[f"{var}_edited"] = (("member", "lat", "lon"), mod[:, k], {"units": units})
        if base is not None:
            data[f"{var}_unmodified"] = (("member", "lat", "lon"), base[:, k],
                                         {"units": units})
        data[f"{var}_baseline_1850_1900"] = (("lat", "lon"), EMU.baseline[var],
                                             {"units": units})
    ds = xr.Dataset(data, coords={"member": np.arange(1, mod.shape[0] + 1),
                                  "lat": EMU.lat, "lon": EMU.lon},
                    attrs=dict(scenario=scen, year=int(year),
                               co2_scale=edit.co2, so2_scale=edit.sul,
                               bc_scale=edit.bc, aerosol_region=edit.region,
                               checkpoint=EMU.meta["checkpoint"],
                               note="CESM2 diffusion emulator output, annual mean, "
                                    "absolute fields"))
    path = os.path.join(tempfile.mkdtemp(), f"emulator_{scen}_{year}.nc")
    ds.to_netcdf(path)
    return path


def generate(scen_label, year, co2, sul, bc, region, n_members, seed, steps, paired,
             progress=gr.Progress()):
    scen = LABEL_TO_SCEN[scen_label]
    year = int(year)
    edit = Edit(co2=float(co2), sul=float(sul), bc=float(bc), region=region)
    edited = not edit.is_identity()
    paired = bool(paired) and edited

    raw = EMU.raw_cond(scen, year)
    cond_base = EMU.normalise(raw)
    cond_mod = EMU.normalise(EMU.apply_edit(raw, edit)) if edited else cond_base

    n_members, seed, steps = int(n_members), int(seed), int(steps)
    run = (n_members, seed, steps)
    key_base = ("base", scen, year) + run
    key_mod = (("edit", scen, year, edit.co2, edit.sul, edit.bc, edit.region) + run
               if edited else key_base)
    progress(0, desc="sampling")
    mod = _cached_sample(key_mod, cond_mod, run, progress, "your scenario")
    base = (_cached_sample(key_base, cond_base, run, progress, "unmodified scenario")
            if paired else None)

    iT, iP = EMU.target_vars.index("TREFHT"), EMU.target_vars.index("PRECT")
    blT, blP = EMU.baseline["TREFHT"], EMU.baseline["PRECT"]
    aT = mod[:, iT].mean(0) - blT
    aP = mod[:, iP].mean(0) - blP
    tag = f"{SCENARIO_LABELS[scen].split(' (')[0]} {year}"

    if paired:
        bT, bP = base[:, iT].mean(0) - blT, base[:, iP].mean(0) - blP
        dT, dP = aT - bT, aP - bP
        limT, limP = _sym_lim(aT, bT), _sym_lim(aP, bP)
        figT = _map_panels([aT, bT, dT],
                           ["Your scenario", "Unmodified scenario", DIFF_TITLE],
                           "RdBu_r", "°C", [limT, limT, _sym_lim(dT, floor=0.05)],
                           f"Surface air temperature anomaly vs 1850–1900 — {tag}, "
                           f"{n_members}-member mean")
        figP = _map_panels([aP, bP, dP],
                           ["Your scenario", "Unmodified scenario", DIFF_TITLE],
                           "BrBG", "mm/day", [limP, limP, _sym_lim(dP, floor=0.02)],
                           f"Precipitation anomaly vs 1850–1900 — {tag}, "
                           f"{n_members}-member mean")
    else:
        panelsT, panelsP, titles = [aT], [aP], ["Emulator"]
        ref = EMU.ref.get(("TREFHT", scen))
        if ref is not None and year in ref[0] and not edited:
            j = list(ref[0]).index(year)
            panelsT.append(ref[1][j] - blT)
            panelsP.append(EMU.ref[("PRECT", scen)][1][j] - blP)
            titles.append("Reference: 25-member emulator mean")
        limT, limP = _sym_lim(*panelsT), _sym_lim(*panelsP)
        figT = _map_panels(panelsT, titles, "RdBu_r", "°C", [limT] * len(panelsT),
                           f"Surface air temperature anomaly vs 1850–1900 — {tag}, "
                           f"{n_members}-member mean")
        figP = _map_panels(panelsP, titles, "BrBG", "mm/day", [limP] * len(panelsP),
                           f"Precipitation anomaly vs 1850–1900 — {tag}, "
                           f"{n_members}-member mean")
    figC = _cond_fig(cond_mod, cond_base, edited)

    # summary
    gT = EMU.gmean(blT)
    gP = EMU.gmean(blP)
    mT, seT = _stats([m - np.stack([blT, blP]) for m in mod], iT)
    mP, seP = _stats([100 * (m[iP] - blP)[None] / gP for m in mod], 0)
    lines = [f"### {tag}",
             f"| | global-mean temperature anomaly | global-mean precipitation change |",
             f"|---|---|---|",
             f"| Your scenario | {_fmt(mT, seT, '°C')} | {_fmt(mP, seP, '%')} |"]
    if paired:
        bmT, bseT = _stats([m - np.stack([blT, blP]) for m in base], iT)
        bmP, bseP = _stats([100 * (m[iP] - blP)[None] / gP for m in base], 0)
        dmT, dseT = _stats([a - b for a, b in zip(mod, base)], iT)
        dmP, dseP = _stats([100 * (a[iP] - b[iP])[None] / gP for a, b in zip(mod, base)], 0)
        lines += [f"| Unmodified scenario | {_fmt(bmT, bseT, '°C')} | {_fmt(bmP, bseP, '%')} |",
                  f"| **Effect of your change** (yours − unmodified) | **{_fmt(dmT, dseT, '°C')}** | "
                  f"**{_fmt(dmP, dseP, '%')}** |"]
    ref = EMU.ref.get(("TREFHT", scen))
    if ref is not None and year in ref[0]:
        j = list(ref[0]).index(year)
        rT = EMU.gmean(ref[1][j]) - gT
        rP = 100 * (EMU.gmean(EMU.ref[("PRECT", scen)][1][j]) - gP) / gP
        lines.append(f"| *Unmodified, 25-member reference* | *{rT:+.2f} °C* | *{rP:+.2f} %* |")
    lines.append("")
    if n_members > 1:
        lines.append(f"± = standard error over {n_members} members.")
    else:
        lines.append("Single member: no error estimate. Global differences of a few "
                     "hundredths of a degree, and local ones of a few tenths, are "
                     "within sampling noise even with paired noise; use more members "
                     "or other seeds before reading anything into them.")
    lines.append(f"Baseline: the emulator's own 1850–1900 climate (global mean "
                 f"{gT:.2f} °C, {gP:.2f} mm/day).")

    rep = EMU.range_report(cond_mod)
    warn = [f"**{COND_NAMES.get(v, v).replace('$_2$', '₂')}**: input reaches "
            f"{r['max']:.2f} (training maximum 1.00) over {100 * r['frac_above']:.1f}% "
            f"of the globe"
            for v, r in rep.items() if r["max"] > 1.02]
    if warn:
        lines += ["", "⚠️ **Outside the training range — this is extrapolation and "
                      "the emulator is known to be unreliable there.**"]
        lines += [f"- {w}" for w in warn]
    if edited and edit.region != "Global (everywhere)":
        lines += ["", "ℹ️ Regional aerosol changes: the emulator's response to a "
                      "regional perturbation is real but often **displaced** from "
                      "the source region (by up to thousands of km). Treat "
                      "local detail with caution."]
    summary = "\n".join(lines)

    nc_path = _write_netcdf(scen, year, edit, mod, base)
    plt.close("all")
    return figT, figP, figC, summary, nc_path


def _years_for(scen_label):
    ys = EMU.years(LABEL_TO_SCEN[scen_label])
    default = 2050 if 2050 in ys else ys[-1]
    return gr.Dropdown(choices=ys, value=default)


# ── layout ────────────────────────────────────────────────────────────────
ABOUT = f"""
# CESM2 emission → climate emulator

Pick an emission scenario and year, scale the emissions, and the emulator
generates annual-mean **surface air temperature** and **precipitation** maps,
the way the CESM2 Earth system model would for those emissions.

The emulator is a diffusion model ({EMU.meta['n_parameters'] / 1e6:.0f} M parameters,
checkpoint `{EMU.meta['checkpoint']}`) trained on CESM2 Large Ensemble
simulations. Every run is a fresh random sample: like the climate model itself
it includes year-to-year weather noise, so use several members for robust patterns.
"""

CAVEATS = """
- **Trained range.** The model was trained on historical and SSP3-7.0 emissions
  (plus single-forcing runs). SSP1-2.6 and SSP2-4.5 were never seen in training,
  and scaling emissions beyond what those scenarios contain is extrapolation.
  The app flags inputs outside the training range.
- **Aerosol removal is under-estimated.** The response to a *difference*
  in aerosol emissions between scenarios is weaker than in CESM2, even
  though total warming is close. Treat the size of aerosol effects as a lower
  bound.
- **Regional changes are displaced.** A regional aerosol change produces a
  response that is regional, but the peak can lie far from the source region.
- **CO₂ is cumulative.** The CO₂ slider scales cumulative emissions since
  1850, i.e. the whole emission history up to the selected year.
- **Paired sampling.** With "compare with unmodified" on, the edited and the
  unmodified scenario share the same random noise. Much of the weather noise
  cancels in the difference, but not all of it: the two samples still drift apart
  during denoising. With one member, local differences of a few tenths of a degree
  are noise. Use several members, or repeat with other seeds, for anything you
  want to interpret.
"""

with gr.Blocks(title="CESM2 emission emulator") as demo:
    gr.Markdown(ABOUT)
    with gr.Accordion("Limitations — please read", open=False):
        gr.Markdown(CAVEATS)
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            first = SCENARIO_LABELS["ssp370"] if "ssp370" in EMU.scenarios \
                else SCENARIO_LABELS[EMU.scenarios[0]]
            scen = gr.Dropdown(list(LABEL_TO_SCEN), value=first, label="Scenario")
            ys0 = EMU.years(LABEL_TO_SCEN[first])
            year = gr.Dropdown(ys0, value=2050 if 2050 in ys0 else ys0[-1],
                               label="Year")
            gr.Markdown("**Emission scaling** (1 = scenario as published)")
            co2 = gr.Slider(0.0, 1.5, value=1.0, step=0.05,
                            label="Cumulative CO₂ (global)")
            sul = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="SO₂ (sulfate aerosol)")
            bc = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Black carbon")
            region = gr.Dropdown(list(REGIONS), value="Global (everywhere)",
                                 label="Apply aerosol scaling in")
            paired = gr.Checkbox(True, label="Compare with the unmodified scenario "
                                             "(same noise; the unmodified run is cached, so only its first use costs time)")
            with gr.Accordion("Sampling", open=False):
                n_mem = gr.Slider(1, MAX_MEMBERS, value=2 if ON_GPU else 1, step=1,
                                  label="Ensemble members")
                seed = gr.Number(0, precision=0, label="Random seed")
                # 25 vs 50 steps tested on CPU: same global mean and pattern
                # correlation within one-member noise; 15 steps leaves ~20% more
                # grid-scale texture, so 25 is the CPU default and the floor is 20.
                steps = gr.Slider(20, 50, value=DEFAULT_STEPS, step=5,
                                  label="Denoising steps (50 = as in the paper "
                                        "evaluation; 25 is about as good at half the cost)")
            btn = gr.Button("Generate", variant="primary")
            if not ON_GPU:
                gr.Markdown(f"*Running on a free CPU: about {SEC_PER_STEP * DEFAULT_STEPS / 60:.0f} min "
                            f"per member at {DEFAULT_STEPS} steps. Unmodified runs are cached; "
                            f"requests are queued one at a time.*")
        with gr.Column(scale=3):
            summary = gr.Markdown()
            with gr.Tabs():
                with gr.Tab("Temperature"):
                    plot_t = gr.Plot(show_label=False)
                with gr.Tab("Precipitation"):
                    plot_p = gr.Plot(show_label=False)
                with gr.Tab("Emission inputs"):
                    plot_c = gr.Plot(show_label=False)
            nc_file = gr.File(label="Download (NetCDF, absolute fields)")

    scen.change(_years_for, scen, year)
    btn.click(generate,
              [scen, year, co2, sul, bc, region, n_mem, seed, steps, paired],
              [plot_t, plot_p, plot_c, summary, nc_file],
              concurrency_limit=1)

if __name__ == "__main__":
    demo.queue(max_size=20).launch()

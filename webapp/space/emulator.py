"""Inference core for the CESM2 emission -> climate emulator web app.

Reproduces the eval_aero.py inference path for ONE year:

    raw emission maps (CO2 cumulative, SO2, BC)
      -> user edits (scale factors, optionally inside a region)
      -> gaussian smoothing per channel (lon wrap, lat reflect)   [= smooth_cond_spatial]
      -> min/max normalisation with the checkpoint's anchors      [= normalize_tensor_cond]
      -> 50-step continuous-DDPM sampling, v-prediction          [= generate_timeseries]
      -> denormalise (TREFHT degC, PRECT mm/day)                  [= DENORM_FN]

Everything the pipeline depends on (anchors, sigma, model shape, norm constants)
is read from the bundle's meta.json, which export_bundle.py writes from the
checkpoint itself. Nothing here is re-fitted.
"""
from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from custom_diffusers.continuous_ddpm import ContinuousDDPM
from models.video_net import UNetModel3D

# Simple lat/lon boxes, lon in [-180, 180]. Deliberately coarse: the emulator's
# response to a regional perturbation is real but spatially displaced, so finer
# boxes would suggest a precision the model does not have.
REGIONS: dict[str, tuple[float, float, float, float] | None] = {
    "Global (everywhere)": None,
    "East Asia":       (20, 50, 95, 145),
    "South Asia":      (5, 35, 65, 95),
    "Southeast Asia":  (-10, 20, 95, 150),
    "Middle East":     (12, 42, 35, 63),
    "Europe":          (35, 70, -10, 40),
    "North America":   (15, 70, -170, -50),
    "South America":   (-56, 15, -82, -34),
    "Africa":          (-35, 37, -20, 52),
}

SCENARIO_LABELS = {
    "hist":   "Historical (1850-2014)",
    "ssp126": "SSP1-2.6 (2015-2100)",
    "ssp245": "SSP2-4.5 (2015-2100)",
    "ssp370": "SSP3-7.0 (2015-2100)",
}


def locate_bundle() -> str:
    """Local bundle dir, or download it from the Hugging Face Hub.

    EMULATOR_BUNDLE_DIR  explicit local directory (wins if set)
    ./bundle             default local directory
    EMULATOR_BUNDLE_REPO Hub model repo id to snapshot_download otherwise
    """
    d = os.environ.get("EMULATOR_BUNDLE_DIR")
    if d:
        return d
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bundle")
    if os.path.exists(os.path.join(here, "meta.json")):
        return here
    repo = os.environ.get("EMULATOR_BUNDLE_REPO")
    if not repo:
        raise FileNotFoundError(
            "No bundle found: set EMULATOR_BUNDLE_DIR, put one in ./bundle, or set "
            "EMULATOR_BUNDLE_REPO to a Hugging Face model repo holding it.")
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo, repo_type="model",
                             token=os.environ.get("HF_TOKEN"))


def area_weights(lat: np.ndarray, shape) -> np.ndarray:
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones(shape[-1])[None, :]
    return w / w.sum()


@dataclass
class Edit:
    co2: float = 1.0       # multiplies the cumulative-CO2 map everywhere
    sul: float = 1.0       # multiplies SO2 emissions (inside `region`)
    bc: float = 1.0        # multiplies BC emissions (inside `region`)
    region: str = "Global (everywhere)"

    def is_identity(self) -> bool:
        return self.co2 == 1.0 and self.sul == 1.0 and self.bc == 1.0


class Emulator:
    def __init__(self, bundle_dir: str, device: str | None = None):
        with open(os.path.join(bundle_dir, "meta.json")) as f:
            self.meta = json.load(f)
        m = self.meta
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = UNetModel3D(**m["model"])
        sd = torch.load(os.path.join(bundle_dir, "weights.pt"),
                        map_location="cpu", weights_only=True)
        self.model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
        self.model = self.model.to(self.device).eval()
        self.scheduler = ContinuousDDPM(**m["scheduler"])

        self.cond_vars: list[str] = m["cond_vars"]
        self.target_vars: list[str] = m["target_vars"]
        self.sigma: list[float] = [float(s) for s in m["cond_smooth_sigma"]]
        self.anchors = {k: (float(v[0]), float(v[1])) for k, v in m["anchors"].items()}

        cb = np.load(os.path.join(bundle_dir, "cond_base.npz"))
        self.lat = cb["lat"].astype(np.float64)
        self.lon = cb["lon"].astype(np.float64)
        self.scenarios = [s for s in m["scenarios"] if f"cond_{s}" in cb]
        self._cond = {s: cb[f"cond_{s}"] for s in self.scenarios}      # (Ty, C, H, W) raw
        self._years = {s: cb[f"years_{s}"].astype(int) for s in self.scenarios}

        bl = np.load(os.path.join(bundle_dir, "baseline.npz"))
        self.baseline = {v: bl[v] for v in self.target_vars if v in bl}
        self.ref = {}                                   # 25-member eval ensemble means
        rp = os.path.join(bundle_dir, "reference.npz")
        if os.path.exists(rp):
            r = np.load(rp)
            for s in self.scenarios:
                for v in self.target_vars:
                    if f"{v}_{s}" in r:
                        self.ref[(v, s)] = (r[f"years_{s}"].astype(int), r[f"{v}_{s}"])

        lon180 = np.where(self.lon > 180, self.lon - 360, self.lon)
        self._lat2d, self._lon2d = np.meshgrid(self.lat, lon180, indexing="ij")
        self.w = area_weights(self.lat, (len(self.lat), len(self.lon)))

    # ── conditioning ────────────────────────────────────────────────────────
    def years(self, scenario: str) -> list[int]:
        return [int(y) for y in self._years[scenario]]

    def raw_cond(self, scenario: str, year: int) -> np.ndarray:
        ys = self._years[scenario]
        i = int(np.argmin(np.abs(ys - year)))
        if ys[i] != year:
            raise ValueError(f"{year} not in bundle for {scenario}")
        return self._cond[scenario][i].astype(np.float32)            # (C, H, W)

    def region_mask(self, region: str) -> np.ndarray:
        box = REGIONS[region]
        if box is None:
            return np.ones_like(self._lat2d, dtype=bool)
        la0, la1, lo0, lo1 = box
        return ((self._lat2d >= la0) & (self._lat2d <= la1)
                & (self._lon2d >= lo0) & (self._lon2d <= lo1))

    def apply_edit(self, raw: np.ndarray, edit: Edit) -> np.ndarray:
        out = raw.copy()
        mask = self.region_mask(edit.region)
        scale = {"CO2": edit.co2, "SUL": edit.sul, "BC": edit.bc}
        for c, var in enumerate(self.cond_vars):
            s = scale.get(var, 1.0)
            if s == 1.0:
                continue
            if var == "CO2":          # cumulative CO2 is scaled globally
                out[c] = out[c] * s
            else:
                out[c] = np.where(mask, out[c] * s, out[c])
        return out

    def normalise(self, raw: np.ndarray) -> np.ndarray:
        """smooth -> min/max, exactly as smooth_cond_spatial + normalize_tensor_cond."""
        out = np.empty_like(raw, dtype=np.float32)
        for c, var in enumerate(self.cond_vars):
            ch = raw[c][None]                                         # (1, H, W)
            sig = self.sigma[c]
            if sig > 0:
                ch = gaussian_filter1d(ch, sigma=sig, axis=-1, mode="wrap")
                ch = gaussian_filter1d(ch, sigma=sig, axis=-2, mode="reflect")
            lo, hi = self.anchors[var]
            if hi <= lo:
                out[c] = -1.0
                continue
            mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
            out[c] = (ch[0] - mid) / half
        return np.nan_to_num(out, nan=-1.0)

    def range_report(self, cond_norm: np.ndarray) -> dict:
        """Per channel: max of the normalised field and the share of the globe
        above +1, i.e. beyond anything that occurred in training."""
        rep = {}
        for c, var in enumerate(self.cond_vars):
            ch = cond_norm[c]
            rep[var] = dict(max=float(ch.max()), min=float(ch.min()),
                            frac_above=float(self.w[ch > 1.0].sum()))
        return rep

    # ── sampling ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, conds: list[np.ndarray], n_members: int, seed: int,
               steps: int | None = None, progress=None) -> list[np.ndarray]:
        """Sample `n_members` for each cond in `conds`, with IDENTICAL noise
        across conds (paired sampling: member k of every cond sees the same
        initial noise and the same per-step noise). Returns, per cond, an array
        (n_members, n_targets, H, W) in physical units."""
        steps = int(steps or self.meta.get("sample_steps", 50))
        n_c = len(conds)
        H, W = conds[0].shape[-2:]
        n_out = self.meta["model"]["out_channels"]
        dev = self.device
        g = torch.Generator(device="cpu").manual_seed(int(seed))

        cond_b = torch.from_numpy(np.stack(conds)).float()            # (n_c, C, H, W)
        cond_b = cond_b.repeat_interleave(n_members, dim=0)            # (n_c*n, C, H, W)
        cond_b = cond_b.unsqueeze(2).to(dev)                           # (B, C, 1, H, W)
        B = cond_b.shape[0]

        def paired_noise():
            z = torch.randn(n_members, n_out, 1, H, W, generator=g)
            return z.repeat(n_c, 1, 1, 1, 1).to(dev)

        sch = self.scheduler
        sch.set_timesteps(steps)
        tgrid = torch.linspace(1.0, 0.0, steps + 1, device=dev)
        x = paired_noise()
        amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if dev.type == "cuda" else contextlib.nullcontext())
        for i in range(steps):
            if progress is not None:
                progress((i + 1) / steps, desc=f"denoising step {i + 1}/{steps}")
            t = sch.log_snr(tgrid[i]).expand(B)
            with amp:
                v = self.model(x, t, cond_map=cond_b)
            v = v.float()
            # == ContinuousDDPM.step, with the noise drawn here so it is paired
            times, times_next = tgrid[i], tgrid[i + 1]
            x0 = sch.predict_start_from_v(x, times, v)
            if sch.clip_sample:
                x0 = torch.clamp(x0, *sch.clip_range)
            mean, var, _ = sch.q_posterior(x0, x, times, t_next=times_next)
            x = mean if times_next == 0 else mean + var.sqrt() * paired_noise()

        out = x.squeeze(2).cpu().numpy()                               # (B, n_out, H, W)
        out = self.denorm(out)
        return [out[k * n_members:(k + 1) * n_members] for k in range(n_c)]

    def denorm(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        for c, var in enumerate(self.target_vars[: x.shape[1]]):
            p = self.meta["target_norm"][var]
            if p["kind"] == "linear":
                x[:, c] = x[:, c] * p["scale"] + p["offset"]
            elif p["kind"] == "log1p":
                x[:, c] = np.expm1(x[:, c] * p["std"] + p["mean"])
            else:
                raise ValueError(f"unknown target norm {p}")
        return x

    def gmean(self, field: np.ndarray) -> float:
        return float((field * self.w).sum())

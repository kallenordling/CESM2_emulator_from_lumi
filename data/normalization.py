# normalization.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, Any
import json
import math
import numpy as np
import xarray as xr
import torch


ArrayLike = Union[xr.DataArray, xr.Dataset, torch.Tensor, np.ndarray]


@dataclass
class NormStats:
    """Statistics for normalization."""
    kind: str  # "zscore" | "minmax" | "robust" | "identity"
    mean: Optional[float] = None
    std: Optional[float] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    median: Optional[float] = None
    iqr: Optional[float] = None
    eps: float = 1e-12
    log1p: bool = False   # apply log1p before scaling
    clip: Optional[Tuple[float, float]] = None  # clip AFTER scaling (e.g., (-5, 5))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "mean": self.mean, "std": self.std,
            "vmin": self.vmin, "vmax": self.vmax,
            "median": self.median, "iqr": self.iqr,
            "eps": self.eps, "log1p": self.log1p,
            "clip": list(self.clip) if self.clip is not None else None,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "NormStats":
        clip = tuple(d["clip"]) if d.get("clip") is not None else None
        return NormStats(
            kind=d["kind"],
            mean=d.get("mean"), std=d.get("std"),
            vmin=d.get("vmin"), vmax=d.get("vmax"),
            median=d.get("median"), iqr=d.get("iqr"),
            eps=float(d.get("eps", 1e-12)),
            log1p=bool(d.get("log1p", False)),
            clip=clip,
        )


def _as_dataarray(x: ArrayLike, name: str = "x") -> xr.DataArray:
    if isinstance(x, xr.DataArray):
        return x
    if isinstance(x, xr.Dataset):
        # Caller should pass a variable, but if not: merge to single array fails; be explicit.
        raise TypeError("Expected xr.DataArray, got xr.Dataset. Select a variable first.")
    if isinstance(x, torch.Tensor):
        return xr.DataArray(x.detach().cpu().numpy(), name=name)
    if isinstance(x, np.ndarray):
        return xr.DataArray(x, name=name)
    raise TypeError(f"Unsupported type: {type(x)}")


def _area_weights_for_lat(da: xr.DataArray, lat_name: str = "lat") -> Optional[xr.DataArray]:
    # Supports either 1D lat coord or 2D lat grid stored as coord.
    if lat_name not in da.coords:
        return None
    lat = da.coords[lat_name]
    # Convert degrees to radians and use cos(lat)
    w = np.cos(np.deg2rad(lat))
    # Broadcast to data shape automatically when used in weighted()
    return xr.DataArray(w, coords=lat.coords, dims=lat.dims)


def compute_stats(
    da: xr.DataArray,
    kind: str = "zscore",
    *,
    dims: Optional[Tuple[str, ...]] = None,
    log1p: bool = False,
    clip: Optional[Tuple[float, float]] = None,
    eps: float = 1e-12,
    robust_quantile: Tuple[float, float] = (0.25, 0.75),
    use_area_weights: bool = False,
    lat_name: str = "lat",
) -> NormStats:
    """
    Compute normalization stats from a reference DataArray.

    dims: dimensions to reduce over (e.g. ("time","lat","lon")). If None -> reduce over all dims.
    use_area_weights: if True and lat coord exists, use cos(lat) weighting for mean/std.
    """
    if dims is None:
        dims = tuple(da.dims)

    x = da
    if log1p:
        # log1p requires non-negative; if negative exist, shift yourself before calling.
        x = xr.apply_ufunc(np.log1p, x)

    if use_area_weights:
        w = _area_weights_for_lat(x, lat_name=lat_name)
    else:
        w = None

    if kind == "identity":
        return NormStats(kind="identity", eps=eps, log1p=log1p, clip=clip)

    if kind == "minmax":
        vmin = float(x.min(dim=dims).item())
        vmax = float(x.max(dim=dims).item())
        return NormStats(kind="minmax", vmin=vmin, vmax=vmax, eps=eps, log1p=log1p, clip=clip)

    if kind == "zscore":
        if w is not None and any(d in x.dims for d in w.dims):
            mean = float(x.weighted(w).mean(dim=dims).item())
            var = float(((x - mean) ** 2).weighted(w).mean(dim=dims).item())
            std = math.sqrt(max(var, 0.0))
        else:
            mean = float(x.mean(dim=dims).item())
            std = float(x.std(dim=dims).item())
        return NormStats(kind="zscore", mean=mean, std=std, eps=eps, log1p=log1p, clip=clip)

    if kind == "robust":
        qlo, qhi = robust_quantile
        med = float(x.median(dim=dims).item())
        q1 = float(x.quantile(qlo, dim=dims).item())
        q3 = float(x.quantile(qhi, dim=dims).item())
        iqr = float(q3 - q1)
        return NormStats(kind="robust", median=med, iqr=iqr, eps=eps, log1p=log1p, clip=clip)

    raise ValueError(f"Unknown kind={kind}")


def apply_stats(da: xr.DataArray, stats: NormStats) -> xr.DataArray:
    """Normalize da using provided stats."""
    x = da
    if stats.log1p:
        x = xr.apply_ufunc(np.log1p, x)

    if stats.kind == "identity":
        y = x

    elif stats.kind == "minmax":
        denom = (stats.vmax - stats.vmin) if (stats.vmax is not None and stats.vmin is not None) else None
        if denom is None:
            raise ValueError("minmax stats missing vmin/vmax")
        if abs(denom) < stats.eps:
            y = xr.zeros_like(x)
        else:
            y = (x - stats.vmin) / denom

    elif stats.kind == "zscore":
        if stats.mean is None or stats.std is None:
            raise ValueError("zscore stats missing mean/std")
        if abs(stats.std) < stats.eps:
            y = xr.zeros_like(x)
        else:
            y = (x - stats.mean) / stats.std

    elif stats.kind == "robust":
        if stats.median is None or stats.iqr is None:
            raise ValueError("robust stats missing median/iqr")
        if abs(stats.iqr) < stats.eps:
            y = xr.zeros_like(x)
        else:
            y = (x - stats.median) / stats.iqr

    else:
        raise ValueError(f"Unknown stats.kind={stats.kind}")

    if stats.clip is not None:
        lo, hi = stats.clip
        y = y.clip(min=lo, max=hi)
    return y


class Normalizer:
    def __init__(self, stats: dict):
        self.stats = stats

    @classmethod
    def load_json(cls, path: str) -> "Normalizer":
        with open(path, "r") as f:
            return cls(json.load(f))

    def normalize(self, x: xr.DataArray, varname: str) -> xr.DataArray:
        """
        Normalize x according to stats[varname].
        Supported methods: "zscore", "robust" (median/IQR), "percentile" (p_low/p_high).
        Optional: log1p, clip.
        """
        if varname not in self.stats:
            raise KeyError(f"Missing stats for var '{varname}'. Available: {list(self.stats.keys())}")

        st = self.stats[varname]
        method = st.get("method", "zscore")
        log1p = bool(st.get("log1p", False))
        clip = st.get("clip", None)

        # ensure float
        x = x.astype("float32")

        # log1p if requested (only safe for non-negative)
        if log1p:
            # If negatives exist, log1p will create NaNs; we preserve them for diagnostics.
            x = xr.apply_ufunc(np.log1p, x)

        if method == "zscore":
            mean = float(st["mean"])
            std = float(st["std"])
            eps = float(st.get("eps", 1e-6))
            denom = max(std, eps)
            y = (x - mean) / denom

        elif method == "robust":
            median = float(st["median"])
            iqr = float(st["iqr"])
            eps = float(st.get("eps", 1e-6))
            denom = max(iqr, eps)
            y = (x - median) / denom

        elif method == "percentile":
            p_low = float(st["p_low"])
            p_high = float(st["p_high"])
            eps = float(st.get("eps", 1e-12))

            # IMPORTANT: if we already applied log1p to x above,
            # we must apply the same transform to the bounds.
            if log1p:
                p_low = float(np.log1p(p_low))
                p_high = float(np.log1p(p_high))

            denom = max(p_high - p_low, eps)
            y01 = (x - p_low) / denom
            y01 = y01.clip(0.0, 1.0)     # keep within training range
            y = y01 * 2.0 - 1.0          # map to [-1, 1]

        else:
            raise ValueError(f"Unknown normalization method '{method}' for var '{varname}'")

        # Optional clip (usually you won't use clip for percentile method; left here for completeness)
        if clip is not None:
            lo, hi = clip
            y = y.clip(lo, hi)

        return y

    # --------- Optional: helper for quick stats on numpy arrays ----------
    @staticmethod
    def quick_array_stats(a: np.ndarray) -> dict:
        a = np.asarray(a)
        finite = np.isfinite(a)
        nf = int((~finite).sum())
        if finite.any():
            af = a[finite]
            return {
                "nonfinite": nf,
                "p01": float(np.percentile(af, 1)),
                "p50": float(np.percentile(af, 50)),
                "p99": float(np.percentile(af, 99)),
                "min": float(np.min(af)),
                "max": float(np.max(af)),
                "mean": float(np.mean(af)),
                "std": float(np.std(af)),
            }
        return {
            "nonfinite": nf,
            "p01": np.nan, "p50": np.nan, "p99": np.nan,
            "min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan,
        }

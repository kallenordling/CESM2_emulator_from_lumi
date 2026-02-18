import os
import random
from typing import Any, Optional
from functools import lru_cache

from omegaconf import OmegaConf
import torch
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
import dask
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA

# Constants for the minimum and maximum of our datasets
MIN_MAX_CONSTANTS = {"TREFHT": (-85.0, 60.0), "pr": (0.0, 6.0)}

# Convert from kelvin to celsius and from kg/m^2/s to mm/day
PREPROCESS_FN = {"TREFHT": lambda x: x - 273.15, "pr": lambda x: x * 86400}
fit_minmax = lambda x: (np.nanmin(x), np.nanmax(x))
# Normalization and Inverse Normalization functions
NORM_FN = {
    "TREFHT": lambda x: (x - 4.5) / 21.0,
    "pr": lambda x: np.cbrt(x),
}
DENORM_FN = {
    "TREFHT": lambda x: x * 21.0 + 4.5,
    "pr": lambda x: x**3,
}

# These functions transform the range of the data to [-1, 1]
MIN_MAX_FN = {"TREFHT": lambda x: x}

def norm_zscore(da: xr.DataArray) -> xr.DataArray:
    """
    Z-score normalization: (x - mean) / std, computed over all values globally.

    Parameters
    ----------
    da : xr.DataArray
        Input data array.

    Returns
    -------
    xr.DataArray
        Normalized array with mean ≈ 0 and std ≈ 1, same shape and coordinates.
    """
    vals=da.values.flatten()
    mask = vals > 1e-5

    mu    = float(vals[mask].mean())
    sigma = float(vals[mask].std())
    return ((da - mu) / max(sigma, 1e-30)).astype("float32")

def min_max_norm(x: Any, min_val: float, max_val: float) -> Any:
    """Normalizes a data array to the range [-1, 1]"""
    return (x - min_val) / (max_val - min_val)


def min_max_denorm(x: Any, min_val: float, max_val: float) -> Any:
    """Inverse normalizes a data array from the range [-1, 1] to [min_val, max_val]"""
    return x * (max_val - min_val) + min_val


def preprocess(ds: xr.DataArray) -> xr.DataArray:
    """Preprocesses a data array"""

    # The name of the variable is contained within the dataarray
    return PREPROCESS_FN[ds.name](ds)


EMISSIONS_PATH = "/scratch/project_462001112/emulator_data/emissions_new.nc"

def scale_cumulative_linear(da: xr.DataArray):
    """Collapse to spatial mean per year, normalize to [-1, 1], broadcast back.
    Preserves temporal signal perfectly; every grid cell gets the same
    value per year (the global mean emission level for that year).
    Best for well-mixed gases like CO2."""
    spatial_dims = [d for d in da.dims if d != "year"]
    ts = da.mean(dim=spatial_dims)  # [year]
    lo = float(ts.min(skipna=True))
    hi = float(ts.max(skipna=True))
    normed = (2.0 * (ts - lo) / max(hi - lo, 1e-30) - 1.0)
    return normed.broadcast_like(da).astype("float32")


def scale_spatial_log10(da: xr.DataArray, floor=1e-30, lo_pct=1.0, hi_pct=99.0):
    """Log10 normalization to [-1, 1] preserving spatial structure.
    Percentiles computed only on non-zero cells to avoid ocean domination.
    Near-zero cells (ocean) mapped to -1.
    Best for regionally varying emissions like SO2."""
    positive = da.where(da > floor)
    lx = np.log10(positive)
    lo = float(lx.quantile(lo_pct / 100.0, skipna=True))
    hi = float(lx.quantile(hi_pct / 100.0, skipna=True))
    z = (lx - lo) / max(hi - lo, 1e-30)
    result = (2.0 * z.clip(0, 1) - 1.0)
    result = xr.where(da <= floor, -1.0, result)
    return result.fillna(-1.0).astype("float32")

def scale_emis_0_1_log10(da: xr.DataArray, low_pct=1.0, high_pct=99.0, floor=1e-30):
    # TOMCAT emissions: non-negative
    x = da.clip(min=0)
    # avoid log(0)
    x = xr.where(x > 0, x, floor)

    lx = np.log10(x)

    lo = lx.quantile(low_pct/100.0, skipna=True)
    hi = lx.quantile(high_pct/100.0, skipna=True)

    z = (lx - lo) / (hi - lo)
    return z.fillna(0).astype("float32")

def scale_emis_m1_p1_log10(da: xr.DataArray, low_pct=1.0, high_pct=99.99999999, floor=1e-30):
    z01 = scale_emis_0_1_log10(da, low_pct, high_pct, floor)
    return (2.0 * z01 - 1.0).astype("float32")


def scale_quantile_transform(da: xr.DataArray, n_quantiles=1000, floor=1e-30):
    """sklearn QuantileTransformer: rank-based normalization to [-1, 1].

    - Smooth and monotonic — no sudden jumps from hard clipping
    - Handles extreme skew naturally (rank-based, not value-based)
    - Preserves spatial structure (high-emission cells get higher values)
    - Preserves temporal trend (increasing emissions → increasing values)
    - Ocean/near-zero cells are handled separately → mapped to -1
    """
    shape = da.shape
    vals = da.values.copy()

    # Separate ocean/near-zero cells
    real_mask = vals #vals > floor

    if real_mask.sum() == 0:
        # All zeros — return flat -1
        return xr.DataArray(
            np.full(shape, -1.0, dtype=np.float32),
            dims=da.dims, coords=da.coords,
        )

    # Fit QuantileTransformer only on non-zero values
    real_vals = vals[real_mask].reshape(-1, 1)

    qt = QuantileTransformer(
        n_quantiles=min(n_quantiles, len(real_vals)),
        output_distribution='uniform',
        random_state=42,
    )
    qt.fit(real_vals)

    # Transform ALL non-zero values → [0, 1] → [-1, 1]
    transformed = np.full(shape, -1.0, dtype=np.float32)  # ocean = -1
    transformed[real_mask] = (
        2.0 * qt.transform(vals[real_mask].reshape(-1, 1)).ravel() - 1.0
    )

    return xr.DataArray(
        transformed, dims=da.dims, coords=da.coords,
    ).astype("float32")

# ---------------------------------------------------------------------------
# PCA denoising helpers
# ---------------------------------------------------------------------------

def fit_pca_denoise(
    data: np.ndarray,
    n_components: int,
    var_name: str = "",
) -> tuple[np.ndarray, PCA]:
    """Fit PCA on a single-variable spatial field and return the denoised
    reconstruction together with the fitted PCA object.

    Args:
        data:         Float32 array of shape ``(T, H, W)`` — one variable,
                      all timesteps.
        n_components: Number of leading EOF components to retain.  Values
                      above ``min(T, H*W)`` are silently clamped.
        var_name:     Optional name used only for the diagnostic print.

    Returns:
        denoised: Reconstructed array, same shape ``(T, H, W)``.
        pca:      Fitted :class:`sklearn.decomposition.PCA` object that can
                  be passed to :func:`apply_pca_denoise` for new data.
    """
    T, H, W = data.shape
    flat = data.reshape(T, H * W).astype(np.float64)  # PCA needs float64

    n_components = min(n_components, T, H * W)
    pca = PCA(n_components=n_components, whiten=False)
    scores = pca.fit_transform(flat)           # (T, n_components)
    recon = pca.inverse_transform(scores)      # (T, H*W)

    var_explained = pca.explained_variance_ratio_.sum() * 100
    print(
        f"[PCA] {var_name}: kept {n_components} components → "
        f"{var_explained:.2f}% variance explained"
    )

    return recon.reshape(T, H, W).astype(np.float32), pca


def apply_pca_denoise(data: np.ndarray, pca: PCA) -> np.ndarray:
    """Project new data through an already-fitted PCA and reconstruct.

    Args:
        data: Float32 array of shape ``(T, H, W)``.
        pca:  PCA object previously returned by :func:`fit_pca_denoise`.

    Returns:
        Denoised array of the same shape.
    """
    T, H, W = data.shape
    flat = data.reshape(T, H * W).astype(np.float64)
    scores = pca.transform(flat)
    recon = pca.inverse_transform(scores)
    return recon.reshape(T, H, W).astype(np.float32)


def pca_denoise_dataset(
    tensor: torch.Tensor,           # (n_vars, T, H, W)
    n_components: int,
    var_names: Optional[list] = None,
    pca_objects: Optional[list] = None,
) -> tuple[torch.Tensor, list[PCA]]:
    """Apply PCA denoising to every variable channel of a tensor.

    If ``pca_objects`` is ``None`` the PCA is *fitted* on this data (use for
    the first realization / training time).  Otherwise the supplied fitted
    PCAs are *applied* without refitting (use for subsequent realizations and
    at generation time).

    Args:
        tensor:      Shape ``(n_vars, T, H, W)``.
        n_components: Components to retain (only used when fitting).
        var_names:   Optional list of variable names for diagnostic prints.
        pca_objects: Pre-fitted PCA objects or ``None``.

    Returns:
        denoised_tensor: Same shape as input.
        pca_objects:     List of fitted :class:`PCA` objects (one per var).
    """
    n_vars = tensor.shape[0]
    var_names = var_names or [str(i) for i in range(n_vars)]
    np_data = tensor.numpy()
    denoised = np.empty_like(np_data)
    fitted_pcas: list[PCA] = []

    for v in range(n_vars):
        channel = np_data[v]  # (T, H, W)
        if pca_objects is None:
            recon, pca = fit_pca_denoise(channel, n_components, var_names[v])
            fitted_pcas.append(pca)
        else:
            recon = apply_pca_denoise(channel, pca_objects[v])
            fitted_pcas.append(pca_objects[v])
        denoised[v] = recon

    return torch.from_numpy(denoised), fitted_pcas


# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_emissions_minmax():
    """
    Lataa emissions.nc vain kerran ja palauttaa min/max-arvot
    CO2_em_anthro:lle ja sul:lle.
    """
    ds_emis = xr.open_dataset(EMISSIONS_PATH)

    minmax = {}
    for var in ["CO2", "SO2"]:
        da = ds_emis[var]
        min_val = float(da.min())
        max_val = float(da.max())
        minmax[var] = (min_val, max_val)

    ds_emis.close()
    return minmax


def normalize(ds: xr.DataArray) -> xr.DataArray:
    """Normalizes a data array"""

    if ds.name in ["CO2", "SO2", "SUL"]:
        result = norm_zscore(ds).fillna(0) #scale_quantile_transform(ds).fillna(0)
        return result

    # Other variables use predefined normalization functions
    norm = NORM_FN[ds.name](ds)
    return norm.fillna(0)

def denorm(ds: xr.DataArray) -> xr.DataArray:
    norm = DENORM_FN[ds.name](ds)

    min_val, max_val = MIN_MAX_CONSTANTS[ds.name]
    # norm = min_max_denorm(norm, min_val, max_val)
    return norm


class ClimateDataset(Dataset):
    def __init__(
        self,
        seq_len: int,
        realizations: list[str],
        data_dir: str,
        target_vars: list[str],
        cond_file: str,
        cond_vars: list[str],
        # ── PCA denoising (set to None to disable) ──────────────────────────
        n_components_target: Optional[int] = None,
        n_components_cond: Optional[int] = None,
    ):
        self.seq_len = seq_len
        self.realizations = realizations

        self.data_dir = data_dir

        # Necessary to convert vars into a Python list
        self.vars = OmegaConf.to_object(target_vars) if not isinstance(target_vars, list) else target_vars
        self.cond_vars = OmegaConf.to_object(cond_vars) if not isinstance(cond_vars, list) else cond_vars

        # PCA configuration and state
        # n_components_* controls how many EOFs to retain (None = PCA off)
        self.n_components_target = n_components_target
        self.n_components_cond = n_components_cond
        # Fitted PCA objects – populated on first load_data call, then reused
        self._pca_target: Optional[list[PCA]] = None
        self._pca_cond: Optional[list[PCA]] = None

        # Store one dataset (out of memory) as an xarray dataset for metadata
        # Store a different dataset as a torch tensor for speed
        self.xr_data: xr.Dataset
        self.tensor_data: torch.Tensor
        self.cond_file = cond_file
        # Load an example realization right off the bat
        self.load_data(self.realizations[0])
        self.lats = 0

    def estimate_num_batches(self, batch_size: int) -> int:
        """Estimates the number of batches in the dataset."""
        return len(self) * len(self.realizations) // batch_size

    def load_data(self, realization: str):
        """Loads the data from the specified paths and returns it as an xarray Dataset."""

        realization_dir = os.path.join(self.data_dir, realization, "*.nc")

        hist_years = list(range(1850, 2015, 5))  # every 5th year
        future_years = list(range(2015, 2101))   # every year
        selected_years = hist_years + future_years

        dataset = xr.open_mfdataset(realization_dir, combine="by_coords").sortby("year")
        self.lats = dataset.lat
        # Only select the variables we are interested in
        dataset = dataset[self.vars]

        # Apply preprocessing and normalization
        self.xr_data = dataset.map(preprocess).map(normalize)#.sel(year=selected_years)

        self.tensor_data = self.convert_xarray_to_tensor(self.xr_data)

        # ── PCA denoising on target ──────────────────────────────────────────
        if self.n_components_target is not None:
            # tensor_data shape: (n_vars, T, H, W)
            self.tensor_data, self._pca_target = pca_denoise_dataset(
                self.tensor_data,
                n_components=self.n_components_target,
                var_names=self.vars,
                pca_objects=self._pca_target,   # None on first call → fits
            )

        cond_file = os.path.join(self.data_dir, self.cond_file)
        self.dataset_cond = xr.open_dataset(cond_file)
        self.dataset_cond = self.dataset_cond[self.cond_vars]
        self.dataset_cond = self.dataset_cond.map(normalize)#.sel(year=selected_years)

        self.tensor_data_cond = self.convert_xarray_to_tensor(self.dataset_cond)

        # ── PCA denoising on conditioning ────────────────────────────────────
        if self.n_components_cond is not None:
            self.tensor_data_cond, self._pca_cond = pca_denoise_dataset(
                self.tensor_data_cond,
                n_components=self.n_components_cond,
                var_names=self.cond_vars,
                pca_objects=self._pca_cond,     # None on first call → fits
            )

        # Print normalized stats
        cond_tensor = self.tensor_data_cond
        for i, var in enumerate(self.cond_vars):
            vals = cond_tensor[i]
            print(f"{var}: min={vals.min():.4f} max={vals.max():.4f} "
                  f"std={vals.std():.4f} unique_range={vals.max() - vals.min():.4f}")

        # Save diagnostic spatial plots (only on first load)
        diag_dir = os.path.join(self.data_dir, "diagnostics")
        if not os.path.isdir(diag_dir):
            os.makedirs(diag_dir, exist_ok=True)
            self._save_cond_diagnostics(diag_dir)

    def _save_cond_diagnostics(self, diag_dir: str):
        """Save spatial maps of normalized conditioning data for visual inspection."""
        years_to_show = [self.dataset_cond.year.values[0], 2015, 2050,
                         self.dataset_cond.year.values[-1]]
        years_to_show = [y for y in years_to_show
                         if y in self.dataset_cond.year.values]

        for var in self.cond_vars:
            da = self.dataset_cond[var]
            n_years = len(years_to_show)

            fig, axes = plt.subplots(1, n_years, figsize=(5 * n_years, 4))
            if n_years == 1:
                axes = [axes]

            for col, yr in enumerate(years_to_show):
                ax = axes[col]
                data = da.sel(year=yr).values
                im = ax.imshow(data, aspect='auto', cmap='RdBu_r',
                               vmin=-1, vmax=1, origin='lower')
                ax.set_title(f"year={yr}\nmin={data.min():.3f} max={data.max():.3f}\n"
                             f"mean={data.mean():.3f} std={data.std():.3f}",
                             fontsize=9)
                plt.colorbar(im, ax=ax, shrink=0.8)

            fig.suptitle(f"{var} — Normalized cond_map (what model sees)", fontsize=13)
            plt.tight_layout()
            save_path = os.path.join(diag_dir, f"cond_normalized_{var}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[DIAG] Saved {save_path}")

        # Also plot spatial-mean time series
        fig, axes = plt.subplots(len(self.cond_vars), 1,
                                  figsize=(12, 4 * len(self.cond_vars)))
        if len(self.cond_vars) == 1:
            axes = [axes]

        for i, var in enumerate(self.cond_vars):
            da = self.dataset_cond[var]
            spatial_dims = [d for d in da.dims if d != "year"]
            ts = da.mean(dim=spatial_dims)
            years = da.year.values

            ax = axes[i]
            ax.plot(years, ts.values, 'b-', linewidth=2)
            ax.set_title(f"{var} — Spatial mean of normalized cond_map\n"
                         f"range: [{float(ts.min()):.3f}, {float(ts.max()):.3f}]",
                         fontsize=12)
            ax.set_xlabel("Year")
            ax.set_ylabel("Normalized value")
            ax.set_ylim(-1.15, 1.15)
            ax.axhline(-1, color='gray', ls='--', alpha=0.4)
            ax.axhline(1, color='gray', ls='--', alpha=0.4)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(diag_dir, "cond_timeseries.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[DIAG] Saved {save_path}")

        # ── PCA scree / reconstruction diagnostics ───────────────────────────
        self._save_pca_diagnostics(diag_dir)

    def _save_pca_diagnostics(self, diag_dir: str):
        """Save scree plots and before/after spatial maps for PCA denoising."""

        pca_sets = [
            ("target", self._pca_target, self.vars,
             self.tensor_data, self.xr_data),
            ("cond",   self._pca_cond,   self.cond_vars,
             self.tensor_data_cond, self.dataset_cond),
        ]

        for tag, pca_list, var_names, tensor, xr_ref in pca_sets:
            if pca_list is None:
                continue  # PCA not enabled for this set

            for v_idx, (pca, vname) in enumerate(zip(pca_list, var_names)):
                # ── scree plot ──────────────────────────────────────────────
                cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(np.arange(1, len(cumvar) + 1), cumvar, 'o-', ms=4)
                ax.axhline(90, color='orange', ls='--', label='90 %')
                ax.axhline(95, color='red',    ls='--', label='95 %')
                ax.set_xlabel("Number of components")
                ax.set_ylabel("Cumulative variance explained (%)")
                ax.set_title(f"PCA scree — {tag}/{vname}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                scree_path = os.path.join(diag_dir,
                                           f"pca_scree_{tag}_{vname}.png")
                plt.savefig(scree_path, dpi=120, bbox_inches='tight')
                plt.close()
                print(f"[DIAG] Saved {scree_path}")

                # ── before / after spatial map ──────────────────────────────
                # Pick the most recent year for illustration
                mid_t = tensor.shape[1] // 2
                raw_map   = tensor[v_idx, mid_t].numpy()       # (H, W) raw
                # recon via PCA
                channel   = tensor[v_idx].numpy()              # (T, H, W)
                recon_all = apply_pca_denoise(channel, pca)    # (T, H, W)
                recon_map = recon_all[mid_t]                   # (H, W)
                resid_map = raw_map - recon_map

                vmin, vmax = raw_map.min(), raw_map.max()
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                for ax, data, title in zip(
                    axes,
                    [raw_map, recon_map, resid_map],
                    ["Raw (normalized)", f"PCA recon ({pca.n_components_} comps)",
                     "Residual (noise removed)"],
                ):
                    im = ax.imshow(data, aspect='auto', cmap='RdBu_r',
                                   vmin=vmin, vmax=vmax, origin='lower')
                    ax.set_title(title, fontsize=10)
                    plt.colorbar(im, ax=ax, shrink=0.8)
                fig.suptitle(
                    f"PCA denoising — {tag}/{vname}  "
                    f"(t_idx={mid_t}, "
                    f"{pca.explained_variance_ratio_.sum()*100:.1f}% var kept)",
                    fontsize=12,
                )
                plt.tight_layout()
                map_path = os.path.join(diag_dir,
                                        f"pca_map_{tag}_{vname}.png")
                plt.savefig(map_path, dpi=120, bbox_inches='tight')
                plt.close()
                print(f"[DIAG] Saved {map_path}")

    def convert_xarray_to_tensor(self, ds: xr.Dataset) -> torch.Tensor:
        """Generate a tensor of data from an xarray dataset"""
        #print(ds)
        # Stacks the data variables ('pr', 'tas', ...) into a single dimension
        stacked_ds = ds.to_stacked_array(
            new_dim="var", sample_dims=["year", "lon", "lat"]
        ).transpose("var", "year", "lat", "lon")
        #print(stacked_ds.to_numpy())
        # Convert the numpy array to a torch tensor
        tensor_data = torch.tensor(stacked_ds.to_numpy(), dtype=torch.float32)

        return tensor_data
    def get_cond_from_coords(self, coord_dict):
        years = coord_dict["year"]
        ds = self.dataset_cond.sel(year=years)
        tensor = self.convert_xarray_to_tensor(ds)
        # Apply the already-fitted conditioning PCA if available
        if self._pca_cond is not None:
            tensor, _ = pca_denoise_dataset(
                tensor,
                n_components=self.n_components_cond,
                var_names=self.cond_vars,
                pca_objects=self._pca_cond,
            )
        return tensor

    def get_pca_state(self) -> dict:
        """Return the fitted PCA objects so they can be saved alongside a
        checkpoint and restored for consistent generation.

        Returns a dict with keys ``'target'`` and ``'cond'``, each holding
        a list of :class:`sklearn.decomposition.PCA` objects (or ``None``).
        """
        return {
            "target": self._pca_target,
            "cond": self._pca_cond,
        }

    def set_pca_state(self, state: dict) -> None:
        """Restore PCA objects from a previously saved state dict.

        Call this before :meth:`load_data` when loading a checkpoint for
        generation so that the same projection is used as during training.
        """
        self._pca_target = state.get("target")
        self._pca_cond = state.get("cond")

    def convert_tensor_to_xarray(
        self, tensor: torch.Tensor, coords: xr.DataArray = None
    ) -> xr.Dataset:
        """Generate an xarray dataset from a tensor of data"""

        assert len(tensor.shape) == 4, "Tensor must have shape (var, time, lat, lon)"

        np_data = tensor.cpu().numpy()

        # Convert the numpy array to a dictionary of xr.DataArrays
        # with the same names as the original dataset
        data_vars = {
            var_name: (["time", "lat", "lon"], np_data[i])
            for i, var_name in enumerate(self.xr_data.data_vars.keys())
        }

        # Create the dataset with the same coordinates as the original dataset
        # Note: The original time values are lost and just start at 0 instead
        ds = xr.Dataset(
            data_vars,
            coords={
                "time": np.arange(np_data.shape[1]),
                "lat": np.linspace(-90, 90, np_data.shape[2]),
                "lon": np.linspace(0, 360, np_data.shape[3]),
            },
        ).map(denorm)

        # If we are provided time coords, create a new time coordinate
        if coords is not None:
            ds = ds.assign_coords(coords)
        return ds

    def __len__(self):
        return len(self.xr_data.year) - self.seq_len + 1

    def __getitem__(self, idx: int):
        """Defines how to get a specific index from the dataset"""
        return self.tensor_data[:, idx : idx + self.seq_len],self.tensor_data_cond[:, idx : idx + self.seq_len]


class ClimateDataLoader:
    def __init__(
        self,
        dataset: ClimateDataset,
        accelerator: Accelerator,
        batch_size: int,
        **dataloader_kwargs: dict[str, Any],
    ):
        self.dataset = dataset
        self.accelerator = accelerator
        self.batch_size = batch_size
        self.dataloader_kwargs = dataloader_kwargs

    def __len__(self):
        return self.dataset.estimate_num_batches(self.batch_size)

    def generate(self) -> torch.Tensor:
        # Iterate through each realization in our dataset
        random.shuffle(self.dataset.realizations)

        for realization in self.dataset.realizations:
            # Load a realization of data into memory
            self.dataset.load_data(realization)

            # Wrap a dataloader around it and generate the data
            dl = self.accelerator.prepare(
                DataLoader(
                    self.dataset, batch_size=self.batch_size, **self.dataloader_kwargs
                )
            )

            for sample in dl:
                yield sample
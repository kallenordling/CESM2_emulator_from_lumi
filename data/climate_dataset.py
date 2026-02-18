import os
import random
from typing import Any
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
    mu    = float(da.mean(skipna=True))
    sigma = float(da.std(skipna=True))
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
    ):
        self.seq_len = seq_len
        self.realizations = realizations

        self.data_dir = data_dir

        # Necessary to convert vars into a Python list
        self.vars = OmegaConf.to_object(target_vars) if not isinstance(target_vars, list) else target_vars
        self.cond_vars = OmegaConf.to_object(cond_vars) if not isinstance(cond_vars, list) else cond_vars
        # Store one dataset (out of memory) as an xarray dataset for metadata
        # Store a different dataset as a torch tensor for speed
        self.xr_data: xr.Dataset
        self.tensor_data: torch.Tensor
        self.cond_file=cond_file
        # Load an example realization right off the bat
        #print('load_data')
        self.load_data(self.realizations[0])
        self.lats=0

    def estimate_num_batches(self, batch_size: int) -> int:
        """Estimates the number of batches in the dataset."""
        return len(self) * len(self.realizations) // batch_size

    def load_data(self, realization: str):
        """Loads the data from the spe
        cified paths and returns it as an xarray Dataset."""

        realization_dir = os.path.join(self.data_dir, realization, "*.nc")

        # Open up the dataset and make sure it's sorted by time
        #print(realization_dir)
        hist_years = list(range(1850, 2015, 5))  # every 5th year
        future_years = list(range(2015, 2101))  # every year
        selected_years = hist_years + future_years
        #xr_data = xr_data.sel(year=selected_years)
        dataset = xr.open_mfdataset(realization_dir, combine="by_coords").sortby("year")
        self.lats=dataset.lat
        # Only select the variables we are interested in
        dataset = dataset[self.vars]

        # Apply preprocessing and normalization
        self.xr_data = dataset.map(preprocess).map(normalize).sel(year=selected_years)

        #if self.spatial_resolution is not None:
        #    with dask.config.set(**{'array.slicing.split_large_chunks' : False}):
        #        self.xr_data = self.xr_data.coarsen(lon=3, lat=2).mean()

        self.tensor_data = self.convert_xarray_to_tensor(self.xr_data)
        cond_file=os.path.join(self.data_dir, self.cond_file)
        self.dataset_cond =xr.open_dataset(cond_file)
        self.dataset_cond = self.dataset_cond[self.cond_vars]
        #print(self.dataset_cond)
        self.dataset_cond = self.dataset_cond.map(normalize).sel(year=selected_years)



        self.tensor_data_cond = self.convert_xarray_to_tensor(self.dataset_cond)
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
        # select those years from the conditioning dataset
        ds = self.dataset_cond.sel(year=years)
        return self.convert_xarray_to_tensor(ds)

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
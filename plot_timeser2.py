import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# ---------- styling (journal-friendly) ----------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

BASELINE = slice(1850, 1900)        # reference period
YR_MIN, YR_MAX = 1850, 2100        # x-axis limits
VAR = "TREFHT"

def calcmean(da: xr.DataArray) -> xr.DataArray:
    """Area-weighted mean over lat/lon (cos(lat) weighting)."""
    weights = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(weights).mean(("lon", "lat"))

def load_anomalies(dir_path: Path) -> xr.DataArray:
    """Load all members from dir, compute area mean + baseline anomalies."""
    series = []
    years = None
    for nc_path in sorted(dir_path.glob("*.nc")):
        print("open", nc_path)
        ds = xr.open_dataset(nc_path)
        da = calcmean(ds[VAR])+273.15

        # guard for baseline existence
        base = da.sel(year=BASELINE)
        if base.size == 0:
            raise ValueError(f"Baseline years {BASELINE} not found in {nc_path.name}")

        anom = da - base.mean()

        # Keep as DataArray indexed by 'year'
        anom = anom.sortby("year")
        if years is None:
            years = anom["year"]
        else:
            # align years across members
            anom = anom.reindex(year=years, method=None)

        # name each member by filename stem
        series.append(anom.assign_coords(member=nc_path.stem).expand_dims("member"))

    # stack to (member, year)
    ens = xr.concat(series, dim="member").isel(time=0)
    return ens

def plot_ensemble(ax, ens: xr.DataArray, color, label):
    """Plot members (light), mean (bold), and 5–95% band."""
    years = ens["year"].values
    # Plot all members
    for m in ens["member"].values:
        ax.plot(years, ens.sel(member=m).values, lw=0.8, alpha=0.25, color=color)

    # Mean and quantiles
    mean = ens.mean("member")
    q05 = ens.quantile(0.05, dim="member")
    q95 = ens.quantile(0.95, dim="member")
    print(ens)
    print(q05)
    ax.fill_between(years, q05, q95, alpha=0.15, linewidth=0, color=color)
    ax.plot(years, mean, lw=2.2, color=color, label=label)

# ---------- paths ----------
#dir_ssp370_co2 = Path("/scratch/project_462001112/emulator_data/gen_co2_ssp370_TREFHT_1850-2100")
dir_ssp370_aero = Path("/scratch/project_462001112/emulator_data/gen_ssp370_aero_v2_TREFHT_1850-2100")
#dir_ssp126_co2 = Path("/scratch/project_462001112/emulator_data/gen_co2_ssp126_TREFHT_1850-2100")
dir_ssp126_aero = Path("/scratch/project_462001112/emulator_data/gen_ssp126_aero_v2_TREFHT_1850-2100")
dir_ramip_aero = Path("/scratch/project_462001112/emulator_data/gen_ssp37_ssp126_aero_v2_TREFHT_1850-2100")
# ---------- load ----------
#ens_ssp370_co2  = load_anomalies(dir_ssp370_co)
#ens_ssp370_co2_v2  = load_anomalies(dir_ssp370_co_v2)
ens_ssp370_aero  = load_anomalies(dir_ssp370_aero)
ens_ssp126_aero = load_anomalies(dir_ssp126_aero)
ens_ramip_aero = load_anomalies(dir_ramip_aero)
#ens_ssp126_co2 = load_anomalies(dir_ssp126_co2)
#ens_aero = load_anomalies(dir_aero)
#ens_aero3 = load_anomalies(dir_aero3)
# Align years across scenarios (intersection to be safe)
#common_years = np.intersect1d(ens_te1t["year"].values, ens_ssp126["year"].values)
#ens_te1t = ens_te1t.sel(year=common_years)
#ens_ssp126 = ens_ssp126.sel(year=common_years)

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(7.2, 4.2))

#plot_ensemble(ax, ens_ssp370_co2,  color="#1fff79",label="SSP3–7.0 (ensemble) CO2")
#plot_ensemble(ax, ens_ssp370_co2_v2,  color="#1fffff",label="SSP126 (ensemble) CO2+sulfate")
plot_ensemble(ax, ens_ssp370_aero, color="#cc2b2b", label="SSP3–7.0 (ensemble) Co2+sulfate")
#plot_ensemble(ax, ens_ssp126_co2, color="#1f4e79", label="SSP1–2.6 (ensemble co2)")
plot_ensemble(ax, ens_ssp126_aero, color="#1fff79", label="SSP1–2.6 (ensemble,co2+sul)")
plot_ensemble(ax, ens_ramip_aero, color="#1fff79", label="SSP1370-ssp126aer")
#plot_ensemble(ax, ens_aero, color="#1f4e00", label="aero constrain (ensemble)")
#plot_ensemble(ax, ens_aero3, color="#1f4e02", label="aero3 constrain (ensemble)")
# Baseline band
ax.axvspan(BASELINE.start, BASELINE.stop, color="0.9", alpha=0.5, lw=0, zorder=0)

# Zero line
ax.axhline(0, lw=1.0, color="0.2", alpha=0.6)

ax.set_xlim(YR_MIN, YR_MAX)
ax.set_xlabel("Year")
# Note: anomalies of K == °C, so label °C
ax.set_ylabel("Global mean T anomaly (°C relative to 1850–1900)",fontsize=8)

ax.margins(x=0)

# small ticks every 25 years
ax.set_xticks(np.arange(1850, 2110, 25))

###plot CMIP6 references

hist_cmip=xr.open_dataset("/scratch/project_462001112/emulator_data/cmip6/historical.nc")
ssp1=xr.open_dataset("/scratch/project_462001112/emulator_data/cmip6/ssp126.nc")
ssp3=xr.open_dataset("/scratch/project_462001112/emulator_data/cmip6/ssp370.nc")

common_members3 = np.intersect1d(hist_cmip.member.values, ssp3.member.values)
common_members1 = np.intersect1d(hist_cmip.member.values, ssp1.member.values)

ds_merged_ssp3 = xr.concat([hist_cmip.sel(member=common_members3), ssp3.sel(member=common_members3)], dim="year")
ds_merged_ssp1 = xr.concat([hist_cmip.sel(member=common_members1), ssp1.sel(member=common_members1)], dim="year")
cmip_ssp3=calcmean(ds_merged_ssp3)
cmip_ssp1=calcmean(ds_merged_ssp1)
cmip_ssp3=cmip_ssp3-cmip_ssp3.sel(year=BASELINE).mean('year')
cmip_ssp1=cmip_ssp1-cmip_ssp1.sel(year=BASELINE).mean('year')
print(ds_merged_ssp1)
ax.plot(cmip_ssp3.year, cmip_ssp3.mean('member').tas, lw=2.2, linestyle='--', color="#cc2b2b", label="SSP3–7.0 (CMIP6 CESM2)")
ax.plot(cmip_ssp1.year, cmip_ssp1.mean('member').tas, lw=2.2, linestyle='--', color="#1f4e79", label="SSP1–2.6 (CMIP6 CESM2)")


##PLOT training data

data_dir = "//scratch/project_462001112/emulator_data/"       # e.g. tas, pr
realizations = ['r10i1181p1f1','r10i1231p1f1','r10i1251p1f1','r10i1281p1f1','r10i1301p1f1','r1i1001p1f1','r1i1231p1f1','r1i1251p1f1']
for i,r in enumerate(realizations):
    ds=xr.open_mfdataset(data_dir+r+"/*.nc")
    ds=calcmean(ds)
    ds=ds-ds.sel(year=BASELINE).mean('year')
    if i==0:
        ax.plot(ds.year,ds.TREFHT,'k',linewidth=0.5,label="Training dataset")    
    else:
        ax.plot(ds.year,ds.TREFHT,'k',linewidth=0.5)

##ramip
ds=xr.open_dataset('tas_Amon_CESM2_ssp370-126aer_r1i1p1f1_gn_201501-207912.nc').groupby('time.year').mean()#.isel(member=0)
print(ds)
ds=calcmean(ds.tas)
print(ds)
ds=ds-cmip_ssp1.sel(year=BASELINE).mean('year')
print(ds)
ax.plot(ds.year,ds.values,'purple',linewidth=3,label="RAMIP")      
ax.legend(frameon=False, ncols=2, handlelength=2.5)
#fig.tight_layout()
plt.title("Emulated global temperature, trainin set includes only ssp370")
fig.savefig("timeseries_temperature_anomaly_ssp3_co2_emulated.png")
fig.savefig("timeseries_temperature_anomaly_ssp3_co2_emulated.svg")  # vector for journals
plt.close(fig)

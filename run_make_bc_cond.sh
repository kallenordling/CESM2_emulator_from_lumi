#!/bin/bash
# CPU-only: build the BC (black carbon) 3rd-cond-channel files, inside the LAIF
# singularity container (project venv injected; needs xarray + xesmf, same as
# run_concat_ssp126.sh). Two stages + verification:
#
#   1. data/make_aerosol_files.py <exp> --species BC
#      input4mips CEDS/IAMC BC → annual Gt/gridpoint on the native 360x720 grid
#      → BC_per_gridpoint_{hist,ssp370,ssp126,ssp245}.nc   (hist clipped ≤2014)
#   2. data/concat_and_regrid.py --build-bc
#      hist≤2014 + scenario≥2015 splice (decadal IAMC years → annual interp),
#      xesmf bilinear periodic regrid to the 192x288 CESM2 grid, then BC is
#      injected as a 3rd data_var into COPIES of the existing cond files:
#        emissions_{hist,ssp370,ssp245,aaer,ghg}_only_timefixed_bc.nc
#        emissions_ssp126_only_timefixed_co2fix_bc.nc
#      ghg BC = pinned constant 1850 field (mirrors SUL); aaer BC varies.
#      ORIGINALS ARE NOT TOUCHED — old TREFHT runs stay reproducible.
#   3. verify: CO2/SUL byte-identical to originals, BC present/no-NaN,
#      ghg constant, aaer varying. Exits 1 on any failure.
#
# Usage on LUMI (from the repo dir, after `git pull`):
#     bash run_make_bc_cond.sh
# If the login node blocks singularity, wrap it:
#     srun --account=project_462001328 --partition=small --time=00:45:00 \
#          --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G \
#          bash run_make_bc_cond.sh
set -euo pipefail

DATA_ROOT=/scratch/project_462001328/emulator_data
# Grid template (only lat/lon read) — same default as run_concat_ssp126.sh.
TARGET="${DATA_ROOT}/emissions_ssp126_only_timefixed.nc"

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif

_VENV_SITE=$(realpath /projappl/project_462001328/venvs/diffesm_laif 2>/dev/null \
             || echo /projappl/project_462001328/venvs/diffesm_laif)/lib/python3.12/site-packages
export SINGULARITYENV_PYTHONPATH="${_VENV_SITE}"
export PYTHONNOUSERSITE=1

echo "[bc-cond] SIF=${SIF}"
echo "[bc-cond] stage 1: input4mips BC → BC_per_gridpoint_*.nc …"
for exp in hist ssp370 ssp126 ssp245; do
    echo "[bc-cond]   make_aerosol_files.py ${exp} --species BC"
    singularity exec "${SIF}" python data/make_aerosol_files.py "${exp}" --species BC
done

echo "[bc-cond] stage 2: splice + regrid + inject BC into *_bc.nc cond files …"
singularity exec "${SIF}" python data/concat_and_regrid.py \
    --build-bc \
    --target   "${TARGET}" \
    --data_dir "${DATA_ROOT}/"

echo "[bc-cond] stage 3: verifying outputs …"
singularity exec "${SIF}" python - "${DATA_ROOT}" <<'EOF'
import sys
import numpy as np
import xarray as xr

D = sys.argv[1]
pairs = {  # original (untouched) -> new _bc file
    f"{D}/emissions_hist_only_timefixed.nc":          f"{D}/emissions_hist_only_timefixed_bc.nc",
    f"{D}/emissions_ssp370_only_timefixed.nc":        f"{D}/emissions_ssp370_only_timefixed_bc.nc",
    f"{D}/emissions_ssp245_only_timefixed.nc":        f"{D}/emissions_ssp245_only_timefixed_bc.nc",
    f"{D}/emissions_ssp126_only_timefixed_co2fix.nc": f"{D}/emissions_ssp126_only_timefixed_co2fix_bc.nc",
    f"{D}/emissions_ghg_only_timefixed.nc":           f"{D}/emissions_ghg_only_timefixed_bc.nc",
    f"{D}/emissions_aaer_only_timefixed.nc":          f"{D}/emissions_aaer_only_timefixed_bc.nc",
}
fail = []
for orig, new in pairs.items():
    name = new.split("/")[-1]
    o, n = xr.open_dataset(orig), xr.open_dataset(new)
    for v in ("CO2", "SUL"):
        d = float(np.abs(o[v].values - n[v].values).max())
        if d != 0.0:
            fail.append(f"{name}: {v} differs from original (max abs {d:g})")
    if "BC" not in n:
        fail.append(f"{name}: BC variable missing"); continue
    bc = n["BC"]
    if bc.shape != n["SUL"].shape:
        fail.append(f"{name}: BC shape {bc.shape} != SUL {n['SUL'].shape}")
    if bool(np.isnan(bc.values).any()):
        fail.append(f"{name}: BC contains NaN")
    tdim = [d_ for d_ in bc.dims if d_ not in ("lat", "lon")][0]
    gsum = bc.sum(dim=("lat", "lon")).values
    const = bool(np.all(gsum == gsum[0]))
    if "ghg" in name and not const:
        fail.append(f"{name}: ghg BC must be constant in time")
    if "aaer" in name and const:
        fail.append(f"{name}: aaer BC must vary in time")
    t = n[tdim].values
    picks = [t[0], t[len(t)//2], t[-1]]
    spots = ", ".join(f"{int(y)}={float(bc.sel({tdim: y}).sum()):.5f}" for y in picks)
    print(f"  OK {name}: BC global-sum Gt/yr [{spots}] constant={const}")
    o.close(); n.close()

if fail:
    print("VERIFY FAILED:")
    for f in fail:
        print("  " + f)
    sys.exit(1)
print("VERIFY OK: all 6 *_bc.nc cond files good; originals untouched.")
EOF

echo "[bc-cond] done — configs/config_data.yaml + eval_aero.py already point at the *_bc.nc files."

# Design: BC 3rd cond channel + PRECT 2nd output channel

Status: DESIGN ONLY (architect). Engineer implements. Branch `precip-bc`.
Both changes alter UNet channel counts → **fresh training required, existing
2-channel TREFHT checkpoints will NOT load** (input_conv, cond_input_proj,
SpatialCondEncoder first conv, and final conv all change shape). The two
features are independent toggles but, since each forces a retrain, recommend
landing them in ONE fresh run to amortize compute. Land BC first in review
order (cond-side only, smaller blast radius), then PRECT.

All current cond integration points assume exactly 2 channels at HARD-CODED
indices (CO2=0, SUL=1). The per-channel-CFG infra is "general in principle"
but several call sites slice `[:, 1:2]` / `[:, 1]` literally — those are the
real work.

---

## PART A — BC as 3rd conditioning channel (CO2, SUL, BC)

### A1. Cond-file builder
BC must become a **3rd `data_var` inside the same `emissions_*_only_timefixed.nc`
files** the loader/eval already read (one merged file per scenario, vars
CO2+SUL today). Reuse the SO2 path:

- `data/make_aerosol_files.py` is hard-wired to SO2 and renames the var to
  `SUL` (`rename_to_co2` at make_aerosol_files.py:85-91; `ANTHRO_PATTERN`
  branches at :29-41). **Parameterize species**: add a `--species {SO2,BC}`
  arg, drive `ANTHRO_PATTERN` glob + output var-name + output filename from it.
  - hist BC: `BC-em-anthro_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn_*.nc`
  - ssp370 BC: `BC-em-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-AIM-ssp370-1-1_gn_201501-210012.nc`
  - **Splice year:** BC CEDS-2025 hist runs to 2023, SO2 CEDS-2017 stops at
    2014. To keep cond consistent with existing files, **clip BC hist to
    ≤2014** and take ssp370 IAMC BC for ≥2015 (do NOT use 2015-2023 CEDS BC).
    Same CEDS→IAMC junction step risk as SUL — mitigated downstream by the
    smoothing+PCA (A3), same as SUL's 2015 junction.
  - Keep surface-anthro only (AIR branch stays commented, matching SUL). BC-em-
    AIR-anthro is optional; defer (adds aviation BC the SUL channel omits).
  - BC is NOT cumulative (annual emissions, like SUL). Leave the cumsum
    commented exactly as in the SUL builder (make_aerosol_files.py:186).
  - Output: `BC_per_gridpoint_<exp>.nc` with var `BC`.

- `data/concat_and_regrid.py`: add `BC_HIST`/`BC_SSP` alongside CO2/SO2
  (mirror :35-36, :94-103), include `ds_bc` in the `xr.merge` at :106, and
  in the hist/ghg/aaer split (:148-170):
  - **ghg** (`ds_ghg`): zero BC and pin to year-0, exactly like SUL
    (`ds_ghg['BC'] *= 0; ds_ghg['BC'] += ds['BC'].isel(year=0)`, mirror
    :156-157). GHG holds all aerosols fixed.
  - **aaer** (`ds_aero`): keep BC varying (only CO2 is zeroed at :159-160).
    This is physically correct: CESM2 LENS2 AAER single-forcing varies ALL
    anthropogenic aerosols incl. BC — so the AAER *target* already contains a
    BC signal the model currently has no input for. Adding the BC channel
    closes that gap (relevant to the under-learned-aerosol-fingerprint
    diagnosis, MEMORY: model_skill_diagnosis).
  - hist: BC carried as-is.
- Rebuild all four `emissions_*_only_timefixed.nc` once via this script.

### A2. normalize() / clip / minmax — `data/climate_dataset.py`
- `normalize()` name gate at :327 `if ds.name in ["CO2", "SUL"]` → add `"BC"`.
- `_CLIP_PCTL` dict :288 → add `"BC": (5, 95)`. BC is heavy-tailed combustion
  emissions, same hotspot geography as SO2; SUL's 5-95 reasoning
  (cond_normalization_diag: 1-99 flattens aerosol contrast to the -1 floor)
  applies identically. Do NOT use CO2's (1,99).
- `_get_emissions_minmax()` var loop :303 `for var in ["CO2","SO2","SUL","sul"]`
  → add `"BC"`. The `@lru_cache` is keyed only on EMISSIONS_PATHS so it picks
  up the new var automatically once the files contain it.

### A3. PCA + smoothing (config_data.yaml)
- `n_components_cond: [30, 5]` → `[30, 5, 5]`. BC shares SUL's spatial
  character (combustion sources), so 5 EOFs — same rationale as SUL (drops the
  CEDS→IAMC junction EOF, config_data.yaml:19). Per-channel list already
  threads through `pca_denoise_dataset` (climate_dataset.py:240-248) and
  per-scenario persistence (multi_experiment_dataset.py:227-242) with no code
  change — it's length-`n_cond_vars` generic.
- `cond_smooth_sigma: [0, 2]` → `[0, 2, 2]`. BC has the same shipping-lane/
  flight-path inventory artefacts as SO2 → σ=2 (config_data.yaml:24).
  `smooth_cond_spatial` is already per-channel (climate_dataset.py:145).

### A4. Per-channel CFG dropout — `trainer/unetTrainer.py`
- Add `cfg_bc_drop_prob` (config_aero.yaml) read at :264-266 next to
  `cfg_co2_drop_prob`/`cfg_sul_drop_prob`. Recommend **0.3** (match SUL — BC is
  an aerosol channel, want CO2-only batches frequent).
- In `get_loss` add a 3rd dropout block after :1261 mirroring the SUL block:
  ```
  if self.cfg_bc_drop_prob > 0:
      drop_bc = (torch.rand(...) < self.cfg_bc_drop_prob) & joint_mask
      cond_map_input[drop_bc, 2] = NULL_COND_VALUE
      cond_dropped = cond_dropped | drop_bc
  ```
  Keep the SAME `joint_mask` (hist+ssp370 only, :1237-1243): aaer has constant
  CO2 (dropping it kills the only joint signal), ghg has constant aerosols.
  Dropping BC there teaches nothing. BC varies in hist/ssp370 → correct set.

### A5. Eval per-channel CFG — `eval_aero.py`
`generate_timeseries` hard-codes the 2-channel decomposition (:393-396 build
co2_only/sul_only by nulling index 1/0; :424-426 the 3-pass `pred = null +
w_co2(co2-null) + w_sul(sul-null)`). Generalize to BC:
- Build `cond_bc_only` (null indices 0 and 1, keep 2). Concat **four**
  conditionings `[co2_only, sul_only, bc_only, null]` (:419-421 → 4*B forward).
  Split into 4 (:423).
- `pred = null + w_co2(co2-null) + w_sul(sul-null) + w_bc(bc-null)` (:424-426).
- Add `guidance_bc` arg (:359) + module default `GUIDANCE_BC=1.0` (:203) + CLI
  `--guidance-bc` (:1895) + plumb at :2131. Production default 1.0 (direct
  conditioning; per-channel guidance is a diagnostic A/B knob, MEMORY:
  cfg_inference_tuning says inference-time CFG tuning does NOT fix bias —
  keep at 1.0).
- COND_VARS must become `[CO2, SUL, BC]` (read from config_data.yaml at
  eval_aero.py:2104; verify it reads `cond_vars` from data cfg, else hard-add).

### A6. EBM / interaction index hard-codes (low priority)
- `_compute_ebm_loss` reads `cond_map[:, 1:2]` as SUL (:1032). EBM is DROPPED
  (scaling 0, config_aero.yaml:100; MEMORY ebm_term_near_inactive) — leave as
  is; if ever re-enabled add a BC term `cond_map[:, 2:3]`.
- `_interaction_term` (:1089-1090) nulls index 1 for co2only / index 0 for
  sulonly — a 2-way CO2-vs-SUL split. With BC present this is incomplete.
  Minimal fix: make co2only null **both** aerosol channels (`co2only[:, 1:] =
  NULL`) so the term stays a clean CO2-vs-combined-aerosol additivity prior on
  hist. Interaction is a soft hist-only prior (scaling adapts from 0.01); this
  is sufficient.

---

## PART B — PRECT as 2nd output channel (TREFHT, PRECT)

Diffusion denoises the target, so a 2-var target means the noised image has 2
channels: **`in_channels = out_channels = 2`**. cond is injected separately.

### B1. Transform + normalization — `data/climate_dataset.py:24-34`
PRECT is monthly CAM h0 in m/s, annual-meaned, heavy-tailed. Add explicit
`"PRECT"` entries (do NOT reuse `"pr"`, units differ):
- `PREPROCESS_FN["PRECT"] = lambda x: x * 8.64e7`  (m/s → mm/day:
  ×1000 m→mm ×86400 s→day).
- `NORM_FN["PRECT"]  = lambda x: (np.log1p(x) - PRECT_LOG_MEAN) / PRECT_LOG_STD`
  log1p compresses the positive tail (no epsilon needed; log1p(0)=0, and
  mm/day ≥ 0). Then z-score so the channel sits at ~unit variance to balance
  MSE against TREFHT's `(x-4.5)/21`.
- `DENORM_FN["PRECT"] = lambda x: np.expm1(x * PRECT_LOG_STD + PRECT_LOG_MEAN)`
  (returns mm/day; eval converts back if °C-equivalent units wanted).
- `PRECT_LOG_MEAN`/`PRECT_LOG_STD`: fixed module constants (mirror TREFHT's
  hard-coded 4.5/21). **Estimate once** from a clean realization
  (log1p(mm/day) mean & std over hist+ssp370) and bake in. Provide a 10-line
  `scripts/estimate_prect_norm.py` (engineer) — do NOT compute at runtime
  (keeps norm deterministic + resume-invariant). Expected ballpark mu≈0.7,
  sigma≈0.9; **confirm from data, do not trust these numbers**. Blocked on the
  NaN-poisoned re-download — write the code, fill constants when data lands.
- `normalize()`/`denorm()` already dispatch by `ds.name` (:325-345); no
  structural change. `MIN_MAX_CONSTANTS` (:24) is only used by `denorm`'s
  unused tail — add a `"PRECT"` entry for safety but it's inert.

### B2. Dataset: load two target-var trees — `data/climate_dataset.py:495-502`
TREFHT and PRECT are staged in separate dirs
(`training_data/TREFHT/<scen>/<real>/` vs `training_data/PRECT/<scen>/<real>/`).
`load_data` opens ONE `data_dir/<real>/*.nc` and does `dataset[self.vars]`
(:495-501) — selecting both vars from one mfdataset fails when they live apart.

**Recommended (no extra disk):** allow `data_dir` to be a list (one dir per
target var). In load_data, open each, select its var, `xr.merge` on shared
coords, then proceed. Pair by **identical realization dir name** and intersect
on `selected_years` (:481-516 already intersects years). The existing
`convert_xarray_to_tensor` stacks vars by `to_stacked_array` (:795) → stays
generic; **ensure deterministic var order = target_vars order** (stacked-array
ordering is alphabetical by default → TREFHT, PRECT happens to invert; pin
order explicitly by reindexing the merged ds to `self.vars` before stacking).
- Alternative (simpler code, costs disk + a staging job): pre-merge to
  `training_data/COMBINED/<scen>/<real>/*.nc` with both vars; then `data_dir`
  stays a string. Prefer the list approach to avoid duplicating ~30 members.
- **Engineer must verify** TREFHT and PRECT realization dir names match
  exactly across the two trees (MEMORY resume_precip_bc_branch: lens2/LENS2
  naming). Mismatch → silent member-misalignment.
- `n_components_target: null` (config_data.yaml:18) stays null — no target PCA.
- Climatology `get_baseline_mean` (:837-893) already returns `(1, n_vars, 1,
  H, W)` — generic over n_vars, no change. PRECT "anomaly vs 1850-1900" is
  computed identically (mean precip baseline); physically fine.

### B3. Loss formulation — `trainer/unetTrainer.py`
Primary MSE `calc_mse_loss` (:62-73) averages over ALL channels equally. With
both channels ≈unit-variance after B1, equal weight is acceptable; but precip
is noisier. Add **optional per-channel target weights**:
- New config `target_var_weights: [1.0, w_pr]` (w_pr≈0.5 to start). Apply in
  `calc_mse_loss` by weighting the channel dim before the mean (multiply
  `spatial_loss[:, c] *= weight[c]`). Pass weights from trainer (read in
  `_init_loss_schedule_and_flags`). Keep default `[1.0, 1.0]` so behaviour is
  unchanged when omitted.

**Aux losses are TEMPERATURE physics — restrict to channel 0 (TREFHT):**
- `_compute_tcre_loss` (:1038-1081): `tcre_pred` and the climatology
  subtraction span all channels; the `.mean(dim=(1,2,3,4))` at :1075 would
  blend precip into the TCRE slope. **Slice ch0**: at the top set `tcre_pred =
  tcre_pred[:, 0:1]` (and `cond_map[..., 0:1]` is already CO2). TCRE is
  CO2→ΔT only; precip has no TCRE.
- `_compute_ebm_loss` (:1018-1036): `pred_anomaly` → `pred_anomaly[:, 0:1]`.
  (Inert anyway, scaling 0.)
- `_compute_interaction_loss` / `_interaction_term` (:1083-1142): the additivity
  prior is about the forced temperature response → operate on ch0. Slice
  `pred_x0_cond[mask][:, 0:1]` etc. (or accept a small over-constraint on
  precip and leave full — but recommend ch0 to avoid mis-teaching precip
  additivity).
- `_compute_gmean_loss` (:1144-1159): pin ch0 global mean only (`pred_x0_cond[:,
  0:1]`, `target_gm` already ch-agnostic gmean — recompute target_gm on ch0).
  Inert by default (scaling 0).
- `cond_loss` (anomaly regression, :1400) and `_scenario_ensemble_mean`
  (:995-1009): these are plain field-matching / ensemble-averaging — **keep on
  BOTH channels**. The cond_loss supervises the precip forced response too,
  which is what we want. `anom_signal`/`anom_error` metrics will then blend
  channels; acceptable, or split per-channel logging (nice-to-have).
- `_precompute_tcre_slope` (:831-930) reads `t[0]` (ch0) already (:875-878) —
  no change. `_precompute_interaction_target` (:958-968) reads `t[0]` — fine.

### B4. Model — config only
`UNetModel3D` is fully channel-parametric (input_conv :725, final conv
:971 `nn.Conv3d(model_dim, out_channels, 1)`, SpatialCondEncoder `in_ch =
cond_channels` :616). No model code change — just `in_channels: 2,
out_channels: 2` in config_aero.yaml. The single diffusion process over a
2-channel image is standard (shared noise schedule, joint denoise).

### B5. Eval — `eval_aero.py`
- `generate_timeseries`: `gen = torch.randn(B, 1, 1, H, W)` (:390) →
  `(B, out_channels, 1, H, W)`. Read `out_channels` from `cfg.model`.
- Return shape becomes `(out_channels, T, H, W)`: the `gen.squeeze(1).squeeze(1)`
  at :433 assumes 1 channel — keep the channel dim
  (`gen.squeeze(2)` → (B, C, H, W)), stack to (C, T, H, W).
- Denorm hard-code `gen_norm * 21.0 + 4.5` (:2135) → per-channel via
  `DENORM_FN`. Apply ch0 with TREFHT denorm for the existing TREFHT writer/
  plots (unchanged outputs), ch1 with PRECT denorm.
- **Minimal-risk path:** add `--target-var {TREFHT,PRECT}` selecting the output
  channel index; default TREFHT (ch0) → existing eval/NetCDF/plot code
  (TREFHT_* vars, :1090-1206; the `°C` ΔT-vs-cumCO2 / TCRE machinery) is
  untouched. A second eval pass with `--target-var PRECT` (ch1, mm/day units,
  swap axis labels :724/:1341, anomaly baseline still 1850-1900) produces
  PRECT_* outputs. This avoids forking the whole NetCDF writer. PRECT metrics:
  global-mean precip anomaly, pattern correlation vs CESM2, wet/dry-region
  bias; the TCRE-curve panel is temperature-only — skip it for PRECT.

### B6. Backward compatibility
Stated up top: in/out 1→2 and cond 2→3 both change conv shapes → no warm-start
from v0.1.0 TREFHT checkpoints. `_PERSISTED_FIELDS` (:99-114) and PCA
persistence are unaffected (they don't encode channel count). Fresh
`save_name` (config_aero.yaml:56) → starts at epoch 0. The `_build_save_dict`/
`load` path is channel-agnostic so new 2/3-channel ckpts round-trip normally.

---

## PART C — Config diffs

### configs/config_aero.yaml
```yaml
model:
    in_channels: 2        # was 1  (TREFHT, PRECT)
    out_channels: 2       # was 1
    cond_channels: 3      # was 2  (CO2, SUL, BC)
trainer:
  hyperparameters:
    cfg_bc_drop_prob: 0.3            # NEW — match SUL
    target_var_weights: [1.0, 0.5]   # NEW — [TREFHT, PRECT]; default [1,1] if absent
    save_name: run_bcprect.pt        # fresh name → epoch 0
```

### configs/config_data.yaml
```yaml
target_vars:
  - TREFHT
  - PRECT                            # NEW
cond_vars:
  - CO2
  - SUL
  - BC                               # NEW
n_components_cond: [30, 5, 5]        # was [30, 5] — BC=5 EOFs
cond_smooth_sigma: [0, 2, 2]         # was [0, 2]  — BC σ=2
# Each experiment_config data_dir → list of two trees, e.g.:
#   data_dir:
#     - /scratch/.../training_data/TREFHT/hist
#     - /scratch/.../training_data/PRECT/hist
# cond_file unchanged (now contains BC as 3rd var after rebuild)
```

---

## PART D — Ordered implementation tasks (smallest-risk first)

**Additive / no-retrain (can land + verify independently):**
1. `make_aerosol_files.py`: add `--species` param (SO2/BC), build
   `BC_per_gridpoint_<exp>.nc`. Verify global totals print sane (A1).
2. `concat_and_regrid.py`: merge BC, ghg-zero / aaer-keep BC, rebuild the four
   `emissions_*_only_timefixed.nc`. Diff CO2/SUL channels vs old files →
   must be byte-identical; BC is purely additive (A1).
3. `estimate_prect_norm.py` + bake `PRECT_LOG_MEAN/STD` constants (B1).
   Blocked on clean PRECT data; write code now, fill constants later.

**Cond-side code (needs retrain — BC):**
4. `climate_dataset.py`: BC in `normalize` gate, `_CLIP_PCTL`,
   `_get_emissions_minmax` (A2). Unit-test `normalize` on a BC slice.
5. `trainer/unetTrainer.py`: `cfg_bc_drop_prob` + 3rd dropout block (A4);
   `_interaction_term` null-both-aerosols (A6).
6. `eval_aero.py`: 4-way CFG decomposition + `guidance_bc` + COND_VARS (A5).
7. config diffs for BC (cond_channels 3, cond_vars, n_components_cond,
   cond_smooth_sigma).

**Output-side code (needs retrain — PRECT):**
8. `climate_dataset.py`: PRECT transform entries (B1); `data_dir` list +
   two-tree merge + pinned var order (B2).
9. `trainer/unetTrainer.py`: `target_var_weights` in `calc_mse_loss`; slice aux
   losses (TCRE/EBM/interaction/gmean) to ch0 (B3).
10. `eval_aero.py`: out_channels-shaped `gen`, per-channel denorm,
    `--target-var` selector (B5).
11. config diffs for PRECT (in/out 2, target_vars, data_dir lists).

**Validate:**
12. Fresh train (BC+PRECT together). First-ckpt checks: ckpt has 3-channel
    cond + 2-channel io; `ckpt["PCA"]["per_scenario"]` has 3 entries/scenario;
    cond diagnostics PNGs show a sane BC map. Isolated-fork A/B (MEMORY
    diagnostic_ab_tooling): one arm BC-on vs current 2-channel base — measure
    aaer patcorr (does the BC input lift the under-learned aerosol fingerprint
    from 0.42?) and that TREFHT bias/TCRE is not regressed. PRECT: pattern
    correlation + global-mean precip trend vs CESM2.

## Risks & cheapest falsification
- **BC adds no skill / hurts aerosol channel.** Cheapest test: 2-channel base
  vs 3-channel BC fork at equal epochs, compare aaer/ssp370 patcorr + TREFHT
  TCRE. If BC patcorr contribution ≈0 and SUL fingerprint unchanged, BC is
  redundant (SUL already proxies combustion geography) — keep it only if
  precip benefits.
- **PRECT destabilizes TREFHT** via shared backbone / unbalanced MSE. Falsify
  early: watch TREFHT VAL/MSE in the first ~30 epochs of the joint run vs the
  TREFHT-only baseline at the same epoch; if TREFHT degrades, lower
  `target_var_weights[1]` or decouple (separate heads — larger redesign).
- **PRECT norm constants wrong** (tail not compressed) → precip MSE dominates
  or vanishes. Falsify by histogram of normalized PRECT (should be ~N(0,1), no
  fat tail) before training.
- **Var-order bug** in the two-tree merge → channels swapped (precip denoised
  as temperature). Assert `merged.data_vars` order == `self.vars` in load_data.
- **BC CEDS→IAMC 2015 junction** re-introduces a 2015 spike (cf. SUL, MEMORY
  aaer_2015_spike). Mitigated by σ=2 smooth + 5-EOF PCA; verify on the BC cond
  timeseries PNG that 2014/2015 is continuous.

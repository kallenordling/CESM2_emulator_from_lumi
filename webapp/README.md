# Web app: edit emissions, generate climate maps

A Gradio app, hosted as a Hugging Face Space, where anyone can pick a scenario
and year, scale CO₂ / SO₂ / black-carbon emissions (globally or in one region),
and get emulated annual-mean temperature and precipitation maps. They can also
download the result as NetCDF.

```
GitHub (this repo)                    Hugging Face
──────────────────                    ────────────
webapp/space/  ──GitHub Action──▶     Space  (app.py, runs the sampler)
models/, custom_diffusers/ (copied)          │ reads EMULATOR_BUNDLE_REPO
                                             ▼
webapp/bundle/ ──upload once─────▶    Model repo (weights + emission maps)
```

The code is deployed from GitHub. The 200 MB bundle goes to the Hub instead,
because large binaries are kept out of this repo's history on purpose.

## Files

| file | what it does |
|---|---|
| `export_bundle.py` | checkpoint → `bundle/` (EMA weights, cond anchors, σ, raw emission maps, baseline + 25-member reference maps). Refuses non-minmax / PCA checkpoints and checks the app's cond normalisation against `data/climate_dataset.py` |
| `space/emulator.py` | inference core: edit → smooth → min/max → 50-step sampling → denorm, with paired noise |
| `space/app.py` | the Gradio UI |
| `build_space.sh` | assembles the Space: `space/` plus `models/video_net.py`, `models/rotary_embedding.py`, `custom_diffusers/continuous_ddpm.py` |
| `../.github/workflows/deploy-space.yml` | pushes the assembled Space on every change to `main` |

## 1. Build the bundle (once per checkpoint)

From the repo root, with the LUMI scratch mounted:

```bash
python webapp/export_bundle.py \
    --ckpt     ~/mnt/lumi_sc3/runs_mmlin/ens25/run_mmlin_co2fix_960.pt \
    --eval-dir ~/mnt/lumi_sc3/eval_output/manual/ens25_mmlin_ep960 \
    --cond-dir ~/mnt/lumi_sc3/emulator_data \
    --out      webapp/bundle
```

`--eval-dir` must be the ens25 eval of the same checkpoint. It provides the
emulator's own 1850–1900 baseline and the reference maps.

## 2. Run locally

```bash
bash webapp/build_space.sh webapp/_build/space --with-bundle
cd webapp/_build/space && pip install gradio -r requirements.txt && python app.py
```

## 3. Publish

```bash
# a) weights + data -> a model repo (once per checkpoint)
hf upload <hf-user>/cesm2-emulator-bundle webapp/bundle . --repo-type model

# b) the Space: create it once on huggingface.co (SDK: Gradio), then in its
#    Settings set the variable EMULATOR_BUNDLE_REPO=<hf-user>/cesm2-emulator-bundle
#    (plus a secret HF_TOKEN if the model repo is private).
#    Hardware: free "CPU basic" (2 vCPU, 16 GB) — about 4 min per member at
#    50 steps; the unmodified run behind "compare" is cached. ZeroGPU (PRO)
#    would cut that to seconds with no code change.

# c) GitHub: Settings -> Secrets and variables -> Actions
#    secret HF_TOKEN (write), variable HF_SPACE=<hf-user>/cesm2-emulator
#    Then push to main, or run the workflow by hand ("Run workflow").
```

To link it from GitHub, add a badge to the top-level README:
`[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/<hf-user>/cesm2-emulator)`

## Scientific guard rails built into the app

- Inputs are smoothed and normalised with the checkpoint's own anchors, never
  refitted. Any cell above +1 (beyond the training maximum) is outlined, and the
  summary warns that the result is extrapolation.
- With "compare with unmodified" on, the edited and unmodified runs share the
  same initial and per-step noise. This cancels much of the weather noise, but
  not all: in a 1-member test, local differences had a median of 0.08 K and a
  99th percentile of 0.4 K. The unmodified run is cached, because a fixed seed
  makes it reproducible.
- Anomalies are relative to the emulator's own 1850–1900 climate (25-member
  mean), so no mean-state offset leaks into them.
- A "Limitations" panel covers the known weak spots: unseen SSPs, the
  under-response to aerosol differences, and displaced regional responses.

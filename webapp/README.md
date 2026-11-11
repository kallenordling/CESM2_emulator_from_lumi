# Web app: edit emissions, generate climate maps

A Gradio app, hosted on CSC Rahti, where anyone can pick a scenario and year,
scale CO₂ / SO₂ / black-carbon emissions (globally or in one region), and get
emulated annual-mean temperature and precipitation maps. They can also download
the result as NetCDF.

```
GitHub (this repo)                         CSC Rahti (OpenShift)
──────────────────                         ─────────────────────
push to release-mse-minmax ──webhook──▶    BuildConfig: docker build webapp/Dockerfile
                                                 │ new image
                                                 ▼
                                           Deployment: app.py, 2 CPU / 4 GiB
                                                 │ reads at startup
Hugging Face (private model repo) ◀──────────────┘ EMULATOR_BUNDLE_REPO + HF_TOKEN
nordlin1/cesm2-emulator-bundle: weights + emission maps (from webapp/bundle)
```

The code deploys from GitHub. The 245 MB bundle sits in a private Hugging
Face model repo instead, because large binaries are kept out of this repo's
history on purpose.

## Files

| file | what it does |
|---|---|
| `export_bundle.py` | checkpoint → `bundle/` (EMA weights, cond anchors, σ, raw emission maps, baseline + 25-member reference maps). Refuses non-minmax / PCA checkpoints and checks the app's cond normalisation against `data/climate_dataset.py` |
| `space/emulator.py` | inference core: edit → smooth → min/max → 50-step sampling → denorm, with paired noise |
| `space/app.py` | the Gradio UI |
| `Dockerfile` | the container (context = repo root; copies the real `models/` + scheduler code). Runs as Rahti's random non-root UID |
| `rahti/rahti.yaml` | ImageStream, BuildConfig (GitHub, Docker strategy), Deployment, Service, Route (edge TLS) |
| `rahti/deploy.sh` | one-command deploy/update on Rahti, optionally registers the GitHub webhook |
| `build_space.sh` | assembles `space/` + model code into one folder, for a local run (or a Hugging Face Space, which now needs PRO) |
| `smoke_test.py` | CPU check: unmodified sample vs the 25-member reference |

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

## 3. Publish on CSC Rahti

Once only:

```bash
# a) weights + data -> a private Hugging Face model repo (once per checkpoint)
hf auth login
hf upload nordlin1/cesm2-emulator-bundle webapp/bundle . --repo-type model --private

# b) Rahti access: https://my.csc.fi -> your CSC project -> Rahti -> "Apply for access"
#    then: https://console.rahti.csc.fi -> your name -> "Copy Login Command"
oc login https://api.2.rahti.csc.fi:6443 --token=...

# c) deploy: namespace name (globally unique), CSC project number, and hook up GitHub
bash webapp/rahti/deploy.sh cesm2-emulator <csc-project-number> --github-webhook
```

After that, every push to `release-mse-minmax` rebuilds the image on Rahti and
redeploys it. To deploy from another branch, change `ref:` in `rahti/rahti.yaml`.
Re-running `deploy.sh` is safe: it applies any changes and rebuilds.

**Cost.** Rahti bills Cloud Billing Units on max(requested, used). The app
requests 0.5 core and 3 GiB, about 3,900 BU/month while running. Scale to zero
with `oc scale deployment/cesm2-emulator --replicas=0` when it isn't needed.

**Speed.** Two CPU cores: about 2 min per member at the default 25 steps. The
unmodified run behind "compare" is cached.

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

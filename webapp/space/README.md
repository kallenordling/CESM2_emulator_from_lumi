---
title: CESM2 Emission Emulator
emoji: 🌍
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.31.0
python_version: "3.11"
app_file: app.py
pinned: false
suggested_hardware: cpu-basic
---

# CESM2 emission → climate emulator

Interactive front end for a diffusion-model emulator of the CESM2 Earth system
model. Choose an emission scenario and year, scale CO₂, SO₂ and black-carbon
emissions (globally or in one region), and generate annual-mean surface air
temperature and precipitation maps.

The model weights and input data are loaded from the Hugging Face model repo
named in the `EMULATOR_BUNDLE_REPO` Space variable.

Source code: https://github.com/kallenordling/CESM2_emulator_from_lumi (`webapp/`).

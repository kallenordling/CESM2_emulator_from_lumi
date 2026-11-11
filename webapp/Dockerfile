# Container for the web app, for CSC Rahti (OpenShift) or any Docker host.
# Build context = the REPO ROOT (the app imports the real model code):
#
#     docker build -f webapp/Dockerfile -t cesm2-emulator .
#     docker run -p 7860:7860 -e EMULATOR_BUNDLE_REPO=nordlin1/cesm2-emulator-bundle \
#                -e HF_TOKEN=... cesm2-emulator
#
# The weights are NOT baked in: the app downloads the bundle from the Hugging
# Face model repo at startup (EMULATOR_BUNDLE_REPO, plus HF_TOKEN when private).
#
# Rahti runs containers as a RANDOM non-root UID in group 0, so every directory
# the app writes to (HF cache, Gradio temp, matplotlib/cartopy config) lives
# under /app/home and is made group-0 writable.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/app/home \
    XDG_DATA_HOME=/app/home/.local/share \
    HF_HOME=/app/home/hf \
    GRADIO_TEMP_DIR=/app/home/gradio \
    MPLCONFIGDIR=/app/home/mpl \
    CARTOPY_DATA_DIR=/app/cartopy \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False \
    OMP_NUM_THREADS=4

WORKDIR /app

COPY webapp/space/requirements.txt /app/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install gradio==6.31.0 -r /app/requirements.txt

# Coastlines are fetched from Natural Earth on first use; do it now so the
# running container needs neither internet for them nor a writable cartopy dir.
RUN python -c "import cartopy, cartopy.feature as f; \
cartopy.config['data_dir'] = '/app/cartopy'; \
[list(f.COASTLINE.with_scale(s).geometries()) for s in ('110m', '50m')]"

COPY webapp/space/app.py webapp/space/emulator.py /app/
COPY models/video_net.py models/rotary_embedding.py /app/models/
COPY custom_diffusers/continuous_ddpm.py /app/custom_diffusers/
RUN touch /app/models/__init__.py /app/custom_diffusers/__init__.py \
 && mkdir -p /app/home/hf /app/home/gradio /app/home/mpl /app/home/.local/share \
 && chgrp -R 0 /app && chmod -R g=u /app

USER 1001
EXPOSE 7860
CMD ["python", "app.py"]

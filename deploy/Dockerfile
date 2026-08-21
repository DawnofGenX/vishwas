# VeriSafe webhook-server image (Roadmap T4.4) — deliberately thin.
#
# The app is stdlib-first: with no heavy wheels installed, every gated stage
# (cv2, yara, pe-lief, docling, ...) simply reports "unavailable" in /health's
# deps list and degrades gracefully (see GAPS_AND_ENABLEMENT.md). Heavy
# optional trees stay HOST-SIDE by design — this host provisions them into
# isolated dirs (PEP-668 box) rather than baking multi-GB layers; operators
# who want them in-image can append the pip installs noted at the bottom.
#
# Model weights are NOT copied: they are volume-mounted read-only at
# /opt/verisafe/models by deploy/docker-compose.yml and wired via
# scripts/provision_weight_env.sh at container start.
FROM python:3.12-slim

# curl: healthcheck (GET /health). bash: already in debian slim, listed for
# clarity since the compose entrypoint and launcher are bash scripts.
#
# numpy: the ONE hard third-party import on the boot path —
# verisafe.capabilities/__init__ eagerly imports deepfake_video.py, which does
# a top-level `import numpy`. torch/einops are NOT installed: the model
# adapters guard those imports (this host itself runs with a broken torch
# shim), and the affected capability stages degrade to "unavailable" in
# /health's deps list. Pinned to the host-validated version.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl bash \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir numpy==2.4.6

WORKDIR /repo
COPY src/ src/
COPY scripts/ scripts/

ENV PYTHONPATH=/repo/src \
    VERISAFE_LOG_LEVEL=INFO

# Container-internal webhook port (compose maps nothing for verisafe; OpenWA
# reaches it over the compose network). Matches deploy/docker-compose.yml.
EXPOSE 2785

# No ENTRYPOINT here: deploy/docker-compose.yml supplies the entrypoint
# (weight provisioning + scripts/run_verisafe.sh launcher). Standalone run:
#   docker run --rm -p 2785:2785 verisafe \
#     bash scripts/run_verisafe.sh webhook --port 2785
#
# Optional heavy deps (huge; prefer the host-side trees documented in
# docs/DEPLOYMENT.md instead):
# RUN pip install --no-cache-dir opencv-python-headless pefile lief yara-x

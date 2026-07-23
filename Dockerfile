# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# The base image has no `git`, so hatch-vcs can't derive the version from
# history — hand it in via the VERSION build arg (the release tag, sans "v").
# Defaults to 0.0.0+local so a locally built image is distinguishable on
# /health from a real release rather than claiming to be a plain 0.0.0.
ARG VERSION=0.0.0+local

COPY . /app

# The version is exported for this RUN only. As an `ENV` it would join the cache
# key of every layer below it (so a release build, whose VERSION always differs,
# could never reuse one) and would persist into the runtime image, where the
# unscoped name would stamp itself onto any setuptools-scm package installed
# later. hatch-vcs does not read the scoped SETUPTOOLS_SCM_PRETEND_VERSION_FOR_*
# form, so the unscoped one is the only option — confining it to this
# instruction is what keeps the blast radius to this install.
# The cache mount keeps wheel downloads across builds even when this layer runs.
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION="${VERSION}" \
    uv pip install --system ".[server,cli]"

FROM python:3.11-slim AS runtime-base

# Only the installed package and its console script — uv (~64 MB of the old
# single-stage image) is a build tool and never runs here.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/argus-forge /usr/local/bin/argus-forge

# Both docker-compose.yaml and the README's `docker run -v ...:/data/out` use
# this path. Without a default the published image starts, answers /health with
# "ok", and then refuses every functional request for want of a root — a
# healthy-looking service the frontend cannot use.
ENV ARGUS_FORGE_EXPORT_ROOT=/data/out

# Demo-safe by default: /config renders but never writes, and every /run route is
# refused with 403. The default (`runtime`) image ships no trainer — no torch, no
# sd-scripts — so a run could only ever fail, and the port is published on
# 0.0.0.0 where a run is real code execution on the host (see runner.py's trust
# note) and an unauthenticated /config would overwrite the curator's
# metadata.jsonl. The default therefore has to be the safe one: `docker run` of
# this image is as locked down as `docker compose up`, and .env.example's
# "Defaults to 1" is true of the image itself rather than only of the compose
# file. The `train` variant below keeps this default on purpose — carrying a
# trainer makes flipping the flag *meaningful*, not automatic; arming /run stays
# a deliberate step on a trusted host. See README "Demo-safe mode".
ENV ARGUS_FORGE_READONLY=1

EXPOSE 8103

CMD ["argus-forge", "serve", "--port", "8103", "--cors"]

# ── train variant ────────────────────────────────────────────────────────────
# The runtime stack `POST /run` needs to actually execute a forged kohya
# train.sh: torch (CUDA wheels), accelerate, and a pinned kohya-ss/sd-scripts
# checkout. Published as ghcr.io/smk762/argus-forge:<version>-train — a distinct
# multi-GB tag, so the default `latest`/`<version>` stays the thin
# config-renderer and nobody pulls the trainer stack by accident (issue #24).
# Build: docker build --target train .
# No CUDA base image: torch's cu124 wheels bundle the CUDA runtime as nvidia-*
# pip packages, so python:3.11-slim serves both variants and only the host
# driver (nvidia-container-toolkit, `--gpus all`) is needed at run time.
FROM runtime-base AS train

# kohya-ss/sd-scripts v0.11.1, pinned by commit so the tag can't move under us.
ARG SD_SCRIPTS_SHA=6721028c79ee85a78b3a06dfd8954dae310a1cce

# opencv-python (an sd-scripts dependency) links libGL/glib at import time.
# gcc + libc6-dev are not build tools here but a RUNTIME dependency: on a GPU
# host, triton (via bitsandbytes, which kohya's default AdamW8bit optimizer
# imports) JIT-compiles a CUDA driver stub at import time and dies without a
# working C toolchain — libc6-dev spelled out because --no-install-recommends
# drops it and gcc alone then fails on <stdlib.h>. CPU-only CI never takes that
# path (bitsandbytes imports fine without a GPU), so the smoke test gates on a
# test compile instead.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Tarball by commit SHA instead of a git clone: pinned by construction, and the
# image never needs git. The forged train.sh cds into $SD_SCRIPTS_DIR to run
# sdxl_train_network.py, so the checkout itself stays in the image.
ADD https://github.com/kohya-ss/sd-scripts/archive/${SD_SCRIPTS_SHA}.tar.gz /tmp/sd-scripts.tar.gz
RUN mkdir -p /opt/sd-scripts \
    && tar -xzf /tmp/sd-scripts.tar.gz -C /opt/sd-scripts --strip-components=1 \
    && rm /tmp/sd-scripts.tar.gz

# torch first, alone: the pin sd-scripts v0.11.1 recommends, from the cu124
# index. Its own layer because it is by far the largest (several GB) and only
# moves on an sd-scripts bump — requirements.txt churn below never re-downloads
# it. uv is mounted, not installed: same build-tool hygiene as the builder stage.
RUN --mount=from=ghcr.io/astral-sh/uv:latest,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system torch==2.6.0 torchvision==0.21.0 \
      --index-url https://download.pytorch.org/whl/cu124

# sd-scripts' own pins (accelerate, diffusers, transformers, ...). The file
# ends in `-e .`, which resolves relative to the cwd — hence the cd; that
# editable install is what makes `import library.train_util` work everywhere.
RUN --mount=from=ghcr.io/astral-sh/uv:latest,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    cd /opt/sd-scripts && uv pip install --system -r requirements.txt

# runner.py spawns train.sh with {**os.environ, **req.env}, so this default
# reaches the script and a caller no longer has to pass SD_SCRIPTS_DIR per run.
ENV SD_SCRIPTS_DIR=/opt/sd-scripts

# ── default ──────────────────────────────────────────────────────────────────
# The thin config-renderer, deliberately LAST: a bare `docker build .` (and the
# compose file's default) must keep producing the demo-safe ~200 MB image,
# never the multi-GB trainer.
FROM runtime-base AS runtime

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

FROM python:3.11-slim

# Only the installed package and its console script — uv (~64 MB of the old
# single-stage image) is a build tool and never runs here.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/argus-forge /usr/local/bin/argus-forge

# Both docker-compose.yaml and the README's `docker run -v ...:/data/out` use
# this path. Without a default the published image starts, answers /health with
# "ok", and then refuses every functional request for want of a root — a
# healthy-looking service the frontend cannot use.
ENV ARGUS_FORGE_EXPORT_ROOT=/data/out

EXPOSE 8103

# Full API by default; a GPU-less/public deployment opts out of live training
# with ARGUS_FORGE_READONLY=1 (or `serve --no-run`). See README "Demo-safe mode".
CMD ["argus-forge", "serve", "--port", "8103", "--cors"]

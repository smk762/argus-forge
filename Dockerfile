# syntax=docker/dockerfile:1
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# The base image has no `git`, so hatch-vcs can't derive the version from
# history — hand it in via the VERSION build arg (the release tag, sans "v").
# Defaults to 0.0.0 for local `docker compose` builds. Matches curator/quarry;
# it also keeps .git out of the build context (see .dockerignore).
ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}

COPY . /app

RUN uv pip install --system --no-cache ".[server,cli]"

EXPOSE 8103

# Full API by default; a GPU-less/public deployment opts out of live training
# with ARGUS_FORGE_READONLY=1 (or `serve --no-run`). See README "Demo-safe mode".
CMD ["argus-forge", "serve", "--port", "8103", "--cors"]

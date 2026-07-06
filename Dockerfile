# syntax=docker/dockerfile:1
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# hatch-vcs reads the version from git history at build time, so the image needs
# a `git` binary — python:3.11-slim ships without one, which makes the install
# below fail with "setuptools-scm was unable to detect version". Own layer so it
# caches across source changes.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# hatch-vcs derives the version from git history, so the build context must
# include .git (keep it out of .dockerignore for this image).
COPY . .

RUN uv pip install --system --no-cache ".[server,cli]"

EXPOSE 8103
CMD ["argus-forge", "serve", "--port", "8103", "--cors"]


# syntax=docker/dockerfile:1
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# hatch-vcs derives the version from git history, so the build context must
# include .git (keep it out of .dockerignore for this image).
COPY . .

RUN uv pip install --system --no-cache ".[server,cli]"

EXPOSE 8103
CMD ["argus-forge", "serve", "--port", "8103", "--cors"]


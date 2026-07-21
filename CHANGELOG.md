# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-21

First tagged release — the version that publishes `ghcr.io/smk762/argus-forge`
(issue #15) and can join the suite demo.

### Security

- **Server endpoints now enforce path containment.** `POST /inspect`,
  `/config` and `/run` resolve the request's `export_dir` under the configured
  export root (`--export-root`, `ARGUS_FORGE_EXPORT_ROOT` /
  `FORGE_EXPORT_PATH`), refusing traversal escapes, and refusing outright when
  the root is unset. Previously any caller could name any absolute path and
  have a `forge/` tree written into it. Request paths are canonically
  root-relative; absolute paths are tolerated only when already inside the root
  (the studio UI echoes back the `export_dir` forge reported). The **CLI stays
  unconstrained** by design. A malformed path (e.g. an embedded NUL) is a 400
  rather than an unhandled 500 traceback.
- **`--cors` no longer means `allow_origins=["*"]` with credentials**, which
  made Starlette reflect any origin back and defeated the allow-list entirely.
  It now allows only the localhost dev frontend; additional origins via
  `--cors-origin` / `FORGE_CORS_ORIGINS`, and a credential-less wildcard via
  `--cors-any`. A literal `"*"` in the allow-list takes the same
  credential-less path instead of silently becoming origin reflection.
- **Unsafe methods are gated on `Origin`** via the suite's shared
  `WriteGuard` (`argus_cortex.server`). CORS is not a write boundary: a
  cross-origin POST with a CORS-safelisted content type gets no preflight, so
  any page a user visits could drive an unauthenticated LAN server into forging
  configs or starting a run. Origin absent (curl, CLI, server-to-server), same
  host:port, or allow-listed passes; anything else is 403. The credential-less
  wildcard grants anonymous reads but never a cross-site write.
- `/health` reports `export_root` alongside `training`.

### Added

- Manifest-aware LoRA config forging for `kohya` / `onetrainer` / `diffusers`,
  with hyperparameters seeded from the same selection-insight heuristics the
  argus-curator `/curate` UI shows.
- `argus-forge` CLI: `inspect`, `config`, `trainers`, `schema`, `serve`, `run`.
- FastAPI micro-server on `:8103` (`server` extra): `/health`, `/trainers`,
  `/inspect`, `/config`, `/run`, `/runs`, `/run/{id}`, `/run/{id}/stream`,
  `/run/{id}/cancel`.
- `run` verb: shells out to the forged `train.sh` and streams NDJSON progress
  (#12), on a job registry so runs outlive the connection (#13).
- **Demo-safe mode** (#16): `serve --no-run` / `ARGUS_FORGE_READONLY=1` serves
  `/config` normally but refuses `POST /run` with 403. `GET /health` reports
  `training: enabled|disabled` so a frontend can disable its train affordance
  up front instead of discovering the refusal by clicking. Required for the
  public demo, whose host is GPU-less by design (argus-halo#7).
- Container image published to GHCR on `v*` tags (#15), plus a
  `docker-compose.yaml` for local runs.
- `argus-cortex[server]>=0.2.0` is a `server`-extra dependency, for the suite's
  shared write-guard helpers.

### Changed

- The image now takes its version from the `VERSION` build-arg
  (`SETUPTOOLS_SCM_PRETEND_VERSION`) instead of reading git history, matching
  argus-curator and argus-quarry. This drops the `git` dependency from the
  image and lets `.git` leave the build context.
- Added `.dockerignore`. The build context previously carried `.venv` (60 MB),
  `.git`, and the pytest/ruff caches straight into the image layer.

### Fixed

- Install `git` in the Docker image so hatch-vcs could detect the version
  (#11) — since superseded by the build-arg above.
- kohya nested subsets, basename-collision caption guard, path remapping, CORS
  docs, emitter parity CI (#6).
- manifest 2.0: accept the new version and resolve rows via `exported_path` (#9).
- Run lifecycle: terminal, bounded, and cancel-safe; a cancel emits a distinct
  `cancelled` event rather than reading as a failure (#14).

[Unreleased]: https://github.com/smk762/argus-forge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/smk762/argus-forge/releases/tag/v0.1.0

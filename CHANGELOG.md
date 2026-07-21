# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-21

First tagged release — the version that publishes `ghcr.io/smk762/argus-forge`
(issue #15) and can join the suite demo.

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

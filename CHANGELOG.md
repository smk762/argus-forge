# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- **BREAKING: `argus_forge.jobs` is gone**; the job registry now lives at
  `argus_forge.server.jobs` (issue #17). `from argus_forge.jobs import Job,
  JobRegistry` — the import a 0.1.0 consumer would have written — raises
  `ModuleNotFoundError`. There is no compatibility shim: the module is an
  in-process implementation detail of the HTTP layer, not part of the wire
  contract, and 0.1.0 shipped days ago. Update the import path.

### Changed

- **The viewer fan-out is built on `anyio` typed memory object streams** instead
  of a hand-rolled `asyncio.Queue[object]` plus an end-of-stream sentinel.
  Closing the producer's half of a viewer's stream ends its `async for`, so the
  sentinel — and the unverifiable `assert isinstance(...)` that `python -O`
  compiled out — are gone. The registry only ever existed to serve the HTTP
  layer, so it now lives beside it and `anyio` stays out of a CLI-only install;
  `anyio>=4` is declared explicitly on the `server` extra rather than relied on
  transitively via starlette. `argus_forge.server` resolves its FastAPI entry
  points lazily so the registry still imports on `anyio` alone, without dragging
  in fastapi/starlette/argus-cortex.
  - The bounded, **drop-oldest** subscriber policy (`MAX_SUBSCRIBER_LAG`) is
    preserved explicitly, since anyio's streams apply backpressure when full and
    one stalled `/stream` reader must never throttle the trainer's stdout. The
    wire format is unchanged: `GET /run/{id}/stream` emits byte-for-byte what it
    did. Note `MAX_SUBSCRIBER_LAG` must now be `>= 1` — `asyncio.Queue(maxsize=0)`
    meant *unbounded*, but `max_buffer_size=0` buffers nothing and would drop
    every event, so the two spellings mean opposite things at `0`.

### Fixed

- A viewer's receive channel is closed on **every** exit from `subscribe()`, not
  only the live one. Replaying an already-finished run, or disconnecting part-way
  through the backlog, left an unclosed `MemoryObjectReceiveStream` — one
  `ResourceWarning` per `GET /run/{id}/stream`, and a hard failure under
  `-W error::ResourceWarning` or `PYTHONDEVMODE=1`. Replaying a finished run no
  longer opens a channel at all, since nothing can publish to it.
- `_finalize` now drops its subscribers as well as closing them. A half-open
  `/stream` reader never resumes its generator, so it never ran `subscribe()`'s
  cleanup and kept a closed channel — plus up to `MAX_SUBSCRIBER_LAG` buffered
  events — pinned to the finished job for as long as the registry retained it.
- `Job.cancel()` no longer returns early when another cancel is in flight; it
  waits for it. `JobRegistry.shutdown()` could otherwise report every trainer
  reaped while a request-side cancel was still inside `runner._terminate`'s
  SIGTERM→SIGKILL grace, and the loop closing under it left the trainer running
  detached.
- `Job.cancel()` no longer swallows a cancellation aimed at *itself*. The blanket
  `suppress(asyncio.CancelledError)` around `await task` caught the caller's own
  cancel too, so a shutdown cut short by uvicorn's `timeout_graceful_shutdown`
  reported success.
- `subscribe()` finds the retained `start` event by identity on the head of the
  buffer rather than `in`, which ran `RunEvent.__eq__` across the whole
  2000-event window — ~2 ms of event-loop time per reconnect, and only on long
  runs, which is exactly when reconnects happen.

## [0.1.0] - 2026-07-21

First tagged release — the version that publishes `ghcr.io/smk762/argus-forge`
(issue #15) and can join the suite demo.

### Security

- **Server endpoints now enforce path containment.** `POST /inspect`,
  `/config` and `/run` require the request's `export_dir` to name a directory
  under the configured export root (`--export-root`,
  `ARGUS_FORGE_EXPORT_ROOT`; `FORGE_EXPORT_PATH` is a legacy alias), refusing
  traversal escapes and refusing outright when the root is unset. Previously
  any caller could name any absolute path and have a `forge/` tree written into
  it. Request paths are canonically root-relative; absolute paths are tolerated
  only when already inside the root (the studio UI echoes back the `export_dir`
  forge reported). The **CLI stays unconstrained** by design.
  - Containment is decided on the fully symlink-resolved path, so a symlink out
    of the root is refused rather than followed — but the path handed to the
    emitters keeps the caller's spelling, since
    `manifest.resolve_export_dir` owns that policy and `path_map` prefixes are
    matched against it. A symlinked root (`/data/out -> /mnt/big/out`) keeps
    rewriting correctly.
  - The check runs on the *absolutised* candidate — exactly the path returned
    and then opened — so "checked" and "used" are always the same file.
    Resolving the raw candidate instead validated a different one, because
    `os.path.abspath` cancels `..` lexically while `resolve()` cancels it
    against the symlink's target: with an in-root `d -> <root>/x/y`, the request
    `d/../../evil` resolved inside the root (so it passed) yet abspath'd to
    `<root>/../evil`, and that escaped path was what every endpoint went on to
    read, write and execute.
  - The export root itself is not a valid `export_dir`: a blank or `"."` field
    would otherwise merge every sibling export into one dataset and write a
    `forge/` tree at the top of the shared volume.
  - A malformed path is a 400, not an unhandled 500 traceback — including a
    symlink loop, which `pathlib` reports as `RuntimeError` rather than an
    `OSError`.
- **Manifest `abs_path` caption sources are fenced too.** `collect_captions`
  copies sidecars from beside the *source* images, and `abs_path` is manifest-
  supplied — as untrusted on a server as the request itself. Without the fence
  any readable `.txt` on the host was copied into the shared dataset volume and,
  for trainers that inline captions, echoed back in the `/config` response.
  Refused sources are reported as a warning. The CLI stays unconstrained.
- **`--cors` no longer means `allow_origins=["*"]` with credentials**, which
  made Starlette reflect any origin back and defeated the allow-list entirely.
  It now allows the localhost dev frontend; `--cors-origin` /
  `FORGE_CORS_ORIGINS` **add** further origins rather than replacing it, and
  `--cors-any` is a credential-less wildcard. A literal `"*"` in the allow-list
  takes the same credential-less path instead of silently becoming origin
  reflection. Allow-list entries are normalised, so a trailing slash or stray
  whitespace no longer produces a list that can never match.
- **Unsafe methods are gated on `Origin`** via the suite's shared
  `WriteGuard` (`argus_cortex.server`). CORS is not a write boundary: a
  cross-origin POST with a CORS-safelisted content type gets no preflight, so
  any page a user visits could drive an unauthenticated LAN server into forging
  configs or starting a run. Origin absent (curl, CLI, server-to-server), same
  host:port, or allow-listed passes; anything else is 403. The wildcard grants
  anonymous reads to everyone while origins the operator named explicitly keep
  their write grant — the pairing a public demo needs, since every forge
  endpoint that does anything is a POST.
- **`PATH` joins the refused `/run` env keys.** The command is the bare name
  `bash`, so a caller-supplied `PATH` pointing at a directory holding its own
  `./bash` replaced the forged script entirely — the deny-list's stated purpose
  ("refuses the env vars that would let a request redirect *what* runs") did not
  hold without it.
- `/health` reports `export_root` alongside `training`, and answers even when
  the configured root cannot be resolved, so a liveness probe never 500s on a
  misconfigured or hung volume. It reports the root only when it is actually
  usable: `resolve()` is non-strict, so an unmounted volume (the image run
  without `-v`, the commonest misconfiguration) used to answer "ok" with a root
  while every request 400'd on "not a directory". The value is the
  un-dereferenced spelling `/inspect` also returns, since `path_map` prefixes
  are matched against it.

### Added

- Manifest-aware LoRA config forging for `kohya` / `onetrainer` / `diffusers`,
  with hyperparameters seeded from the same selection-insight heuristics the
  argus-curator `/curate` UI shows.
- `argus-forge` CLI: `inspect`, `config`, `trainers`, `schema`, `serve`, `run`.
- FastAPI micro-server on `:8103` (`server` extra): `/health`, `/trainers`,
  `/inspect`, `/config`, `/run`, `/runs`, `/run/{id}`, `/run/{id}/stream`,
  `/run/{id}/cancel`.
- `run` verb: shells out to the forged `train.sh` and streams NDJSON progress
  (#12), on a job registry so runs outlive the connection (#14).
- **Demo-safe mode** (#16): `serve --no-run` / `ARGUS_FORGE_READONLY=1` renders
  configs but never trains and never writes — every `/run` route is refused
  with 403 (as middleware, so the refusal precedes body validation and covers
  routes added later), and `POST /config` is forced to `dry_run`. `GET /health`
  reports `training: enabled|disabled` so a frontend can disable its train
  affordance up front instead of discovering the refusal by clicking. Required
  for the public demo, whose host is GPU-less by design (argus-halo#7).
- Container image published to GHCR on `v*` tags (#15), plus a
  `docker-compose.yaml` for local runs. **The image itself** defaults to
  demo-safe mode (`ENV ARGUS_FORGE_READONLY=1`), not just the compose stack: it
  ships no trainer, and it publishes the port on 0.0.0.0 where a run is real
  code execution and an unauthenticated `/config` would overwrite the curator's
  `metadata.jsonl`, so a bare `docker run` has to be as locked down as
  `docker compose up`.
- `argus-cortex[server]>=0.2.0` is a `server`-extra dependency, for the suite's
  shared write-guard and env-flag helpers.
- `caption_source_root`: the containment root for manifest `abs_path` caption
  sources. A keyword argument to `forge_config`, deliberately **not** a field on
  `ForgeRequest` — a security boundary a request could name is one it could
  widen to `/`. The server passes its export root; the CLI passes `None`
  (unconstrained — it is the operator's own shell).

### Changed

- The image now takes its version from the `VERSION` build-arg
  (`SETUPTOOLS_SCM_PRETEND_VERSION`) instead of reading git history, matching
  argus-curator and argus-quarry. This drops the `git` dependency from the
  image and lets `.git` leave the build context. The variable is scoped to the
  install step rather than being an `ENV`, so it neither pollutes the cache key
  of later layers nor persists into the runtime container.
- The image is multi-stage and defaults `ARGUS_FORGE_EXPORT_ROOT=/data/out`:
  `uv` (~64 MB) and the source tree no longer ship, and the published image
  works from `docker run -v ...:/data/out` alone instead of starting healthy
  and refusing every request. 534 MB → **211 MB**.
- `ARGUS_FORGE_READONLY` is read by `create_app` rather than only by the `serve`
  command, so every ASGI entry point honours it. Being a *protection* flag it
  fails **safe**: an unrecognised value (`=y`, `=enabled`) warns and keeps the
  guard on, where `env_flag`'s enable-a-feature default of "off" would have
  silently enabled training and `/config` writes on a host that is
  unauthenticated and public by assumption. Still never fatal — under compose's
  `restart: unless-stopped` a hard exit is a crash loop. Only an explicit
  `0`/`false`/`no`/`off` allows runs.
- Added `.dockerignore`. The build context previously carried `.venv` (60 MB),
  `.git`, and the pytest/ruff caches straight into the image layer. Patterns
  use `**/` where needed — `.dockerignore` has no implicit recursion — and the
  compose dataset volume (`./out`) is excluded.

### Fixed

- Install `git` in the Docker image so hatch-vcs could detect the version
  (#11) — since superseded by the build-arg above.
- kohya nested subsets, basename-collision caption guard, path remapping, CORS
  docs, emitter parity CI (#6).
- manifest 2.0: accept the new version and resolve rows via `exported_path` (#9).
- Run lifecycle: terminal, bounded, and cancel-safe; a cancel emits a distinct
  `cancelled` event rather than reading as a failure (#14).
- Containment and `/health` no longer run blocking `stat`/`realpath` syscalls on
  the asyncio event loop: the root is resolved once at startup and per-request
  checks happen inside the existing `asyncio.to_thread` hop, so a slow or hung
  export volume cannot stall in-flight `/run/{id}/stream` progress streams.
- `serve`'s startup warnings resolve the same env fallbacks `create_app` does,
  so `FORGE_CORS_ORIGINS` no longer produces a "CORS is disabled" warning about
  a server that has CORS enabled.
- A manifest row whose `abs_path` is empty (or `/`) no longer 500s `POST
  /config`. Only `exported_path` is validated, so `abs_path` can name no file at
  all, and `Path("").with_suffix(...)` raised straight through the catch-all. A
  row with no readable source path simply has no sidecar to collect.
- Demo-safe `POST /config` now says in `warnings` that it was forced to a dry
  run, so a caller that asked for a real write learns it did not happen from the
  body rather than by noticing every file's `path` is null.
- `--cors-origin` / `FORGE_CORS_ORIGINS` no longer drop the localhost:3000
  defaults when a bare `--cors` was not also passed. They *add* to the dev
  frontend (as the README says) rather than replacing it, for the write-guard
  trust list as well as the CORS allow-list — naming a production origin must
  not cost you the studio frontend you were already developing against.
- The release workflow derives the image version from the tag itself rather
  than `docker/metadata-action`'s `version` output, which falls back to the
  literal string `latest` for a non-semver `v*` tag; and `latest` is no longer
  repointed by a prerelease tag.

[Unreleased]: https://github.com/smk762/argus-forge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/smk762/argus-forge/releases/tag/v0.1.0

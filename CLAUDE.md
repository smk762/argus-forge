# CLAUDE.md — argus-forge

Guidance for AI agents working in this repo. Human-facing usage lives in [README.md](README.md); this file is the orientation an agent needs to change code safely.

## What this is

The **training bridge** in the Argus suite: it turns a curated dataset export (images + `manifest.jsonl` + `.txt` caption sidecars) into **runnable** LoRA training configs for kohya_ss, OneTrainer, and diffusers.

```
argus-quarry -> argus-curator -> argus-lens -> argus-forge -> your trainer -> argus-proof
  acquire        curate/export    caption       configs        LoRA           validate
```

The hard part is not rendering TOML — it is being faithful to what the curator exported and what a trainer will actually read (caption pairing, kohya's non-recursive globs, container↔host path translation, base-checkpoint resolution). Read [README.md](README.md) for the *why* behind each of those; it is unusually detailed and is the source of truth for behaviour.

## Argus suite
<!-- argus-suite: keep this block; argus-ci checks for it -->
This repo is part of the Argus suite. The suite map below lists every sibling and what it owns; check it before
building something a sibling may already do. **On every release, update this repo's row and capabilities in
argus-cortex's SUITE.md** (a PR to argus-cortex in the same release).

@../argus-core/SUITE.md

Not checked out? Read https://github.com/smk762/argus-cortex/blob/main/SUITE.md

## Layout

`src/argus_forge/`:

- `models.py` — Pydantic v2 wire types (`ManifestRow`, `ForgeRequest`/`ForgeResult`, `RunRequest`/`RunEvent`/`RunState`, `TrainingParams`). **This is the API contract**; `wire_schema()` here backs `argus-forge schema` and the committed `schema/forge-wire.schema.json`.
- `manifest.py` — read/validate the manifest (major-version-aware), find images, resolve exported locations, detect basename collisions.
- `heuristics.py` — dataset-size × target-category → suggested `TrainingParams` (repeats/epochs/LR/dim/alpha). Must stay in **parity with argus-curator's `/curate` UI**; `tests/test_kohya_parity.py` and `_js_round` guard that.
- `core.py` — `forge_config()`, the orchestrator: path-map resolution + containment, caption collection, calls the emitters.
- `emitters/` — one module per trainer (`kohya`, `onetrainer`, `diffusers`) over `base.py`. Add a trainer here and register it in `emitters/__init__.py`.
- `runner.py` — shells out to the forged `train.sh`, streams NDJSON `RunEvent`s, manages the process group (SIGTERM→SIGKILL).
- `cli.py` — Typer app (`config`, `run`, `inspect`, `trainers`, `schema`, `serve`).
- `server/` — FastAPI micro-server on **:8103** (`app.py` routes + CORS/Origin guard + read-only middleware; `jobs.py` the background-run registry & viewer fan-out). Optional `[server]` extra.

## Commands

```bash
make install   # uv venv + editable install with [dev,server,cli]
make test      # pytest --tb=short -q
make lint      # ruff check + format --check (pinned ruff 0.15.16)
make format    # ruff format + --fix
```

Run a single test: `uv run --no-sync pytest tests/test_core.py::test_name -q`.

## Conventions & gotchas

- **Versioning is git-tag-derived** (`hatch-vcs`). Never hand-edit a version; `src/argus_forge/_version.py` is generated (gitignored). Tag `vX.Y.Z` to release.
- **The wire schema is checked in CI.** If you touch `models.py` types that appear in `wire_schema()`, run `argus-forge schema --check` — CI fails if `schema/forge-wire.schema.json` drifts. Regenerate with `argus-forge schema > schema/forge-wire.schema.json`.
- **Two safety boundaries, both intentional — don't loosen them casually:**
  - The server's `--export-root` fence: a request's `export_dir` must resolve to a path *under* the root (symlinks resolved). The **CLI is deliberately unconstrained** — it's your own shell.
  - Demo-safe / read-only mode (`ARGUS_FORGE_READONLY`, `--no-run`): `/run` refuses with 403 (via middleware, before body validation) and `/config` is forced to `dry_run`. It fails *safe* — an unparseable flag keeps the guard on.
- **`POST /run` is real code execution on the host** (runs a forged `train.sh`). There is no auth; treat reaching the port as shell access. Keep new `/run`-family routes covered by the read-only middleware rather than a per-route guard.
- **Captions are paired carefully:** lens writes `.txt` next to *source* images; forge copies them next to the *exported* copies. On a basename collision it leaves the file uncaptioned rather than mispairing — preserve that behaviour.
- **kohya emits one subset per image directory** (its glob doesn't recurse). A structure-preserving export would otherwise train zero images.
- Shared suite code (taxonomy, wire-schema discipline, the server write-guard/env-flag helpers) lives in **argus-cortex** (`argus-cortex[server]`), not here — reuse it rather than re-implementing.
- `structlog` for logging; Pydantic v2 everywhere; async server + runner (`pytest asyncio_mode = auto`). Ruff line-length 120.

## Branches

`dev` is the default branch and the base for every PR. Cut feature branches from `origin/dev` and bring them up to date by merging `origin/dev`, never `main`. Never open a PR against `main` or push work to it directly: `main` must never be ahead of `dev`. A release fast-forwards `main` to a `dev` commit that passed CI (`git push origin origin/dev:main`), then tags it. If `main` is protected and needs a PR, merge the release PR, then immediately merge `main` back into `dev` so the merge commit is not left on `main` alone.

## CI / release

CI runs via the shared [`argus-ci`](https://github.com/smk762/argus-ci) reusable workflow plus `argus-forge schema --check`. Release publishes to PyPI (OIDC trusted publishing) + GHCR on `v*` tags. This repo is scaffolded from [`argus-pkg-template`](https://github.com/smk762/argus-pkg-template) — run `copier update` to pull tooling changes.

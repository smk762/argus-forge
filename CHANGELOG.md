# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **argus-cortex 0.16** (`>=0.16.1,<0.17`), part of smk762/argus-cortex#212 and #209.
  - `GET /health` is argus-cortex's `ServiceHealth`: `service_version`, `service`, `version`, `status`,
    `started_at`, `wire`, `capabilities` (`["training"]` when runs are allowed, else empty), `checks`, `read_only`
    (demo-safe mode), `build` and `peer`, plus forge's own `export_root` and `training`, unchanged. It is still
    always 200 with `status: "ok"` (forge has no loading phase and no dependency checks), and
    `GET /health?checks=0` answers the same. The route is no longer listed in `/openapi.json` or shown on `/docs`.
  - The `ARGUS_GPU_*` settings (the broker URL and token, the coordinator, the lease path and timeout, the device)
    are also read from the suite settings file (`~/.config/argus/suite.env`) when `argus-forge serve` runs
    directly on a host and the environment leaves them unset or empty. A host whose suite file names a broker now
    leases through it without setting anything for forge; `ARGUS_GPU_COORDINATOR=none` opts out. Forge never copies the
    file into its environment, so the trainer still inherits none of the suite's tokens, and nothing else from the
    file either (`test_suite_settings`).
  - `ARGUS_FORGE_READONLY` is read from the suite file too when the environment does not set it, with the same
    fail-safe rule: a value forge cannot parse keeps demo-safe mode on. The file can turn the mode on, never off
    against the environment.
  - `.env.example` no longer lists `ARGUS_GPU_BROKER_FALLBACK`. The variable still works, and compose still
    defaults it to `lease`.

### Added

- Compose: the `argus-forge` service carries the discovery labels `argus.peer: forge`,
  `argus.service: argus-forge`, `argus.health: /health` and `argus.port: "8103"`, and `ARGUS_PEER=forge`, which
  `/health` reports as `peer`.
  Forge's container is not given the suite settings file, and no compose file here mounts it: the trainer shares
  the container, and the file holds the suite's tokens. The broker URL and token (and `HF_TOKEN` where needed) go
  in forge's own environment.

- **`argus-forge serve` registers with the GPU broker at startup** (`gpu.keep_registered`, in the app's lifespan,
  with the same cached coordinator runs lease from), re-registers every five minutes, and signs off after cancelling
  in-flight runs at shutdown, so argus-iris knows the tenant before its first run. A down broker never blocks startup;
  an unusable `ARGUS_GPU_*` environment is logged and still ends the first run in an `error` event. The CLI registers
  with its run's lease.
- **One active run at a time on `POST /run`.** While a run is active, which means status `running`, including a run
  still queued for its argus-iris `train` lease, another `POST /run` is a 409 and creates nothing, so two clients
  (two studio tabs) can no longer start two runs. The body is the new wire model `RunActiveError`:
  `{"detail": "a run is already active (run <id>, queued or running); wait for it to finish or cancel it",
  "code": "run_active", "active_run": {"run_id", "status"}}`. `detail` stays a string; `code` tells it apart from
  the `Idempotency-Key` 409s (no `code`). The check and the creation are one critical section across all
  requests, so concurrent requests with different keys or none start exactly one run. A replay of the active
  run's own key still returns it (202, `Idempotent-Replayed: true`); a refused request records no key. A
  `dry_run` is exempt. A run in flight at a crash is already restored `failed`, so nothing stale blocks after
  a restart. `ARGUS_FORGE_MAX_ACTIVE_RUNS` (default 1) raises the limit for a multi-GPU broker. Active is
  defined once (`models.ACTIVE_RUN_STATUSES` / `run_is_active`). `schema/forge-wire.schema.json` gains
  `ActiveRun` and `RunActiveError`; `schema/forge-parity.json` is unchanged.

- **`Idempotency-Key` on `POST /run`.** A retried `POST /run` can no longer start a second training run (a duplicate
  holds the GPU for hours). With the optional header (1-200 printable ASCII characters; anything else is a 422 that
  does not echo it), the same key and the same request body (fingerprinted as the validated `RunRequest`, so JSON
  spelling doesn't matter) return the run the first request started, in whatever state it is now, with
  `Idempotent-Replayed: true`; the same key with a different body is a 409, and concurrent requests with one new
  key start exactly one run. Keys are global to the server, kept beside the run records in
  `<state_dir>/idempotency/` (only the key's sha256 is written; in memory without a state dir), and pruned lazily
  after `ARGUS_FORGE_IDEMPOTENCY_RETENTION_S` (default 86400), never while their run is active. Without the header
  `POST /run` is unchanged. The wire schema and `schema/forge-parity.json` are unchanged.

- **Curator speech exports (#58).** `argus_forge.speech.read_speech_export(export_dir, held_out_dir=None)` reads
  curator's `speech_manifest.jsonl` (`SpeechManifestRow`, `speech_manifest_version` 1.x; another major, or a file
  mixing majors, is refused naming the line) into a `SpeechDataset`: train clips (`split` `train` or null) paired
  with their `<stem>.txt` transcripts from `exported_path` / `transcript_path`, held-out reference clips, speaker,
  accent, language and gender per clip, and corpus / licence / provenance per source. A held-out speaker (by
  `speaker_id`, `speaker_label` or `speaker_cluster`) found in training is refused. `SpeechEmitter` is the
  interface the XTTS-v2 emitter (#56) implements; none ships yet. `argus-forge inspect` summarises a speech export
  (`--held-out` for the other part). The image path refuses a speech export, and an image `manifest.jsonl` row with
  a speech profile, instead of reading either as an empty or image set.

- **Pose-dominance warning (curator manifest 2.7).** `ManifestRow` reads curator's optional `pose_labels`
  (`{stance, arms, facing}`, values kept as plain strings so `unknown` and new values parse) and `pose_cluster`
  (argus-puppet's scan-level body-pose cluster); both are null on older rows. When at least 10 rows of the
  images on disk have a cluster and one cluster holds 75% or more of them (puppet's and proof's collapse cut-off,
  uncalibrated on real sets), `ForgeResult.warnings` (and the CLI) gets one line: "N of M posed training images
  share one body pose (standing, arms down, facing front); outputs are likely to collapse onto it — rebalance
  the selection in curator", plus how many images had no measured pose when that is a quarter or more.
  Warning only: repeats, heuristics, emitters and `schema/forge-parity.json` are unchanged. Forge counts
  curator's values and never computes pose metrics or calls puppet (new must-never row and
  `test_rules::test_forge_does_not_compute_pose_metrics_or_call_puppet`). New wire model `PoseLabels`.
- **A run reports its LoRA (#52).** When a run's trainer exits 0, forge finds the LoRA where the forged config
  says it goes (kohya's `output_dir` + `output_name`, diffusers' `--output_dir`), hashes it, and puts it on
  `RunState.lora` (`path`, `export_path`, `sha256`, `size_bytes`) before the run turns `succeeded`.
  `GET /run/{id}/lora` returns it alone: 409 while the run hasn't succeeded, 404 when it left no file. New wire
  model `RunArtifact`.
- **Run records survive a restart (#52).** With `--state-dir` / `ARGUS_FORGE_STATE_DIR`, each run's `RunState`
  is saved as `<dir>/runs/<run_id>.json` when it starts and ends, and reloaded at startup (the newest 64
  finished runs). A restored run's stream replays `start` and its terminal event; one still `running` when
  forge went away comes back `failed`. The image sets `/var/lib/argus-forge` and the compose file mounts the
  `forge-state` volume there. Unset, runs are in memory only, as before.

### Changed

- **argus-cortex 0.14 → 0.15**, pinned `>=0.15,<0.16` (base, `server`, `gpu`, `dev`). `ARGUS_GPU_BROKER_URL` (and
  `ARGUS_GPU_BROKER_TOKEN`) is all the broker needs: cortex selects it from the URL alone and finds the image's
  `/app/gpu-tenant.yaml` itself. Compose no longer defaults `ARGUS_GPU_COORDINATOR` to `none` (it still passes an
  explicit `none` or `lease` through) and drops `ARGUS_GPU_TENANT` and `ARGUS_GPU_TENANT_MANIFEST`; `.env.example`
  leaves `ARGUS_GPU_COORDINATOR` commented out. **Behaviour change:** a host that set `ARGUS_GPU_BROKER_URL` without
  `ARGUS_GPU_COORDINATOR` now leases from the broker; set `ARGUS_GPU_COORDINATOR=none` to keep starting runs at
  once. A `.env` copied from the old `.env.example` still says `ARGUS_GPU_COORDINATOR=none`, which wins: delete that
  line to use the broker by its URL alone.
- **argus-cortex `>=0.13,<0.14`** (was `>=0.6,<0.10`): 0.12 added `TargetProfile.modality` and the speech
  categories, 0.13 the `consent:<release_sha256>` subject ref curator writes into consented speech rows.
  `schema/forge-wire.schema.json` is regenerated: `TargetProfile` gains `modality` and `target_category` admits
  the speech categories (from cortex; argus-studio's generated types follow).
- **Demo-safe mode refuses a writing `/config` (#52).** With training disabled, a `POST /config` whose
  `dry_run` is false or omitted is now a 403 (as `/run` is) instead of a 200 dry run with a warning, so no
  caller mistakes a preview for a write. Send `"dry_run": true` to render. argus-studio already does.

### Fixed

- A diffusers run whose hand-edited `train.sh` passes a relative `--output_dir` now reports its LoRA, read from the
  run's own directory. It is still refused when the script changes directory or the value needs shell expansion.
- `--output_dir PATH` (space-separated) in a diffusers `train.sh` is read like `--output_dir=PATH`.
- An absolute `--output_dir` in a diffusers `train.sh` has its `..` collapsed like a relative one, so a path that
  leaves the export through it no longer gets an `export_path` as if it were inside. An absolute-looking value that
  needs shell expansion (`/runs/$RUN_ID`) is refused, like a relative one, instead of being opened literally.
- A run record is fsynced before it is renamed into place, and its directory after, so a crash can no longer leave
  an empty record where the old one was.
- A run record that cannot be saved no longer leaves `/run/{id}/stream` viewers hanging, and one holding a path that
  is not valid UTF-8 is logged instead of raising.

## [0.3.0] - 2026-11-19

### Added

- `schema/forge-parity.json` (#45): forge's answers to `suggest_training_params` and `dataset_size_status` for every
  category at each size-guide boundary, plus the kohya `dataset.toml` / `config.toml` it renders for demo-mode
  inputs, parsed. argus-studio pins it like a wire schema and checks its hand-written twins (`types.ts`,
  `forgeDemo.ts`) against it. `tests/test_studio_parity.py` fails when it goes stale; regenerate with
  `UPDATE_PARITY=1`.
- **GPU lease for training, through argus-iris (#41).** Every run (`POST /run`, `argus-forge run`) holds one
  lease from argus-cortex's GPU coordinator for the whole of `train.sh`. It is an exclusive whole-card lease:
  kind `train`, `background` priority, not preemptible, pinned to morpheus, 22000 MiB (`FORGE_GPU_VRAM_MB`).
  `ARGUS_GPU_COORDINATOR=broker` with `ARGUS_GPU_BROKER_URL`, `ARGUS_GPU_BROKER_TOKEN` and
  `ARGUS_GPU_TENANT_MANIFEST` asks iris; when iris is unreachable or has no lease routes yet (it is
  observe-only today) the lease comes from `ARGUS_GPU_BROKER_FALLBACK` (`lease` by default). The default
  is `none`, which leaves runs as they were. A lease that is not granted ends the run in an `error` event
  before anything runs. The lease is released before the terminal event, reporting the run's outcome (`ok`,
  `failed`, `oom` or `cancelled`) and the peak VRAM sampled through NVML.
- `gpu-tenant.yaml`, the tenant manifest iris registers. It is validated by cortex's `TenantManifest`, and
  the image ships it at `/app/gpu-tenant.yaml`.
- `gpu` extra (`argus-cortex[gpu,remote,yaml]`), installed in both images.
- `FORGE_TRUSTED_HOSTS` (and `create_app(trusted_hosts=)`): the host names forge is reached by.

### Changed

- argus-cortex `>=0.6,<0.10` (was `>=0.4.7`). Since cortex 0.6, a write from a page on the server's own
  host:port is refused when that host is a bare DNS name (DNS rebinding), unless the name is in
  `FORGE_TRUSTED_HOSTS`. IP literals and `localhost` are unaffected.
- docker-compose mounts the host-wide GPU lock dir (`ARGUS_GPU_LOCK_DIR`, default `/tmp/argus-gpu`) and passes the
  `ARGUS_GPU_*` variables through.

### Fixed

- A GPU lease that fails for a reason other than a timeout or refusal (an unknown `ARGUS_GPU_COORDINATOR`, an
  unwritable lock dir, `broker` without the `gpu` extra) ends the run in an `error` event. It used to escape the
  run with no terminal event: a traceback from `argus-forge run`.
- A path-map target with a trailing slash (`/mnt/x/`) no longer renders doubled slashes (`/mnt/x//a`).
- README: the GPU host in the iris and `FORGE_TRUSTED_HOSTS` examples is the `<GPU_HOST>` placeholder, so the
  argus-handbook docs sync passes its opsec gate again.
- A run's `env` is an allowlist now (`SD_SCRIPTS_DIR`, `DIFFUSERS_SCRIPT`, `CUDA_VISIBLE_DEVICES`, `HF_HOME`,
  `HF_HUB_OFFLINE`). The old blocklist missed `LD_LIBRARY_PATH`, `PYTHONPATH`, `PYTHONSTARTUP` and other hooks that
  change what code runs. Any other key is refused with 400. The trainer no longer inherits `ARGUS_GPU_BROKER_TOKEN`,
  `ARGUS_ADMIN_TOKEN` or `AI_AUDIT_API_TOKEN` from the server's environment.
- `dataset_size_status` matches argus-studio again: a count inside the good range but outside the ideal band is
  "workable …; the sweet spot is ~X", not "in the sweet spot". Studio had changed its wording, so the /forge page
  (forge's `size_hint`) and the /curate export panel showed different hints for the same set.
- README: the symlinked-root example uses a placeholder host path (`/mnt/<NAS>/out`)

## [0.2.0] - 2026-11-13

The first release since 0.1.1. On the image side it adds the `-train` variant
and the image rebuild that a new argus-cortex release triggers. In the package,
the target taxonomy now comes from argus-cortex, which adds the `concept`
category, and every tag now gets a GitHub Release.

### Added

- **`concept` target category** (style / visual-concept LoRAs, issue #28). Its
  training defaults are the same as `setting`: about 2000 total steps,
  `network_dim` 32 and `network_alpha` 16. Its dataset-size guide is ~30–60
  images (light below 20, large above 100). A style needs the most variety,
  otherwise the LoRA learns the subjects instead of the look. argus-studio's
  `types.ts` twin of these tables does not have `concept` yet and needs the same
  values added.

- **A GitHub Release for every tag** (issue #26). `release.yml` creates it once
  the wheel is on PyPI. The notes are that version's `CHANGELOG.md` section (the
  run fails if it is missing), the wheel and sdist are attached, and
  `-`/`rc`/`a<n>`/`b<n>` tags are marked as pre-releases. Re-running the workflow
  updates the existing Release instead of failing.

- **A new argus-cortex rebuilds the published image.** `release.yml` also runs on
  the `cortex-released` dispatch argus-cortex's release sends: only the image job runs,
  rebuilding the newest final `v*` tag without cache and republishing its tags, so the
  image picks up the cortex its pyproject range now resolves to.

- **A `-train` image variant, so `POST /run` can go live on GPU hosts** (issue
  #24). `ghcr.io/smk762/argus-forge:<version>-train` (also `:latest-train`)
  layers torch 2.6 (cu124 wheels), accelerate, and a pinned kohya-ss/sd-scripts
  v0.11.1 checkout (`SD_SCRIPTS_DIR=/opt/sd-scripts` preset) on top of the
  existing image; `docker build --target train` / `FORGE_BUILD_TARGET=train`
  build it locally. Opt-in everywhere: the default `latest`/`<version>` tags
  stay the thin demo-safe config-renderer, and the train image *keeps*
  `ARGUS_FORGE_READONLY=1` — enabling live runs remains a deliberate env flip
  on a trusted host, since reaching an armed `/run` port is equivalent to shell
  access. Release-gated by a CPU-only smoke test
  (`scripts/smoke-train-image.sh`: serve + trainer-stack imports + the
  `prepare_run` validation path); CI has no GPU, so a `-train` tag is
  explicitly not e2e-trained.

### Changed

- **`TargetProfile`, `TargetCategory` and `TargetStyle` come from
  `argus_cortex.taxonomy`** (issue #28), replacing forge's own copy. They are
  still importable from `argus_forge.models`. `argus-cortex>=0.4.7` is now a core
  dependency (it pulls only pydantic), and the `[server]` extra's floor is raised
  to match. The wire schema's `TargetProfile` gains `concept` and cortex's
  docstring. `--category` help on `config` and `inspect` now lists the values
  from the taxonomy.

### Documentation

- README (issue #33): `serve` binds `0.0.0.0:8103` by default with `/run`
  enabled, so the README now says to use `--host 127.0.0.1` or `--no-run`
  unless the network is trusted. It also documents `argus-forge schema`, and
  demo-safe mode now lists exactly what it refuses: unsafe methods on `/run` and
  `/run/...`, while `GET /runs` and `GET /run/{id}` still answer.

### Fixed

- **The `cortex-released` image rebuild no longer fails on a tag older than its
  workflow.** The dispatch runs `release.yml` from the default branch but builds
  the newest release tag's Dockerfile, and v0.1.1's Dockerfile predates the named
  `runtime` and `train` stages, so both image jobs failed with `target stage
  "runtime"/"train" could not be found` (seen on `main` and on `dev`). The
  `release-tag` job now reads the tag's Dockerfile: without a `runtime` stage the
  thin image is built from the default (last) stage, which is the same image the
  old Dockerfile shipped; without a `train` stage the train job skips with a
  notice, since that tag never had a `-train` image.

### Security

- **Images no longer trust the compromised emSign ECC Root CA - G3** (the
  eMudhra CA incident). The build uses uv 0.13.5, which drops that root, and
  requires `certifi>=2026.10.19` in both images. The base image's pip, which is
  copied into the runtime image, is replaced with a hash-pinned pip 26.2.2.
  The system CA config marks the emSign G3 and Comodo AAA roots as distrusted.
  Debian's current `ca-certificates` doesn't ship either one, so that step
  only matters if a later package adds them back.

## [0.1.1] - 2026-07-22

Tagged without a changelog heading at the time; these notes were under
`[Unreleased]` until the 0.2.0 notes were written.

### Removed

- **BREAKING: `argus_forge.jobs` is gone**; the job registry now lives at
  `argus_forge.server.jobs` (issue #17). `from argus_forge.jobs import Job,
  JobRegistry` — the import a 0.1.0 consumer would have written — raises
  `ModuleNotFoundError`. There is no compatibility shim: the module is an
  in-process implementation detail of the HTTP layer, not part of the wire
  contract, and 0.1.0 shipped days ago. Update the import path.

### Changed

- **A `/stream` viewer is now a cursor into the job's shared event buffer**
  rather than a queue of its own (issue #21). The registry retained the same
  2000-event window *twice* — once in `Job._events`, once per viewer — so N
  stalled readers pinned N copies of it (~1 MB each) on top of the shared tail.
  Total retention is now `MAX_BUFFERED_EVENTS` objects regardless of how many
  viewers are attached; a viewer costs a sequence number and an `asyncio.Event`.
  - Drop-oldest for a lagging viewer is now a *consequence* of the one shared
    bound: a cursor that falls behind the window resumes at the oldest retained
    event. `MAX_SUBSCRIBER_LAG` is gone, and with it the second eviction policy
    and the backlog/live split in `subscribe()` — a cursor cannot straddle a
    seam, so "no event dropped or duplicated across it" is structural rather
    than an invariant a comment had to assert. One behaviour did narrow: a
    viewer *stalled part-way through its replay* used to be immune to eviction
    (the replay was a private list copy) and had a second 2000-event channel
    behind it, so its worst-case tolerance was ~4000 events; it is now the one
    2000-event window. A viewer that has caught up is unaffected.
  - `_publish` still never waits on a viewer, so one stalled `/stream` reader
    cannot apply backpressure to the trainer's stdout. The wire format is
    unchanged: `GET /run/{id}/stream` emits byte-for-byte what it did.
  - The `start` event is still retained out-of-band, but keyed on the sequence
    number it occupied rather than on it being the head of the buffer, and it is
    injected the moment a cursor is found to be past it. So a viewer whose
    cursor is *clamped over* `start` — not only one that joined after it rolled
    off — still learns command/cwd, and a viewer can never be handed it twice.
  - The registry only ever existed to serve the HTTP layer, so it now lives
    beside it at `argus_forge.server.jobs`. The fan-out is plain stdlib asyncio
    again — the `anyio` dependency it briefly took on (and the explicit
    `anyio>=4` on the `server` extra) is gone, as is the end-of-stream sentinel
    and its unverifiable `assert isinstance(...)` that `python -O` compiled out.
    `argus_forge.server` resolves its FastAPI entry points lazily, so the
    registry imports without dragging in fastapi/starlette/argus-cortex.

### Fixed

- `subscribe()` no longer registers a cursor on a run that has **already
  finished**. `_finalize` is idempotent, so the sweep that drops half-open
  `/stream` readers has run for the last time by then and could never drop it —
  a client that reconnects to a finished run and stalls mid-replay would pin its
  cursor for as long as the registry retained the job (`MAX_FINISHED_JOBS`).
  Nothing can publish to such a viewer anyway, so it registers nothing.
- `MAX_BUFFERED_EVENTS` is validated as `>= 1` at import, replacing the guard
  that went with `MAX_SUBSCRIBER_LAG`. A zero-length window is a worse mis-tune
  under a cursor than under a queue: the deque keeps nothing while `_published`
  runs away from it, so a clamped cursor indexes off the end and `/stream`
  raises `IndexError` mid-response instead of merely returning an empty body.
- A TestPyPI failure no longer withholds the real PyPI release. `publish-pypi`
  still runs after the TestPyPI rehearsal, but it is gated only on the build and
  install checks. A TestPyPI fault still fails the run. The `dev` extra adds
  `httpx2`, which starlette's `TestClient` needs.
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

[Unreleased]: https://github.com/smk762/argus-forge/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/smk762/argus-forge/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/smk762/argus-forge/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/smk762/argus-forge/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/smk762/argus-forge/releases/tag/v0.1.0

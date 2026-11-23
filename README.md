<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/smk762/argus-cortex/dev/brand/packages/forge/argus-forge-lockup-dark.svg">
    <img alt="argus-forge" src="https://raw.githubusercontent.com/smk762/argus-cortex/dev/brand/packages/forge/argus-forge-lockup-light.svg" height="52">
  </picture>
</h1>

Training bridge: turn curated dataset exports into ready-to-run LoRA training configs (kohya_ss / OneTrainer / diffusers).

Part of the [Argus suite](https://github.com/smk762?tab=repositories&q=argus). argus-curator's `/curate` UI already
computes suggested SDXL hyperparameters from the selected set (size, category); forge closes the gap between
"export" and "train" by handing over a **runnable config** instead of numbers on a screen.

```
argus-quarry  ->  argus-curator  ->  argus-lens  ->  argus-forge  ->  your trainer
  acquire          curate/export      caption          configs          LoRA
```

## What it does

Point it at a curator export directory (images + `manifest.jsonl` + `.txt` caption sidecars) and it emits
trainer-native config files, with hyperparameters seeded from the same selection-insight heuristics the
/curate UI shows (dataset size x target category -> repeats/epochs/LR/dim/alpha):

| Trainer | Emits |
| ------- | ----- |
| `kohya` | `dataset.toml` + `config.toml` (+ `train.sh` for `sdxl_train_network.py`) |
| `onetrainer` | `concepts.json` + partial `config.json` to load in the UI |
| `diffusers` | `metadata.jsonl` (HF imagefolder) + `train.sh` for `train_text_to_image_lora_sdxl.py` |

Along the way it:

- validates the manifest (`manifest_version`-aware — refuses an incompatible major),
- **collects captions**: argus-lens writes `.txt` sidecars next to the *source* images, so forge copies them
  next to the exported copies trainers actually read,
- warns when a flattened export collided on basenames (several manifest rows -> one file on disk) and leaves
  the ambiguous file uncaptioned instead of pairing it with the wrong caption,
- emits one kohya subset per image directory (kohya's glob doesn't recurse — a structure-preserving export
  would otherwise train zero images),
- **warns when one body pose dominates the set**: argus-curator's manifest 2.7 carries argus-puppet's
  scan-level `pose_cluster` and `pose_labels` per row; when at least 10 of the images on disk have a cluster and
  the most common one holds 75% or more of them, forge adds a warning naming the pose (e.g. "standing, arms
  down, facing front") and how many images had no measured pose. Forge only counts what curator wrote: it
  never measures poses or calls puppet, and the warning changes no repeats or configs. Older manifests: no
  warning,
- falls back gracefully to a bare folder of images with no manifest,
- resolves the base checkpoint from the manifest's `target_profile.checkpoint` (or the SDXL base).

Everything lands in `<export>/forge/<trainer>/` (plus `metadata.jsonl` at the dataset root for diffusers).

## Install

```bash
uv pip install "argus-forge[cli]"          # CLI
uv pip install "argus-forge[cli,server]"   # + HTTP server for argus-studio
```

Or run the published image (serves `:8103` with `--cors`, and defaults its export root to `/data/out`):

```bash
docker run --rm -p 8103:8103 -v /path/to/out:/data/out ghcr.io/smk762/argus-forge:latest
docker compose up          # local build; see docker-compose.yaml (and .env.example) for the knobs
```

The image contains no trainer, so it defaults to [demo-safe mode](#demo-safe-mode-no-training) — configs
render, training is refused and nothing is written. That default lives in the image itself
(`ARGUS_FORGE_READONLY=1`), so a bare `docker run` is as locked down as `docker compose up`; the port is
published on 0.0.0.0 and a run is real code execution on the host, so the safe direction is the default one.
Set `ARGUS_FORGE_READONLY=0` on a trusted host once you have a trainer — either mounted, or baked in via
the [train image](#training-image--train).

### Training image (`-train`)

A second image variant carries the training runtime, so `POST /run` can go live on a GPU host instead of
crashing for want of a trainer: torch 2.6 (CUDA 12.4 wheels), accelerate, and a pinned
[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) checkout at `/opt/sd-scripts`
(`SD_SCRIPTS_DIR` is preset, so a forged kohya `train.sh` runs as-is). It is opt-in everywhere: published
only under the `-train` suffix (`ghcr.io/smk762/argus-forge:<version>-train`, `:latest-train`) so
`latest`/`<version>` stay the ~200 MB config-renderer and nobody pulls a multi-GB image by accident.

```bash
docker run --rm --init --gpus all --shm-size=8g \
  -e ARGUS_FORGE_READONLY=0 \
  -p 127.0.0.1:8103:8103 -v /path/to/out:/data/out \
  ghcr.io/smk762/argus-forge:latest-train
# or locally (uncomment `gpus: all` in docker-compose.yaml first):
# FORGE_BUILD_TARGET=train ARGUS_FORGE_READONLY=0 docker compose up --build
```

- **The posture flip stays deliberate.** The train image *also* defaults to `ARGUS_FORGE_READONLY=1`;
  carrying a trainer is what makes flipping it meaningful, not automatic. An armed `/run` executes a forged
  script with no auth — treat reaching the port as shell access on the host (see
  [demo-safe mode](#demo-safe-mode-no-training)). The command above binds loopback for exactly that
  reason; publish wider (`-p 8103:8103`) only on a trusted network — docker's port publishing bypasses
  ufw-style host firewalls, so "the firewall will catch it" is not a safety net.
- **kohya only.** OneTrainer is driven from its own UI, and the diffusers `train.sh` expects a diffusers
  examples checkout (`DIFFUSERS_SCRIPT`) this image doesn't carry.
- The base checkpoint downloads from Hugging Face on first run; mount a cache
  (`-v hf-cache:/root/.cache/huggingface`, or uncomment the `hf-cache` volume in docker-compose.yaml)
  to keep it across containers, and `--shm-size` spares the dataloader docker's 64 MB `/dev/shm`
  default (the compose file sets `shm_size` already).
- The LoRA lands under the export dir (`<export>/forge/kohya/output/`) on the shared volume — point that
  volume (or a `path_map`) at your ComfyUI-layout models tree (`.../loras/`) and the result is immediately
  visible to argus-proof's `GET /models`, no file shuffling.
- The `-train` tag is smoke-tested in CI, **not** e2e-trained — see [CI / Release](#ci--release).

### Sharing the GPU (argus-iris)

Every run, from `POST /run` or `argus-forge run`, holds one GPU lease for the whole of `train.sh`, through
argus-cortex's coordinator. The variables are the suite's own, so they mean the same in every argus service:

| Variable | Meaning |
| -------- | ------- |
| `ARGUS_GPU_BROKER_URL` | [argus-iris](https://github.com/smk762/argus-iris), e.g. `http://<GPU_HOST>:8120`. Set, it is the coordinator; unset, a run starts at once |
| `ARGUS_GPU_BROKER_TOKEN` | the shared caller token. Secret: host `.env` only |
| `ARGUS_GPU_COORDINATOR` | only to override: `none` (start at once) or `lease` (the host-wide lock file, once the card has room) |
| `ARGUS_GPU_BROKER_FALLBACK` | `lease` (default) or `none`, used when iris is unreachable or has no lease routes yet |
| `ARGUS_GPU_LEASE_TIMEOUT_S` | how long a run waits for its lease (default 300 s) before it ends in an `error` event |
| `FORGE_GPU_VRAM_MB` | the lease's footprint (default 22000 MiB) |

The URL and the token are all the broker needs: argus-cortex reads the tenant from `gpu-tenant.yaml` in the working
directory or at `/app/gpu-tenant.yaml`, where the image ships it. `argus-forge serve` registers the tenant with iris
at startup, keeps it registered, and signs off at shutdown; startup never waits on iris. The CLI registers with its
run's lease.

The lease is exclusive: kind `train`, `background` priority, not preemptible, pinned to `<GPU_HOST>`, and sized to
fill a 24 GiB card, so iris admits nothing else beside it. When a run may start is the operator's decision, made
with a schedule window for `argus-forge` in iris's policy file. While it waits, the stream shows
`[argus-forge] waiting for a GPU lease`. The lease is released, with the run's outcome (`ok`, `failed`, `oom`
or `cancelled`) and its peak VRAM, before the terminal `exit` event is sent, so an argus-proof eval that
starts on `exit` finds the GPU free. Peak VRAM needs NVML, from the `gpu` extra that both images include
(`pip install argus-forge[gpu]` otherwise).

## CLI

```bash
argus-forge inspect /data/out                          # what's in this export?
argus-forge inspect /data/out --category concept       # ...judged as a style / concept LoRA
argus-forge config /data/out --trainer kohya           # emit configs
argus-forge config /data/out -t diffusers --dry-run    # preview without writing
argus-forge config /data/out --trigger "zxq person" --network-dim 32 --epochs 8
argus-forge trainers                                   # list emitters
argus-forge schema                                     # write the wire-contract JSON Schema
argus-forge run /data/out --trainer kohya \
  --env SD_SCRIPTS_DIR=~/kohya-ss/sd-scripts             # run the forged train.sh, streaming progress
```

`run` shells out to the forged `train.sh` (kohya / diffusers; OneTrainer is
driven from its own UI) and streams the trainer's output. It exits with the
trainer's own exit code, so it slots into scripts and CI; `--dry-run` prints the
command without executing, and `--json` streams raw NDJSON `RunEvent`s.

### Speech exports

A curator speech export (argus-curator#187) holds WAV/FLAC clips, a `<stem>.txt` transcript beside each, and
`speech_manifest.jsonl` (`SpeechManifestRow`, versioned by `speech_manifest_version`, independent of the image
`manifest_version`). Forge reads 1.x and refuses any other major by name. `inspect` detects one by that file:

```bash
argus-forge inspect /data/voice-train --held-out /data/voice-held   # clips, hours, speakers, sources
```

Each clip is joined from curator's `exported_path` / `transcript_path` onto the export dir as forge sees it
(never re-derived from `rel_path`); curator's `exported_abs_path` is carried, and refused if it names a
different file forge can see. Rows with `split` `train` (or null) are trained on; `held_out` rows, and every row
of a `--held-out` export, are reference clips only. A held-out `speaker_id`, `speaker_label` or `speaker_cluster`
that also appears in training is refused. Speaker, accent, language and gender are carried per clip, and corpus,
licence and provenance per source. An edited `.txt` wins over the manifest's `transcript`. The result,
`argus_forge.speech.read_speech_export` -> `SpeechDataset`, is what a speech emitter (`SpeechEmitter`) takes.
No speech emitter ships yet: XTTS-v2 is #56. The image trainers refuse a speech export.

`--category` overrides the manifest's target category. The values are argus-cortex's shared taxonomy
(`identity`, `wardrobe`, `pose_composition`, `setting`, `concept`), and each one sets its own step budget,
network size and dataset-size guidance.

`schema` writes the JSON Schema for forge's wire contract (the `/inspect`, `/config` and `/run` request and
response types, plus the manifest row), so a client can generate types from it rather than copying them by
hand. It writes to [`schema/forge-wire.schema.json`](schema/forge-wire.schema.json) by default (`-o` to
choose another path); `--check` exits non-zero if that file no longer matches the code, which is what CI runs.

## Server (argus-studio integration)

Start the FastAPI micro-server on **:8103** (peer to lens :8100, curator :8101, quarry :8102):

```bash
argus-forge serve --cors --export-root /data/out
```

`serve` listens on `--host 0.0.0.0 --port 8103` by default, which is **every interface**, with `POST /run`
enabled. Forge has no authentication, and [reaching an enabled `/run` is equivalent to shell
access](#demo-safe-mode-no-training), so unless the network is trusted, pass `--host 127.0.0.1` (studio on
the same machine), `--no-run` (configs only), or both:

```bash
argus-forge serve --host 127.0.0.1 --cors --export-root /data/out
```

`--export-root` is **required** by the API: a request's `export_dir` is untrusted, so it must name a
directory *under* this one, and every `/inspect`, `/config` and `/run` refuses with 400 until it is set.
Without the fence, `POST /config` would forge a tree into any directory the caller names. Also settable as
`ARGUS_FORGE_EXPORT_ROOT` (`FORGE_EXPORT_PATH` is a legacy alias); the published image defaults it to
`/data/out`. The **CLI is unconstrained** by design — it is your own shell, not a request.

Containment is decided on the fully resolved path, so a symlink pointing out of the root is refused rather
than followed, and the root itself is not a valid `export_dir` (a blank field would otherwise mean "treat
the whole shared volume as one dataset"). The path handed to the emitters keeps *your* spelling, though —
`path_map` prefixes are matched against it, so a symlinked root like `/data/out -> /mnt/<NAS>/out` still
rewrites correctly. The manifest's `abs_path` caption sources are fenced to the same root: forge will not
copy a sidecar from outside it into your dataset.

`--cors` matters: the studio frontend calls forge **cross-origin** (browser on :3000 → forge on :8103), and
CORS is opt-in — without it the ExportPanel fails with "Failed to fetch" even though `curl` works fine.
(The Docker image passes `--cors` already; this applies to the pip-installed path.)

A bare `--cors` allows the localhost:3000 dev frontend. Name other origins with `--cors-origin`
(repeatable, or `FORGE_CORS_ORIGINS=a,b`); they are **added to** the localhost defaults, not swapped for
them. Entries are compared to the `Origin` header a browser actually sends, so a trailing slash or stray
whitespace is normalised away rather than silently never matching.

`--cors-any` allows *any* origin **credential-less** — anonymous reads from anywhere, for a public demo. A
literal `*` in the allow-list takes that same path rather than becoming credentialed origin reflection.
It never grants a cross-site *write* to an origin you did not name — but origins you **did** name keep
theirs. That pairing is what a public demo needs, because every endpoint that does anything in forge
(`/inspect`, `/config`) is a POST:

```bash
argus-forge serve --cors-any --cors-origin https://demo.example --no-run --export-root /data/out
```

> CORS is not a write boundary. A cross-origin POST with a CORS-safelisted content type is sent with **no
> preflight**, so any page your browser visits can drive an unauthenticated LAN server; the same-origin
> policy only stops it from *reading* the reply. So unsafe methods are additionally gated on `Origin`:
> absent (curl, the CLI, server-to-server) or same host:port or allow-listed → through; anything else → 403.
> A same host:port `Origin` passes only when the host is an IP literal or `localhost`, or is listed in
> `FORGE_TRUSTED_HOSTS` (comma-separated names, no port). That guards against DNS rebinding, so when you
> use `/docs` through a name like `http://<GPU_HOST>:8103`, list that host name there.

| Route | Purpose |
| ----- | ------- |
| `GET /health` | liveness + version + export root + whether training is enabled |
| `GET /trainers` | supported trainers + emitted files |
| `POST /inspect` | look at an export dir (counts, manifest, suggested params) |
| `POST /config` | render configs; `dry_run: true` returns contents without writing |
| `POST /run` | start the forged `train.sh` on a background job; returns the run's `RunState` (with `run_id`) — the run outlives the request; 409 while another run is active, and an `Idempotency-Key` header makes a retry safe (below) |
| `GET /runs` | list tracked runs |
| `GET /run/{id}` | a run's status (poll for the terminal `status` + `returncode` — the argus-proof join); a succeeded run's `lora` too |
| `GET /run/{id}/lora` | a succeeded run's LoRA: `path`, `export_path` (relative to the export dir), `sha256`, `size_bytes`; 409 while it hasn't succeeded |
| `GET /run/{id}/stream` | attach to a run: NDJSON `RunEvent`s, the retained tail then live; reconnect anytime |
| `POST /run/{id}/cancel` | stop a run (SIGTERM→SIGKILL its process group) |

A run is started once (`POST /run`) and watched — or re-watched after a dropped connection — via
`GET /run/{id}/stream`; a client going away never stops the run. `run_id` (also on the stream's
`X-Training-Run-Id` header) is the join key for the argus-proof handoff. The `/curate` page's ExportPanel in
[argus-studio](https://github.com/smk762/argus-studio) uses `/config` to forge a config right after an export
(`docker compose --profile forge up`).

When a run's trainer exits 0, forge finds the LoRA where the forged config says it goes (kohya's
`output_dir` + `output_name`, diffusers' `--output_dir`) and hashes it before the run turns `succeeded`, so a
poller that sees `succeeded` gets `lora` with it and never has to know a trainer's output layout. A trainer
that exits 0 without leaving that file still ends `succeeded`, with no `lora` and a `message` naming where
forge looked.

**Runs survive a restart** when forge has a state directory (`--state-dir` or `ARGUS_FORGE_STATE_DIR`; the
image sets `/var/lib/argus-forge` and the compose file mounts a volume there). Each run's `RunState` is saved
there as JSON when it starts and when it ends, and reloaded at startup, the newest 64 finished runs kept.
The event log isn't saved: a restored run's stream replays its `start` and its terminal event. A run that
was still `running` when forge went away comes back `failed`, with a message saying its outcome is unknown.
Without a state directory, runs are kept in memory only.

**Retrying `POST /run` is safe with an `Idempotency-Key`.** A dropped connection can leave a client unsure
whether its run started, and a duplicate run holds the GPU for hours. Send a fresh random key (a UUID) per
intended run in the `Idempotency-Key` header (1-200 printable ASCII characters) and retry with the same key:

| Request | Response |
| ------- | -------- |
| no `Idempotency-Key` | a new run (202), or 409 `run_active` while another run is active (below) |
| a key not seen before | 202, a new run, remembered under the key; or 409 `run_active` while another run is active, remembering nothing |
| the same key and the same body | 202 with the run the key started, as it is now (running, succeeded, failed or cancelled), and `Idempotent-Replayed: true`; nothing new starts |
| the same key, a different body | 409; nothing new starts |
| the same key, but its run is no longer retained (past the newest 64 finished runs) | 409; nothing new starts |
| a malformed key | 422 (the value is not echoed) |

"The same body" compares the validated request (sorted keys), so whitespace and key order don't matter, but
`dry_run`, `env` or a different `export_dir` do. A request refused with a 400 records nothing, so its retry is
judged afresh. Keys are **global to the server**: forge has no user identity, so two clients that send the same
key share it. They are kept beside the run records (`<state-dir>/idempotency/`, one JSON file per key holding
the key's sha256, never the key, and written atomically) so a replay after a restart still finds its run;
without a state directory they are in memory only. A key is remembered for
`ARGUS_FORGE_IDEMPOTENCY_RETENTION_S` seconds (default 86400, 24 h), pruned lazily on the next keyed `POST /run`
after that, and never while its run is still going. `Idempotent-Replayed` is exposed to browsers under CORS.

**One run at a time.** While a run is **active**, another `POST /run` is refused with 409 and creates nothing,
so two clients (two studio tabs, a script and a tab) cannot start two runs that then fight over the GPU. Active
means not yet ended: status `running`, which covers a run still **queued** for its argus-iris `train` lease as
well as one whose trainer is going (a queued run counts, or the broker's queue fills with forge's own runs).
`succeeded`, `failed` and `cancelled` are terminal. The 409 body is a `RunActiveError`:

```json
{
  "detail": "a run is already active (run 3f2a…, queued or running); wait for it to finish or cancel it",
  "code": "run_active",
  "active_run": {"run_id": "3f2a…", "status": "running"}
}
```

`detail` is a string like every other forge error, so existing clients still read it; `code: "run_active"` is
the machine-readable marker that tells this 409 apart from the `Idempotency-Key` 409s, which are a plain
`{"detail": ...}` with no `code`. Nothing from the request is echoed. Wait for the active run (poll
`GET /run/{id}` or attach to its stream) or cancel it (`POST /run/{id}/cancel`), then start yours.

With an `Idempotency-Key`, the key is looked at first: a retry of the active run's own request (same key, same
body) still gets its run back (202, `Idempotent-Replayed: true`), never a 409 about itself, and the same key with
a different body is still the key's 409. A request refused because a run is active records no key, so the same
request with the same key starts a run once nothing is active. The check and the creation are one critical
section across every request, keyed or not, so concurrent requests start exactly one run.

A `dry_run` is exempt both ways: it takes no GPU lease and starts no process, so it neither waits for nor blocks a
real run. After a restart nothing stale blocks: a run that was in flight when forge went away is restored
`failed` (or saved `cancelled` by a graceful stop), so it is not active. A host whose GPU broker can place several
runs at once (several GPUs behind argus-iris) can raise the limit with `ARGUS_FORGE_MAX_ACTIVE_RUNS` (default 1;
an unparseable or sub-1 value falls back to 1); the 409 then names the oldest active run.

> The local CLI `argus-forge run` streams live in your terminal and is independent of the server registry.

### Demo-safe mode (no training)

`POST /run` executes a script on the host. It runs a forged `train.sh` on the shared dataset volume — the
same single-user LAN assumption as `/config` — but it is **not sandboxed**: treat reaching the port as
equivalent to shell access on the host.

So a host that should render configs but never train — a public demo, a box with no GPU or no `sd-scripts`
— starts forge in demo-safe mode:

```bash
argus-forge serve --cors --no-run       # or: ARGUS_FORGE_READONLY=1 argus-forge serve --cors
```

`/inspect` and `/config` keep working. Every state-changing request (POST, PUT, PATCH, DELETE) to `/run` or a
path under `/run/` is refused with **403** and a message saying training is disabled on this host. Today
that is `POST /run` and `POST /run/{id}/cancel`. Reads are not refused: `GET /runs` lists only runs restored
from the state directory (none on a host that never trained), and any other id is a 404. The refusal is
middleware, so it lands *before* the request body is validated (on such a host a malformed request deserves
the same answer as a well-formed one), and it covers `/run` routes added later without a per-route guard to
forget.

Demo-safe also means **write**-safe: `POST /config` renders only with `"dry_run": true`. A `/config` that asks
to write (`dry_run` false or left out) is refused with the same **403**, before forge looks at the export dir,
rather than quietly rendered as a preview a caller could mistake for a write and go on to train from
([#52](https://github.com/smk762/argus-forge/issues/52)). Forge has no authentication,
so on a publicly reachable host an ordinary `curl` would otherwise overwrite the curator's
`metadata.jsonl` and leave an executable `train.sh` behind on shared storage.

`GET /health` reports `"training": "enabled" | "disabled"`, so a frontend can disable its train button up
front rather than discovering the refusal by clicking it. **Both published images turn this mode on by
default** (and the compose file with it): the thin image ships no trainer, so a run there could only ever
fail, and the [train image](#training-image--train) keeps the same default so that arming `/run` is always
an explicit choice rather than a side effect of pulling a bigger tag.
`ARGUS_FORGE_READONLY` is a protection flag, so it fails *safe*: a value it cannot parse (`=y`, `=enabled`)
warns and keeps the guard on rather than quietly enabling writes. Only an explicit `0`/`false`/`no`/`off`
turns it off.

### Container ↔ host paths (`path_map`)

When forge runs in the compose stack it sees container paths (`/data/out/...`), but the emitted `train.sh` /
configs are meant to run **on the host**, where those paths don't exist. Tell forge how to translate:

```bash
# per request (CLI: repeatable --path-map; API: "path_map" on POST /config)
argus-forge config /data/out --path-map /data/out=$HOME/argus/out

# or once, via the environment (the compose file can set this from OUTPUT_DIR)
FORGE_PATH_MAP=/data/out=$HOME/argus/out argus-forge serve --cors
```

Every absolute path rendered into configs (`image_dir`, `output_dir`, `--train_data_dir`, OneTrainer concept
paths, ...) gets the longest matching prefix rewritten; the request-level map wins over the env var. The
emitted README notes whether a remap was applied.

## Develop

```bash
make install   # venv + editable install with the "dev,server,cli" extras
make test
make lint
```

## CI / Release

- **CI** runs via the shared [`argus-ci`](https://github.com/smk762/argus-ci) reusable workflow
  (plus `argus-forge schema --check` to keep the committed wire schema honest).
- **Release** publishes to PyPI (OIDC trusted publishing) and GHCR on `v*` tags — the thin image and the
  [`-train` variant](#training-image--train) as independent jobs, so a fault in one never withholds the other.
- The `-train` image is gated by a CPU-only smoke test before it is pushed
  ([`scripts/smoke-train-image.sh`](scripts/smoke-train-image.sh)): the image builds, `serve` answers
  `/health` (still demo-safe), torch/accelerate/sd-scripts import, and a forged kohya config passes the
  `/run` validation path (`run --dry-run`). CI has no GPU, so **no actual training step runs** — a `-train`
  tag is honest about building and validating, not e2e-verified; the first real `accelerate launch` happens
  on your GPU host.
- Once the wheel is on PyPI, the same run creates the **GitHub Release** for the tag. Its notes are that
  version's `CHANGELOG.md` section, and the run fails if the section is missing. The built wheel and sdist
  are attached, and tags like `-rc1`, `a1` or `b1` are marked as pre-releases.
- Versioning is derived from git tags via `hatch-vcs` — tag `vX.Y.Z` to cut a release.

This repo was scaffolded from [`argus-pkg-template`](https://github.com/smk762/argus-pkg-template).
Run `copier update` to pull template changes (CI, release, tooling).

## Roadmap

- argus-proof handoff: post-training validation ([argus-studio#4](https://github.com/smk762/argus-studio/issues/4)).
  `GET /run/{id}` now exposes a run's terminal `status` + `returncode` by `run_id` for the join.
- run registry follow-ups ([#13](https://github.com/smk762/argus-forge/issues/13)): CLI management commands
  (`runs` / `--attach` / `--cancel`) and an optional single-flight guard. (Run records now survive a restart:
  `--state-dir`, [#52](https://github.com/smk762/argus-forge/issues/52).)

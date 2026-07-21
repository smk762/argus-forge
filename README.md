# argus-forge

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
- falls back gracefully to a bare folder of images with no manifest,
- resolves the base checkpoint from the manifest's `target_profile.checkpoint` (or the SDXL base).

Everything lands in `<export>/forge/<trainer>/` (plus `metadata.jsonl` at the dataset root for diffusers).

## Install

```bash
uv pip install "argus-forge[cli]"          # CLI
uv pip install "argus-forge[cli,server]"   # + HTTP server for argus-studio
```

## CLI

```bash
argus-forge inspect /data/out                          # what's in this export?
argus-forge config /data/out --trainer kohya           # emit configs
argus-forge config /data/out -t diffusers --dry-run    # preview without writing
argus-forge config /data/out --trigger "zxq person" --network-dim 32 --epochs 8
argus-forge trainers                                   # list emitters
argus-forge run /data/out --trainer kohya \
  --env SD_SCRIPTS_DIR=~/kohya-ss/sd-scripts             # run the forged train.sh, streaming progress
```

`run` shells out to the forged `train.sh` (kohya / diffusers; OneTrainer is
driven from its own UI) and streams the trainer's output. It exits with the
trainer's own exit code, so it slots into scripts and CI; `--dry-run` prints the
command without executing, and `--json` streams raw NDJSON `RunEvent`s.

## Server (argus-studio integration)

Start the FastAPI micro-server on **:8103** (peer to lens :8100, curator :8101, quarry :8102):

```bash
argus-forge serve --cors
```

`--cors` matters: the studio frontend calls forge **cross-origin** (browser on :3000 → forge on :8103), and
CORS is opt-in — without it the ExportPanel fails with "Failed to fetch" even though `curl` works fine.
(The Docker image passes `--cors` already; this applies to the pip-installed path.)

| Route | Purpose |
| ----- | ------- |
| `GET /health` | liveness + version |
| `GET /trainers` | supported trainers + emitted files |
| `POST /inspect` | look at an export dir (counts, manifest, suggested params) |
| `POST /config` | render configs; `dry_run: true` returns contents without writing |
| `POST /run` | start the forged `train.sh` on a background job; returns the run's `RunState` (with `run_id`) — the run outlives the request |
| `GET /runs` | list tracked runs |
| `GET /run/{id}` | a run's status (poll for the terminal `status` + `returncode` — the argus-proof join) |
| `GET /run/{id}/stream` | attach to a run: NDJSON `RunEvent`s, buffered backlog then live; reconnect anytime |
| `POST /run/{id}/cancel` | stop a run (SIGTERM→SIGKILL its process group) |

A run is started once (`POST /run`) and watched — or re-watched after a dropped connection — via
`GET /run/{id}/stream`; a client going away never stops the run. `run_id` (also on the stream's
`X-Training-Run-Id` header) is the join key for the argus-proof handoff. The `/curate` page's ExportPanel in
[argus-studio](https://github.com/smk762/argus-studio) uses `/config` to forge a config right after an export
(`docker compose --profile forge up`).

> The local CLI `argus-forge run` streams live in your terminal and is independent of the server registry.

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
- **Release** publishes to PyPI (OIDC trusted publishing) and GHCR on `v*` tags.
- Versioning is derived from git tags via `hatch-vcs` — tag `vX.Y.Z` to cut a release.

This repo was scaffolded from [`argus-pkg-template`](https://github.com/smk762/argus-pkg-template).
Run `copier update` to pull template changes (CI, release, tooling).

## Roadmap

- argus-proof handoff: post-training validation ([argus-studio#4](https://github.com/smk762/argus-studio/issues/4)).
  `GET /run/{id}` now exposes a run's terminal `status` + `returncode` by `run_id` for the join.
- run registry follow-ups ([#13](https://github.com/smk762/argus-forge/issues/13)): CLI management commands
  (`runs` / `--attach` / `--cancel`), an optional single-flight guard, and durable run metadata across restarts.

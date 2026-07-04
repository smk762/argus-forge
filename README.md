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
```

## Server (argus-studio integration)

`argus-forge serve` starts a FastAPI micro-server on **:8103** (peer to lens :8100, curator :8101, quarry :8102):

| Route | Purpose |
| ----- | ------- |
| `GET /health` | liveness + version |
| `GET /trainers` | supported trainers + emitted files |
| `POST /inspect` | look at an export dir (counts, manifest, suggested params) |
| `POST /config` | render configs; `dry_run: true` returns contents without writing |

The `/curate` page's ExportPanel in [argus-studio](https://github.com/smk762/argus-studio) uses this to forge a
config right after an export (`docker compose --profile forge up`).

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

- `argus-forge run`: job-runner mode that shells out to the trainer and streams progress
  (NDJSON/SSE, following the suite's conventions) — [argus-studio#3](https://github.com/smk762/argus-studio/issues/3).
- argus-proof handoff: post-training validation ([argus-studio#4](https://github.com/smk762/argus-studio/issues/4)).

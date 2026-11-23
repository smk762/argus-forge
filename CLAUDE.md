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

## Must never happen

Each row names what enforces it. A **Partial** or **UNENFORCED** row links the issue that closes the gap: rely on
review until then, and close it with a check rather than more prose. A rule about code shape across the suite (a banned
import or API) belongs in argus-ci's `suite-rules/` catalogue rather than a test here.

| Never | Enforced by |
|---|---|
| Change a `models.py` type in `wire_schema()` without regenerating `schema/forge-wire.schema.json` | CI post-test `argus-forge schema --check` (`.github/workflows/ci.yml`) |
| Serve an `export_dir`, or copy a manifest caption source, from outside the server's `--export-root` (symlinks resolved) | `test_server::test_traversal_escape_is_refused`, `::test_symlink_plus_dotdot_cannot_escape_the_root`, `::test_absolute_path_outside_the_root_is_refused`, `::test_caption_sources_outside_the_root_are_refused` |
| Read-only mode: run anything, validate a `/run` body before refusing it, write from `/config` (or answer a writing `/config` with anything but 403), or turn off on an unparseable flag | `test_server::test_readonly_refuses_run_before_validating`, `::test_readonly_config_write_is_refused`, `::test_unrecognised_readonly_value_fails_safe_not_fatal` |
| Add a write route that read-only mode doesn't cover (it must sit under `/run`, or be listed with why it writes nothing) | `test_rules::test_every_write_route_is_refused_or_known_safe_under_readonly` |
| Accept a cross-site write, including under `--cors-any` | `test_server::test_cross_site_write_is_refused`, `::test_wildcard_still_refuses_cross_site_writes` |
| Let a caller's env set anything outside `models.RUN_ENV_KEYS` (a loader, shell or interpreter hook changes what `/run` executes), or hand the trainer the suite's tokens | `test_runner::test_prepare_run_rejects_exec_hijacking_env`, `::test_trainer_does_not_inherit_suite_secrets`, `test_server::test_run_blocked_env_is_400` |
| Call `argus_cortex.settings.apply()`, or otherwise copy the suite settings file into the environment: the trainer inherits it, and the file holds the suite's tokens | `test_suite_settings::test_forge_never_applies_the_suite_file_to_the_environment`, `::test_trainer_gets_nothing_from_the_suite_file` |
| Mount the suite settings file (a path ending in `suite.env`) into forge's container in any compose file: the trainer shares the container and could read the tokens from disk | `test_suite_settings::test_no_compose_file_mounts_the_suite_file` |
| Leave a trainer running after its run is cancelled | `test_runner::test_cancelled_run_kills_the_process`, `test_jobs::test_double_cancel_does_not_orphan_the_trainer` |
| Start a trainer without its GPU lease, or after the lease was refused or timed out | `test_gpu::test_run_takes_an_exclusive_background_lease`, `::test_lease_not_granted_is_an_error_and_nothing_runs` |
| Send a run's terminal event while still holding its GPU lease | `test_gpu::test_lease_is_released_before_the_terminal_event` |
| Report a run `succeeded` before its `lora` is set (a poller reads both together) | `test_lora::test_lora_is_set_before_the_run_turns_succeeded` |
| Start a second run for a retried `POST /run` carrying the same `Idempotency-Key` (concurrent retries included, and after a restart with a state dir) | `test_idempotency::test_a_replay_returns_the_same_run_and_says_so`, `::test_a_replay_after_the_run_ended_returns_it_as_it_ended`, `::test_the_same_key_with_a_different_body_is_409_and_starts_nothing`, `::test_concurrent_requests_with_one_new_key_start_exactly_one_run`, `::test_a_key_survives_a_restart_with_a_state_dir`, `::test_a_key_whose_run_was_evicted_is_409_not_a_second_run`, `::test_an_expired_key_is_pruned_but_never_while_its_run_is_active` |
| Start a second run while another is active (queued for its GPU lease or running), concurrent requests with different keys or none included, or answer a retry of the active run's own `Idempotency-Key` with that 409 | `test_one_active_run::test_a_second_run_is_409_while_one_is_running_and_creates_nothing`, `::test_a_run_queued_for_its_gpu_lease_is_active`, `::test_concurrent_requests_with_different_keys_or_none_start_exactly_one_run`, `::test_a_replay_of_the_active_runs_own_key_is_202_not_409`, `::test_a_refused_request_records_no_key_so_its_retry_starts_a_run`, `::test_a_run_in_flight_at_a_crash_does_not_block_new_runs_after_a_restart` |
| Lose a run's record on restart when a state dir is set, or restore a run that was in flight as anything but `failed` | `test_lora::test_runs_survive_a_restart`, `::test_a_run_in_flight_at_a_crash_is_restored_failed` |
| Report a failed or OOM run to the broker as `ok` | `test_gpu::test_failed_run_is_reported_to_the_coordinator`, `::test_through_cortex_broker_client` |
| Let `gpu-tenant.yaml` drift from the lease the runner takes | `test_gpu::test_manifest_validates_and_matches_the_lease` |
| Pair a caption with the wrong image on a basename collision (leave it uncaptioned) | `test_emitters::test_basename_collision_warns_and_skips_caption` |
| Emit an unmapped container path into a config the host trainer reads | `test_emitters::test_no_unmapped_container_path_escapes_any_emitter` |
| Emit a kohya config that trains zero images from a structure-preserving export (one subset per image dir) | `test_emitters::test_kohya_nested_export_gets_subset_per_directory` |
| Let heuristics or the kohya emitter drift from argus-studio's twins (`types.ts` `suggestTrainingParams` / `datasetSizeStatus`, `forgeDemo.ts`) | `test_studio_parity::test_parity_fixture_is_current` keeps `schema/forge-parity.json` in step with the code; argus-studio pins that file and `forgeParity.test.ts` checks its twins against it (its daily wire-drift job re-pulls it from `dev`). Regenerate with `UPDATE_PARITY=1` and change studio's twins along with it |
| Compute pose metrics or call argus-puppet (the pose-dominance warning only counts curator's manifest `pose_cluster` / `pose_labels`) | `test_rules::test_forge_does_not_compute_pose_metrics_or_call_puppet` (no puppet, numeric-stack or HTTP-client import; no puppet endpoint or setting named) |
| Re-implement something argus-cortex provides | `test_rules::test_forge_does_not_reimplement_cortex` (same-named functions and classes; a thin wrapper must import the original) |

## Facts the code doesn't make obvious

- `models.py` is the API contract: `wire_schema()` backs `argus-forge schema` and the committed schema. Regenerate
  with `argus-forge schema`, which writes the file itself; don't redirect stdout into it.
- Versioning is git-tag-derived (`hatch-vcs`): `src/argus_forge/_version.py` is generated and gitignored. Tag `vX.Y.Z`
  to release.
- The **CLI is deliberately unconstrained**: no export-root fence, no read-only mode. It is your own shell.
- `POST /run` is real code execution on the host and there is no auth: treat reaching the port as shell access.
- Lens writes `.txt` next to *source* images; forge copies them next to the *exported* copies.
- kohya's image glob doesn't recurse, which is why it gets one subset per directory.
- A new trainer is a module in `emitters/` registered in `emitters/__init__.py`; it is runnable only if its
  `TrainerInfo.entrypoint` is set.
- A run's GPU lease comes from argus-cortex's coordinator (`gpu.py`); `ARGUS_GPU_BROKER_URL` alone selects the broker
  (`serve` registers the tenant at startup), and with neither it nor `ARGUS_GPU_COORDINATOR` set it is `none`, so
  runs start at once. The lease size (`FORGE_GPU_VRAM_MB`, default 22000) is below the card's 24 GiB because iris keeps
  per-node headroom; a full-card request could never be admitted.
- `/health` is argus-cortex's `ServiceHealth` (`install_health`); forge's own fields (`export_root`, `training`) ride
  in `extra` and must keep their names. It has no checks and no readiness, so it is always 200.
- The suite settings file is read through cortex (`ARGUS_GPU_*`, and `ARGUS_FORGE_READONLY` with its fail-safe
  rule) when forge runs directly on a host; the container is never given the file. Tests never see a developer's file: `tests/conftest.py` sets `ARGUS_ENV_FILE=off`.
- Shared suite code (taxonomy, wire-schema building, the server write-guard and env-flag helpers) lives in
  **argus-cortex** (`argus-cortex[server]`).
- Run one test: `uv run --no-sync pytest tests/test_core.py::test_name -q`. `make help` lists the rest; ruff is pinned
  (line length 120).
- `structlog` for logging; Pydantic v2; async server and runner (`pytest asyncio_mode = auto`).

## Branches

`dev` is the default branch and the base for every PR. Cut feature branches from `origin/dev` and bring them up to date by merging `origin/dev`, never `main`. Never push work to `main` or open a PR against it, except the release PR from `dev`: `main` must never be ahead of `dev`, with one exception. A release fast-forwards `main` to a `dev` commit that passed CI (`git push origin origin/dev:main`), then tags it. If `main` is protected, the release is a reviewed PR from `dev` into `main`, and merging it leaves a merge commit on `main` alone. That merge commit is the one exception to the rule: it leaves `main` one commit ahead of `dev` with an identical tree. Leave it there and don't merge `main` back into `dev`. Anything else on `main` that `dev` lacks is a mistake (`git diff origin/dev origin/main` must be empty). Once `main` carries such a commit it can't be fast-forwarded, so later releases are PRs too.

## CI / release

CI runs via the shared [`argus-ci`](https://github.com/smk762/argus-ci) reusable workflow plus `argus-forge schema --check`. Release publishes to PyPI (OIDC trusted publishing) + GHCR on `v*` tags. This repo is scaffolded from [`argus-pkg-template`](https://github.com/smk762/argus-pkg-template) — run `copier update` to pull tooling changes.

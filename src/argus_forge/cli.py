"""argus-forge CLI — ``config``, ``inspect``, ``trainers``, ``schema``, ``serve``."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog

try:
    import typer
    from typer import Argument, Option
except ImportError as _exc:  # pragma: no cover
    print("CLI requires: pip install argus-forge[cli]", file=sys.stderr)
    raise SystemExit(1) from _exc

app = typer.Typer(
    name="argus-forge",
    help="Training bridge: turn curated dataset exports into ready-to-run LoRA training configs.",
    no_args_is_help=True,
)


@app.callback()
def _cli(verbose: bool = Option(False, "--verbose", "-v", help="Show info/debug logs")) -> None:
    """Keep stdout clean for --json output; ``serve`` re-enables info logs."""
    level = logging.DEBUG if verbose else logging.WARNING
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(level))


@app.command()
def config(
    export_dir: Path = Argument(..., help="Curator export dir (images + manifest.jsonl + .txt sidecars)"),
    trainer: str = Option("kohya", "--trainer", "-t", help="kohya | onetrainer | diffusers"),
    base_model: str | None = Option(None, "--base-model", help="Checkpoint/repo (default: manifest, else SDXL base)"),
    trigger: str | None = Option(None, "--trigger", help="Concept token (default: slug of the export dir name)"),
    output_name: str | None = Option(None, "--output-name", help="LoRA filename stem (default: <dir>-lora)"),
    category: str | None = Option(
        None, "--category", help="Override manifest category: identity|wardrobe|pose_composition|setting"
    ),
    repeats: int | None = Option(None, "--repeats", help="Override per-image repeats"),
    epochs: int | None = Option(None, "--epochs", help="Override epochs"),
    network_dim: int | None = Option(None, "--network-dim", help="Override LoRA rank"),
    network_alpha: int | None = Option(None, "--network-alpha", help="Override LoRA alpha"),
    unet_lr: float | None = Option(None, "--unet-lr", help="Override UNet learning rate"),
    text_encoder_lr: float | None = Option(None, "--text-encoder-lr", help="Override text-encoder learning rate"),
    optimizer: str | None = Option(None, "--optimizer", help="Override optimizer (kohya name, e.g. AdamW8bit)"),
    scheduler: str | None = Option(None, "--scheduler", help="Override LR scheduler"),
    resolution: int | None = Option(None, "--resolution", help="Override training resolution"),
    batch_size: int | None = Option(None, "--batch-size", help="Override batch size"),
    precision: str | None = Option(None, "--precision", help="Override mixed precision (bf16/fp16)"),
    collect_captions: bool = Option(
        True,
        "--collect-captions/--no-collect-captions",
        help="Copy .txt sidecars from manifest source paths into the export",
    ),
    path_map: list[str] = Option(
        [],
        "--path-map",
        help="Rewrite a path prefix in emitted configs, as CONTAINER=HOST (one entry per flag; also FORGE_PATH_MAP env)",
    ),
    dry_run: bool = Option(False, "--dry-run", help="Print rendered files without writing anything"),
    as_json: bool = Option(False, "--json", help="Print the full ForgeResult as JSON"),
) -> None:
    """Emit a ready-to-run training config for a curated export."""
    from argus_forge.core import forge_config, path_map_entry
    from argus_forge.emitters.base import map_path
    from argus_forge.models import ForgeError, ForgeRequest, ParamOverrides

    overrides = ParamOverrides(
        repeats=repeats,
        epochs=epochs,
        network_dim=network_dim,
        network_alpha=network_alpha,
        unet_lr=unet_lr,
        text_encoder_lr=text_encoder_lr,
        optimizer=optimizer,
        scheduler=scheduler,
        resolution=resolution,
        batch_size=batch_size,
        precision=precision,
    )
    try:
        req = ForgeRequest(
            export_dir=str(export_dir),
            trainer=trainer,  # type: ignore[arg-type]  (pydantic validates the literal)
            base_model=base_model,
            trigger=trigger,
            output_name=output_name,
            category=category,  # type: ignore[arg-type]
            overrides=overrides,
            collect_captions=collect_captions,
            # One entry per flag, parsed individually — paths may contain commas.
            path_map=dict(path_map_entry(entry, source="--path-map") for entry in path_map),
            dry_run=dry_run,
        )
        result = forge_config(req)
    except (ForgeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return

    p = result.params
    typer.echo(f"Forged {result.trainer} config for {result.export_dir}")
    typer.echo(
        f"  {p.images} images x {p.repeats} repeats x {p.epochs} epochs "
        f"= {p.total_steps} samples ({p.optimizer_steps} steps @ batch {p.batch_size})"
    )
    typer.echo(f"  dim/alpha {p.network_dim}/{p.network_alpha} · lr {p.unet_lr}/{p.text_encoder_lr} · {p.precision}")
    typer.echo(f"  base model: {result.base_model}")
    typer.echo(f"  trigger:    {result.trigger}")
    if result.captions_collected:
        typer.echo(f"  captions collected from sources: {result.captions_collected}")
    for w in result.warnings:
        typer.echo(f"  ! {w}")
    typer.echo("")
    if dry_run:
        for f in result.files:
            typer.echo(f"----- {f.name} -----")
            typer.echo(f.content)
    else:
        for f in result.files:
            typer.echo(f"  wrote {f.path}")
        run_hint = next((f.path for f in result.files if f.name.endswith("train.sh")), None)
        if run_hint:
            # The next step runs where the *mapped* paths live, so hint with those
            # even though the "wrote" lines above are forge's own (real) locations.
            from argus_forge.core import resolve_path_map

            effective_map = resolve_path_map(req.path_map)
            typer.echo(
                f"\nNext: review {map_path(result.out_dir, effective_map)}/README.md, "
                f"then run {map_path(run_hint, effective_map)}"
            )


@app.command()
def inspect(
    export_dir: Path = Argument(..., help="Curator export dir to inspect"),
    category: str | None = Option(None, "--category", help="Override manifest category"),
    as_json: bool = Option(False, "--json", help="Print DatasetInfo as JSON"),
) -> None:
    """Summarise an export dir: images, captions, manifest, suggested params."""
    from argus_forge.manifest import inspect_export
    from argus_forge.models import ForgeError

    try:
        info, _ = inspect_export(export_dir, category=category)  # type: ignore[arg-type]
    except ForgeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if as_json:
        typer.echo(info.model_dump_json(indent=2))
        return

    typer.echo(f"Export: {info.export_dir}")
    typer.echo(f"  images:   {info.image_count} ({info.caption_count} captioned)")
    if info.manifest_present:
        typer.echo(f"  manifest: v{info.manifest_version}, {info.manifest_rows} rows, {info.missing_from_disk} missing")
    else:
        typer.echo("  manifest: none (bare folder mode)")
    tp = info.target_profile
    typer.echo(f"  profile:  {tp.target_category} / {tp.target_style} / {tp.target_backend or '-'}")
    typer.echo(f"  size:     [{info.size_hint.tone}] {info.size_hint.text}")
    p = info.suggested
    typer.echo(
        f"  suggest:  {p.repeats} repeats x {p.epochs} epochs = {p.total_steps} samples, "
        f"dim/alpha {p.network_dim}/{p.network_alpha}, lr {p.unet_lr}/{p.text_encoder_lr}"
    )


@app.command()
def trainers() -> None:
    """List supported trainers and what forge emits for each."""
    from argus_forge.emitters import TRAINER_INFO

    for info in TRAINER_INFO.values():
        typer.echo(f"{info.id:12s} {info.label}")
        typer.echo(f"{'':12s}   files: {', '.join(info.files)}")
        typer.echo(f"{'':12s}   {info.notes}")


DEFAULT_SCHEMA_PATH = Path("schema/forge-wire.schema.json")


@app.command()
def schema(
    output: Path = Option(DEFAULT_SCHEMA_PATH, "--output", "-o", help="Where to write the JSON Schema"),
    check: bool = Option(False, "--check", help="Exit non-zero if the committed schema is stale (for CI)"),
) -> None:
    """Emit the wire-contract JSON Schema consumers codegen against."""
    import json

    from argus_forge.models import wire_schema

    rendered = json.dumps(wire_schema(), indent=2, sort_keys=True) + "\n"

    if check:
        existing = output.read_text(encoding="utf-8") if output.exists() else ""
        if existing != rendered:
            typer.echo(f"{output} is stale — run `argus-forge schema` and commit the result.", err=True)
            raise typer.Exit(1)
        typer.echo(f"{output} is up to date.")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    typer.echo(f"Wrote wire schema -> {output}")


@app.command()
def serve(
    port: int = Option(8103, "--port", "-p", help="Port to listen on"),
    host: str = Option("0.0.0.0", "--host", help="Host to bind to"),
    cors: bool = Option(False, "--cors", help="Enable CORS (allow all origins)"),
) -> None:
    """Start the argus-forge micro-server (FastAPI) on :8103."""
    try:
        import uvicorn
    except ImportError as _exc:  # pragma: no cover
        typer.echo("Server requires: pip install argus-forge[server]", err=True)
        raise typer.Exit(1) from _exc

    from argus_forge.core import env_path_map
    from argus_forge.models import ForgeError
    from argus_forge.server import create_app

    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))
    try:
        env_path_map()  # fail fast at startup, not as a 400 on every /config
    except ForgeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    if not cors:
        typer.echo(
            "CORS is disabled — browser clients (e.g. the argus-studio frontend on :3000) "
            "will fail with 'Failed to fetch'; pass --cors to allow them.",
            err=True,
        )
    application = create_app(cors=cors)
    uvicorn.run(application, host=host, port=port)


if __name__ == "__main__":
    app()

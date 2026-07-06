"""Forge orchestration: inspect -> resolve params -> emit -> write.

The one entry point is :func:`forge_config`; the CLI and the HTTP server are
thin shells around it.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import structlog

from argus_forge.emitters import EMITTERS, EmitContext
from argus_forge.emitters.base import map_path
from argus_forge.heuristics import apply_overrides
from argus_forge.manifest import (
    FORGE_DIR_NAME,
    caption_path,
    exported_collisions,
    resolve_export_dir,
    resolve_rows,
    scan_export,
)
from argus_forge.models import (
    PATH_MAP_ENV,
    SUPPORTED_EXTS,
    ForgeError,
    ForgeRequest,
    ForgeResult,
    ManifestRow,
)

logger = structlog.get_logger()

# Fallback base checkpoints per target_backend when the manifest carries none.
DEFAULT_BASE_MODELS = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
}


def slugify(name: str) -> str:
    """Directory name -> a safe trigger/output token (``My Set!`` -> ``my_set``)."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "dataset"


def normalize_path_pair(src: str, dst: str, source: str = "path_map") -> tuple[str, str]:
    """Validate and normalize one ``container -> host`` prefix mapping.

    Prefixes are compared without trailing slashes so ``/data/out`` and
    ``/data/out/`` are the same key everywhere (merge, override, longest-match).
    """
    src, dst = src.strip(), dst.strip()
    if not src or not dst:
        raise ForgeError(f"{source}: bad entry {src!r} -> {dst!r} — expected 'container/prefix=host/prefix'")
    src = src.rstrip("/")
    if not src:
        raise ForgeError(f"{source}: cannot remap the filesystem root ('/')")
    return src, dst.rstrip("/") or "/"


def path_map_entry(pair: str, source: str = "path_map") -> tuple[str, str]:
    """Parse a single ``container=host`` string (one --path-map flag value).

    Not comma-split, so paths containing a comma work — pass one flag per entry.
    """
    src, sep, dst = pair.partition("=")
    if not sep:
        raise ForgeError(f"{source}: bad entry {pair.strip()!r} — expected 'container/prefix=host/prefix'")
    return normalize_path_pair(src, dst, source)


def parse_path_map(spec: str, source: str = "path_map") -> dict[str, str]:
    """Parse the comma-separated ``container=host[,...]`` env-var form."""
    return dict(path_map_entry(pair, source) for pair in spec.split(",") if pair.strip())


def env_path_map() -> dict[str, str]:
    """The FORGE_PATH_MAP default map; raises ForgeError if it is malformed."""
    return parse_path_map(os.environ.get(PATH_MAP_ENV, ""), source=f"{PATH_MAP_ENV} env var")


def resolve_path_map(req_map: dict[str, str]) -> dict[str, str]:
    """The request's path_map merged over the FORGE_PATH_MAP env default.

    Request entries are validated and normalized like env entries, so a
    trailing-slash spelling can neither dodge validation nor dodge being
    overridden.
    """
    merged = env_path_map()
    merged.update(normalize_path_pair(src, dst) for src, dst in req_map.items())
    return merged


def collect_captions(
    export_dir: Path,
    rows: list[ManifestRow],
    dry_run: bool,
    resolved: list[Path | None] | None = None,
    skip: set[Path] | None = None,
) -> int:
    """Copy ``.txt`` sidecars written next to the *source* images into the export.

    argus-lens captions the manifest's ``abs_path`` entries, so sidecars land
    beside the originals — not beside the exported copies trainers read. This
    closes that gap. Existing sidecars in the export are never overwritten,
    and files in *skip* (basename collisions — see :func:`exported_collisions`)
    are left uncaptioned rather than paired with a caption that may describe
    different pixels. *resolved* is :func:`resolve_rows` output, passable to
    avoid re-stat()ing every row.
    """
    if resolved is None:
        resolved = resolve_rows(export_dir, rows)
    copied = 0
    for row, dest_img in zip(rows, resolved, strict=True):
        if dest_img is None or (skip and dest_img in skip):
            continue
        src_caption = caption_path(Path(row.abs_path))
        if not src_caption.is_file():
            continue
        dest_caption = caption_path(dest_img)
        if dest_caption.exists():
            continue
        if not dry_run:
            shutil.copy2(src_caption, dest_caption)
        copied += 1
    return copied


def forge_config(req: ForgeRequest) -> ForgeResult:
    """Render (and unless ``dry_run``, write) trainer configs for an export dir."""
    export_dir = resolve_export_dir(req.export_dir)
    if req.trainer not in EMITTERS:
        raise ForgeError(f"unknown trainer: {req.trainer} (expected one of {', '.join(EMITTERS)})")

    path_map = resolve_path_map(req.path_map)

    # One scan up front; reuse its resolved list and image list below instead of
    # re-walking the tree and re-stat()ing every row.
    info, rows, resolved, images = scan_export(export_dir, category=req.category)
    if info.image_count == 0:
        raise ForgeError(f"no images found under {export_dir} (supported: {', '.join(sorted(SUPPORTED_EXTS))})")

    warnings: list[str] = []
    if info.missing_from_disk:
        warnings.append(f"{info.missing_from_disk}/{info.manifest_rows} manifest rows have no exported image on disk")
    if info.manifest_present and info.manifest_rows != info.image_count:
        warnings.append(
            f"manifest lists {info.manifest_rows} images but {info.image_count} were found — forging for what's on disk"
        )

    collisions = exported_collisions(rows, resolved)
    for dest, rels in sorted(collisions.items()):
        try:
            shown = str(dest.relative_to(export_dir))
        except ValueError:  # a row's rel_path escaped the export dir
            shown = str(dest)
        note = f"basename collision: {', '.join(rels)} all resolve to {shown} — "
        note += "the pixels on disk could belong to any of them, so caption pairing is ambiguous "
        note += "(skipped caption collection for it"
        if caption_path(dest).exists():
            note += f"; the existing {caption_path(dest).name} may be mispaired — verify or delete it"
        note += "); re-export so selections land at distinct paths (argus-curator 2.x de-collides shared basenames; a 1.x export can preserve folder structure)"
        warnings.append(note)

    captions_collected = 0
    if req.collect_captions and rows:
        captions_collected = collect_captions(
            export_dir, rows, dry_run=req.dry_run, resolved=resolved, skip=set(collisions)
        )
        if captions_collected and req.dry_run:
            warnings.append(f"dry run: {captions_collected} caption sidecars would be collected from sources")

    # Collection only adds sidecars next to already-counted images, so refresh
    # the caption count in place rather than re-parsing the manifest and
    # re-walking the whole export dir.
    if captions_collected and not req.dry_run:
        info = info.model_copy(update={"caption_count": info.caption_count + captions_collected})

    if info.caption_count == 0:
        warnings.append("no .txt captions found — images will train on the trigger phrase alone")

    profile = info.target_profile
    backend = (profile.target_backend or "sdxl").lower()
    base_model = req.base_model or profile.checkpoint or DEFAULT_BASE_MODELS.get(backend)
    if base_model is None:
        base_model = DEFAULT_BASE_MODELS["sdxl"]
        warnings.append(
            f"no default base model for backend {backend!r} — wrote the SDXL base; override with base_model"
        )
    if backend != "sdxl":
        warnings.append(f"heuristics are tuned for SDXL; manifest targets {backend!r} — review lr/resolution")

    # A checkpoint from the manifest is a container path like everything else;
    # HF repo ids ("stabilityai/...") are not absolute and are left alone.
    base_model_mapped = base_model.startswith("/") and map_path(base_model, path_map) != base_model
    if base_model_mapped:
        base_model = map_path(base_model, path_map)

    trigger = req.trigger or slugify(export_dir.name)
    output_name = req.output_name or f"{slugify(export_dir.name)}-lora"
    params = apply_overrides(info.suggested, req.overrides)

    ctx = EmitContext(
        export_dir=export_dir,
        out_rel=f"{FORGE_DIR_NAME}/{req.trainer}",
        params=params,
        profile=profile,
        base_model=base_model,
        trigger=trigger,
        output_name=output_name,
        images=images,
        warnings=warnings,
        path_map=path_map,
    )
    if base_model_mapped:
        ctx.map_hits += 1  # count it so path_note() reports the remap honestly
    files = EMITTERS[req.trainer](ctx)

    if not req.dry_run:
        for f in files:
            target = export_dir / f.name
            if f.name == "metadata.jsonl" and target.exists():
                warnings.append("overwrote existing metadata.jsonl at the dataset root")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f.content, encoding="utf-8")
            if f.name.endswith(".sh"):
                target.chmod(target.stat().st_mode | 0o111)
            f.path = str(target)

    logger.info(
        "forge_done",
        export_dir=str(export_dir),
        trainer=req.trainer,
        files=[f.name for f in files],
        dry_run=req.dry_run,
        captions_collected=captions_collected,
    )

    return ForgeResult(
        trainer=req.trainer,
        export_dir=str(export_dir),
        out_dir=str(export_dir / FORGE_DIR_NAME / req.trainer),
        files=files,
        params=params,
        dataset=info,
        base_model=base_model,
        trigger=trigger,
        output_name=output_name,
        captions_collected=captions_collected,
        warnings=warnings,
    )

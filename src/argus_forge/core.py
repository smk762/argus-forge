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
from argus_forge.heuristics import apply_overrides
from argus_forge.manifest import (
    FORGE_DIR_NAME,
    caption_path,
    exported_collisions,
    exported_location,
    find_images,
    inspect_export,
)
from argus_forge.models import SUPPORTED_EXTS, ForgeError, ForgeRequest, ForgeResult, ManifestRow

logger = structlog.get_logger()

# Fallback base checkpoints per target_backend when the manifest carries none.
DEFAULT_BASE_MODELS = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
}


def slugify(name: str) -> str:
    """Directory name -> a safe trigger/output token (``My Set!`` -> ``my_set``)."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "dataset"


PATH_MAP_ENV = "FORGE_PATH_MAP"


def parse_path_map(spec: str) -> dict[str, str]:
    """Parse ``container=host[,container=host...]`` into a prefix map."""
    mapping: dict[str, str] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        src, sep, dst = pair.partition("=")
        if not sep or not src.strip() or not dst.strip():
            raise ForgeError(f"bad path_map entry {pair!r} — expected 'container/prefix=host/prefix'")
        mapping[src.strip()] = dst.strip()
    return mapping


def resolve_path_map(req_map: dict[str, str]) -> dict[str, str]:
    """The request's path_map merged over the FORGE_PATH_MAP env default."""
    return {**parse_path_map(os.environ.get(PATH_MAP_ENV, "")), **req_map}


def collect_captions(export_dir: Path, rows: list[ManifestRow], dry_run: bool, skip: set[Path] | None = None) -> int:
    """Copy ``.txt`` sidecars written next to the *source* images into the export.

    argus-lens captions the manifest's ``abs_path`` entries, so sidecars land
    beside the originals — not beside the exported copies trainers read. This
    closes that gap. Existing sidecars in the export are never overwritten,
    and files in *skip* (basename collisions — see :func:`exported_collisions`)
    are left uncaptioned rather than paired with a caption that may describe
    different pixels.
    """
    copied = 0
    for row in rows:
        src_caption = caption_path(Path(row.abs_path))
        if not src_caption.is_file():
            continue
        dest_img = exported_location(export_dir, row)
        if dest_img is None or (skip and dest_img in skip):
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
    export_dir = Path(req.export_dir).expanduser()
    if req.trainer not in EMITTERS:
        raise ForgeError(f"unknown trainer: {req.trainer} (expected one of {', '.join(EMITTERS)})")

    path_map = resolve_path_map(req.path_map)

    info, rows = inspect_export(export_dir, category=req.category)
    if info.image_count == 0:
        raise ForgeError(f"no images found under {export_dir} (supported: {', '.join(sorted(SUPPORTED_EXTS))})")

    warnings: list[str] = []
    if info.missing_from_disk:
        warnings.append(f"{info.missing_from_disk}/{info.manifest_rows} manifest rows have no exported image on disk")
    if info.manifest_present and info.manifest_rows != info.image_count:
        warnings.append(
            f"manifest lists {info.manifest_rows} images but {info.image_count} were found — forging for what's on disk"
        )

    collisions = exported_collisions(export_dir, rows)
    for dest, rels in sorted(collisions.items()):
        warnings.append(
            f"basename collision: {', '.join(rels)} all resolve to {dest.relative_to(export_dir)} — "
            "only one image survived the flattened export and its caption pairing is ambiguous "
            "(skipped caption collection for it); re-export with folder structure preserved"
        )

    captions_collected = 0
    if req.collect_captions and rows:
        captions_collected = collect_captions(export_dir, rows, dry_run=req.dry_run, skip=set(collisions))
        if captions_collected and req.dry_run:
            warnings.append(f"dry run: {captions_collected} caption sidecars would be collected from sources")

    # Re-inspect after collection so caption counts (and diffusers metadata) see them.
    if captions_collected and not req.dry_run:
        info, rows = inspect_export(export_dir, category=req.category)

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
        images=find_images(export_dir),
        warnings=warnings,
        path_map=path_map,
    )
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

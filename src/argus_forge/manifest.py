"""Read curator exports: the ``manifest.jsonl`` handoff plus what's on disk.

An export directory is what ``argus-curator``'s POST /export (or the /curate
ExportPanel) produces: the selected images (structure preserved or flattened),
optional ``.txt`` caption sidecars from argus-lens, and ``manifest.jsonl``.
Forge also degrades gracefully to a bare folder of images with no manifest —
category and checkpoint then come from CLI/API arguments or defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from argus_forge.heuristics import dataset_size_status, suggest_training_params
from argus_forge.models import (
    CAPTION_EXT,
    MANIFEST_VERSION,
    SUPPORTED_EXTS,
    DatasetInfo,
    ForgeError,
    ManifestRow,
    TargetCategory,
    TargetProfile,
)

logger = structlog.get_logger()

MANIFEST_NAME = "manifest.jsonl"

# Forge writes its output under <export_dir>/forge/<trainer>/; anything below
# that must not count as dataset content.
FORGE_DIR_NAME = "forge"


def _manifest_major(version: str) -> str:
    return version.split(".", 1)[0]


def read_manifest(path: Path) -> list[ManifestRow]:
    """Parse ``manifest.jsonl``, refusing an incompatible major version."""
    rows: list[ManifestRow] = []
    expected_major = _manifest_major(MANIFEST_VERSION)
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = ManifestRow.model_validate(json.loads(line))
            except Exception as exc:
                raise ForgeError(f"{path.name}:{lineno}: unreadable manifest row: {exc}") from exc
            if _manifest_major(row.manifest_version) != expected_major:
                raise ForgeError(
                    f"{path.name}:{lineno}: manifest_version {row.manifest_version} is not supported "
                    f"(this build understands {expected_major}.x) — upgrade argus-forge or re-export"
                )
            rows.append(row)
    return rows


def find_images(export_dir: Path) -> list[Path]:
    """Supported images under *export_dir*, skipping forge output and dotdirs.

    Broken symlinks (a symlink-mode export whose source moved) are excluded —
    ``is_file()`` follows the link.
    """
    images: list[Path] = []
    for p in sorted(export_dir.rglob("*")):
        rel = p.relative_to(export_dir)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if rel.parts and rel.parts[0] == FORGE_DIR_NAME:
            continue
        if p.suffix.lower() in SUPPORTED_EXTS and p.is_file():
            images.append(p)
    return images


def caption_path(image: Path) -> Path:
    """The ``.txt`` sidecar location for *image* (argus-lens / kohya convention)."""
    return image.with_suffix(CAPTION_EXT)


def exported_location(export_dir: Path, row: ManifestRow) -> Path | None:
    """Where *row*'s image landed inside the export dir, or None if absent.

    Exports write either ``<dest>/<rel_path>`` (structure preserved) or
    ``<dest>/<basename>`` (flattened) — probe both.
    """
    preserved = export_dir / row.rel_path
    if preserved.is_file():
        return preserved
    flattened = export_dir / Path(row.rel_path).name
    if flattened.is_file():
        return flattened
    return None


def exported_collisions(export_dir: Path, rows: list[ManifestRow]) -> dict[Path, list[str]]:
    """Exported files that two or more manifest rows resolve to.

    A flattened export collides when selections share a basename: the curator
    silently keeps the last-written pixels, so any caption pairing for that
    file is ambiguous. Maps each colliding exported file to the ``rel_path``
    of every row claiming it, in manifest order.
    """
    claims: dict[Path, list[str]] = {}
    for row in rows:
        dest = exported_location(export_dir, row)
        if dest is not None:
            claims.setdefault(dest, []).append(row.rel_path)
    return {dest: rels for dest, rels in claims.items() if len(rels) > 1}


def inspect_export(
    export_dir: Path,
    category: TargetCategory | None = None,
) -> tuple[DatasetInfo, list[ManifestRow]]:
    """Look at an export dir: images, captions, manifest, suggested params."""
    if not export_dir.is_dir():
        raise ForgeError(f"not a directory: {export_dir}")

    manifest_file = export_dir / MANIFEST_NAME
    rows = read_manifest(manifest_file) if manifest_file.is_file() else []

    images = find_images(export_dir)
    captions = sum(1 for img in images if caption_path(img).is_file())
    missing = sum(1 for row in rows if exported_location(export_dir, row) is None)

    profile = rows[0].target_profile.model_copy() if rows else TargetProfile()
    if category is not None:
        profile.target_category = category

    info = DatasetInfo(
        export_dir=str(export_dir),
        image_count=len(images),
        caption_count=captions,
        manifest_present=bool(rows),
        manifest_rows=len(rows),
        manifest_version=rows[0].manifest_version if rows else None,
        missing_from_disk=missing,
        target_profile=profile,
        size_hint=dataset_size_status(len(images), profile.target_category),
        suggested=suggest_training_params(len(images), profile.target_category),
    )
    logger.debug(
        "inspected_export",
        export_dir=str(export_dir),
        images=len(images),
        captions=captions,
        manifest_rows=len(rows),
    )
    return info, rows

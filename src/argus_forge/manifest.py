"""Read curator exports: the ``manifest.jsonl`` handoff plus what's on disk.

An export directory is what ``argus-curator``'s POST /export (or the /curate
ExportPanel) produces: the selected images (structure preserved or flattened),
optional ``.txt`` caption sidecars from argus-lens, and ``manifest.jsonl``.
Forge also degrades gracefully to a bare folder of images with no manifest —
category and checkpoint then come from CLI/API arguments or defaults.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePath, PurePosixPath

import structlog
from pydantic import ValidationError

from argus_forge.heuristics import dataset_size_status, suggest_training_params
from argus_forge.models import (
    CAPTION_EXT,
    MAJORS_REQUIRING_EXPORTED_PATH,
    SUPPORTED_EXTS,
    SUPPORTED_MANIFEST_MAJORS,
    DatasetInfo,
    ForgeError,
    ManifestRow,
    TargetCategory,
    TargetProfile,
    manifest_major,
)

logger = structlog.get_logger()

MANIFEST_NAME = "manifest.jsonl"

# Forge writes its output under <export_dir>/forge/<trainer>/; anything below
# that must not count as dataset content.
FORGE_DIR_NAME = "forge"


def read_manifest(path: Path) -> list[ManifestRow]:
    """Parse ``manifest.jsonl``, rejecting a manifest forge cannot trust.

    Each failure names the offending line: a row whose major version is not in
    :data:`SUPPORTED_MANIFEST_MAJORS`; a row that breaks the
    :class:`ManifestRow` contract (a version that requires ``exported_path`` but
    omits it, or an ``exported_path`` that is empty/absolute/escapes the export
    root — validated on the model); and a file that mixes major versions, which
    is a corrupt concatenation rather than one handoff.
    """
    rows: list[ManifestRow] = []
    understood = ", ".join(f"{m}.x" for m in SUPPORTED_MANIFEST_MAJORS)
    first_major: str | None = None
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = ManifestRow.model_validate(json.loads(line))
            except ValidationError as exc:
                detail = "; ".join(e["msg"].removeprefix("Value error, ") for e in exc.errors())
                raise ForgeError(f"{path.name}:{lineno}: {detail}") from exc
            except Exception as exc:
                raise ForgeError(f"{path.name}:{lineno}: unreadable manifest row: {exc}") from exc
            major = manifest_major(row.manifest_version)
            if major not in SUPPORTED_MANIFEST_MAJORS:
                raise ForgeError(
                    f"{path.name}:{lineno}: manifest_version {row.manifest_version} is not supported "
                    f"(this build understands {understood}) — upgrade argus-forge or re-export"
                )
            if first_major is None:
                first_major = major
            elif major != first_major:
                raise ForgeError(
                    f"{path.name}:{lineno}: manifest_version {row.manifest_version} mixes major {major}.x with "
                    f"the file's earlier {first_major}.x rows — a manifest must be one version; re-export"
                )
            rows.append(row)
    return rows


def _is_dataset_member(rel: PurePath) -> bool:
    """Whether *rel* (a path relative to the export dir) is one forge would train
    on: a supported image, not inside a dotdir, not under forge's own output.

    Shared by :func:`find_images` (what's on disk) and :func:`exported_location`
    (where a row resolves) so the two views can never disagree about whether a
    file counts — otherwise a row could resolve to a file the trainer never sees.
    """
    if any(part.startswith(".") for part in rel.parts):
        return False
    if rel.parts and rel.parts[0] == FORGE_DIR_NAME:
        return False
    return rel.suffix.lower() in SUPPORTED_EXTS


def find_images(export_dir: Path) -> list[Path]:
    """Supported images under *export_dir*, skipping forge output and dotdirs.

    Broken symlinks (a symlink-mode export whose source moved) are excluded —
    ``is_file()`` follows the link.
    """
    images: list[Path] = []
    for p in sorted(export_dir.rglob("*")):
        if _is_dataset_member(p.relative_to(export_dir)) and p.is_file():
            images.append(p)
    return images


def caption_path(image: Path) -> Path:
    """The ``.txt`` sidecar location for *image* (argus-lens / kohya convention)."""
    return image.with_suffix(CAPTION_EXT)


def exported_location(export_dir: Path, row: ManifestRow) -> Path | None:
    """Where *row*'s image landed inside the export dir, or None if absent.

    Resolution is chosen by the *major version*, not by whether ``exported_path``
    happens to be set: a row whose major is in
    :data:`MAJORS_REQUIRING_EXPORTED_PATH` carries ``exported_path`` — the exact
    destination the curator wrote (flattened exports de-collide shared basenames
    to ``stem-<hash>.ext``, so it cannot be re-derived from ``rel_path``) — and
    is resolved from it with no probing; a miss means the file has since gone
    from disk. Older rows predate the field: exports wrote either
    ``<dest>/<rel_path>`` (structure preserved) or ``<dest>/<basename>``
    (flattened) — probe both.

    The exported_path branch also requires the destination to be a dataset
    member (:func:`_is_dataset_member`), so a row can never resolve to a file
    that :func:`find_images` would exclude and the trainer would never see.
    """
    if manifest_major(row.manifest_version) in MAJORS_REQUIRING_EXPORTED_PATH:
        if row.exported_path is None:  # unreachable: guaranteed by ManifestRow validation
            return None
        exported = export_dir / row.exported_path
        if exported.is_file() and _is_dataset_member(PurePosixPath(row.exported_path)):
            return exported
        return None
    preserved = export_dir / row.rel_path
    if preserved.is_file():
        return preserved
    flattened = export_dir / Path(row.rel_path).name
    if flattened.is_file():
        return flattened
    return None


def resolve_rows(export_dir: Path, rows: list[ManifestRow]) -> list[Path | None]:
    """:func:`exported_location` for every row, in manifest order.

    Resolution costs 1-2 stat() calls per row, so callers that need both the
    collision set and caption destinations should resolve once and share.
    """
    return [exported_location(export_dir, row) for row in rows]


def exported_collisions(rows: list[ManifestRow], resolved: list[Path | None]) -> dict[Path, list[str]]:
    """Exported files that two or more *distinct* manifest rows resolve to.

    A 1.x flattened export collides when selections share a basename: the
    curator silently kept the last-written pixels, so any caption pairing for
    that file is ambiguous. 2.x exports de-collide destinations curator-side,
    so this is a defensive guard for 1.x and hand-built export dirs.

    Maps each colliding exported file to the ``rel_path`` of every row claiming
    it, in manifest order. Rows with an identical ``rel_path`` are one selection
    listed twice, not a collision — they are deduplicated, not flagged.
    """
    claims: dict[Path, dict[str, None]] = {}
    for row, dest in zip(rows, resolved, strict=True):
        if dest is not None:
            claims.setdefault(dest, {})[row.rel_path] = None
    return {dest: list(rels) for dest, rels in claims.items() if len(rels) > 1}


def scan_export(
    export_dir: Path,
    category: TargetCategory | None = None,
) -> tuple[DatasetInfo, list[ManifestRow], list[Path | None], list[Path]]:
    """One pass over an export dir: the derived :class:`DatasetInfo` plus the
    intermediates it is built from — ``resolved`` (row -> disk path, in manifest
    order) and ``images`` (on-disk dataset files). Returning them lets a caller
    that needs both (:func:`argus_forge.core.forge_config`) avoid re-walking the
    tree and re-stat()ing every row; :func:`inspect_export` is the plain
    ``(info, rows)`` view for callers that don't.
    """
    if not export_dir.is_dir():
        raise ForgeError(f"not a directory: {export_dir}")

    manifest_file = export_dir / MANIFEST_NAME
    rows = read_manifest(manifest_file) if manifest_file.is_file() else []

    images = find_images(export_dir)
    captions = sum(1 for img in images if caption_path(img).is_file())
    resolved = resolve_rows(export_dir, rows)

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
        missing_from_disk=resolved.count(None),
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
    return info, rows, resolved, images


def inspect_export(
    export_dir: Path,
    category: TargetCategory | None = None,
) -> tuple[DatasetInfo, list[ManifestRow]]:
    """Look at an export dir: images, captions, manifest, suggested params."""
    info, rows, _, _ = scan_export(export_dir, category)
    return info, rows

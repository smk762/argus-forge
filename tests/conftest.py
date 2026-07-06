from __future__ import annotations

import base64
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from argus_forge.models import MAJORS_REQUIRING_EXPORTED_PATH, manifest_major

# A real 1x1 transparent PNG. Forge never decodes images (it only matches
# extensions), but a valid file keeps fixtures honest.
PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)

ExportFactory = Callable[..., Path]


@pytest.fixture(autouse=True)
def _isolate_path_map_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """forge_config reads FORGE_PATH_MAP; a value exported in the developer's
    shell (as the README suggests for serve) must not leak into tests."""
    monkeypatch.delenv("FORGE_PATH_MAP", raising=False)


@pytest.fixture
def export_factory(tmp_path: Path) -> ExportFactory:
    """Build a curator-shaped export dir: images, optional sidecars, manifest.

    ``captions`` sidecars are written next to the *exported* images;
    ``source_captions`` next to the *source* images only (exercising the
    collect-captions gap that argus-lens leaves).
    """

    def make(
        n: int = 20,
        captions: int = 0,
        source_captions: int = 0,
        manifest: bool = True,
        category: str = "identity",
        checkpoint: str | None = None,
        preserve_structure: bool = False,
        manifest_version: str = "2.0",
        name: str = "myset",
    ) -> Path:
        export = tmp_path / name
        export.mkdir(parents=True, exist_ok=True)
        source = tmp_path / "source"
        source.mkdir(exist_ok=True)

        rows = []
        for i in range(n):
            rel = f"sub/img_{i:03d}.png" if preserve_structure else f"img_{i:03d}.png"
            src = source / f"img_{i:03d}.png"
            src.write_bytes(PNG_1PX)
            if i < source_captions:
                src.with_suffix(".txt").write_text(f"source caption {i}", encoding="utf-8")

            dest = export / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(PNG_1PX)
            if i < captions:
                dest.with_suffix(".txt").write_text(f"caption {i}", encoding="utf-8")

            row = {
                "manifest_version": manifest_version,
                "rel_path": rel,
                "abs_path": str(src),
                "target_profile": {
                    "target_style": "photo",
                    "target_backend": "sdxl",
                    "checkpoint": checkpoint,
                    "target_category": category,
                },
                "primary_face_cluster": None,
                "primary_face_pose": None,
                "score": 0.9,
                "similar_group": i,
            }
            # Majors that require it carry the real destination; 1.x predates it.
            # Uses the same rule as production so the fixture can't drift from it.
            if manifest_major(manifest_version) in MAJORS_REQUIRING_EXPORTED_PATH:
                row["exported_path"] = rel
            rows.append(row)

        if manifest:
            lines = "\n".join(json.dumps(r) for r in rows)
            (export / "manifest.jsonl").write_text(lines + "\n", encoding="utf-8")
        return export

    return make


def decollided_export(tmp_path: Path, with_source_captions: bool = False) -> Path:
    """A manifest-2.x flattened export the curator had to de-collide.

    Two selections shared a basename, so rel_paths ``a/IMG_0001.png`` and
    ``b/IMG_0001.png`` land at *distinct* ``exported_path`` destinations
    ``IMG_0001.png`` and ``IMG_0001-9fc3d2.png``. This is the shape where
    exported_path resolution diverges from the rel_path probe (the whole point
    of manifest 2.x), which ``export_factory`` — writing exported_path == rel_path
    — cannot express, so the resolver and emitter tests both build it here.
    """
    export = tmp_path / "flat2"
    export.mkdir()
    rows = []
    for sub, exported, caption in (
        ("a", "IMG_0001.png", "caption a"),
        ("b", "IMG_0001-9fc3d2.png", "caption b"),
    ):
        (export / exported).write_bytes(PNG_1PX)
        src = tmp_path / "sources" / sub / "IMG_0001.png"
        if with_source_captions:
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_bytes(PNG_1PX)
            src.with_suffix(".txt").write_text(caption, encoding="utf-8")
        rows.append(
            {
                "manifest_version": "2.0",
                "rel_path": f"{sub}/IMG_0001.png",
                "abs_path": str(src),
                "exported_path": exported,
            }
        )
    (export / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return export

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from pathlib import Path

import pytest

# A real 1x1 transparent PNG. Forge never decodes images (it only matches
# extensions), but a valid file keeps fixtures honest.
PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)

ExportFactory = Callable[..., Path]


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
        manifest_version: str = "1.0",
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

            rows.append(
                {
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
            )

        if manifest:
            lines = "\n".join(json.dumps(r) for r in rows)
            (export / "manifest.jsonl").write_text(lines + "\n", encoding="utf-8")
        return export

    return make

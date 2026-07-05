"""Emitter parity guard: the kohya emitter vs. checked-in golden files.

argus-studio's demo mode ships a hand-maintained twin of this emitter
(``frontend/src/components/curator/forgeDemo.ts``) that renders the same
kohya TOML client-side. These goldens are the shared reference: they are
rendered with ``path_map`` pointing the export at ``/data/out`` — the demo's
placeholder image dir — so the two outputs stay diffable key-for-key.

Same spirit as test_heuristics.py's "parity cases mirroring
suggestTrainingParams": if a golden here changes, sync forgeDemo.ts.

Regenerate after an intentional emitter change with:

    UPDATE_GOLDEN=1 uv run --no-sync pytest tests/test_kohya_parity.py
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest
from conftest import ExportFactory

from argus_forge.core import forge_config
from argus_forge.models import ForgeRequest

GOLDEN_DIR = Path(__file__).parent / "golden" / "kohya"

# The demo builder's DEMO_IMAGE_DIR — goldens are rendered as if the export
# lived there so they carry no tmp-path noise.
DEMO_IMAGE_DIR = "/data/out"

# (image count, category) pairs, mirroring the heuristics parity cases.
CASES = [(27, "identity"), (20, "pose_composition"), (30, "setting")]


def _render(export_factory: ExportFactory, count: int, category: str) -> dict[str, str]:
    export = export_factory(n=count, category=category)
    result = forge_config(
        ForgeRequest(
            export_dir=str(export),
            trainer="kohya",
            dry_run=True,
            collect_captions=False,
            path_map={str(export): DEMO_IMAGE_DIR},
        )
    )
    by_basename = {f.name.rsplit("/", 1)[-1]: f.content for f in result.files}
    return {name: by_basename[name] for name in ("dataset.toml", "config.toml")}


@pytest.mark.parametrize(("count", "category"), CASES)
def test_kohya_emitter_matches_golden(export_factory: ExportFactory, count: int, category: str) -> None:
    rendered = _render(export_factory, count, category)
    update = os.environ.get("UPDATE_GOLDEN", "").lower() in ("1", "true", "yes")
    for name, content in rendered.items():
        golden_path = GOLDEN_DIR / f"{category}_{count}_{name}"
        if update:
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(content, encoding="utf-8")
        assert golden_path.is_file(), f"missing golden {golden_path.name} — run with UPDATE_GOLDEN=1"
        golden = golden_path.read_text(encoding="utf-8")
        tomllib.loads(content)  # whatever else drifts, the emitter must render valid TOML
        assert content == golden, (
            f"{golden_path.name} drifted from the emitter. If the emitter change is intentional, "
            "regenerate with UPDATE_GOLDEN=1 and sync argus-studio's forgeDemo.ts to match."
        )

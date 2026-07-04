from __future__ import annotations

from pathlib import Path

import pytest
from conftest import PNG_1PX, ExportFactory

from argus_forge.manifest import find_images, inspect_export, read_manifest
from argus_forge.models import ForgeError


def test_inspect_with_manifest(export_factory: ExportFactory) -> None:
    export = export_factory(n=27, captions=5, category="identity")
    info, rows = inspect_export(export)
    assert info.image_count == 27
    assert info.caption_count == 5
    assert info.manifest_present and info.manifest_rows == 27
    assert info.manifest_version == "1.0"
    assert info.missing_from_disk == 0
    assert info.target_profile.target_category == "identity"
    assert info.suggested.images == 27
    assert info.suggested.repeats == 6
    assert len(rows) == 27


def test_inspect_bare_folder(export_factory: ExportFactory) -> None:
    export = export_factory(n=8, manifest=False)
    info, rows = inspect_export(export)
    assert not info.manifest_present
    assert info.manifest_version is None
    assert rows == []
    assert info.target_profile.target_category == "identity"  # default profile
    assert info.size_hint.tone == "low"


def test_inspect_category_override(export_factory: ExportFactory) -> None:
    export = export_factory(n=20, category="identity")
    info, _ = inspect_export(export, category="setting")
    assert info.target_profile.target_category == "setting"
    assert info.suggested.network_dim == 32


def test_inspect_missing_files_counted(export_factory: ExportFactory) -> None:
    export = export_factory(n=10)
    (export / "img_000.png").unlink()
    info, _ = inspect_export(export)
    assert info.image_count == 9
    assert info.missing_from_disk == 1


def test_inspect_preserved_structure(export_factory: ExportFactory) -> None:
    export = export_factory(n=6, preserve_structure=True)
    info, _ = inspect_export(export)
    assert info.image_count == 6
    assert info.missing_from_disk == 0


def test_manifest_major_version_rejected(export_factory: ExportFactory) -> None:
    export = export_factory(n=3, manifest_version="2.0")
    with pytest.raises(ForgeError, match="manifest_version 2.0"):
        inspect_export(export)


def test_manifest_minor_version_accepted(export_factory: ExportFactory) -> None:
    export = export_factory(n=3, manifest_version="1.7")
    info, _ = inspect_export(export)
    assert info.manifest_version == "1.7"


def test_manifest_bad_row_reports_line(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text('{"manifest_version": "1.0", "rel_path": "a.png", "abs_path": "/a.png"}\nnot json\n')
    with pytest.raises(ForgeError, match="manifest.jsonl:2"):
        read_manifest(path)


def test_find_images_skips_forge_output_and_dotdirs(export_factory: ExportFactory) -> None:
    export = export_factory(n=4)
    (export / "forge" / "kohya").mkdir(parents=True)
    (export / "forge" / "kohya" / "sample.png").write_bytes(PNG_1PX)
    (export / ".cache").mkdir()
    (export / ".cache" / "thumb.png").write_bytes(PNG_1PX)
    assert len(find_images(export)) == 4

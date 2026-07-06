from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import PNG_1PX, ExportFactory, decollided_export
from pydantic import ValidationError

from argus_forge.manifest import find_images, inspect_export, read_manifest, resolve_rows, scan_export
from argus_forge.models import ForgeError, ManifestRow


def test_inspect_with_manifest(export_factory: ExportFactory) -> None:
    export = export_factory(n=27, captions=5, category="identity")
    info, rows = inspect_export(export)
    assert info.image_count == 27
    assert info.caption_count == 5
    assert info.manifest_present and info.manifest_rows == 27
    assert info.manifest_version == "2.0"
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
    export = export_factory(n=3, manifest_version="3.0")
    with pytest.raises(ForgeError, match="manifest_version 3.0"):
        inspect_export(export)


def test_manifest_minor_version_accepted(export_factory: ExportFactory) -> None:
    export = export_factory(n=3, manifest_version="2.7")
    info, _ = inspect_export(export)
    assert info.manifest_version == "2.7"


def test_manifest_legacy_1x_accepted(export_factory: ExportFactory) -> None:
    """1.x rows have no exported_path; destinations come from the rel_path probes."""
    export = export_factory(n=4, manifest_version="1.7")
    info, rows = inspect_export(export)
    assert info.manifest_version == "1.7"
    assert all(row.exported_path is None for row in rows)
    assert info.missing_from_disk == 0


def test_manifest_2x_row_without_exported_path_rejected(tmp_path: Path) -> None:
    export = tmp_path / "bad2x"
    export.mkdir()
    (export / "a.png").write_bytes(PNG_1PX)
    row = {"manifest_version": "2.0", "rel_path": "a.png", "abs_path": "/src/a.png"}
    (export / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ForgeError, match="manifest.jsonl:1.*exported_path"):
        inspect_export(export)


def test_exported_path_resolves_decollided_flattened_export(tmp_path: Path) -> None:
    """2.x flattened exports de-collide shared basenames to stem-<hash>.ext;
    resolution must follow exported_path, not re-derive from rel_path."""
    export = decollided_export(tmp_path)
    info, parsed = inspect_export(export)
    assert info.missing_from_disk == 0
    resolved = resolve_rows(export, parsed)
    assert [p.name for p in resolved] == ["IMG_0001.png", "IMG_0001-9fc3d2.png"]


def test_exported_path_miss_is_not_probed(tmp_path: Path) -> None:
    """A 2.x row whose exported_path is gone counts missing even if a file
    matching the legacy rel_path derivation exists — no fallback probing."""
    export = tmp_path / "gone"
    export.mkdir()
    (export / "IMG_0001.png").write_bytes(PNG_1PX)  # matches basename(rel_path), not exported_path
    row = {
        "manifest_version": "2.0",
        "rel_path": "b/IMG_0001.png",
        "abs_path": "/src/b/IMG_0001.png",
        "exported_path": "IMG_0001-9fc3d2.png",
    }
    (export / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    info, _ = inspect_export(export)
    assert info.missing_from_disk == 1


def test_manifest_2x_empty_exported_path_rejected(tmp_path: Path) -> None:
    """An empty exported_path is malformed, not a silent miss: '' would resolve
    to the export dir itself and the row would be undercounted as absent."""
    export = tmp_path / "empty2x"
    export.mkdir()
    (export / "a.png").write_bytes(PNG_1PX)
    row = {"manifest_version": "2.0", "rel_path": "a.png", "abs_path": "/src/a.png", "exported_path": ""}
    (export / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ForgeError, match="manifest.jsonl:1.*exported_path is empty"):
        inspect_export(export)


@pytest.mark.parametrize("bad", ["/abs/IMG.png", "../sibling/IMG.png", "sub/../../IMG.png"])
def test_manifest_2x_exported_path_escape_rejected(tmp_path: Path, bad: str) -> None:
    """An absolute or ``..`` exported_path escapes the export root; it is rejected
    at read time so a caption is never written outside the export dir."""
    export = tmp_path / "escape2x"
    export.mkdir()
    (export / "IMG.png").write_bytes(PNG_1PX)
    row = {"manifest_version": "2.0", "rel_path": "IMG.png", "abs_path": "/src/IMG.png", "exported_path": bad}
    (export / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ForgeError, match="manifest.jsonl:1.*relative path inside the export root"):
        inspect_export(export)


def test_manifest_mixed_major_versions_rejected(tmp_path: Path) -> None:
    """A handoff is one version; a file mixing majors is a corrupt concatenation
    and must be refused rather than resolved row-by-row with mixed strategies."""
    export = tmp_path / "mixed"
    export.mkdir()
    (export / "a.png").write_bytes(PNG_1PX)
    (export / "b.png").write_bytes(PNG_1PX)
    rows = [
        {"manifest_version": "1.0", "rel_path": "a.png", "abs_path": "/src/a.png"},
        {"manifest_version": "2.0", "rel_path": "b.png", "abs_path": "/src/b.png", "exported_path": "b.png"},
    ]
    (export / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    with pytest.raises(ForgeError, match="manifest.jsonl:2.*mixes major"):
        inspect_export(export)


def test_exported_path_outside_dataset_view_counts_missing(tmp_path: Path) -> None:
    """resolve_rows must agree with find_images: an exported_path landing under
    forge/ (or a dotdir, or a non-image) would be dropped from training, so it
    is reported missing rather than counted present."""
    export = tmp_path / "oddloc"
    export.mkdir()
    (export / "forge").mkdir()
    (export / "forge" / "x.png").write_bytes(PNG_1PX)  # excluded by find_images
    (export / "keep.png").write_bytes(PNG_1PX)
    rows = [
        {"manifest_version": "2.0", "rel_path": "keep.png", "abs_path": "/s/keep.png", "exported_path": "keep.png"},
        {"manifest_version": "2.0", "rel_path": "x.png", "abs_path": "/s/x.png", "exported_path": "forge/x.png"},
    ]
    (export / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    info, _ = inspect_export(export)
    assert info.image_count == 1  # find_images excludes forge/
    assert info.missing_from_disk == 1  # the forge/ row does not count as present


def test_manifest_1x_row_with_exported_path_is_probed_not_read(tmp_path: Path) -> None:
    """Resolution is version-based: a 1.x row is probed from rel_path even if it
    carries an exported_path value (the field is a 2.x concept)."""
    export = tmp_path / "legacy_ep"
    export.mkdir()
    (export / "IMG.png").write_bytes(PNG_1PX)  # present at basename(rel_path)
    row = {
        "manifest_version": "1.0",
        "rel_path": "x/IMG.png",
        "abs_path": "/src/x/IMG.png",
        "exported_path": "nonexistent.png",  # would resolve missing if it were read
    }
    (export / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    info, _ = inspect_export(export)
    assert info.missing_from_disk == 0  # probed rel_path basename; ignored exported_path


def test_manifest_1x_nested_preserved_structure_resolves(export_factory: ExportFactory) -> None:
    """1.x nested rel_paths resolve via the preserve-structure probe; guards the
    legacy branch the now-2.0 preserve-structure fixtures no longer exercise."""
    export = export_factory(n=4, preserve_structure=True, manifest_version="1.7")
    info, rows = inspect_export(export)
    assert all(row.exported_path is None for row in rows)
    assert info.image_count == 4
    assert info.missing_from_disk == 0


def test_scan_export_exposes_reusable_resolved_and_images(export_factory: ExportFactory) -> None:
    """scan_export returns the resolved + images it already built (so forge_config
    resolves and walks the tree once), and its summary view equals inspect_export."""
    export = export_factory(n=5, captions=2)
    info, rows, resolved, images = scan_export(export)
    assert len(resolved) == len(rows) == 5
    assert images == find_images(export)
    assert info.missing_from_disk == resolved.count(None)
    assert (info, rows) == inspect_export(export)


def test_manifest_row_model_enforces_exported_path_contract() -> None:
    """The invariant lives on ManifestRow, so direct construction / API
    deserialization cannot produce a malformed 2.x row that later code trusts."""
    with pytest.raises(ValidationError):  # 2.x requires exported_path
        ManifestRow(manifest_version="2.0", rel_path="a.png", abs_path="/a.png")
    with pytest.raises(ValidationError):  # exported_path may not escape the root
        ManifestRow(manifest_version="2.0", rel_path="a.png", abs_path="/a.png", exported_path="../a.png")
    # A 1.x row without the field is valid.
    assert ManifestRow(manifest_version="1.0", rel_path="a.png", abs_path="/a.png").exported_path is None


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

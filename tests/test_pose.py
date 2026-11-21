"""The pose-dominance warning, from curator manifest 2.7's ``pose_cluster`` / ``pose_labels``."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from conftest import PNG_1PX, ExportFactory
from typer.testing import CliRunner

from argus_forge.cli import app
from argus_forge.core import forge_config
from argus_forge.manifest import read_manifest
from argus_forge.models import TARGET_CATEGORIES, ForgeRequest
from argus_forge.pose import POSE_DOMINANCE_SHARE, POSE_MIN_CLUSTERED, pose_dominance_warning

STANDING = {"stance": "standing", "arms": "down", "facing": "front"}
UNKNOWN = {"stance": "unknown", "arms": "unknown", "facing": "unknown"}
MARK = "share one body pose"


def _export(tmp_path: Path, poses: list[tuple[int | None, dict | None]], category: str = "identity") -> Path:
    """A 2.7 export: one image per ``(pose_cluster, pose_labels)``, written to a hand-built manifest."""
    export = tmp_path / "set"
    export.mkdir(parents=True)
    rows = []
    for i, (cluster, labels) in enumerate(poses):
        name = f"img_{i:03d}.png"
        (export / name).write_bytes(PNG_1PX)
        rows.append(
            {
                "manifest_version": "2.7",
                "rel_path": name,
                "abs_path": f"/source/{name}",
                "exported_path": name,
                "target_profile": {"target_style": "photo", "target_backend": "sdxl", "target_category": category},
                "score": 0.9,
                "similar_group": i,
                "pose_labels": labels,
                "pose_cluster": cluster,
            }
        )
    (export / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return export


def _forge(export: Path) -> list[str]:
    req = ForgeRequest(export_dir=str(export), trainer="kohya", dry_run=True, collect_captions=False)
    return forge_config(req).warnings


def _pose_warnings(export: Path) -> list[str]:
    return [w for w in _forge(export) if MARK in w]


def test_cut_offs_are_the_suite_values() -> None:
    assert POSE_MIN_CLUSTERED == 10
    assert POSE_DOMINANCE_SHARE == 0.75


def test_old_manifest_without_pose_fields_reads_null_and_never_warns(export_factory: ExportFactory) -> None:
    export = export_factory(n=40)
    rows = read_manifest(export / "manifest.jsonl")
    assert all(r.pose_cluster is None and r.pose_labels is None for r in rows)
    assert _pose_warnings(export) == []


def test_below_the_row_minimum_does_not_warn(tmp_path: Path) -> None:
    export = _export(tmp_path, [(0, STANDING)] * (POSE_MIN_CLUSTERED - 1))
    assert _pose_warnings(export) == []


def test_at_the_row_minimum_warns(tmp_path: Path) -> None:
    export = _export(tmp_path, [(0, STANDING)] * POSE_MIN_CLUSTERED)
    assert _pose_warnings(export) == [
        "10 of 10 posed training images share one body pose (standing, arms down, facing front); "
        "outputs are likely to collapse onto it — rebalance the selection in curator"
    ]


def test_just_under_the_threshold_does_not_warn(tmp_path: Path) -> None:
    # 14/20 = 0.70 < 0.75
    export = _export(tmp_path, [(0, STANDING)] * 14 + [(1, None)] * 3 + [(2, None)] * 3)
    assert _pose_warnings(export) == []


def test_at_the_threshold_warns(tmp_path: Path) -> None:
    # 15/20 = 0.75, the cut-off itself
    export = _export(tmp_path, [(0, STANDING)] * 15 + [(1, None)] * 3 + [(2, None)] * 2)
    (note,) = _pose_warnings(export)
    assert note.startswith("15 of 20 posed training images share one body pose (standing, arms down, facing front)")


def test_the_dominant_cluster_need_not_be_zero(tmp_path: Path) -> None:
    # Ids are scan-level: the export's biggest cluster may be any id.
    export = _export(tmp_path, [(3, STANDING)] * 12 + [(0, None)] * 2)
    (note,) = _pose_warnings(export)
    assert note.startswith("12 of 14 ")


def test_null_clusters_are_excluded_and_reported(tmp_path: Path) -> None:
    # 12 clustered, all one pose; 20 with no measured pose are neither counted nor dilute the share.
    export = _export(tmp_path, [(0, STANDING)] * 12 + [(None, UNKNOWN)] * 10 + [(None, None)] * 10)
    (note,) = _pose_warnings(export)
    assert note.startswith("12 of 12 posed training images")
    assert note.endswith("(20 of 32 images had no measured pose and are not counted)")


def test_null_clusters_do_not_count_towards_the_minimum(tmp_path: Path) -> None:
    export = _export(tmp_path, [(0, STANDING)] * (POSE_MIN_CLUSTERED - 1) + [(None, STANDING)] * 30)
    assert _pose_warnings(export) == []


def test_a_small_unmeasured_share_is_not_mentioned(tmp_path: Path) -> None:
    export = _export(tmp_path, [(0, STANDING)] * 20 + [(None, None)] * 2)
    (note,) = _pose_warnings(export)
    assert "no measured pose" not in note


def test_labels_all_unknown_or_null_name_no_pose(tmp_path: Path) -> None:
    export = _export(tmp_path, [(0, UNKNOWN)] * 6 + [(0, None)] * 6)
    assert _pose_warnings(export) == [
        "12 of 12 posed training images share one body pose; "
        "outputs are likely to collapse onto it — rebalance the selection in curator"
    ]


def test_mixed_labels_name_the_most_common_known_value_of_each(tmp_path: Path) -> None:
    poses = (
        [(0, {"stance": "seated", "arms": "unknown", "facing": "left"})] * 4
        + [(0, {"stance": "unknown", "arms": "up", "facing": "unknown"})] * 5
        + [(0, {"stance": "standing", "arms": "unknown", "facing": "unknown"})] * 3
        # Another cluster's labels must not leak into the dominant cluster's description.
        + [(1, {"stance": "lying", "arms": "out", "facing": "back"})] * 3
    )
    (note,) = _pose_warnings(_export(tmp_path, poses))
    assert note.startswith("12 of 15 posed training images share one body pose (seated, arms up, facing left);")


def test_label_values_and_keys_forge_does_not_know_are_tolerated(tmp_path: Path) -> None:
    odd = {"stance": "kneeling", "arms": "down", "facing": "front", "torso": "twisted"}
    (note,) = _pose_warnings(_export(tmp_path, [(0, odd)] * 10))
    assert "(kneeling, arms down, facing front)" in note


def test_rows_missing_from_disk_are_not_counted(tmp_path: Path) -> None:
    export = _export(tmp_path, [(0, STANDING)] * 10 + [(1, None)] * 4)
    (export / "img_000.png").unlink()  # 9 of 13 on disk in cluster 0: 0.69, under the cut-off
    assert _pose_warnings(export) == []


@pytest.mark.parametrize("category", TARGET_CATEGORIES)
def test_every_category_warns(tmp_path: Path, category: str) -> None:
    assert len(_pose_warnings(_export(tmp_path, [(0, STANDING)] * 20, category=category))) == 1


def test_the_warning_changes_nothing_else(tmp_path: Path) -> None:
    """Same images, with and without a dominant pose: identical files and params, one extra warning."""
    export = _export(tmp_path, [(0, STANDING)] * 20)
    req = ForgeRequest(export_dir=str(export), trainer="kohya", dry_run=True, collect_captions=False)
    with_pose = forge_config(req)
    shutil.rmtree(export)  # the same path again, so every rendered path matches
    _export(tmp_path, [(None, None)] * 20)
    without = forge_config(req)
    assert with_pose.params == without.params
    assert [(f.name, f.content) for f in with_pose.files] == [(f.name, f.content) for f in without.files]
    assert [w for w in with_pose.warnings if MARK not in w] == without.warnings
    assert len(with_pose.warnings) == len(without.warnings) + 1


def test_pure_function_accepts_no_rows() -> None:
    assert pose_dominance_warning([]) is None


def test_cli_prints_the_warning(tmp_path: Path) -> None:
    export = _export(tmp_path, [(0, STANDING)] * 10)
    result = CliRunner().invoke(app, ["config", str(export), "--trainer", "kohya", "--dry-run"])
    assert result.exit_code == 0
    assert f"! 10 of 10 posed training images {MARK}" in result.output

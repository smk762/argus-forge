from __future__ import annotations

import json
from pathlib import Path

from conftest import ExportFactory
from typer.testing import CliRunner

from argus_forge.cli import app

runner = CliRunner()


def test_trainers_lists_all() -> None:
    result = runner.invoke(app, ["trainers"])
    assert result.exit_code == 0
    for trainer in ("kohya", "onetrainer", "diffusers"):
        assert trainer in result.output


def test_inspect(export_factory: ExportFactory) -> None:
    export = export_factory(n=27, captions=3)
    result = runner.invoke(app, ["inspect", str(export)])
    assert result.exit_code == 0
    assert "27 (3 captioned)" in result.output
    assert "manifest: v2.0" in result.output


def test_inspect_json(export_factory: ExportFactory) -> None:
    export = export_factory(n=5)
    result = runner.invoke(app, ["inspect", str(export), "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output)["image_count"] == 5


def test_config_dry_run(export_factory: ExportFactory) -> None:
    export = export_factory(n=10)
    result = runner.invoke(app, ["config", str(export), "--trainer", "kohya", "--dry-run"])
    assert result.exit_code == 0
    assert "dataset.toml" in result.output
    assert not (export / "forge").exists()


def test_config_writes_and_reports(export_factory: ExportFactory) -> None:
    export = export_factory(n=10)
    result = runner.invoke(app, ["config", str(export), "--trainer", "kohya", "--trigger", "zxq"])
    assert result.exit_code == 0
    assert "wrote" in result.output
    assert (export / "forge/kohya/train.sh").exists()


def test_config_missing_dir_fails(tmp_path: Path) -> None:
    result = runner.invoke(app, ["config", str(tmp_path / "missing")])
    assert result.exit_code == 1
    assert "Error:" in result.output


def test_config_bad_trainer_fails(export_factory: ExportFactory) -> None:
    export = export_factory(n=3)
    result = runner.invoke(app, ["config", str(export), "--trainer", "nope"])
    assert result.exit_code == 1


def test_schema_write_and_check(tmp_path: Path) -> None:
    out = tmp_path / "schema.json"
    assert runner.invoke(app, ["schema", "--output", str(out)]).exit_code == 0
    schema = json.loads(out.read_text())
    assert schema["title"] == "argus-forge wire contract"
    assert "ForgeResult" in schema["$defs"]
    assert runner.invoke(app, ["schema", "--output", str(out), "--check"]).exit_code == 0
    out.write_text("{}")
    assert runner.invoke(app, ["schema", "--output", str(out), "--check"]).exit_code == 1

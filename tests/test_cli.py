from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import ExportFactory, forge_stub
from typer.testing import CliRunner

from argus_forge.cli import _run_exit_status, app
from argus_forge.models import ARGUS_ROOT_ENV, CORS_ORIGINS_ENV, READONLY_ENV

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


def test_run_streams_output_and_exit_code(tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "echo training...\n")
    result = runner.invoke(app, ["run", str(export), "--trainer", "kohya"])
    assert result.exit_code == 0
    assert "training..." in result.output
    assert "finished (exit 0)" in result.output


def test_run_propagates_trainer_exit_code(tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "exit 5\n")
    result = runner.invoke(app, ["run", str(export), "--trainer", "kohya"])
    assert result.exit_code == 5


def test_run_dry_run_prints_command_without_executing(tmp_path: Path) -> None:
    sentinel = tmp_path / "ran"
    export = forge_stub(tmp_path, "kohya", f'touch "{sentinel}"\n')
    result = runner.invoke(app, ["run", str(export), "--trainer", "kohya", "--dry-run"])
    assert result.exit_code == 0
    assert "dry run" in result.output
    assert not sentinel.exists()


def test_run_missing_config_errors(tmp_path: Path) -> None:
    export = tmp_path / "exp"
    export.mkdir()
    result = runner.invoke(app, ["run", str(export), "--trainer", "kohya"])
    assert result.exit_code == 1
    assert "no forged config" in result.output


def test_run_bad_trainer_is_clean_error_not_traceback(tmp_path: Path) -> None:
    """An unknown --trainer must fail like `config` does (exit 1 + Error line),
    not leak an uncaught pydantic ValidationError traceback."""
    result = runner.invoke(app, ["run", str(tmp_path), "--trainer", "nope"])
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_run_signal_death_reports_128_plus_n(tmp_path: Path) -> None:
    """A trainer killed by a signal exits 128+N (SIGTERM -> 143), per shell
    convention, not a modulo-256 mangling of the negative return code."""
    export = forge_stub(tmp_path, "kohya", "kill -TERM $$\n")
    result = runner.invoke(app, ["run", str(export), "--trainer", "kohya"])
    assert result.exit_code == 143


def test_run_json_streams_ndjson_events(tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "echo hi\n")
    result = runner.invoke(app, ["run", str(export), "--trainer", "kohya", "--json"])
    assert result.exit_code == 0
    types = [json.loads(line)["type"] for line in result.output.splitlines() if line.strip().startswith("{")]
    assert types[0] == "start" and types[-1] == "exit"


def test_run_exit_status_mapping() -> None:
    assert _run_exit_status(0, False) == 0
    assert _run_exit_status(3, False) == 3
    assert _run_exit_status(-9, False) == 137  # SIGKILL -> 128+9
    assert _run_exit_status(None, True) == 1  # errored, never exited
    assert _run_exit_status(None, False) == 0
    assert _run_exit_status(0, True) == 1  # error seen -> never report success


def test_schema_write_and_check(tmp_path: Path) -> None:
    out = tmp_path / "schema.json"
    assert runner.invoke(app, ["schema", "--output", str(out)]).exit_code == 0
    schema = json.loads(out.read_text())
    assert schema["title"] == "argus-forge wire contract"
    assert "ForgeResult" in schema["$defs"]
    assert runner.invoke(app, ["schema", "--output", str(out), "--check"]).exit_code == 0
    out.write_text("{}")
    assert runner.invoke(app, ["schema", "--output", str(out), "--check"]).exit_code == 1


# --- serve wiring: the flag/env seam between Typer and create_app ---


def _serve_kwargs(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> dict:
    """Invoke `serve` and capture the kwargs it hands create_app.

    The server tests call create_app directly, so without this the whole
    translation layer — flag precedence, env fallbacks, the warnings — is
    untested, which is exactly where its bugs live.
    """
    captured: dict = {}

    def fake_create_app(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("argus_forge.server.create_app", fake_create_app)
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    result = runner.invoke(app, ["serve", *argv])
    assert result.exit_code == 0, result.output
    captured["_stderr"] = result.output
    return captured


def test_serve_passes_export_root(monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = _serve_kwargs(monkeypatch, ["--export-root", "/data/out"])
    assert kwargs["export_root"] == "/data/out"
    assert "No export root" not in kwargs["_stderr"]


def test_serve_warns_when_no_export_root(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "No export root" in _serve_kwargs(monkeypatch, [])["_stderr"]


def test_serve_export_root_env_silences_the_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ARGUS_ROOT_ENV, "/data/out")
    assert "No export root" not in _serve_kwargs(monkeypatch, [])["_stderr"]


def test_serve_cors_flags_are_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = _serve_kwargs(monkeypatch, ["--cors", "--cors-origin", "https://a.example", "--cors-any"])
    assert kwargs["cors"] is True
    assert kwargs["cors_origins"] == ["https://a.example"]
    assert kwargs["cors_allow_any"] is True


def test_serve_cors_warning_respects_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """The warning must describe the app that actually gets built: with
    FORGE_CORS_ORIGINS set, create_app enables CORS, so claiming it is disabled
    sends the operator to debug a working config."""
    assert "CORS is disabled" in _serve_kwargs(monkeypatch, [])["_stderr"]
    monkeypatch.setenv(CORS_ORIGINS_ENV, "https://a.example")
    assert "CORS is disabled" not in _serve_kwargs(monkeypatch, [])["_stderr"]


def test_serve_no_run_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = _serve_kwargs(monkeypatch, ["--no-run"])
    assert kwargs["allow_run"] is False
    assert "Demo-safe mode" in kwargs["_stderr"]


def test_serve_readonly_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(READONLY_ENV, "1")
    assert _serve_kwargs(monkeypatch, [])["allow_run"] is False


def test_serve_readonly_env_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both the image and the compose file default this to 1, so 0 is the
    explicit opt-out an operator sets once a trainer is mounted."""
    monkeypatch.setenv(READONLY_ENV, "0")
    assert _serve_kwargs(monkeypatch, [])["allow_run"] is True


def test_serve_fails_safe_on_a_typoed_readonly_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mistyped protection flag must warn and keep the guard ON, not exit
    non-zero (under compose's `restart: unless-stopped` a hard exit is a crash
    loop) and not silently enable writes on an assumed-public host."""
    monkeypatch.setenv(READONLY_ENV, "enabled")
    kwargs = _serve_kwargs(monkeypatch, [])
    assert kwargs["allow_run"] is False
    assert "Demo-safe mode" in kwargs["_stderr"]

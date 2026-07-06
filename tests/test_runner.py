from __future__ import annotations

from pathlib import Path

import pytest

from argus_forge.models import ForgeError, RunEvent, RunRequest
from argus_forge.runner import RUNNABLE_TRAINERS, astream_run, prepare_run


def _forge_script(tmp_path: Path, trainer: str, body: str) -> Path:
    """A minimal forged export: forge/<trainer>/train.sh with *body*.

    Hand-built (not via forge_config) so the script is a trivial, runnable stub
    instead of a real ``accelerate launch`` line.
    """
    out = tmp_path / "exp" / "forge" / trainer
    out.mkdir(parents=True)
    (out / "train.sh").write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    return tmp_path / "exp"


async def _collect(req: RunRequest, **kw: str) -> list[RunEvent]:
    return [ev async for ev in astream_run(req, **kw)]


async def test_run_streams_start_logs_exit(tmp_path: Path) -> None:
    export = _forge_script(tmp_path, "kohya", "echo hello\necho world\n")
    events = await _collect(RunRequest(export_dir=str(export), trainer="kohya"), run_id="rid123")

    assert events[0].type == "start"
    assert events[0].command == ["bash", str(export / "forge/kohya/train.sh")]
    assert events[0].cwd == str(export / "forge/kohya")
    assert [e.message for e in events if e.type == "log"] == ["hello", "world"]
    assert events[-1].type == "exit" and events[-1].returncode == 0
    assert all(e.run_id == "rid123" for e in events)  # the join key is stable


async def test_run_reports_nonzero_exit(tmp_path: Path) -> None:
    export = _forge_script(tmp_path, "kohya", "echo boom\nexit 3\n")
    events = await _collect(RunRequest(export_dir=str(export), trainer="kohya"))
    assert events[-1].type == "exit" and events[-1].returncode == 3


async def test_run_dry_run_does_not_execute(tmp_path: Path) -> None:
    sentinel = tmp_path / "ran"
    export = _forge_script(tmp_path, "kohya", f'touch "{sentinel}"\n')
    events = await _collect(RunRequest(export_dir=str(export), trainer="kohya", dry_run=True))
    assert [e.type for e in events] == ["start"]
    assert not sentinel.exists()


async def test_run_passes_env_through(tmp_path: Path) -> None:
    export = _forge_script(tmp_path, "kohya", 'echo "val=$FORGE_TEST_VAR"\n')
    events = await _collect(RunRequest(export_dir=str(export), trainer="kohya", env={"FORGE_TEST_VAR": "xyz"}))
    assert "val=xyz" in [e.message for e in events if e.type == "log"]


def test_prepare_run_missing_config_raises(tmp_path: Path) -> None:
    export = tmp_path / "exp"
    export.mkdir()
    with pytest.raises(ForgeError, match="no forged config"):
        prepare_run(RunRequest(export_dir=str(export), trainer="kohya"))


def test_prepare_run_rejects_unrunnable_trainer(tmp_path: Path) -> None:
    assert "onetrainer" not in RUNNABLE_TRAINERS  # no train.sh emitted
    with pytest.raises(ForgeError, match="produces no train.sh"):
        prepare_run(RunRequest(export_dir=str(tmp_path), trainer="onetrainer"))

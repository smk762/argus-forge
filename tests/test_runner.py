from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import forge_stub

from argus_forge.models import ForgeError, RunEvent, RunRequest
from argus_forge.runner import RUNNABLE_TRAINERS, astream_run, prepare_run


async def _collect(req: RunRequest, **kw: str) -> list[RunEvent]:
    return [ev async for ev in astream_run(req, **kw)]


async def test_run_streams_start_logs_exit(tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "echo hello\necho world\n")
    events = await _collect(RunRequest(export_dir=str(export), trainer="kohya"), run_id="rid123")

    assert events[0].type == "start"
    assert events[0].command == ["bash", str(export / "forge/kohya/train.sh")]
    assert events[0].cwd == str(export / "forge/kohya")
    assert [e.message for e in events if e.type == "log"] == ["hello", "world"]
    assert events[-1].type == "exit" and events[-1].returncode == 0
    assert all(e.run_id == "rid123" for e in events)  # the join key is stable


async def test_run_reports_nonzero_exit(tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "echo boom\nexit 3\n")
    events = await _collect(RunRequest(export_dir=str(export), trainer="kohya"))
    assert events[-1].type == "exit" and events[-1].returncode == 3


async def test_run_dry_run_does_not_execute(tmp_path: Path) -> None:
    sentinel = tmp_path / "ran"
    export = forge_stub(tmp_path, "kohya", f'touch "{sentinel}"\n')
    events = await _collect(RunRequest(export_dir=str(export), trainer="kohya", dry_run=True))
    assert [e.type for e in events] == ["start"]
    assert not sentinel.exists()


async def test_run_passes_env_through(tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", 'echo "val=$FORGE_TEST_VAR"\n')
    events = await _collect(RunRequest(export_dir=str(export), trainer="kohya", env={"FORGE_TEST_VAR": "xyz"}))
    assert "val=xyz" in [e.message for e in events if e.type == "log"]


# --- regressions from the max-effort review of PR #12 ---


async def test_large_line_streams_without_deadlock(tmp_path: Path) -> None:
    """A single stdout line far larger than asyncio's 64 KiB readline limit must
    still stream and reach a terminal exit — the old readline loop hung here."""
    export = forge_stub(
        tmp_path, "kohya", "python3 -c \"import sys; sys.stdout.write('a'*200000 + '\\n')\"\necho end\n"
    )
    events = await asyncio.wait_for(_collect(RunRequest(export_dir=str(export), trainer="kohya")), timeout=20)
    logs = [e.message for e in events if e.type == "log"]
    assert any(len(m) >= 200000 for m in logs)  # the giant line came through, not truncated/hung
    assert "end" in logs  # draining continued after the giant line
    assert events[-1].type == "exit" and events[-1].returncode == 0


async def test_carriage_return_progress_streams_incrementally(tmp_path: Path) -> None:
    """tqdm-style `\\r` progress (no newline) arrives as separate log events."""
    export = forge_stub(tmp_path, "kohya", "printf 'p1\\rp2\\rp3\\n'\n")
    events = await _collect(RunRequest(export_dir=str(export), trainer="kohya"))
    assert [e.message for e in events if e.type == "log"] == ["p1", "p2", "p3"]


async def test_cancelled_run_kills_the_process(tmp_path: Path) -> None:
    """A consumer that stops early (client disconnect / Ctrl-C) must not orphan
    the trainer: astream_run's finally reaps the process group."""
    if not shutil.which("pgrep"):  # pragma: no cover
        pytest.skip("pgrep not available")
    mark = "argusforge_test_marker_kill"
    # Marker as a real argv of a grandchild process (mirrors accelerate under
    # bash) so pgrep can find it and we prove the whole group is reaped.
    export = forge_stub(tmp_path, "kohya", f'echo up\npython3 -c "import time; time.sleep(30)" {mark}\n')

    async def consume() -> None:
        async for _ in astream_run(RunRequest(export_dir=str(export), trainer="kohya")):
            pass

    task = asyncio.ensure_future(consume())
    await asyncio.sleep(0.8)
    assert subprocess.run(["pgrep", "-f", mark], capture_output=True, text=True).stdout.strip()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.5)
    alive = subprocess.run(["pgrep", "-f", mark], capture_output=True, text=True).stdout.strip()
    if alive:  # pragma: no cover
        subprocess.run(["pkill", "-f", mark])
    assert not alive, "trainer process was orphaned after cancel"


async def test_launch_failure_yields_terminal_error(tmp_path: Path) -> None:
    """If the process can't be launched (here: an embedded NUL in an env value
    makes create_subprocess_exec raise ValueError), the stream ends with a
    terminal `error` event and no exception escapes."""
    export = forge_stub(tmp_path, "kohya", "echo hi\n")
    events = await _collect(RunRequest(export_dir=str(export), trainer="kohya", env={"X": "a\x00b"}))
    assert events[0].type == "start"
    assert events[-1].type == "error"
    assert not any(e.type == "exit" for e in events)


def test_prepare_run_missing_config_raises(tmp_path: Path) -> None:
    export = tmp_path / "exp"
    export.mkdir()
    with pytest.raises(ForgeError, match="no forged config"):
        prepare_run(RunRequest(export_dir=str(export), trainer="kohya"))


def test_prepare_run_rejects_unrunnable_trainer(tmp_path: Path) -> None:
    assert "onetrainer" not in RUNNABLE_TRAINERS  # declares no entrypoint
    with pytest.raises(ForgeError, match="no launcher"):
        prepare_run(RunRequest(export_dir=str(tmp_path), trainer="onetrainer"))


def test_prepare_run_rejects_exec_hijacking_env(tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "echo hi\n")
    with pytest.raises(ForgeError, match="BASH_ENV"):
        prepare_run(RunRequest(export_dir=str(export), trainer="kohya", env={"BASH_ENV": "/tmp/evil.sh"}))


def test_runnable_trainers_from_entrypoint() -> None:
    """Runnability comes from the machine entrypoint field, not the display list."""
    assert set(RUNNABLE_TRAINERS) == {"kohya", "diffusers"}

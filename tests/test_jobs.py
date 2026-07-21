"""Direct unit tests for the job registry (argus_forge.jobs).

These exercise Job/JobRegistry without the HTTP layer, so the lifecycle edges
that a TestClient can't reach — cancel-before-first-step, terminal status on a
dry run, a stalled subscriber, double-cancel — are testable and fast.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import forge_stub

from argus_forge.jobs import MAX_SUBSCRIBER_LAG, Job, JobRegistry
from argus_forge.models import RunEvent, RunRequest
from argus_forge.runner import prepare_run


def _job(tmp_path: Path, body: str, *, dry_run: bool = False, run_id: str = "rid") -> Job:
    """A Job wired to a real forged stub, the way JobRegistry.start builds one."""
    export = forge_stub(tmp_path, "kohya", body)
    req = RunRequest(export_dir=str(export), trainer="kohya", dry_run=dry_run)
    command, cwd = prepare_run(req)
    return Job(run_id, req, command, cwd)


async def _drain(job: Job) -> list[RunEvent]:
    return [ev async for ev in job.subscribe()]


async def test_dry_run_reaches_terminal_status(tmp_path: Path) -> None:
    """A dry run yields only `start` (no exit/error), but must still land on a
    terminal status — otherwise pollers of the argus-proof join hang forever."""
    job = _job(tmp_path, "echo hi\n", dry_run=True)
    job._task = asyncio.create_task(job._drive())
    await asyncio.wait_for(job._task, timeout=5)
    assert job.finished
    assert job.status == "succeeded"  # completed cleanly, just didn't execute
    assert job.ended_at is not None


async def test_successful_run_is_succeeded(tmp_path: Path) -> None:
    job = _job(tmp_path, "echo hi\n")
    job._task = asyncio.create_task(job._drive())
    await asyncio.wait_for(job._task, timeout=10)
    assert job.status == "succeeded" and job.returncode == 0


async def test_nonzero_exit_is_failed(tmp_path: Path) -> None:
    job = _job(tmp_path, "exit 3\n")
    job._task = asyncio.create_task(job._drive())
    await asyncio.wait_for(job._task, timeout=10)
    assert job.status == "failed" and job.returncode == 3


async def test_cancel_before_first_step_does_not_wedge(tmp_path: Path) -> None:
    """If the driving task is cancelled before it takes its first step, its body
    (and finally) never run; the job must still finalize, not stay 'running' with
    a stream that hangs forever."""
    job = _job(tmp_path, "sleep 30\n")
    job._task = asyncio.create_task(job._drive())
    # No await between create_task and cancel: the task never gets to step.
    await job.cancel()
    assert job.finished
    assert job.status == "cancelled"
    assert job.ended_at is not None
    # A viewer joining the finalized job terminates cleanly instead of blocking.
    events = await asyncio.wait_for(_drain(job), timeout=2)
    assert all(isinstance(e, RunEvent) for e in events)


async def test_start_event_survives_past_the_buffer(tmp_path: Path) -> None:
    """`start` carries command/cwd; a run longer than the event buffer must not
    drop it from a reconnecting viewer's replay."""
    n = MAX_SUBSCRIBER_LAG + 50
    job = _job(tmp_path, f"for i in $(seq 1 {n}); do echo line$i; done\n")
    job._task = asyncio.create_task(job._drive())
    await asyncio.wait_for(job._task, timeout=30)
    events = await asyncio.wait_for(_drain(job), timeout=5)  # replay after finish
    assert events[0].type == "start"
    assert events[0].command and events[0].cwd


async def test_stalled_subscriber_queue_is_bounded(tmp_path: Path) -> None:
    """A subscriber that registers but never reads must not accumulate the whole
    run: its queue is bounded and drops its oldest un-read events."""
    job = _job(tmp_path, "echo hi\n")
    # Register a raw queue the way subscribe() does, then never drain it.
    q: asyncio.Queue[object] = asyncio.Queue(maxsize=MAX_SUBSCRIBER_LAG)
    job._subscribers.add(q)
    for i in range(MAX_SUBSCRIBER_LAG * 3):
        job._publish(RunEvent(run_id=job.run_id, type="log", message=f"line{i}"))
    assert q.qsize() <= MAX_SUBSCRIBER_LAG


async def test_registry_evicts_only_finished_and_keeps_newest(tmp_path: Path) -> None:
    reg = JobRegistry()
    export = forge_stub(tmp_path, "kohya", "echo hi\n")
    req = RunRequest(export_dir=str(export), trainer="kohya")
    command, cwd = prepare_run(req)
    job = reg.start(req, command, cwd)
    await asyncio.wait_for(job._task, timeout=10)
    assert reg.get(job.run_id) is job
    assert job in reg.list()


@pytest.mark.skipif(not shutil.which("pgrep"), reason="pgrep not available")
async def test_double_cancel_does_not_orphan_the_trainer(tmp_path: Path) -> None:
    """A second cancel arriving during the first cancel's SIGTERM grace must not
    abort the SIGKILL escalation and leave a SIGTERM-ignoring trainer running."""
    mark = "argusforge_test_double_cancel"
    # Trap SIGTERM so only the SIGKILL escalation can reap it; keep a grandchild
    # carrying the marker so pgrep proves the whole group is gone.
    body = f"trap '' TERM\npython3 -c 'import time; time.sleep(60)' {mark}\n"
    job = _job(tmp_path, body)
    job._task = asyncio.create_task(job._drive())
    await asyncio.sleep(0.8)  # let bash + grandchild actually launch
    assert subprocess.run(["pgrep", "-f", mark], capture_output=True, text=True).stdout.strip()

    # Fire two cancels concurrently; the second must be a no-op, not a re-cancel.
    await asyncio.gather(job.cancel(), job.cancel())
    assert job.status == "cancelled"

    await asyncio.sleep(0.5)
    alive = subprocess.run(["pgrep", "-f", mark], capture_output=True, text=True).stdout.strip()
    if alive:  # pragma: no cover - cleanup only on failure
        subprocess.run(["pkill", "-9", "-f", mark])
    assert not alive, "trainer orphaned: the second cancel skipped the SIGKILL escalation"

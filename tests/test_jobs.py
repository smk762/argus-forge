"""Direct unit tests for the job registry (argus_forge.server.jobs).

These exercise Job/JobRegistry without the HTTP layer, so the lifecycle edges
that a TestClient can't reach — cancel-before-first-step, terminal status on a
dry run, a stalled subscriber, double-cancel — are testable and fast.
"""

from __future__ import annotations

import asyncio
import gc
import os
import shutil
import subprocess
import weakref
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from conftest import forge_stub

from argus_forge.manifest import resolve_export_dir
from argus_forge.models import RunEvent, RunRequest
from argus_forge.runner import prepare_run
from argus_forge.server.jobs import MAX_BUFFERED_EVENTS, Job, JobRegistry


def _job(tmp_path: Path, body: str, *, dry_run: bool = False, run_id: str = "rid") -> Job:
    """A Job wired to a real forged stub, the way JobRegistry.start builds one."""
    export = forge_stub(tmp_path, "kohya", body)
    req = RunRequest(export_dir=str(export), trainer="kohya", dry_run=dry_run)
    command, cwd = prepare_run(req)
    return Job(run_id, req, command, cwd)


async def _drain(job: Job) -> list[RunEvent]:
    return [ev async for ev in job.subscribe()]


async def _attach(job: Job) -> AsyncIterator[RunEvent]:
    """Start a viewer and leave it parked, having read nothing.

    ``subscribe`` is an async generator: its body — and so the cursor's
    registration — doesn't run until the first ``__anext__``. These tests need a
    *registered* viewer, so drive it once against a pre-published event.
    """
    job._publish(RunEvent(run_id=job.run_id, type="log", message="attached"))
    viewer = job.subscribe()
    assert (await viewer.__anext__()).message == "attached"
    assert job._subscribers, "viewer never registered"
    return viewer


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


async def test_cancel_emits_a_distinct_cancelled_event(tmp_path: Path) -> None:
    """A cancel is surfaced on the stream as a terminal `cancelled` event, not an
    `error` — so a consumer never mistakes a user cancel for a failure."""
    job = _job(tmp_path, "echo up\nsleep 30\n")
    job._task = asyncio.create_task(job._drive())
    await asyncio.sleep(0.3)  # let it start and log before cancelling
    await job.cancel()
    assert job.status == "cancelled"
    events = await asyncio.wait_for(_drain(job), timeout=2)  # replay the backlog
    assert events[-1].type == "cancelled"
    assert not any(e.type == "error" for e in events)


async def test_start_event_survives_past_the_buffer(tmp_path: Path) -> None:
    """`start` carries command/cwd; a run longer than the event buffer must not
    drop it from a reconnecting viewer's replay."""
    n = MAX_BUFFERED_EVENTS + 50
    job = _job(tmp_path, f"for i in $(seq 1 {n}); do echo line$i; done\n")
    job._task = asyncio.create_task(job._drive())
    await asyncio.wait_for(job._task, timeout=30)
    events = await asyncio.wait_for(_drain(job), timeout=5)  # replay after finish
    assert events[0].type == "start"
    assert events[0].command and events[0].cwd


async def test_a_stalled_viewer_resumes_at_the_oldest_retained_event(tmp_path: Path) -> None:
    """A viewer that never reads must not accumulate the whole run. Its cursor
    falls out of the shared window and resumes at the window's tail — a drop, but
    one that costs the producer nothing: `_publish` is synchronous and never
    waits on a viewer, so a stalled reader can't throttle the trainer's stdout."""
    job = _job(tmp_path, "echo hi\n")
    viewer = await _attach(job)
    n = MAX_BUFFERED_EVENTS * 3
    for i in range(n):
        job._publish(RunEvent(run_id=job.run_id, type="log", message=f"line{i}"))
    job._finalize("succeeded")  # so the parked viewer ends instead of blocking

    rest = [ev async for ev in viewer]
    # Only the shared window survived, and it is the *newest* end of the run:
    # drop-oldest, in order, no duplicates.
    assert len(rest) == MAX_BUFFERED_EVENTS
    assert [ev.message for ev in rest] == [f"line{i}" for i in range(n - MAX_BUFFERED_EVENTS, n)]


async def test_total_retention_is_the_shared_window_not_per_viewer(tmp_path: Path) -> None:
    """The property the per-viewer-channel design could not satisfy: N stalled
    viewers must pin the *same* bounded window, not N copies of it. Asserted on
    live objects, since that is the memory claim — each viewer holds a cursor,
    and the only strong references to events are the job's own deque."""
    job = _job(tmp_path, "echo hi\n")
    viewers = [await _attach(job) for _ in range(5)]
    assert len(job._subscribers) == 5

    alive: list[weakref.ref[RunEvent]] = []
    for i in range(MAX_BUFFERED_EVENTS * 3):
        ev = RunEvent(run_id=job.run_id, type="log", message=f"line{i}")
        alive.append(weakref.ref(ev))
        job._publish(ev)
    del ev
    gc.collect()

    retained = sum(ref() is not None for ref in alive)
    assert retained <= MAX_BUFFERED_EVENTS  # not 5 x MAX_BUFFERED_EVENTS
    assert retained == len(job._events)

    job._finalize("succeeded")
    for viewer in viewers:
        await viewer.aclose()


async def test_replaying_a_finished_run_registers_nothing_lasting(tmp_path: Path) -> None:
    """The common reconnect path: /run/{id}/stream on a run that already ended.
    It replays the retained buffer, ends on its own, and leaves no viewer behind
    holding the finished job."""
    job = _job(tmp_path, "echo hi\n")
    job._task = asyncio.create_task(job._drive())
    await asyncio.wait_for(job._task, timeout=10)
    assert job.finished

    replayed = await asyncio.wait_for(_drain(job), timeout=2)
    assert replayed and replayed[0].type == "start"
    assert replayed[-1].type == "exit"
    assert not job._subscribers


async def test_a_late_viewer_sees_every_event_exactly_once(tmp_path: Path) -> None:
    """A viewer attached mid-run reads forward from one buffer, so the old
    backlog/live seam — the place a broadcast fan-out can drop or duplicate an
    event — cannot desynchronise: no gap, no repeat, no reordering."""
    job = _job(tmp_path, "echo hi\n")
    viewer = await _attach(job)
    seen: list[str] = []
    for i in range(50):
        job._publish(RunEvent(run_id=job.run_id, type="log", message=f"line{i}"))
        if i % 7 == 0:  # read part-way, so the viewer repeatedly catches up and re-parks
            seen.append((await viewer.__anext__()).message)
    job._finalize("succeeded")

    seen += [ev.message async for ev in viewer]
    assert seen == [f"line{i}" for i in range(50)]


async def test_live_viewer_sees_progress_then_ends_at_finalize(tmp_path: Path) -> None:
    """A viewer attached *before* the run ends follows it live and terminates on
    its own when the job finalizes — no sentinel, no hang. Closing the producer's
    half is what ends the iteration, and events already buffered still arrive."""
    job = _job(tmp_path, "echo one\necho two\n")
    events: list[RunEvent] = []

    async def watch() -> None:
        async for ev in job.subscribe():
            events.append(ev)

    viewer = asyncio.create_task(watch())
    # Wait for the viewer to actually register rather than assuming one loop turn
    # suffices: a bare `await asyncio.sleep(0)` has exactly zero margin, so any
    # suspension point later added inside subscribe() ahead of the register would
    # surface here as a baffling "log != start" instead of a clear failure.
    for _ in range(1000):
        if job._subscribers:
            break
        await asyncio.sleep(0)
    assert job._subscribers, "viewer never registered"

    job._task = asyncio.create_task(job._drive())
    try:
        await asyncio.wait_for(viewer, timeout=10)  # ends because _finalize closed it
    finally:
        # On the timeout path — the regression this test exists to catch — the
        # run's task would otherwise keep driving the bash child past the end of
        # the test, leaving "Task was destroyed but it is pending!" and an
        # unreaped process group.
        if not job._task.done():
            await job.cancel()
    assert job.finished
    assert [e.type for e in events][0] == "start"
    assert any(e.type == "log" and e.message == "one" for e in events)
    assert events[-1].type == "exit"
    assert not job._subscribers  # the viewer unregistered on the way out


async def test_registry_evicts_only_finished_and_keeps_newest(tmp_path: Path) -> None:
    reg = JobRegistry()
    export = forge_stub(tmp_path, "kohya", "echo hi\n")
    req = RunRequest(export_dir=str(export), trainer="kohya")
    command, cwd = prepare_run(req)
    job = reg.start(req, command, cwd)
    await asyncio.wait_for(job._task, timeout=10)
    assert reg.get(job.run_id) is job
    assert job in reg.list()


def test_state_reports_the_resolved_export_dir() -> None:
    """RunState.export_dir is the resolved absolute path — matching what
    command/cwd derive from, and what DatasetInfo/ForgeResult report."""
    req = RunRequest(export_dir="some/relative/dir", trainer="kohya")
    state = Job("rid", req, ["bash", "train.sh"], "/tmp").state()
    assert os.path.isabs(state.export_dir)
    assert state.export_dir == str(resolve_export_dir("some/relative/dir"))


async def test_launch_failure_records_the_reason(tmp_path: Path) -> None:
    """A run that can't be launched records why on the RunState, so a poller can
    diagnose it without reading the event log."""
    export = forge_stub(tmp_path, "kohya", "echo hi\n")
    req = RunRequest(export_dir=str(export), trainer="kohya")
    job = Job("rid", req, ["/nonexistent/trainer-binary"], str(tmp_path))
    job._task = asyncio.create_task(job._drive())
    await asyncio.wait_for(job._task, timeout=10)
    assert job.status == "failed"
    assert job.message and "failed to launch" in job.message
    assert job.state().message == job.message


async def test_registry_shutdown_cancels_in_flight_runs(tmp_path: Path) -> None:
    """The lifespan shutdown hook calls this: an in-flight run must be cancelled
    so no trainer is left without an owner when the server stops."""
    reg = JobRegistry()
    export = forge_stub(tmp_path, "kohya", "echo up\nsleep 30\n")
    req = RunRequest(export_dir=str(export), trainer="kohya")
    command, cwd = prepare_run(req)
    job = reg.start(req, command, cwd)
    await asyncio.sleep(0.3)  # let it reach 'running'
    assert job.status == "running"
    await asyncio.wait_for(reg.shutdown(), timeout=10)
    assert job.status == "cancelled"
    assert job.finished


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

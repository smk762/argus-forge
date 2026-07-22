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
from collections.abc import AsyncGenerator
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


async def _attach(job: Job, label: str = "attached") -> AsyncGenerator[RunEvent, None]:
    """Register a viewer and drive it up to the head of the buffer.

    ``subscribe`` is an async generator: its body — and so the cursor's
    registration — doesn't run until the first ``__anext__``. So publish a
    uniquely-labelled event and pull until this viewer has seen *that* event,
    which leaves it registered and caught up, so its next pull genuinely parks.
    A cursor starts at the window's floor, not at the head, so a second viewer
    replays what is retained before reaching its own label — pass distinct
    labels when attaching several, or the wait below ends on the wrong event.
    """
    job._publish(RunEvent(run_id=job.run_id, type="log", message=label))
    viewer = job.subscribe()
    while (await viewer.__anext__()).message != label:
        pass
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
    and the only strong references to events are the job's own deque.

    Each viewer joins a run that is *already* at a full window, at a different
    point, and then the window is rolled past every join point. That staggering
    is what gives the test teeth: attach them all while the buffer holds one
    event and every design looks identical, because a per-viewer queue then holds
    references to the same objects the shared deque does rather than a generation
    of its own. Reverting to that design must fail here, at ~6x the window.
    """
    job = _job(tmp_path, "echo hi\n")
    alive: list[weakref.ref[RunEvent]] = []

    def publish_a_full_window(tag: str) -> None:
        # In a function so the loop variable can't outlive it and keep an event
        # alive past the collect below.
        for i in range(MAX_BUFFERED_EVENTS):
            ev = RunEvent(run_id=job.run_id, type="log", message=f"{tag}{i}")
            alive.append(weakref.ref(ev))
            job._publish(ev)

    viewers: list[AsyncGenerator[RunEvent, None]] = []
    try:
        for k in range(5):
            publish_a_full_window(f"gen{k}_")
            viewers.append(await _attach(job, f"attached{k}"))
        assert len(job._subscribers) == 5
        publish_a_full_window("final_")  # roll the window past every join point
        gc.collect()

        retained = sum(ref() is not None for ref in alive)
        # Not 6 x MAX_BUFFERED_EVENTS, which is what one generation per viewer
        # plus the shared tail would cost. Each parked viewer holds only the
        # event it is suspended on, and those are the untracked `attached*` ones.
        assert retained <= MAX_BUFFERED_EVENTS
        assert retained == len(job._events)
    finally:
        # In a finally: an assertion above must not also leak five suspended
        # generators into loop teardown, on top of the failure being reported.
        job._finalize("succeeded")
        for viewer in viewers:
            await viewer.aclose()


async def test_a_parked_viewer_is_woken_by_the_next_publish(tmp_path: Path) -> None:
    """A viewer that has caught up parks on its wakeup, and `_publish` must wake
    it — every attached viewer, not just one. Without that, a live /stream reader
    receives nothing until the run ends, so this asserts the event arrives *while
    the job is still running* rather than being flushed out by `_finalize`."""
    job = _job(tmp_path, "echo hi\n")
    viewers = [await _attach(job, f"attached{k}") for k in range(3)]
    # Each `_attach` publishes, so the earlier viewers are behind again by the
    # time the last one is up. One marker brings every cursor to the head.
    job._publish(RunEvent(run_id=job.run_id, type="log", message="sync"))
    for v in viewers:
        while (await v.__anext__()).message != "sync":
            pass

    pending = [asyncio.ensure_future(v.__anext__()) for v in viewers]
    for _ in range(100):  # let every viewer reach `await viewer.wakeup.wait()`
        await asyncio.sleep(0)
    assert not any(p.done() for p in pending), "viewers should be parked, not holding a buffered event"

    job._publish(RunEvent(run_id=job.run_id, type="log", message="live"))
    try:
        for p in pending:
            assert (await asyncio.wait_for(p, timeout=2)).message == "live"
        assert not job.finished  # delivered live, not flushed by the finalize below
    finally:
        job._finalize("succeeded")
        for viewer in viewers:
            await viewer.aclose()


async def test_finalize_releases_every_viewer_without_resuming_it(tmp_path: Path) -> None:
    """`_finalize` must both wake its viewers and drop them, and the two cover
    different readers.

    Waking is what lets a *parked* reader end its own iteration instead of
    hanging on a run that will never publish again. Dropping is for the reader
    that never resumes at all — a half-open connection — whose generator never
    runs `subscribe()`'s cleanup; `_finalize` is idempotent, so if it doesn't
    unregister that cursor here, nothing ever will.
    """
    job = _job(tmp_path, "echo hi\n")
    half_open = await _attach(job, "half-open")  # registered, then never resumed
    parked = await _attach(job, "parked")  # caught up, so its next pull parks
    seen: list[RunEvent] = []

    async def watch() -> None:
        async for ev in parked:
            seen.append(ev)

    task = asyncio.create_task(watch())
    for _ in range(100):
        await asyncio.sleep(0)
    assert not task.done(), "viewer should be parked on its wakeup"
    assert len(job._subscribers) == 2

    job._finalize("succeeded")
    await asyncio.wait_for(task, timeout=2)  # the wakeup is what ends it
    assert not job._subscribers, "a viewer that never resumed is still registered"
    await half_open.aclose()


async def test_start_survives_a_cursor_clamped_past_it(tmp_path: Path) -> None:
    """`start` carries command/cwd, so a viewer must see it even when its cursor
    is clamped forward over the position `start` held. A viewer registered before
    the run produced anything is the sharp case: it has no retained event to
    anchor on, and `_drive` can publish a whole window in one synchronous burst
    (nothing awaits between the lines of a single stdout read)."""
    job = _job(tmp_path, "echo hi\n")
    viewer = job.subscribe()
    first = asyncio.ensure_future(viewer.__anext__())
    for _ in range(100):
        if job._subscribers:
            break
        await asyncio.sleep(0)
    assert job._subscribers, "viewer never registered"

    job._publish(RunEvent(run_id=job.run_id, type="start", command=["bash", "train.sh"], cwd="/tmp"))
    for i in range(MAX_BUFFERED_EVENTS + 50):  # burst the cursor out of the window
        job._publish(RunEvent(run_id=job.run_id, type="log", message=f"line{i}"))
    job._finalize("succeeded")

    assert (await asyncio.wait_for(first, timeout=2)).command == ["bash", "train.sh"]
    rest = [ev async for ev in viewer]
    assert [ev.type for ev in rest].count("start") == 0  # injected once, not twice


async def test_start_is_not_duplicated_when_it_is_not_the_head(tmp_path: Path) -> None:
    """The retained `start` is keyed on the sequence it occupied, not on it
    happening to be the first thing in the buffer — otherwise any event published
    ahead of it makes every viewer see the run start twice."""
    job = _job(tmp_path, "echo hi\n")
    job._publish(RunEvent(run_id=job.run_id, type="log", message="pre"))
    job._publish(RunEvent(run_id=job.run_id, type="start", command=["bash"], cwd="/tmp"))
    job._publish(RunEvent(run_id=job.run_id, type="log", message="post"))
    job._finalize("succeeded")

    events = await asyncio.wait_for(_drain(job), timeout=2)
    assert [ev.type for ev in events] == ["log", "start", "log"]


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

    # And a client that gives up part-way must leave nothing behind either. This
    # is the half-open reader `_finalize`'s sweep exists for — but `_finalize` is
    # idempotent and has already run, so a cursor registered here would never be
    # dropped. Nothing can publish to it, so subscribe() must not register one.
    partial = job.subscribe()
    assert (await partial.__anext__()).type == "start"
    assert not job._subscribers
    await partial.aclose()


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
    its own when the job finalizes — no sentinel, no hang. What ends the
    iteration is the viewer re-reading `finished` on its next pass; events
    already in the buffer are drained ahead of it, so nothing is lost."""
    job = _job(tmp_path, "echo one\necho two\n")
    events: list[RunEvent] = []

    async def watch() -> None:
        async for ev in job.subscribe():
            events.append(ev)

    viewer = asyncio.create_task(watch())
    # Wait for the viewer to actually register rather than assuming one loop turn
    # suffices: `subscribe` is an async generator, so its body doesn't run — and
    # the cursor isn't registered — until the first `__anext__`. Starting the run
    # before that point would make this a replay test rather than a live one.
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

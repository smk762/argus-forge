"""In-process registry of training runs, so a run outlives the connection.

``POST /run`` starts a :class:`Job` on a background task that drives
:func:`argus_forge.runner.astream_run` independently of any HTTP request. The
request (and later reconnects) are just *viewers* that subscribe to the job's
event stream; a client disconnecting drops a viewer, not the run. The job keeps
recent events buffered so a reconnecting viewer sees history + live progress,
records terminal status for polling (the argus-proof join), and can be
cancelled explicitly.

Scope: single-server. Given a state directory (``ARGUS_FORGE_STATE_DIR`` /
``--state-dir``), every run's :class:`RunState` is also kept on disk, one JSON
file per run (:class:`RunStore`), so a restart keeps finished runs, their LoRA
included, and a poller can still learn how a run ended. The event log is not
kept: a run restored from disk replays its ``start`` and its terminal event
only. A run that was still ``running`` when the server went away is restored as
``failed``: a graceful stop cancels it first (so it is saved ``cancelled``), and
after a crash nothing is left driving it (the trainer runs in its own session
and may still be running detached — see :mod:`argus_forge.runner`). Without a
state directory, runs are in memory only, as before.

This lives under ``server/`` rather than in the core package because the
registry exists only to serve the HTTP layer — a CLI-only install has no reason
to carry it. It needs nothing from the ``server`` extra, though:
:mod:`argus_forge.server` resolves the FastAPI entry points lazily, so importing
this module does not drag in fastapi/starlette/argus-cortex with it. Its fan-out
is plain stdlib asyncio; the only third-party imports are the package's own base
dependencies (structlog, and pydantic via :mod:`argus_forge.models`).
"""

from __future__ import annotations

import asyncio
import os
import re
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path

import structlog
from argus_cortex.timestamps import utc_now_iso
from pydantic import ValidationError

from argus_forge.manifest import resolve_export_dir
from argus_forge.models import ForgeError, RunArtifact, RunEvent, RunRequest, RunState, RunStatus, run_is_active
from argus_forge.runner import astream_run, lora_artifact, new_run_id

logger = structlog.get_logger()

# Recent events retained per run for reconnecting viewers. The tail of the log
# is bounded; the run's `start` event is retained out-of-band (see Job._start)
# so command/cwd survive on the stream even past this window.
#
# Every viewer reads out of this one deque through a cursor (see _Viewer), so N
# attached /stream readers — however far behind — cost N cursors, not N copies of
# the window. It bounds one job: the registry retains MAX_FINISHED_JOBS finished
# ones, each with a window of its own.
MAX_BUFFERED_EVENTS = 2000
if MAX_BUFFERED_EVENTS < 1:  # pragma: no cover - guards a mis-tune, not a runtime path
    # A zero-length deque accepts every append and keeps none, so `_published`
    # would run away from a window that is always empty and a cursor clamped to
    # its floor would index off the end. Tune the window, never to nothing.
    raise ValueError("MAX_BUFFERED_EVENTS must be >= 1: a zero-length window leaves a cursor nothing to read")
# Finished runs kept in the registry (most-recent-first) before eviction.
MAX_FINISHED_JOBS = 64


# The reason a run restored as `running` is reported failed.
INTERRUPTED_MESSAGE = (
    "forge stopped while this run was in progress, so its outcome is unknown "
    "(after a crash its trainer may still be running detached)"
)

# A run id as new_run_id makes it; anything else in the state dir is not ours.
_RUN_ID = re.compile(r"[0-9a-f]{32}")


class RunStore:
    """Run records on disk: ``<state_dir>/runs/<run_id>.json``, one RunState each.

    Writes are atomic (a temp file renamed over the record), so a crash mid-save
    leaves the previous record, never half of one. A failure to save is logged,
    not raised: losing a record must not fail or wedge the run it describes.
    """

    def __init__(self, state_dir: Path) -> None:
        self.dir = state_dir / "runs"

    def _path(self, run_id: str) -> Path:
        return self.dir / f"{run_id}.json"

    def save(self, state: RunState) -> None:
        path = self._path(state.run_id)
        tmp = path.with_name(f".{path.name}.tmp")
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp.write_text(state.model_dump_json(), encoding="utf-8")
            os.replace(tmp, path)
        except (OSError, ValueError) as exc:  # ValueError: text that isn't valid UTF-8 (a path's surrogates)
            logger.error("run_record_save_failed", run_id=state.run_id, error=repr(exc))

    def delete(self, run_id: str) -> None:
        try:
            self._path(run_id).unlink(missing_ok=True)
        except OSError as exc:
            logger.error("run_record_delete_failed", run_id=run_id, error=repr(exc))

    def load(self) -> list[RunState]:
        """Every readable record; an unreadable or foreign file is logged and skipped."""
        if not self.dir.is_dir():
            return []
        states: list[RunState] = []
        for path in sorted(self.dir.glob("*.json")):
            if not _RUN_ID.fullmatch(path.stem):
                continue
            try:
                state = RunState.model_validate_json(path.read_bytes())
            except (OSError, ValidationError) as exc:
                logger.warning("run_record_unreadable", path=str(path), error=repr(exc))
                continue
            if state.run_id != path.stem:
                logger.warning("run_record_mismatched", path=str(path), run_id=state.run_id)
                continue
            states.append(state)
        return states


class _Viewer:
    """One viewer's position in the job's shared event buffer, plus a wakeup.

    ``pos`` is an *absolute* sequence number, not a deque index: the job counts
    every event it ever published (``Job._published``), so the retained window is
    ``[_published - len(_events), _published)``. A viewer that falls further
    behind than the window simply finds its ``pos`` below that floor and resumes
    at the oldest retained event — the drop is a consequence of the one shared
    bound, not a second eviction policy running per viewer.

    ``sent_start`` tracks whether this viewer has been handed the run's `start`
    event yet, so the out-of-band copy (``Job._start``) can be injected exactly
    once, whenever the cursor turns out to be past the sequence it occupied.

    Identity-keyed (no ``__eq__``), so viewers live in a set.
    """

    __slots__ = ("pos", "sent_start", "wakeup")

    def __init__(self, pos: int) -> None:
        self.pos = pos
        self.sent_start = False
        self.wakeup = asyncio.Event()


class Job:
    """One training run: the background task driving it, a bounded event buffer,
    and the set of viewers whose cursors read out of that buffer."""

    def __init__(
        self, run_id: str, req: RunRequest, command: list[str], cwd: str, store: RunStore | None = None
    ) -> None:
        self.run_id = run_id
        self.req = req
        self.command = command
        self.cwd = cwd
        # The resolved absolute export dir, matching DatasetInfo/ForgeResult (and
        # what command/cwd derive from) rather than the raw request spelling.
        self.export_dir = str(resolve_export_dir(req.export_dir))
        self.status: RunStatus = "running"
        self.returncode: int | None = None
        self.message: str | None = None
        self.started_at = utc_now_iso("auto")
        self.ended_at: str | None = None
        self.lora: RunArtifact | None = None
        self._store = store
        self._events: deque[RunEvent] = deque(maxlen=MAX_BUFFERED_EVENTS)
        # Count of every event ever published, so a viewer's cursor can be an
        # absolute sequence number that survives eviction from `_events`.
        self._published = 0
        # The `start` event is retained out-of-band so a viewer reconnecting to a
        # long run still learns command/cwd even after it rolls off `_events`.
        # Its sequence number goes with it: that — not `start` happening to be
        # the head of the deque — is what tells a cursor whether it is past it.
        self._start: RunEvent | None = None
        self._start_seq = -1
        self._subscribers: set[_Viewer] = set()
        self._done = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._cancelling = False

    @classmethod
    def restore(cls, state: RunState, store: RunStore | None = None) -> Job:
        """A finished job rebuilt from its saved record (see the module docstring).

        A record still saying ``running`` is closed as ``failed`` and saved
        that way. Its stream replays ``start`` and one terminal event.
        """
        req = RunRequest(export_dir=state.export_dir, trainer=state.trainer)
        job = cls(state.run_id, req, list(state.command), state.cwd or "", store)
        job.export_dir = state.export_dir
        job.started_at = state.started_at
        job.returncode = state.returncode
        job.message = state.message
        job.lora = state.lora
        job._publish(RunEvent(run_id=job.run_id, type="start", command=job.command, cwd=state.cwd))
        if state.status == "running":
            job.message = INTERRUPTED_MESSAGE
            job._publish(RunEvent(run_id=job.run_id, type="error", message=job.message))
            job._finalize("failed")  # saves the corrected record
            return job
        if state.status == "cancelled":
            job._publish(RunEvent(run_id=job.run_id, type="cancelled", message=job.message))
        elif job.returncode is not None:
            job._publish(RunEvent(run_id=job.run_id, type="exit", returncode=job.returncode))
        elif state.status == "failed":
            job._publish(RunEvent(run_id=job.run_id, type="error", message=job.message))
        job.status = state.status
        job.ended_at = state.ended_at
        job._done.set()
        return job

    def save(self) -> None:
        if self._store is not None:
            self._store.save(self.state())

    @property
    def finished(self) -> bool:
        return self._done.is_set()

    @property
    def active(self) -> bool:
        """Still queued for its GPU lease or running (:func:`~argus_forge.models.run_is_active`).

        A cancel still unwinding (the trainer inside its SIGTERM grace) is active
        too: the status turns terminal only once the task has really ended.
        """
        return run_is_active(self.status)

    @property
    def holds_a_slot(self) -> bool:
        """Whether this run counts against the one-active-run rule (``POST /run``).

        An active run that is not a dry run: a dry run takes no GPU lease and
        starts no process (it ends straight after its ``start`` event), so it
        neither blocks a real run nor is blocked by one.
        """
        return self.active and not self.req.dry_run

    def state(self) -> RunState:
        return RunState(
            run_id=self.run_id,
            trainer=self.req.trainer,
            export_dir=self.export_dir,
            status=self.status,
            returncode=self.returncode,
            started_at=self.started_at,
            ended_at=self.ended_at,
            command=self.command,
            cwd=self.cwd,
            message=self.message,
            lora=self.lora,
        )

    @property
    def _oldest_seq(self) -> int:
        """Sequence number of the oldest event still retained.

        The window is ``[_oldest_seq, _published)``. This holds only while
        `_published` counts exactly the appends to `_events`, which is why both
        live in `_publish` and nothing else may touch `_events`.
        """
        return self._published - len(self._events)

    def _publish(self, ev: RunEvent) -> None:
        if ev.type == "start":
            self._start = ev
            self._start_seq = self._published
        self._events.append(ev)  # bounded: the deque evicts its own oldest
        self._published += 1
        for viewer in self._subscribers:
            viewer.wakeup.set()

    async def subscribe(self) -> AsyncIterator[RunEvent]:
        """Yield the run's retained events, then live ones until it ends.

        The viewer is a cursor into the job's single bounded buffer, so there is
        no backlog/live split to keep consistent: it reads forward from wherever
        it starts, and "live" just means it caught up with the head and parked on
        its wakeup. Nothing can be reordered, duplicated or lost across a
        boundary that no longer exists, and a viewer that never reads pins one
        cursor rather than its own copy of the window. A viewer that falls
        further behind than the window resumes at its tail — the drop is a
        consequence of the one shared bound, not a second eviction policy.

        A viewer joining a finished run just replays what is retained. The
        `start` event is injected from its out-of-band copy the moment the cursor
        is found to be past the sequence it occupied — whether it rolled off the
        tail or the cursor was clamped over it — so command/cwd are always the
        first thing seen, exactly once.
        """
        viewer = _Viewer(self._oldest_seq)
        # Registering on a finished run would strand the cursor: `_finalize` is
        # idempotent, so the sweep that drops half-open readers has already run
        # and can never run again. There is nothing to register for either —
        # nothing publishes after `_done` is set. No await between the read above
        # and this decision, so `finished` cannot change under us.
        if not self.finished:
            self._subscribers.add(viewer)
        try:
            while True:
                # Clear *before* reading anything: a publish landing after this
                # point re-sets the wakeup, so the park below returns at once
                # rather than sleeping through an event we haven't seen.
                viewer.wakeup.clear()
                # Read `finished` before draining, not after: everything
                # published by a finished run is already in the buffer, so a
                # drain that starts after this read cannot miss a later event.
                # This read is also what ends a viewer that `_finalize` woke and
                # unregistered — *not* the wakeup it set, which the clear above
                # has already discarded. Keep the two in this order.
                done = self.finished
                while viewer.pos < self._published:
                    oldest = self._oldest_seq
                    if viewer.pos < oldest:
                        viewer.pos = oldest  # fell out of the shared window; resume at its tail
                    if not viewer.sent_start and self._start is not None and viewer.pos > self._start_seq:
                        # The cursor is past `start`: it rolled off the window, or
                        # the clamp above stepped over it. Hand over the retained
                        # copy without moving the cursor, so nothing else is lost.
                        viewer.sent_start = True
                        yield self._start
                        continue
                    ev = self._events[viewer.pos - oldest]
                    viewer.pos += 1
                    viewer.sent_start = viewer.sent_start or ev is self._start
                    yield ev  # suspends here; the cursor is what keeps our place
                if done:
                    return
                await viewer.wakeup.wait()
        finally:
            self._subscribers.discard(viewer)  # a no-op if we never registered

    def _finalize(self, status: RunStatus) -> None:
        """Record terminal state exactly once and release every viewer.

        Idempotent: the first caller wins. ``_drive``'s ``finally`` is the normal
        path, but ``cancel`` also calls this so a task cancelled before it ever
        ran (its ``finally`` never fires) can't wedge the job as ``running``.
        """
        if self._done.is_set():
            return
        self.status = status
        self.ended_at = utc_now_iso("auto")
        self._done.set()
        # Wake every viewer: each one drains what is left in the buffer, sees the
        # `finished` it read on its next pass, and ends its own iteration.
        for viewer in self._subscribers:
            viewer.wakeup.set()
        # Then drop them. A viewer whose generator is never resumed again — a
        # half-open /stream reader — never runs subscribe()'s finally, so without
        # this its cursor stays registered for as long as the registry retains
        # the finished job (MAX_FINISHED_JOBS). subscribe()'s own discard is then
        # a no-op.
        #
        # Unregistering can't strand a viewer, but *not* because the wakeup set
        # above survives — subscribe() clears it at the top of its next pass.
        # What ends the viewer is that the same pass re-reads `finished` before
        # draining and returns on it. That ordering is the load-bearing one; the
        # wakeup only ensures a parked viewer gets a pass at all.
        self._subscribers.clear()
        # Last: nothing retries a finalize, so if saving raised any earlier the
        # viewers would stay parked forever.
        self.save()

    async def _drive(self) -> None:
        status: RunStatus = "succeeded"  # a run that ends without a terminal event (dry_run) still succeeded
        try:
            async for ev in astream_run(self.req, run_id=self.run_id, resolved=(self.command, self.cwd)):
                self._publish(ev)
                if ev.type == "exit":
                    self.returncode = ev.returncode
                    status = "succeeded" if ev.returncode == 0 else "failed"
                elif ev.type == "error":
                    status = "failed"
                    self.message = ev.message  # a launch failure carries its reason here
            if status == "succeeded" and not self.req.dry_run:
                # Before _finalize, so a poller that sees `succeeded` sees the LoRA
                # too. Off the loop: it hashes a file of hundreds of MB.
                try:
                    self.lora = await asyncio.to_thread(lora_artifact, self.req)
                except ForgeError as exc:
                    self.message = f"the trainer exited 0, but forge found no LoRA: {exc}"
                    logger.warning("run_lora_missing", run_id=self.run_id, error=str(exc))
        except asyncio.CancelledError:
            # Explicit cancel: astream_run's finally already reaped the process
            # group. Emit a terminal `cancelled` event (not `error`, so a stream
            # consumer never reads a user cancel as a failure) and re-raise so the
            # task ends cancelled (so cancel()/shutdown() observe a true cancel).
            self.message = "run cancelled"
            self._publish(RunEvent(run_id=self.run_id, type="cancelled", message=self.message))
            self._finalize("cancelled")
            raise
        except Exception as exc:  # pragma: no cover - defensive; astream_run handles its own
            logger.exception("job_failed", run_id=self.run_id)
            self.message = f"run failed: {exc}"
            self._publish(RunEvent(run_id=self.run_id, type="error", message=self.message))
            status = "failed"
        self._finalize(status)

    async def cancel(self) -> None:
        # Guard against a re-entrant cancel (a double-clicked UI, a retry after
        # the request-side wait): re-cancelling a task already unwinding inside
        # runner._terminate's SIGTERM grace would abort its SIGKILL escalation
        # and orphan the trainer. There is no await between the read and the set.
        if self._cancelling:
            # Wait for the in-flight cancel instead of returning straight away:
            # shutdown() must not report every trainer reaped while another
            # caller is still inside runner._terminate's SIGTERM->SIGKILL grace.
            # The finally below guarantees _done is set once the task has ended
            # (at once, or from the task's done callback), so this cannot wedge.
            await self._done.wait()
            return
        self._cancelling = True
        task = self._task
        try:
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    # `task` ending cancelled is the expected outcome here and
                    # must not propagate. But if *this* coroutine was itself
                    # cancelled (uvicorn hitting timeout_graceful_shutdown), a
                    # bare suppress would report a completed shutdown for one
                    # that was actually cut short — so let that one through.
                    current = asyncio.current_task()
                    if current is not None and current.cancelling():
                        raise
        finally:
            # If the task was cancelled before its first step, _drive's body — and
            # so its finalize — never ran; make sure the job doesn't stay
            # "running". But only once the task has really ended: if *this*
            # coroutine was cancelled while the task is still unwinding inside
            # runner._terminate's grace, finalizing now would release everyone
            # parked on _done (shutdown included) while the trainer is still
            # alive, and let _drive publish its `cancelled` after _done. Defer to
            # the task's own end instead; _finalize is idempotent, so whichever of
            # _drive's handler and this callback runs first wins.
            if task is None or task.done():
                self._finalize("cancelled")
            else:
                task.add_done_callback(lambda _t: self._finalize("cancelled"))


class JobRegistry:
    """Runs keyed by run_id, with bounded retention of finished ones."""

    def __init__(self, state_dir: Path | None = None) -> None:
        """*state_dir* keeps run records across restarts (see :class:`RunStore`); None keeps them in memory only.

        Loads the saved records (blocking, so call it at startup, not per request).
        """
        self._jobs: dict[str, Job] = {}
        self._store = RunStore(state_dir) if state_dir is not None else None
        if self._store is not None:
            for state in sorted(self._store.load(), key=lambda st: st.started_at):
                self._jobs[state.run_id] = Job.restore(state, self._store)
            self._evict()

    def start(self, req: RunRequest, command: list[str], cwd: str) -> Job:
        """Create a job and launch it on a background task (needs a running loop).

        The task is independent of the caller, so the run continues after the
        request that started it returns or disconnects.
        """
        job = Job(new_run_id(), req, command, cwd, self._store)
        job.save()  # so a restart mid-run knows the run existed
        job._task = asyncio.create_task(job._drive())
        self._jobs[job.run_id] = job
        self._evict()
        return job

    def get(self, run_id: str) -> Job | None:
        return self._jobs.get(run_id)

    def list(self) -> list[Job]:
        return list(self._jobs.values())

    def active_runs(self) -> list[Job]:
        """Runs that count against the one-active-run rule (see :attr:`Job.holds_a_slot`), oldest first.

        Never one restored from disk: a run that was in flight when forge went
        away is restored ``failed`` (:meth:`Job.restore`), so a stale record
        cannot block new runs after a restart.
        """
        return sorted((job for job in self._jobs.values() if job.holds_a_slot), key=lambda job: job.started_at)

    async def shutdown(self) -> None:
        """Cancel every in-flight run (server stopping) so no trainer is left
        without an owner in this process. Cancels concurrently so total time is
        one SIGTERM grace period, not one per run (which would overrun the
        container's stop grace and leave the later runs un-reaped)."""
        jobs = [job for job in self._jobs.values() if not job.finished]
        # return_exceptions: one failing cancel must not abandon the others
        # mid-flight (gather would raise at once while they are still reaping).
        results = await asyncio.gather(*(job.cancel() for job in jobs), return_exceptions=True)
        for job, result in zip(jobs, results, strict=True):
            if isinstance(result, BaseException):
                logger.error("job_cancel_failed", run_id=job.run_id, error=repr(result))

    def _evict(self) -> None:
        finished = sorted((j for j in self._jobs.values() if j.finished), key=lambda j: j.ended_at or "")
        for job in finished[: max(0, len(finished) - MAX_FINISHED_JOBS)]:
            self._jobs.pop(job.run_id, None)
            if self._store is not None:
                self._store.delete(job.run_id)

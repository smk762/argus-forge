"""In-process registry of training runs, so a run outlives the connection.

``POST /run`` starts a :class:`Job` on a background task that drives
:func:`argus_forge.runner.astream_run` independently of any HTTP request. The
request (and later reconnects) are just *viewers* that subscribe to the job's
event stream; a client disconnecting drops a viewer, not the run. The job keeps
recent events buffered so a reconnecting viewer sees history + live progress,
records terminal status for polling (the argus-proof join), and can be
cancelled explicitly.

Scope: in-process and single-server. A server restart forgets in-flight runs
(and, because the trainer runs in its own session, can leave it running detached
— see :mod:`argus_forge.runner`); durable run metadata is a follow-up.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import structlog

from argus_forge.models import RunEvent, RunRequest, RunState, RunStatus
from argus_forge.runner import astream_run, new_run_id

logger = structlog.get_logger()

# Recent events retained per run for reconnecting viewers. The tail of the log
# is bounded; the run's `start` event is retained out-of-band (see Job._start)
# so command/cwd survive on the stream even past this window.
MAX_BUFFERED_EVENTS = 2000
# How far a single /stream viewer may fall behind before its queue starts
# dropping its oldest un-read events. Bounds the memory one stalled reader can
# pin, independently of MAX_BUFFERED_EVENTS (which bounds only the shared tail).
MAX_SUBSCRIBER_LAG = MAX_BUFFERED_EVENTS
# Finished runs kept in the registry (most-recent-first) before eviction.
MAX_FINISHED_JOBS = 64

_SENTINEL = object()  # queued to a subscriber to signal end-of-stream


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _offer(q: asyncio.Queue[object], item: object) -> None:
    """Enqueue *item* for a subscriber without ever blocking the producer.

    A viewer that has fallen ``MAX_SUBSCRIBER_LAG`` events behind (a stalled or
    half-open /stream reader) drops its oldest un-read event to make room, so
    one slow consumer can't grow the server's memory without bound. The
    end-of-stream sentinel is enqueued the same way, so it is never lost.
    """
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        with contextlib.suppress(asyncio.QueueEmpty):
            q.get_nowait()
        q.put_nowait(item)


class Job:
    """One training run: the background task driving it, a bounded event buffer,
    and the set of live subscribers to broadcast to."""

    def __init__(self, run_id: str, req: RunRequest, command: list[str], cwd: str) -> None:
        self.run_id = run_id
        self.req = req
        self.command = command
        self.cwd = cwd
        self.status: RunStatus = "running"
        self.returncode: int | None = None
        self.started_at = _now()
        self.ended_at: str | None = None
        self._events: deque[RunEvent] = deque(maxlen=MAX_BUFFERED_EVENTS)
        # The `start` event is retained out-of-band so a viewer reconnecting to a
        # long run still learns command/cwd even after it rolls off `_events`.
        self._start: RunEvent | None = None
        self._subscribers: set[asyncio.Queue[object]] = set()
        self._done = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._cancelling = False

    @property
    def finished(self) -> bool:
        return self._done.is_set()

    def state(self) -> RunState:
        return RunState(
            run_id=self.run_id,
            trainer=self.req.trainer,
            export_dir=self.req.export_dir,
            status=self.status,
            returncode=self.returncode,
            started_at=self.started_at,
            ended_at=self.ended_at,
            command=self.command,
            cwd=self.cwd,
        )

    def _publish(self, ev: RunEvent) -> None:
        if ev.type == "start":
            self._start = ev
        self._events.append(ev)
        for q in self._subscribers:
            _offer(q, ev)

    async def subscribe(self) -> AsyncIterator[RunEvent]:
        """Yield the buffered backlog, then live events until the run ends.

        Snapshot + register happen with no ``await`` between them, so the split
        between backlog and live queue is atomic — no event is dropped or
        duplicated across it. A viewer joining a finished run just replays the
        retained buffer. The `start` event is prepended if it has already rolled
        off the bounded tail, so command/cwd are always the first thing seen.
        """
        q: asyncio.Queue[object] = asyncio.Queue(maxsize=MAX_SUBSCRIBER_LAG)
        backlog = list(self._events)
        if self._start is not None and self._start not in backlog:
            backlog.insert(0, self._start)
        done = self._done.is_set()
        self._subscribers.add(q)
        try:
            for ev in backlog:
                yield ev
            if done:
                return
            while True:
                item = await q.get()
                if item is _SENTINEL:
                    return
                assert isinstance(item, RunEvent)
                yield item
        finally:
            self._subscribers.discard(q)

    def _finalize(self, status: RunStatus) -> None:
        """Record terminal state exactly once and release every viewer.

        Idempotent: the first caller wins. ``_drive``'s ``finally`` is the normal
        path, but ``cancel`` also calls this so a task cancelled before it ever
        ran (its ``finally`` never fires) can't wedge the job as ``running``.
        """
        if self._done.is_set():
            return
        self.status = status
        self.ended_at = _now()
        self._done.set()
        for q in list(self._subscribers):
            _offer(q, _SENTINEL)

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
        except asyncio.CancelledError:
            # Explicit cancel: astream_run's finally already reaped the process
            # group; record it, tell viewers, and re-raise so the task ends
            # cancelled (so cancel()/shutdown() observe a true cancellation).
            self._publish(RunEvent(run_id=self.run_id, type="error", message="run cancelled"))
            self._finalize("cancelled")
            raise
        except Exception as exc:  # pragma: no cover - defensive; astream_run handles its own
            logger.exception("job_failed", run_id=self.run_id)
            self._publish(RunEvent(run_id=self.run_id, type="error", message=f"run failed: {exc}"))
            status = "failed"
        self._finalize(status)

    async def cancel(self) -> None:
        # Guard against a re-entrant cancel (a double-clicked UI, a retry after
        # the request-side wait): re-cancelling a task already unwinding inside
        # runner._terminate's SIGTERM grace would abort its SIGKILL escalation
        # and orphan the trainer. There is no await between the read and the set.
        if self._cancelling:
            return
        self._cancelling = True
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        # If the task was cancelled before its first step, _drive's body — and
        # so its finalize — never ran; make sure the job doesn't stay "running".
        self._finalize("cancelled")


class JobRegistry:
    """Runs keyed by run_id, with bounded retention of finished ones."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def start(self, req: RunRequest, command: list[str], cwd: str, run_id: str | None = None) -> Job:
        """Create a job and launch it on a background task (needs a running loop).

        The task is independent of the caller, so the run continues after the
        request that started it returns or disconnects.
        """
        job = Job(run_id or new_run_id(), req, command, cwd)
        job._task = asyncio.create_task(job._drive())
        self._jobs[job.run_id] = job
        self._evict()
        return job

    def get(self, run_id: str) -> Job | None:
        return self._jobs.get(run_id)

    def list(self) -> list[Job]:
        return list(self._jobs.values())

    async def shutdown(self) -> None:
        """Cancel every in-flight run (server stopping) so no trainer is left
        without an owner in this process. Cancels concurrently so total time is
        one SIGTERM grace period, not one per run (which would overrun the
        container's stop grace and leave the later runs un-reaped)."""
        await asyncio.gather(*(job.cancel() for job in list(self._jobs.values())))

    def _evict(self) -> None:
        finished = sorted((j for j in self._jobs.values() if j.finished), key=lambda j: j.ended_at or "")
        for job in finished[: max(0, len(finished) - MAX_FINISHED_JOBS)]:
            self._jobs.pop(job.run_id, None)

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

This lives under ``server/`` rather than in the core package because the
registry exists only to serve the HTTP layer, and its fan-out is built on
``anyio`` — a dependency the ``server`` extra already carries (via starlette)
but a CLI-only install has no reason to acquire.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    EndOfStream,
    WouldBlock,
    create_memory_object_stream,
)
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from argus_forge.manifest import resolve_export_dir
from argus_forge.models import RunEvent, RunRequest, RunState, RunStatus
from argus_forge.runner import astream_run, new_run_id

logger = structlog.get_logger()

# Recent events retained per run for reconnecting viewers. The tail of the log
# is bounded; the run's `start` event is retained out-of-band (see Job._start)
# so command/cwd survive on the stream even past this window.
MAX_BUFFERED_EVENTS = 2000
# How far a single /stream viewer may fall behind before its channel starts
# dropping its oldest un-read events. Bounds the memory one stalled reader can
# pin, independently of MAX_BUFFERED_EVENTS (which bounds only the shared tail).
MAX_SUBSCRIBER_LAG = MAX_BUFFERED_EVENTS
# Finished runs kept in the registry (most-recent-first) before eviction.
MAX_FINISHED_JOBS = 64


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(eq=False)  # identity-keyed, so viewers live in a set
class _Subscriber:
    """One viewer's bounded channel onto the job's events.

    Both halves of the memory object stream are held here: the receive half is
    the viewer's, but the producer needs it too, to enforce the drop-oldest lag
    policy that anyio's streams don't provide (see :meth:`offer`).
    """

    send: MemoryObjectSendStream[RunEvent]
    receive: MemoryObjectReceiveStream[RunEvent]

    @classmethod
    def open(cls) -> _Subscriber:
        return cls(*create_memory_object_stream[RunEvent](max_buffer_size=MAX_SUBSCRIBER_LAG))

    def offer(self, ev: RunEvent) -> None:
        """Hand *ev* to this viewer without ever blocking the producer.

        anyio's memory object streams apply backpressure when full — they block,
        or raise ``WouldBlock`` from ``send_nowait`` — which here would let one
        stalled /stream reader throttle the trainer's stdout. So a viewer that
        has fallen ``MAX_SUBSCRIBER_LAG`` events behind (a stalled or half-open
        reader) drops its own oldest un-read event to make room instead, which
        bounds the memory one slow consumer can pin.

        Dropping is safe from the producer side: ``WouldBlock`` means the buffer
        is full and no task is waiting to receive, so nothing is racing us for
        that oldest item.
        """
        try:
            self.send.send_nowait(ev)
        except WouldBlock:
            with contextlib.suppress(WouldBlock, EndOfStream, ClosedResourceError):
                self.receive.receive_nowait()  # drop the oldest un-read event
            with contextlib.suppress(WouldBlock, BrokenResourceError, ClosedResourceError):
                self.send.send_nowait(ev)
        except (BrokenResourceError, ClosedResourceError):
            pass  # the viewer went away between its last read and unregistering

    def close(self) -> None:
        """Close the send half: the viewer's ``async for`` ends once it has
        drained what is already buffered. This replaces an end-of-stream
        sentinel, so nothing out-of-band can be dropped by the lag policy."""
        self.send.close()


class Job:
    """One training run: the background task driving it, a bounded event buffer,
    and the set of live subscribers to broadcast to."""

    def __init__(self, run_id: str, req: RunRequest, command: list[str], cwd: str) -> None:
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
        self.started_at = _now()
        self.ended_at: str | None = None
        self._events: deque[RunEvent] = deque(maxlen=MAX_BUFFERED_EVENTS)
        # The `start` event is retained out-of-band so a viewer reconnecting to a
        # long run still learns command/cwd even after it rolls off `_events`.
        self._start: RunEvent | None = None
        self._subscribers: set[_Subscriber] = set()
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
            export_dir=self.export_dir,
            status=self.status,
            returncode=self.returncode,
            started_at=self.started_at,
            ended_at=self.ended_at,
            command=self.command,
            cwd=self.cwd,
            message=self.message,
        )

    def _publish(self, ev: RunEvent) -> None:
        if ev.type == "start":
            self._start = ev
        self._events.append(ev)
        for sub in self._subscribers:
            sub.offer(ev)

    async def subscribe(self) -> AsyncIterator[RunEvent]:
        """Yield the buffered backlog, then live events until the run ends.

        Snapshot + register happen with no ``await`` between them, so the split
        between backlog and live channel is atomic — no event is dropped or
        duplicated across it. A viewer joining a finished run just replays the
        retained buffer. The `start` event is prepended if it has already rolled
        off the bounded tail, so command/cwd are always the first thing seen.
        """
        sub = _Subscriber.open()
        backlog = list(self._events)
        if self._start is not None and self._start not in backlog:
            backlog.insert(0, self._start)
        done = self._done.is_set()
        self._subscribers.add(sub)
        try:
            for ev in backlog:
                yield ev
            if done:
                return
            async with sub.receive:
                async for ev in sub.receive:
                    yield ev
        finally:
            self._subscribers.discard(sub)
            sub.close()  # release the producer's half too, so nothing outlives the viewer

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
        for sub in list(self._subscribers):
            sub.close()

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

    def start(self, req: RunRequest, command: list[str], cwd: str) -> Job:
        """Create a job and launch it on a background task (needs a running loop).

        The task is independent of the caller, so the run continues after the
        request that started it returns or disconnects.
        """
        job = Job(new_run_id(), req, command, cwd)
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

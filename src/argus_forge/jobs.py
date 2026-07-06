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

# Recent events retained per run for reconnecting viewers (start/terminal events
# are few and effectively always retained; this bounds the log tail).
MAX_BUFFERED_EVENTS = 2000
# Finished runs kept in the registry (most-recent-first) before eviction.
MAX_FINISHED_JOBS = 64

_SENTINEL = object()  # queued to a subscriber to signal end-of-stream


def _now() -> str:
    return datetime.now(UTC).isoformat()


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
        self._subscribers: set[asyncio.Queue[object]] = set()
        self._done = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

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
        self._events.append(ev)
        for q in self._subscribers:
            q.put_nowait(ev)

    async def subscribe(self) -> AsyncIterator[RunEvent]:
        """Yield the buffered backlog, then live events until the run ends.

        Snapshot + register happen with no ``await`` between them, so the split
        between backlog and live queue is atomic — no event is dropped or
        duplicated across it. A viewer joining a finished run just replays the
        retained buffer.
        """
        q: asyncio.Queue[object] = asyncio.Queue()
        backlog = list(self._events)
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

    async def _drive(self) -> None:
        try:
            async for ev in astream_run(self.req, run_id=self.run_id, resolved=(self.command, self.cwd)):
                self._publish(ev)
                if ev.type == "exit":
                    self.returncode = ev.returncode
                    self.status = "succeeded" if ev.returncode == 0 else "failed"
                elif ev.type == "error":
                    self.status = "failed"
        except asyncio.CancelledError:
            # Explicit cancel: astream_run's finally already reaped the process
            # group; record it and let viewers finish cleanly.
            self.status = "cancelled"
            self._publish(RunEvent(run_id=self.run_id, type="error", message="run cancelled"))
        except Exception as exc:  # pragma: no cover - defensive; astream_run handles its own
            logger.exception("job_failed", run_id=self.run_id)
            self.status = "failed"
            self._publish(RunEvent(run_id=self.run_id, type="error", message=f"run failed: {exc}"))
        finally:
            self.ended_at = _now()
            self._done.set()
            for q in list(self._subscribers):
                q.put_nowait(_SENTINEL)

    async def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task


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
        without an owner in this process."""
        for job in list(self._jobs.values()):
            await job.cancel()

    def _evict(self) -> None:
        finished = sorted((j for j in self._jobs.values() if j.finished), key=lambda j: j.ended_at or "")
        for job in finished[: max(0, len(finished) - MAX_FINISHED_JOBS)]:
            self._jobs.pop(job.run_id, None)

"""Job-runner mode: shell out to the forged trainer and stream progress.

``argus-forge config`` writes a runnable launcher (``train.sh``) under
``<export_dir>/forge/<trainer>/``; this module executes it and streams the
trainer's output as :class:`RunEvent` NDJSON lines. The CLI ``run`` verb and the
server's ``POST /run`` are thin shells around :func:`astream_run`.

Trust model: ``/run`` executes a script forge generated, on the shared dataset
volume — the single-user LAN assumption of ``/config``. But note this is *not*
sandboxed: the environment is caller-supplied and a trainer script runs real
code, so treat reaching the port as equivalent to shell access on the host.
A request's env may set only :data:`~argus_forge.models.RUN_ENV_KEYS` (where
the trainer checkout lives, which GPU and model cache): every loader, shell and
interpreter hook is refused. The child inherits the server's environment less
the suite's own secrets (:data:`SECRET_ENV_KEYS`). This is defence-in-depth, not
a sandbox.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import uuid
from collections.abc import AsyncIterator

import structlog
from argus_cortex.gpu import GpuCoordinator

from argus_forge import gpu
from argus_forge.emitters import TRAINER_INFO
from argus_forge.manifest import FORGE_DIR_NAME, resolve_export_dir
from argus_forge.models import RUN_ENV_KEYS, ForgeError, RunEvent, RunRequest

logger = structlog.get_logger()

# Trainers forge can run: those that declare a launcher entrypoint. Derived from
# the machine field TrainerInfo.entrypoint (not the human ``files`` list), so a
# new runnable trainer only needs to set entrypoint and relabelling ``files``
# can't silently break runnability.
RUNNABLE_TRAINERS: tuple[str, ...] = tuple(t for t, info in TRAINER_INFO.items() if info.entrypoint)

# The suite's own credentials in the server's environment. A trainer script
# never needs them, and it runs third-party code (sd-scripts, pip packages), so
# they are dropped from the environment it inherits.
SECRET_ENV_KEYS: frozenset[str] = frozenset({"ARGUS_GPU_BROKER_TOKEN", "ARGUS_ADMIN_TOKEN", "AI_AUDIT_API_TOKEN"})


def child_env(req: RunRequest) -> dict[str, str]:
    """The trainer's environment: the server's own, less :data:`SECRET_ENV_KEYS`, plus the request's allowed keys."""
    base = {k: v for k, v in os.environ.items() if k not in SECRET_ENV_KEYS}
    return {**base, **req.env}


# Bytes read per stdout pull. Reading in chunks (not readline) means no single
# line can exceed a buffer limit, and carriage-return progress bars stream.
_READ_CHUNK = 8192
# A run of output with no newline/carriage-return longer than this is force-cut
# into a log event so a pathological separator-less stream can't grow unbounded.
_MAX_SEGMENT = 1 << 20


def new_run_id() -> str:
    """A fresh training_run_id — the join key for a run's events and eval results."""
    return uuid.uuid4().hex


def prepare_run(req: RunRequest) -> tuple[list[str], str]:
    """Validate *req* and return the ``(command, cwd)`` to launch.

    Raises :class:`ForgeError` if the trainer has no launcher, if the caller-
    supplied env contains a blocked key, or if the config has not been forged
    yet — call this *before* streaming so a caller can surface the failure up
    front (a 400, a non-zero exit) instead of mid-stream.
    """
    entrypoint = TRAINER_INFO[req.trainer].entrypoint
    if not entrypoint:
        raise ForgeError(f"trainer {req.trainer!r} has no launcher to run (runnable: {', '.join(RUNNABLE_TRAINERS)})")
    refused = sorted(k for k in req.env if k not in RUN_ENV_KEYS)
    if refused:
        raise ForgeError(
            f"env may not set {', '.join(refused)}: a run's env may set only {', '.join(sorted(RUN_ENV_KEYS))}"
        )
    export_dir = resolve_export_dir(req.export_dir)
    script = export_dir / FORGE_DIR_NAME / req.trainer / entrypoint
    if not script.is_file():
        raise ForgeError(
            f"no forged config to run at {script} — run `argus-forge config {export_dir} --trainer {req.trainer}` first"
        )
    return ["bash", str(script)], str(script.parent)


async def _iter_output_lines(stream: asyncio.StreamReader) -> AsyncIterator[str]:
    """Yield output split on newline OR carriage return, reading in chunks.

    Chunked (not readline) so no line-length limit applies and ``\\r`` progress
    bars stream incrementally; CR/LF are single ASCII bytes that never appear
    inside a UTF-8 multibyte sequence, so splitting on them at the byte level is
    safe. A forced cut (see ``_MAX_SEGMENT``) is moved back to a character
    boundary so it cannot split a multibyte sequence. Empty segments (a bare ``\\r\\n`` boundary, trailing separators) are
    skipped. The whole stream is always drained to EOF — the reader never stops
    early, so the child can't wedge on a full pipe.
    """
    buf = bytearray()
    while True:
        chunk = await stream.read(_READ_CHUNK)
        if not chunk:
            break
        buf.extend(chunk)
        while True:
            idx = _first_separator(buf)
            if idx < 0:
                if len(buf) >= _MAX_SEGMENT:  # bound memory on separator-less output
                    cut = _char_boundary(buf, _MAX_SEGMENT)
                    yield bytes(buf[:cut]).decode("utf-8", "replace")
                    del buf[:cut]
                    continue
                break
            if idx > 0:  # skip empties from adjacent separators
                yield bytes(buf[:idx]).decode("utf-8", "replace")
            del buf[: idx + 1]
    if buf:
        yield bytes(buf).decode("utf-8", "replace")


def _char_boundary(buf: bytearray, cut: int) -> int:
    """*cut* moved back (at most 3 bytes) so it doesn't land inside a UTF-8 sequence.

    A byte of the form ``10xxxxxx`` continues the previous character; cutting
    before one would decode both halves as U+FFFD. Invalid UTF-8 (a longer run
    of continuation bytes) is cut at *cut* as before.
    """
    for back in range(4):
        i = cut - back
        if i <= 0:
            break
        if i >= len(buf) or (buf[i] & 0xC0) != 0x80:
            return i
    return cut


def _first_separator(buf: bytearray) -> int:
    """Index of the first CR or LF in *buf*, or -1."""
    cr = buf.find(b"\r")
    lf = buf.find(b"\n")
    if cr < 0:
        return lf
    if lf < 0:
        return cr
    return min(cr, lf)


# How long a run's process group gets after SIGTERM before SIGKILL.
_TERM_GRACE_S = 5.0
_GROUP_POLL_S = 0.05


async def _terminate(proc: asyncio.subprocess.Process) -> None:
    """Stop the child's process group so a cancelled/failed run leaves nothing
    orphaned (``accelerate`` is a grandchild of ``bash``; the child is a group
    leader via ``start_new_session``, so its pgid is its pid — recorded at spawn,
    not looked up from the leader, which may already be reaped).

    Escalation is decided by the *group*, not the leader: bash typically exits on
    SIGTERM at once while its trainer workers linger, and a leader that already
    exited can leave workers behind. SIGTERM is sent synchronously so it fires
    even while the surrounding task is being cancelled; SIGKILL follows in a
    ``finally`` if anything in the group outlives the grace, so a second
    cancellation arriving during the grace cannot skip it."""
    pgid = proc.pid
    if not _signal_group(pgid, signal.SIGTERM):
        return  # nothing left in the group
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _TERM_GRACE_S
        while _signal_group(pgid, 0) and loop.time() < deadline:
            if proc.returncode is None:
                # Wake on the leader's exit, but keep polling the group after it.
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=_GROUP_POLL_S)
            else:
                await asyncio.sleep(_GROUP_POLL_S)
    finally:
        _signal_group(pgid, signal.SIGKILL)


def _signal_group(pgid: int, sig: int) -> bool:
    """Send *sig* to process group *pgid*; False if the group no longer exists."""
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class _RunFailed(Exception):
    """Raised inside the GPU lease so the coordinator records a failed (or OOM) run, then swallowed."""


# Trainer output that means the card ran out of memory; the broker learns from an `oom` release.
_OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError", "CUBLAS_STATUS_ALLOC_FAILED")


async def astream_run(
    req: RunRequest,
    *,
    run_id: str | None = None,
    resolved: tuple[list[str], str] | None = None,
    coordinator: GpuCoordinator | None = None,
) -> AsyncIterator[RunEvent]:
    """Launch the forged trainer and yield its lifecycle + log events.

    Emits ``start`` (the resolved command), then one ``log`` per output line,
    then a terminal event: ``error`` if the GPU lease was not granted or the
    process could not be launched, else ``exit`` with the return code.
    ``dry_run`` stops after ``start``. Every path ends in exactly one terminal
    event, and the child's process group is always reaped (finally), even on
    cancellation / consumer disconnect.

    The trainer runs under a GPU lease from *coordinator* (default: the
    environment's, see :mod:`argus_forge.gpu`; ``none`` when unset, which
    grants at once). The lease is released *before* the terminal event is
    yielded, so whoever acts on ``exit`` (an argus-proof eval) finds the GPU free.

    *resolved* is a pre-computed ``(command, cwd)`` (from :func:`prepare_run`)
    the server threads in to avoid re-validating; the CLI passes None.
    """
    run_id = run_id or new_run_id()
    command, cwd = resolved if resolved is not None else prepare_run(req)

    yield RunEvent(run_id=run_id, type="start", command=command, cwd=cwd)
    if req.dry_run:
        return

    coord = coordinator if coordinator is not None else gpu.coordinator()
    gated = coord.name != "none"
    if gated:
        yield RunEvent(run_id=run_id, type="log", message=f"[argus-forge] waiting for a GPU lease ({coord.name})")

    terminal: RunEvent | None = None
    try:
        async with gpu.lease_training(coord, run_id) as lease:
            if gated:
                where = f" on {lease.node}" if lease.node else ""
                yield RunEvent(
                    run_id=run_id, type="log", message=f"[argus-forge] GPU lease granted via {lease.via}{where}"
                )
            oom = False
            async with contextlib.aclosing(_run_process(command, cwd, child_env(req), run_id)) as events:
                async for ev in events:
                    if ev.type == "log":
                        oom = oom or any(m in (ev.message or "") for m in _OOM_MARKERS)
                        yield ev
                    else:
                        terminal = ev
            if terminal is not None and (terminal.type == "error" or terminal.returncode != 0):
                why = "out of memory" if oom else (terminal.message or f"exit {terminal.returncode}")
                raise _RunFailed(why)
    except gpu.LeaseNotGranted as exc:
        terminal = RunEvent(run_id=run_id, type="error", message=f"GPU lease not granted: {exc}")
    except _RunFailed:
        pass
    if terminal is not None:
        yield terminal


async def _run_process(command: list[str], cwd: str, env: dict[str, str], run_id: str) -> AsyncIterator[RunEvent]:
    """Run *command* and yield one ``log`` per output line, then ``exit`` (or ``error`` if it could not launch)."""
    logger.info("run_start", run_id=run_id, command=command, cwd=cwd)
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,  # own process group so we can reap grandchildren
        )
    except (OSError, ValueError) as exc:  # bash off PATH, bad cwd, NUL in an env value
        yield RunEvent(run_id=run_id, type="error", message=f"failed to launch {command[0]}: {exc}")
        return

    try:
        assert proc.stdout is not None
        async for line in _iter_output_lines(proc.stdout):
            yield RunEvent(run_id=run_id, type="log", message=line)
        returncode = await proc.wait()
        logger.info("run_done", run_id=run_id, returncode=returncode)
        yield RunEvent(run_id=run_id, type="exit", returncode=returncode)
    finally:
        await _terminate(proc)

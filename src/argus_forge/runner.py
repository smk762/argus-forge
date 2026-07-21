"""Job-runner mode: shell out to the forged trainer and stream progress.

``argus-forge config`` writes a runnable launcher (``train.sh``) under
``<export_dir>/forge/<trainer>/``; this module executes it and streams the
trainer's output as :class:`RunEvent` NDJSON lines. The CLI ``run`` verb and the
server's ``POST /run`` are thin shells around :func:`astream_run`.

Trust model: ``/run`` executes a script forge generated, on the shared dataset
volume — the single-user LAN assumption of ``/config``. But note this is *not*
sandboxed: the environment is caller-supplied and a trainer script runs real
code, so treat reaching the port as equivalent to shell access on the host.
:data:`BLOCKED_ENV_KEYS` refuses the env vars that would let a request silently
redirect *what* runs (before the forged script); it is defence-in-depth, not a
sandbox.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import uuid
from collections.abc import AsyncIterator

import structlog

from argus_forge.emitters import TRAINER_INFO
from argus_forge.manifest import FORGE_DIR_NAME, resolve_export_dir
from argus_forge.models import ForgeError, RunEvent, RunRequest

logger = structlog.get_logger()

# Trainers forge can run: those that declare a launcher entrypoint. Derived from
# the machine field TrainerInfo.entrypoint (not the human ``files`` list), so a
# new runnable trainer only needs to set entrypoint and relabelling ``files``
# can't silently break runnability.
RUNNABLE_TRAINERS: tuple[str, ...] = tuple(t for t, info in TRAINER_INFO.items() if info.entrypoint)

# Env vars refused from caller-supplied env: each lets a request redirect what
# executes before/instead of the forged script — bash sources BASH_ENV/ENV for
# non-interactive shells, the dynamic loader honours LD_PRELOAD/LD_AUDIT, and
# PATH decides which binary the bare name "bash" below even resolves to (set it
# to a directory holding an attacker's ./bash and the forged script never runs).
BLOCKED_ENV_KEYS: frozenset[str] = frozenset({"BASH_ENV", "ENV", "LD_PRELOAD", "LD_AUDIT", "PATH"})

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
    blocked = sorted(k for k in req.env if k in BLOCKED_ENV_KEYS)
    if blocked:
        raise ForgeError(f"env may not set {', '.join(blocked)} — they redirect what code runs, not just where it runs")
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
    safe. Empty segments (a bare ``\\r\\n`` boundary, trailing separators) are
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
                    yield bytes(buf[:_MAX_SEGMENT]).decode("utf-8", "replace")
                    del buf[:_MAX_SEGMENT]
                    continue
                break
            if idx > 0:  # skip empties from adjacent separators
                yield bytes(buf[:idx]).decode("utf-8", "replace")
            del buf[: idx + 1]
    if buf:
        yield bytes(buf).decode("utf-8", "replace")


def _first_separator(buf: bytearray) -> int:
    """Index of the first CR or LF in *buf*, or -1."""
    cr = buf.find(b"\r")
    lf = buf.find(b"\n")
    if cr < 0:
        return lf
    if lf < 0:
        return cr
    return min(cr, lf)


async def _terminate(proc: asyncio.subprocess.Process) -> None:
    """Stop the child's process group so a cancelled/failed run leaves nothing
    orphaned (``accelerate`` is a grandchild of ``bash``; the child is a group
    leader via ``start_new_session``). SIGTERM is sent synchronously so it fires
    even while the surrounding task is being cancelled; SIGKILL escalates if it
    lingers."""
    if proc.returncode is not None:
        return
    _signal_group(proc, signal.SIGTERM)
    with contextlib.suppress(Exception):
        await asyncio.wait_for(asyncio.shield(proc.wait()), timeout=5)
    if proc.returncode is None:
        _signal_group(proc, signal.SIGKILL)


def _signal_group(proc: asyncio.subprocess.Process, sig: int) -> None:
    if proc.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(proc.pid), sig)


async def astream_run(
    req: RunRequest,
    *,
    run_id: str | None = None,
    resolved: tuple[list[str], str] | None = None,
) -> AsyncIterator[RunEvent]:
    """Launch the forged trainer and yield its lifecycle + log events.

    Emits ``start`` (the resolved command), then one ``log`` per output line,
    then a terminal event: ``error`` if the process could not be launched, else
    ``exit`` with the return code. ``dry_run`` stops after ``start``. Every path
    ends in exactly one terminal event, and the child's process group is always
    reaped (finally), even on cancellation / consumer disconnect.

    *resolved* is a pre-computed ``(command, cwd)`` (from :func:`prepare_run`)
    the server threads in to avoid re-validating; the CLI passes None.
    """
    run_id = run_id or new_run_id()
    command, cwd = resolved if resolved is not None else prepare_run(req)

    yield RunEvent(run_id=run_id, type="start", command=command, cwd=cwd)
    if req.dry_run:
        return

    env = {**os.environ, **req.env}
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

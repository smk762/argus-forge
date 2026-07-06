"""Job-runner mode: shell out to the forged trainer and stream progress.

``argus-forge config`` writes a runnable ``train.sh`` under
``<export_dir>/forge/<trainer>/``; this module executes it and streams the
trainer's output as :class:`RunEvent` NDJSON lines. The CLI ``run`` verb and the
server's ``POST /run`` are thin shells around :func:`astream_run`.

Trust model: this shells out to a script forge itself generated, on the shared
dataset volume — the same single-user LAN assumption as ``/config``'s filesystem
writes. The command is fixed (``bash train.sh``); only the environment (where
the trainer checkout lives) is caller-supplied.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import structlog

from argus_forge.emitters import TRAINER_INFO
from argus_forge.manifest import FORGE_DIR_NAME
from argus_forge.models import ForgeError, RunEvent, RunRequest

logger = structlog.get_logger()

TRAIN_SCRIPT = "train.sh"

# Trainers forge can run — those whose emitted files include a train.sh. Derived
# from the catalogue so a new runnable trainer needs no change here (OneTrainer,
# driven from its own UI, has no train.sh and is excluded automatically).
RUNNABLE_TRAINERS: tuple[str, ...] = tuple(t for t, info in TRAINER_INFO.items() if TRAIN_SCRIPT in info.files)


def new_run_id() -> str:
    """A fresh training_run_id — the join key for a run's events and eval results."""
    return uuid.uuid4().hex


def prepare_run(req: RunRequest) -> tuple[list[str], str]:
    """Validate *req* and return the ``(command, cwd)`` to launch.

    Raises :class:`ForgeError` if the trainer has no train.sh to run, or if the
    config has not been forged yet — call this *before* streaming so a caller can
    surface the failure up front (a 400, a non-zero exit) instead of mid-stream.
    """
    if req.trainer not in RUNNABLE_TRAINERS:
        raise ForgeError(
            f"trainer {req.trainer!r} produces no {TRAIN_SCRIPT} to run (runnable: {', '.join(RUNNABLE_TRAINERS)})"
        )
    export_dir = Path(os.path.abspath(Path(req.export_dir).expanduser()))
    script = export_dir / FORGE_DIR_NAME / req.trainer / TRAIN_SCRIPT
    if not script.is_file():
        raise ForgeError(
            f"no forged config to run at {script} — run `argus-forge config {export_dir} --trainer {req.trainer}` first"
        )
    return ["bash", str(script)], str(script.parent)


async def astream_run(req: RunRequest, *, run_id: str | None = None) -> AsyncIterator[RunEvent]:
    """Launch the forged trainer and yield its lifecycle + log events.

    Emits a ``start`` event (the resolved command), then one ``log`` event per
    line of trainer output, then an ``exit`` event with the return code. On
    ``dry_run`` it stops after ``start``. :func:`prepare_run` is called up front,
    so a missing/invalid config raises before any event is yielded.
    """
    run_id = run_id or new_run_id()
    command, cwd = prepare_run(req)

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
        )
    except OSError as exc:  # e.g. bash not on PATH
        yield RunEvent(run_id=run_id, type="error", message=f"failed to launch {command[0]}: {exc}")
        return

    assert proc.stdout is not None
    try:
        async for raw in proc.stdout:
            yield RunEvent(run_id=run_id, type="log", message=raw.decode("utf-8", "replace").rstrip("\r\n"))
    except ValueError as exc:  # a single line longer than the stream buffer
        yield RunEvent(run_id=run_id, type="error", message=f"log stream error: {exc}")

    returncode = await proc.wait()
    logger.info("run_done", run_id=run_id, returncode=returncode)
    yield RunEvent(run_id=run_id, type="exit", returncode=returncode)

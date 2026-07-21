"""FastAPI micro-server for argus-forge (peer to argus-curator on :8103).

Routes:

    GET  /health           -> {status, service, version, training}
    GET  /trainers         -> list[TrainerInfo]
    POST /inspect          -> DatasetInfo    (read-only look at an export dir)
    POST /config           -> ForgeResult    (render configs; dry_run for preview)
    POST /run              -> RunState        (start a background run, return its id)
    GET  /runs             -> list[RunState]  (registry of tracked runs)
    GET  /run/{id}         -> RunState        (status; poll for terminal result)
    GET  /run/{id}/stream  -> NDJSON stream   (attach: backlog + live events)
    POST /run/{id}/cancel  -> RunState        (stop the run's process group)

Config generation is filesystem work on the shared dataset volume — the same
trust model as argus-curator's /export (single-user LAN tool; the UI sends
container paths like /data/out/...). ``POST /run`` shells out to a script forge
generated (see :mod:`argus_forge.runner`) on a background job that outlives the
request (see :mod:`argus_forge.jobs`): it returns the run's id immediately, and
progress is watched (and re-watched) via ``GET /run/{id}/stream``.

Because a run is real code execution on the host (see the trust note in
:mod:`argus_forge.runner`), a host that should render configs but never train —
a public demo, a GPU-less box — starts the app with ``allow_run=False``. Config
generation stays fully available; ``POST /run`` refuses with 403. The mode is
advertised on ``GET /health`` as ``training``, so a frontend can disable its
train button rather than discovering the 403 by clicking it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

try:
    from fastapi import Depends, FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
except ImportError as exc:  # pragma: no cover
    raise ImportError("Server requires: pip install argus-forge[server]") from exc

from argus_forge import __version__
from argus_forge.core import forge_config
from argus_forge.emitters import TRAINER_INFO
from argus_forge.jobs import Job, JobRegistry
from argus_forge.manifest import inspect_export, resolve_export_dir
from argus_forge.models import (
    DatasetInfo,
    ForgeError,
    ForgeRequest,
    ForgeResult,
    InspectRequest,
    RunEvent,
    RunRequest,
    RunState,
    TrainerInfo,
)
from argus_forge.runner import prepare_run


class NDJSONResponse(StreamingResponse):
    """StreamingResponse fixed to NDJSON, so the route's OpenAPI advertises
    ``application/x-ndjson`` instead of the default (misleading) JSON."""

    media_type = "application/x-ndjson"


async def _ndjson(events: AsyncIterator[RunEvent]) -> AsyncIterator[str]:
    async for event in events:
        yield event.model_dump_json() + "\n"


def _stream(job: Job) -> NDJSONResponse:
    """An NDJSON view of *job* — backlog then live events — tagged with its run id."""
    return NDJSONResponse(_ndjson(job.subscribe()), headers={"X-Training-Run-Id": job.run_id})


def create_app(
    cors: bool = False,
    cors_origins: list[str] | None = None,
    allow_run: bool = True,
) -> FastAPI:
    """Create the forge FastAPI application.

    With ``allow_run=False`` (demo-safe mode) the app renders configs but never
    trains: ``POST /run`` is 403 and ``GET /health`` reports ``training:
    disabled``.
    """
    registry = JobRegistry()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # No startup work; on shutdown, cancel every in-flight run so no trainer
        # is left without an owner in this process.
        try:
            yield
        finally:
            await registry.shutdown()

    app = FastAPI(
        title="Argus Forge",
        description="Training bridge: curated exports in, ready-to-run LoRA training configs out.",
        version=__version__,
        lifespan=lifespan,
    )

    if cors:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins or ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            # Non-simple headers are hidden from cross-origin JS unless exposed;
            # the run-id join key rides here on GET /run/{id}/stream.
            expose_headers=["X-Training-Run-Id"],
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "argus-forge",
            "version": __version__,
            # Lets a client disable its train affordance up front instead of
            # learning about demo-safe mode from a 403.
            "training": "enabled" if allow_run else "disabled",
        }

    @app.get("/trainers", response_model=list[TrainerInfo])
    async def trainers() -> list[TrainerInfo]:
        return list(TRAINER_INFO.values())

    @app.post("/inspect", response_model=DatasetInfo)
    async def inspect(req: InspectRequest) -> DatasetInfo:
        try:
            info, _ = await asyncio.to_thread(inspect_export, resolve_export_dir(req.export_dir), req.category)
        except ForgeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return info

    @app.post("/config", response_model=ForgeResult)
    async def config(req: ForgeRequest) -> ForgeResult:
        try:
            return await asyncio.to_thread(forge_config, req)
        except ForgeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"forge failed: {exc}") from exc

    TRAINING_DISABLED = {403: {"description": "training is disabled on this host"}}

    @app.post("/run", response_model=RunState, status_code=202, responses=TRAINING_DISABLED)
    async def run(req: RunRequest) -> RunState:
        # Refuse before validating: on a demo-safe host the answer is the same
        # for a well-formed request as a broken one, and a 400 would imply the
        # request could have worked.
        if not allow_run:
            raise HTTPException(
                status_code=403,
                detail="training is disabled on this host — POST /config still renders configs to run elsewhere",
            )
        # Validate (off the event loop) up front so a missing/invalid config is a
        # 400. The run then executes on a background job that outlives this
        # request; the caller gets its id back and watches via GET /run/{id}/stream.
        try:
            command, cwd = await asyncio.to_thread(prepare_run, req)
        except ForgeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return registry.start(req, command, cwd).state()

    @app.get("/runs", response_model=list[RunState])
    async def runs() -> list[RunState]:
        return [job.state() for job in registry.list()]

    def _require_job(run_id: str) -> Job:
        """Resolve a run by id or 404 — shared by the three /run/{id} endpoints."""
        job = registry.get(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
        return job

    NOT_FOUND = {404: {"description": "no such run"}}

    @app.get("/run/{run_id}", response_model=RunState, responses=NOT_FOUND)
    async def run_status(job: Job = Depends(_require_job)) -> RunState:
        return job.state()

    @app.get("/run/{run_id}/stream", response_class=NDJSONResponse, responses=NOT_FOUND)
    async def run_stream(job: Job = Depends(_require_job)) -> NDJSONResponse:
        return _stream(job)

    @app.post("/run/{run_id}/cancel", response_model=RunState, responses=NOT_FOUND)
    async def run_cancel(job: Job = Depends(_require_job)) -> RunState:
        await job.cancel()
        return job.state()

    return app

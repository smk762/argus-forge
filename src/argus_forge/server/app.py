"""FastAPI micro-server for argus-forge (peer to argus-curator on :8103).

Routes:

    GET  /health           -> {status, service, version}
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


def create_app(cors: bool = False, cors_origins: list[str] | None = None) -> FastAPI:
    """Create the forge FastAPI application."""
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
        return {"status": "ok", "service": "argus-forge", "version": __version__}

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

    @app.post("/run", response_model=RunState, status_code=202)
    async def run(req: RunRequest) -> RunState:
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

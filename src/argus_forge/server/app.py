"""FastAPI micro-server for argus-forge (peer to argus-curator on :8103).

Routes:

    GET  /health    -> {status, service, version}
    GET  /trainers  -> list[TrainerInfo]
    POST /inspect   -> DatasetInfo       (read-only look at an export dir)
    POST /config    -> ForgeResult       (render configs; dry_run for preview)
    POST /run       -> NDJSON stream     (shell out to the forged trainer)

Config generation is filesystem work on the shared dataset volume — the same
trust model as argus-curator's /export (single-user LAN tool; the UI sends
container paths like /data/out/...). /run additionally shells out to a script
forge generated; see :mod:`argus_forge.runner`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
except ImportError as exc:  # pragma: no cover
    raise ImportError("Server requires: pip install argus-forge[server]") from exc

from pathlib import Path

from argus_forge import __version__
from argus_forge.core import forge_config
from argus_forge.emitters import TRAINER_INFO
from argus_forge.manifest import inspect_export
from argus_forge.models import (
    DatasetInfo,
    ForgeError,
    ForgeRequest,
    ForgeResult,
    InspectRequest,
    RunRequest,
    TrainerInfo,
)
from argus_forge.runner import astream_run, new_run_id, prepare_run


def create_app(cors: bool = False, cors_origins: list[str] | None = None) -> FastAPI:
    """Create the forge FastAPI application."""
    app = FastAPI(
        title="Argus Forge",
        description="Training bridge: curated exports in, ready-to-run LoRA training configs out.",
        version=__version__,
    )

    if cors:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins or ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
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
            info, _ = await asyncio.to_thread(inspect_export, Path(req.export_dir).expanduser(), req.category)
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

    @app.post("/run")
    async def run(req: RunRequest) -> StreamingResponse:
        # Validate before opening the stream so a missing/invalid config is a 400,
        # not a broken NDJSON body (HTTP status is fixed once streaming starts).
        try:
            prepare_run(req)
        except ForgeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        run_id = new_run_id()

        async def body() -> AsyncIterator[str]:
            async for event in astream_run(req, run_id=run_id):
                yield event.model_dump_json() + "\n"

        return StreamingResponse(
            body(),
            media_type="application/x-ndjson",
            headers={"X-Training-Run-Id": run_id},
        )

    return app

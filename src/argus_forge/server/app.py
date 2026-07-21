"""FastAPI micro-server for argus-forge (peer to argus-curator on :8103).

Routes:

    GET  /health           -> {status, service, version, export_root, training}
    GET  /trainers         -> list[TrainerInfo]
    POST /inspect          -> DatasetInfo    (read-only look at an export dir)
    POST /config           -> ForgeResult    (render configs; dry_run for preview)
    POST /run              -> RunState        (start a background run, return its id)
    GET  /runs             -> list[RunState]  (registry of tracked runs)
    GET  /run/{id}         -> RunState        (status; poll for terminal result)
    GET  /run/{id}/stream  -> NDJSON stream   (attach: backlog + live events)
    POST /run/{id}/cancel  -> RunState        (stop the run's process group)

Config generation is filesystem work on the shared dataset volume, and a
request's ``export_dir`` is **untrusted**: it resolves under the configured
export root (``--export-root`` / ``ARGUS_FORGE_EXPORT_ROOT`` /
``FORGE_EXPORT_PATH``), refusing traversal escapes, and the endpoints refuse
outright when no root is configured. Without that fence, ``POST /config`` writes
a ``forge/`` tree into any directory the caller names. The CLI stays
unconstrained by design — it is the operator's own shell.

``POST /run`` shells out to a script forge generated (see
:mod:`argus_forge.runner`) on a background job that outlives the request (see
:mod:`argus_forge.jobs`): it returns the run's id immediately, and progress is
watched (and re-watched) via ``GET /run/{id}/stream``.

Because a run is real code execution on the host (see the trust note in
:mod:`argus_forge.runner`), a host that should render configs but never train —
a public demo, a GPU-less box — starts the app with ``allow_run=False``. Config
generation stays fully available; ``POST /run`` refuses with 403. The mode is
advertised on ``GET /health`` as ``training``, so a frontend can disable its
train button rather than discovering the 403 by clicking it.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

try:
    from argus_cortex.server import WriteGuard, cross_site_refuse
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

# Browser origins allowed by a bare --cors: the argus-studio dev frontend.
_LOCALHOST_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]


def _resolve_within(root: Path, requested: str) -> Path:
    """Resolve *requested* under *root*, refusing path-traversal escapes.

    ``requested`` is canonically relative to the root; an absolute path is
    tolerated only when it already lies inside the root, since the studio UI
    echoes back the absolute ``export_dir`` forge itself reported.
    """
    try:
        root = root.resolve()
        candidate = (root / requested).resolve() if not os.path.isabs(requested) else Path(requested).resolve()
    except (ValueError, OSError) as exc:
        # e.g. an embedded NUL byte — a 400, not an unhandled 500 traceback.
        raise HTTPException(status_code=400, detail=f"malformed path: {requested}") from exc
    if candidate != root and root not in candidate.parents:
        raise HTTPException(status_code=400, detail=f"path escapes the export root: {requested}")
    return candidate


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
    cors_allow_any: bool = False,
    export_root: str | None = None,
    allow_run: bool = True,
) -> FastAPI:
    """Create the forge FastAPI application.

    Request-supplied ``export_dir`` values are untrusted: they resolve under
    *export_root* (refusing traversal escapes), and the endpoints refuse
    outright when it is not configured.

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

    if cors_origins is None and (env_origins := os.environ.get("FORGE_CORS_ORIGINS")):
        cors_origins = [o.strip() for o in env_origins.split(",") if o.strip()]

    # A literal wildcard is only honoured by browsers without credentials; with
    # allow_credentials=True Starlette reflects any Origin back, which defeats
    # the allow-list entirely. An explicit "*" in the allow-list means the same
    # thing as --cors-any, so it takes the same safe path rather than silently
    # becoming origin reflection.
    wildcard = cors_allow_any or bool(cors_origins and "*" in cors_origins)
    # Origins the operator has actually named. The wildcard grants anonymous
    # READ access from anywhere, but never a cross-site write: a public demo
    # must not double as a way to make this host render configs or train.
    trusted_origins: list[str] = [] if wildcard else list(cors_origins or (_LOCALHOST_ORIGINS if cors else []))

    # CORS is not a write boundary — see cross_site_refuse for why an unauthed
    # LAN/localhost server must gate unsafe methods on Origin itself. Registered
    # before CORSMiddleware so CORS ends up the outer layer (add_middleware
    # inserts at 0) and can still annotate a refused write, so the caller sees a
    # readable 403 rather than an opaque CORS error.
    app.add_middleware(WriteGuard, refuse=cross_site_refuse(trusted_origins))

    if cors or cors_origins or cors_allow_any:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"] if wildcard else (cors_origins or _LOCALHOST_ORIGINS),
            allow_credentials=not wildcard,
            allow_methods=["*"],
            allow_headers=["*"],
            # Non-simple headers are hidden from cross-origin JS unless exposed;
            # the run-id join key rides here on GET /run/{id}/stream.
            expose_headers=["X-Training-Run-Id"],
        )

    # The ARGUS_* name is the deployment-facing one (argus-halo); FORGE_* is kept
    # for the compose file, matching curator's CURATOR_* pair.
    root = export_root or os.environ.get("ARGUS_FORGE_EXPORT_ROOT") or os.environ.get("FORGE_EXPORT_PATH")

    def _contained(export_dir: str) -> str:
        """A request's ``export_dir``, proven to lie under the export root."""
        if not root:
            raise HTTPException(
                status_code=400,
                detail="no export root configured (set ARGUS_FORGE_EXPORT_ROOT or pass --export-root)",
            )
        if not Path(root).is_dir():
            raise HTTPException(status_code=400, detail=f"export root is not a directory: {root}")
        return str(_resolve_within(Path(root), export_dir))

    @app.get("/health")
    async def health() -> dict[str, str | None]:
        return {
            "status": "ok",
            "service": "argus-forge",
            "version": __version__,
            "export_root": str(Path(root).resolve()) if root else None,
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
            info, _ = await asyncio.to_thread(
                inspect_export, resolve_export_dir(_contained(req.export_dir)), req.category
            )
        except ForgeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return info

    @app.post("/config", response_model=ForgeResult)
    async def config(req: ForgeRequest) -> ForgeResult:
        # Rewrite to the containment-checked path, so everything downstream
        # (emitters, caption collection, the forge/ tree) inherits the fence.
        req = req.model_copy(update={"export_dir": _contained(req.export_dir)})
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
        req = req.model_copy(update={"export_dir": _contained(req.export_dir)})
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

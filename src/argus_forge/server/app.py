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
request's ``export_dir`` is **untrusted**: it must name a directory *strictly
under* the configured export root (``--export-root`` /
``ARGUS_FORGE_EXPORT_ROOT`` / ``FORGE_EXPORT_PATH``), and the endpoints refuse
outright when no root is configured. Without that fence, ``POST /config`` writes
a ``forge/`` tree into any directory the caller names. The CLI stays
unconstrained by design — it is the operator's own shell.

Containment is decided on the fully symlink-resolved path, so a symlink out of
the root cannot be followed; but the path handed downstream keeps the caller's
spelling, because :func:`argus_forge.manifest.resolve_export_dir` owns that
policy ("absolute but NOT symlink-resolved") and ``path_map`` prefixes are
matched against it. Manifest ``abs_path`` caption sources are fenced too — see
``caption_source_root`` on :class:`~argus_forge.models.ForgeRequest`.

``POST /run`` shells out to a script forge generated (see
:mod:`argus_forge.runner`) on a background job that outlives the request (see
:mod:`argus_forge.jobs`): it returns the run's id immediately, and progress is
watched (and re-watched) via ``GET /run/{id}/stream``.

Because a run is real code execution on the host (see the trust note in
:mod:`argus_forge.runner`), a host that should render configs but never train —
a public demo, a GPU-less box — starts the app with ``allow_run=False``. That
mode refuses every ``/run`` route with 403 *before* the body is validated (a
:class:`WriteGuard`, so a route added later is covered without a guard to
forget), and forces ``POST /config`` to ``dry_run``: an unauthenticated public
host renders configs but never writes to the shared volume. The mode is
advertised on ``GET /health`` as ``training``, so a frontend can disable its
train button rather than discovering the 403 by clicking it.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path

try:
    from argus_cortex.server import FALSY, TRUTHY, WriteGuard, cross_site_refuse, env_flag
    from fastapi import Depends, FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from starlette.datastructures import Headers
    from starlette.types import Scope
except ImportError as exc:  # pragma: no cover
    raise ImportError("Server requires: pip install argus-forge[server]") from exc

from argus_forge import __version__
from argus_forge.core import forge_config
from argus_forge.emitters import TRAINER_INFO
from argus_forge.jobs import Job, JobRegistry
from argus_forge.manifest import inspect_export, resolve_export_dir
from argus_forge.models import (
    ARGUS_ROOT_ENV,
    CORS_ORIGINS_ENV,
    LEGACY_ROOT_ENV,
    READONLY_ENV,
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


def _normalize_origin(origin: str) -> str:
    """An ``Origin`` header value in the form browsers actually send it.

    Trailing slashes and surrounding whitespace are a common way to write an
    allow-list entry that can never match, since Starlette and the write guard
    both compare the header verbatim.
    """
    return origin.strip().rstrip("/")


def _dedupe(origins: Iterable[str]) -> list[str]:
    """*origins* with duplicates dropped, first occurrence winning."""
    seen: set[str] = set()
    out: list[str] = []
    for origin in origins:
        if origin and origin not in seen:
            seen.add(origin)
            out.append(origin)
    return out


def _resolve_within(root: Path, resolved_root: Path, requested: str) -> Path:
    """*requested* as a path under *root*, refusing traversal escapes.

    ``requested`` is canonically relative to the root; an absolute path is
    tolerated when it lies inside the root, since the studio UI echoes back the
    absolute ``export_dir`` forge itself reported.

    Containment is decided on the fully resolved path (*resolved_root* is the
    root's own realpath), so a symlink pointing out of the root is refused
    rather than followed. The value *returned* keeps the caller's spelling —
    only made absolute — because :func:`resolve_export_dir` owns that policy and
    ``path_map`` prefixes are matched against the un-dereferenced form.

    The check is run on the *absolutised* candidate, i.e. on exactly the path
    this function returns and every caller then opens. Resolving the raw
    candidate instead would validate a different file: ``os.path.abspath``
    cancels ``..`` lexically while ``resolve()`` cancels it against the symlink's
    *target*, so for a root containing ``d -> <root>/x/y`` the request
    ``d/../../evil`` resolves to ``<root>/evil`` (inside, so it would pass) yet
    abspaths to ``<root>/../evil`` (outside, and that is the path actually used).
    Deciding on the returned form keeps "checked" and "used" the same file.
    """
    candidate = Path(requested) if os.path.isabs(requested) else root / requested
    absolute = Path(os.path.abspath(candidate))
    try:
        probe = absolute.resolve()
    except (ValueError, OSError, RuntimeError) as exc:
        # An embedded NUL (ValueError), an unreadable component (OSError), or a
        # symlink loop — which pathlib raises as RuntimeError, *not* an OSError,
        # on every Python this package supports. A 400, not a 500 traceback.
        raise HTTPException(status_code=400, detail=f"malformed path: {requested}") from exc
    if probe == resolved_root:
        # An empty or "." export_dir lands here. Treating it as "the whole
        # shared volume is one dataset" would merge every sibling export and
        # write a forge/ tree at the root, so it is a request error.
        raise HTTPException(
            status_code=400,
            detail="export_dir must name a directory under the export root, not the root itself",
        )
    if resolved_root not in probe.parents:
        raise HTTPException(status_code=400, detail=f"path escapes the export root: {requested}")
    return absolute


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


TRAINING_DISABLED_DETAIL = (
    "training is disabled on this host — POST /config still renders configs (dry-run) to run elsewhere"
)


def env_readonly() -> bool:
    """Whether demo-safe mode is on, per ``ARGUS_FORGE_READONLY``.

    This is a *protection* flag, so it fails **safe**: a set-but-unrecognised
    value (a typo like ``=y`` or ``=enabled``) keeps the guard on rather than
    silently enabling training and writes on a host that is unauthenticated and
    public by assumption. ``env_flag``'s "unrecognised means off" suits
    enable-a-feature flags, where off is the safe direction; here it is not.
    Unset, or an explicit falsy value (``0``/``false``/``no``/``off``), allows
    runs. Never fatal — under compose's ``restart: unless-stopped`` a hard exit
    would be a crash loop.
    """
    raw = os.environ.get(READONLY_ENV, "").strip().lower()
    on = env_flag(READONLY_ENV)  # recognised truthy; also logs the typo warning
    if raw and raw not in FALSY and raw not in TRUTHY:
        return True
    return on


def _refuse_training(scope: Scope, headers: Headers) -> str | None:
    """Refuse every unsafe method on a ``/run`` route (demo-safe mode).

    Path-based rather than per-route, so a ``/run`` endpoint added later is
    covered without a guard to forget, and so the refusal lands *before* the
    body is validated — on such a host a malformed request deserves the same
    answer as a well-formed one, and a 422 would imply it could have worked.
    """
    path = scope.get("path", "")
    if path == "/run" or path.startswith("/run/"):
        return TRAINING_DISABLED_DETAIL
    return None


def create_app(
    cors: bool = False,
    cors_origins: list[str] | None = None,
    cors_allow_any: bool = False,
    export_root: str | None = None,
    allow_run: bool | None = None,
) -> FastAPI:
    """Create the forge FastAPI application.

    Request-supplied ``export_dir`` values are untrusted: they must name a
    directory under *export_root*, and the endpoints refuse outright when it is
    not configured. *export_root* falls back to ``ARGUS_FORGE_EXPORT_ROOT`` then
    ``FORGE_EXPORT_PATH``.

    CORS: *cors* allows the localhost studio frontend, *cors_origins* names
    additional origins (defaulting to ``FORGE_CORS_ORIGINS``), and
    *cors_allow_any* — or a literal ``"*"`` among the origins — grants
    credential-less reads to anyone. Named origins keep their cross-site write
    grant even alongside the wildcard, so a public demo can still drive its own
    frontend.

    *allow_run* defaults to ``ARGUS_FORGE_READONLY``. With ``allow_run=False``
    (demo-safe mode) the app renders configs but never trains or writes: every
    ``/run`` route is 403, ``POST /config`` is forced to ``dry_run``, and
    ``GET /health`` reports ``training: disabled``.
    """
    registry = JobRegistry()

    if allow_run is None:
        allow_run = not env_readonly()

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

    raw_origins = cors_origins if cors_origins is not None else (os.environ.get(CORS_ORIGINS_ENV) or "").split(",")
    origins = _dedupe(_normalize_origin(o) for o in raw_origins)

    # A literal wildcard is only honoured by browsers without credentials; with
    # allow_credentials=True Starlette reflects any Origin back, which defeats
    # the allow-list entirely. An explicit "*" in the allow-list means the same
    # thing as --cors-any, so it takes the same safe path rather than silently
    # becoming origin reflection.
    wildcard = cors_allow_any or "*" in origins
    cors_enabled = bool(cors or origins or cors_allow_any)
    # Origins the operator actually named. --cors-origin *adds* to the localhost
    # dev frontend rather than replacing it, so the defaults ride along whenever
    # CORS is on at all — naming a production origin must not silently cost you
    # the studio frontend you were already developing against. The wildcard does
    # not erase this list either: it grants anonymous reads to everyone while the
    # named origins keep the cross-site write grant they were given explicitly.
    named = _dedupe([*(_LOCALHOST_ORIGINS if cors_enabled else []), *(o for o in origins if o != "*")])

    # Demo-safe mode: refuse every /run route before the body is validated.
    # Registered first so it is the innermost guard — a hostile cross-origin
    # caller is told it is cross-origin, not what this host can do.
    if not allow_run:
        app.add_middleware(WriteGuard, refuse=_refuse_training)

    # CORS is not a write boundary — see cross_site_refuse for why an unauthed
    # LAN/localhost server must gate unsafe methods on Origin itself. Registered
    # before CORSMiddleware so CORS ends up the outer layer (add_middleware
    # inserts at 0) and can still annotate a refused write, so the caller sees a
    # readable 403 rather than an opaque CORS error.
    app.add_middleware(WriteGuard, refuse=cross_site_refuse(named))

    if cors_enabled:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"] if wildcard else named,
            allow_credentials=not wildcard,
            allow_methods=["*"],
            allow_headers=["*"],
            # Non-simple headers are hidden from cross-origin JS unless exposed;
            # the run-id join key rides here on GET /run/{id}/stream.
            expose_headers=["X-Training-Run-Id"],
        )

    raw_root = export_root or os.environ.get(ARGUS_ROOT_ENV) or os.environ.get(LEGACY_ROOT_ENV)
    root = Path(raw_root) if raw_root else None
    # Resolved once: it cannot change for the life of the app, and doing it per
    # request would put a realpath walk on the event loop (including on /health,
    # which a liveness probe hits constantly). A root that cannot be resolved at
    # all — a symlink loop, an unreadable parent — leaves the API closed rather
    # than raising out of /health.
    resolved_root: Path | None = None
    root_error: str | None = None
    if root is None:
        root_error = f"no export root configured (set {ARGUS_ROOT_ENV} or pass --export-root)"
    else:
        try:
            resolved_root = root.resolve()
        except (ValueError, OSError, RuntimeError):
            root_error = f"export root cannot be resolved: {raw_root}"

    # What /health advertises. Two things it must NOT do: claim a root that no
    # request can actually use, and hand back a spelling that defeats path_map.
    #
    # `resolve()` is non-strict, so an unmounted volume (the published image run
    # without -v, the commonest misconfiguration) resolves fine — health would
    # answer "ok" with a root while every /inspect, /config and /run 400s on
    # "not a directory". One is_dir() at startup, off the request path, keeps the
    # probe honest. It is deliberately a snapshot: a volume mounted later still
    # works, because _fenced re-checks per request; health merely under-reports.
    #
    # The spelling is the *un-dereferenced* absolute form, the same policy
    # resolve_export_dir owns and that /inspect returns, because path_map
    # prefixes are matched against it — reporting the realpath of a symlinked
    # root (/data/out -> /mnt/big/out) would make a client that echoes this value
    # back silently lose every rewrite.
    health_root: str | None = None
    if root is not None and resolved_root is not None and root.is_dir():
        health_root = str(Path(os.path.abspath(root)))

    def _fenced(export_dir: str) -> Path:
        """A request's ``export_dir``, proven to lie under the export root.

        Called from inside the ``asyncio.to_thread`` hop, like every other
        filesystem touch in this module — ``resolve()``/``is_dir()`` are
        blocking syscalls and the export root is by design a shared (often
        network-mounted) volume.
        """
        if root is None or resolved_root is None:
            raise HTTPException(status_code=400, detail=root_error)
        if not root.is_dir():
            raise HTTPException(status_code=400, detail=f"export root is not a directory: {raw_root}")
        return _resolve_within(root, resolved_root, export_dir)

    # Fixed for the life of the app, so it is built once rather than stat'ing
    # the export root on every liveness probe.
    health = {
        "status": "ok",
        "service": "argus-forge",
        "version": __version__,
        "export_root": health_root,
        # Lets a client disable its train affordance up front instead of
        # learning about demo-safe mode from a 403.
        "training": "enabled" if allow_run else "disabled",
    }

    @app.get("/health")
    async def health_route() -> dict[str, str | None]:
        return health

    @app.get("/trainers", response_model=list[TrainerInfo])
    async def trainers() -> list[TrainerInfo]:
        return list(TRAINER_INFO.values())

    @app.post("/inspect", response_model=DatasetInfo)
    async def inspect(req: InspectRequest) -> DatasetInfo:
        def work() -> DatasetInfo:
            info, _ = inspect_export(resolve_export_dir(str(_fenced(req.export_dir))), req.category)
            return info

        try:
            return await asyncio.to_thread(work)
        except ForgeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/config", response_model=ForgeResult)
    async def config(req: ForgeRequest) -> ForgeResult:
        # Whether the caller asked for a real write and demo-safe mode took it
        # away. Reported as a warning below: a 200 whose files all have a null
        # path is otherwise the only clue that nothing reached the volume.
        forced_dry_run = not allow_run and not req.dry_run
        if not allow_run:
            # A demo-safe host is unauthenticated and publicly reachable by
            # assumption, and a non-dry /config overwrites the curator's
            # metadata.jsonl and plants an executable train.sh on the shared
            # volume. Render, never write.
            req = req.model_copy(update={"dry_run": True})

        def work() -> ForgeResult:
            # Rewrite to the containment-checked path, so everything downstream
            # (emitters, caption collection, the forge/ tree) inherits the
            # fence. caption_source_root fences the *other* untrusted path a
            # request reaches: the manifest's abs_path caption sources. It is a
            # server-side argument, never a request field, so no caller can widen
            # its own fence.
            fenced = _fenced(req.export_dir)
            result = forge_config(
                req.model_copy(update={"export_dir": str(fenced)}),
                caption_source_root=str(resolved_root),
            )
            if forced_dry_run:
                result.warnings.append(
                    "demo-safe mode: rendered dry-run only — nothing was written to the export dir "
                    "(GET /health reports training: disabled)"
                )
            return result

        try:
            return await asyncio.to_thread(work)
        except ForgeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            # A containment refusal is already the right status; do not let the
            # catch-all below relabel it as a 500.
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"forge failed: {exc}") from exc

    TRAINING_DISABLED = {403: {"description": "training is disabled on this host"}}

    @app.post("/run", response_model=RunState, status_code=202, responses=TRAINING_DISABLED)
    async def run(req: RunRequest) -> RunState:
        # Demo-safe mode never reaches here: the WriteGuard above refuses every
        # /run route before the body is validated.
        def work() -> tuple[RunRequest, list[str], str]:
            fenced = req.model_copy(update={"export_dir": str(_fenced(req.export_dir))})
            command, cwd = prepare_run(fenced)
            return fenced, command, cwd

        # Validate (off the event loop) up front so a missing/invalid config is a
        # 400. The run then executes on a background job that outlives this
        # request; the caller gets its id back and watches via GET /run/{id}/stream.
        try:
            fenced, command, cwd = await asyncio.to_thread(work)
        except ForgeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return registry.start(fenced, command, cwd).state()

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

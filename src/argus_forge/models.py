"""Pydantic models — the forge stage's data contract.

argus-forge sits between argus-curator's export and an actual trainer run.
Its input contract is the curator handoff manifest (``manifest.jsonl``, one
row per selected image, each row stamped with ``manifest_version`` and the
shared :class:`TargetProfile`); its output contract is :class:`ForgeResult`
(the trainer-native files it rendered plus the resolved hyperparameters).

``TargetProfile`` / ``TargetCategory`` mirror ``argus_curator.models``
verbatim — same taxonomy, no remapping (or, eventually, hoist both into a
shared ``argus-core`` package).
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field, model_validator

TargetStyle = Literal["photo", "anime"]
TargetCategory = Literal["identity", "wardrobe", "pose_composition", "setting"]

TrainerId = Literal["kohya", "onetrainer", "diffusers"]

TRAINERS: tuple[TrainerId, ...] = ("kohya", "onetrainer", "diffusers")

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Major versions of the curator handoff manifest this build understands.
# argus-curator stamps every row with its MANIFEST_VERSION; forge refuses a
# manifest whose major is not listed here instead of misreading it.
# 2.x: rows carry ``exported_path`` (the real destination under the export
#      root, de-collided by the curator) and exist only for files whose
#      transfer succeeded. 1.x: legacy — destinations are re-derived from
#      ``rel_path`` by probing.
SUPPORTED_MANIFEST_MAJORS: tuple[str, ...] = ("1", "2")

# Majors whose rows must carry ``exported_path``. Kept as an explicit set beside
# SUPPORTED_MANIFEST_MAJORS — rather than derived as "any major that isn't 1" —
# so adding a future major is a deliberate decision about whether it uses
# exported_path, not a silent consequence of a comparison.
MAJORS_REQUIRING_EXPORTED_PATH: frozenset[str] = frozenset({"2"})


def manifest_major(version: str) -> str:
    """The major component of a ``manifest_version`` string (``'2.7' -> '2'``)."""
    return version.split(".", 1)[0]


# Caption sidecar extension (argus-lens writes these next to the images).
CAPTION_EXT = ".txt"

# Env var holding a default path_map as "container=host[,container=host...]"
# (the compose file sets it from OUTPUT_DIR). Request-level maps win over it.
PATH_MAP_ENV = "FORGE_PATH_MAP"

# Deployment knobs, named here (not in the server module) so the CLI can talk
# about them without importing FastAPI, and so there is one spelling of each.
# Containment root for request-supplied paths; ARGUS_* is the deployment-facing
# name (argus-halo), FORGE_EXPORT_PATH the legacy alias.
ARGUS_ROOT_ENV = "ARGUS_FORGE_EXPORT_ROOT"
LEGACY_ROOT_ENV = "FORGE_EXPORT_PATH"
# Demo-safe mode: render configs, never train and never write.
READONLY_ENV = "ARGUS_FORGE_READONLY"
# Comma-separated browser origins allowed to call the API.
CORS_ORIGINS_ENV = "FORGE_CORS_ORIGINS"

# Every env var the server reads. Tests clear these so a value exported in a
# developer's shell cannot decide whether the security tests pass.
SERVER_ENV_VARS: tuple[str, ...] = (
    PATH_MAP_ENV,
    ARGUS_ROOT_ENV,
    LEGACY_ROOT_ENV,
    READONLY_ENV,
    CORS_ORIGINS_ENV,
)


class ForgeError(RuntimeError):
    """A user-facing failure: bad input dir, unreadable manifest, bad request."""


class TargetProfile(BaseModel):
    """What the dataset was curated *for* — inherited verbatim from argus-curator."""

    target_style: TargetStyle = "photo"
    target_backend: str | None = "sdxl"
    checkpoint: str | None = None
    target_category: TargetCategory = "identity"


class ManifestRow(BaseModel):
    """One line of the curator's ``manifest.jsonl`` (the fields forge consumes).

    Unknown keys are ignored so a minor-version manifest with extra columns
    still parses; a major-version bump is rejected up front in
    :func:`argus_forge.manifest.read_manifest`.
    """

    manifest_version: str
    rel_path: str
    abs_path: str
    # Where the file actually landed under the export root (posix, relative).
    # Required on 2.x rows — flattened exports de-collide basenames to
    # ``stem-<hash>.ext``, so it cannot be re-derived from rel_path. Absent
    # on 1.x rows, which predate the field.
    exported_path: str | None = None
    target_profile: TargetProfile = Field(default_factory=TargetProfile)
    primary_face_cluster: str | None = None
    primary_face_pose: str | None = None
    score: float = 0.0
    similar_group: int = 0

    @model_validator(mode="after")
    def _check_exported_path(self) -> ManifestRow:
        """Enforce the ``exported_path`` contract on every row, however built.

        This lives on the model (not only in :func:`read_manifest`) so the
        invariant holds for direct construction and any future deserialization
        path, and so :func:`argus_forge.manifest.exported_location` can trust
        it. A version whose major is in :data:`MAJORS_REQUIRING_EXPORTED_PATH`
        must carry ``exported_path``; when present it must be a non-empty
        relative path that stays inside the export root (no absolute path, no
        ``..``) — it is joined onto the export dir and caption sidecars are
        written beside the result.
        """
        if self.exported_path is not None:
            if not self.exported_path.strip():
                raise ValueError("exported_path is empty")
            rel = PurePosixPath(self.exported_path)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"exported_path {self.exported_path!r} must be a relative path inside the export root")
        elif manifest_major(self.manifest_version) in MAJORS_REQUIRING_EXPORTED_PATH:
            raise ValueError(
                f"manifest_version {self.manifest_version} row has no exported_path "
                "— the manifest is malformed; re-export with argus-curator"
            )
        return self


class TrainingParams(BaseModel):
    """Resolved hyperparameters for one LoRA run.

    ``total_steps`` counts sample presentations (images x repeats x epochs) —
    the number the curate UI displays; ``optimizer_steps`` divides by batch
    size, which is what trainers actually count.
    """

    images: int
    repeats: int
    epochs: int
    total_steps: int
    optimizer_steps: int
    network_dim: int
    network_alpha: int
    unet_lr: float
    text_encoder_lr: float
    optimizer: str
    scheduler: str
    resolution: int
    batch_size: int
    precision: str


class ParamOverrides(BaseModel):
    """Optional user overrides; anything unset keeps the heuristic value."""

    repeats: int | None = Field(default=None, ge=1)
    epochs: int | None = Field(default=None, ge=1)
    network_dim: int | None = Field(default=None, ge=1)
    network_alpha: int | None = Field(default=None, ge=1)
    unet_lr: float | None = Field(default=None, gt=0)
    text_encoder_lr: float | None = Field(default=None, gt=0)
    optimizer: str | None = None
    scheduler: str | None = None
    resolution: int | None = Field(default=None, ge=64)
    batch_size: int | None = Field(default=None, ge=1)
    precision: str | None = None


class SizeHint(BaseModel):
    """Dataset-size guidance for the target category (mirrors the curate UI)."""

    tone: Literal["empty", "low", "good", "high"]
    text: str


class DatasetInfo(BaseModel):
    """What forge found in an export directory."""

    export_dir: str
    image_count: int
    caption_count: int
    manifest_present: bool
    manifest_rows: int = 0
    manifest_version: str | None = None
    # Manifest rows whose exported image was not found on disk.
    missing_from_disk: int = 0
    target_profile: TargetProfile = Field(default_factory=TargetProfile)
    size_hint: SizeHint
    suggested: TrainingParams


class InspectRequest(BaseModel):
    """POST /inspect — look at an export dir without writing anything."""

    export_dir: str
    category: TargetCategory | None = None


class ForgeRequest(BaseModel):
    """POST /config — render trainer configs for an export directory."""

    export_dir: str
    trainer: TrainerId
    base_model: str | None = None
    trigger: str | None = None
    output_name: str | None = None
    category: TargetCategory | None = None
    overrides: ParamOverrides = Field(default_factory=ParamOverrides)
    # Copy caption sidecars that argus-lens wrote next to the *source* images
    # (manifest abs_path) into the export dir, where trainers expect them.
    collect_captions: bool = True
    # Containment root for those sidecar *sources*. ``abs_path`` comes from the
    # manifest, which on a server is as untrusted as the request itself: without
    # this, any readable ``.txt`` on the host is copied into the shared volume
    # and — for trainers that inline captions — echoed back in the response.
    # The server sets it to its export root; None (the CLI) means unconstrained,
    # since the CLI is the operator's own shell.
    caption_source_root: str | None = None
    # Prefix rewrites for absolute paths rendered into configs, e.g.
    # {"/data/out": "/home/you/argus/out"} when forge runs in a container but
    # the trainer runs on the host. Longest prefix wins; merged over the
    # FORGE_PATH_MAP env var ("container=host,container2=host2").
    path_map: dict[str, str] = Field(default_factory=dict)
    # Render and return file contents without touching the filesystem.
    dry_run: bool = False


class GeneratedFile(BaseModel):
    """One rendered file. ``name`` is relative to the export dir; ``path`` is
    the absolute location once written (None on dry runs)."""

    name: str
    path: str | None = None
    content: str


class ForgeResult(BaseModel):
    """Everything a caller (CLI, UI) needs to show or run the forged config."""

    trainer: TrainerId
    export_dir: str
    out_dir: str
    files: list[GeneratedFile]
    params: TrainingParams
    dataset: DatasetInfo
    base_model: str
    trigger: str
    output_name: str
    captions_collected: int = 0
    warnings: list[str] = Field(default_factory=list)


class RunRequest(BaseModel):
    """POST /run — execute a forged trainer config by shelling out to its train.sh."""

    export_dir: str
    trainer: TrainerId
    # Extra environment for the trainer process — typically where its checkout
    # lives, e.g. {"SD_SCRIPTS_DIR": "/home/you/kohya-ss/sd-scripts"} for kohya
    # or {"DIFFUSERS_SCRIPT": "/home/you/.../train_...py"} for diffusers.
    env: dict[str, str] = Field(default_factory=dict)
    # Resolve and report the command without executing it.
    dry_run: bool = False


RunEventType = Literal["start", "log", "exit", "error", "cancelled"]


class RunEvent(BaseModel):
    """One NDJSON line streamed from a run.

    ``run_id`` (the training_run_id) is stable for the whole run and is the join
    key for downstream eval (argus-proof). ``type`` selects which fields are set:
    ``start`` carries ``command`` + ``cwd``; ``log`` a line of trainer output in
    ``message``; ``exit`` the ``returncode``; ``error`` a failure ``message``.
    ``cancelled`` is the terminal event for a run stopped by request — distinct
    from ``error`` so a consumer never mistakes a user cancel for a failure.
    """

    run_id: str
    type: RunEventType
    message: str | None = None
    command: list[str] | None = None
    cwd: str | None = None
    returncode: int | None = None


RunStatus = Literal["running", "succeeded", "failed", "cancelled"]


class RunState(BaseModel):
    """A run's status in the server's job registry (GET /run/{id}, GET /runs).

    Outlives the connection that started the run, so a caller can poll for the
    terminal ``status`` + ``returncode`` (the argus-proof handoff) or reconnect
    to the live stream by ``run_id`` long after the launching request is gone.
    """

    run_id: str
    trainer: TrainerId
    export_dir: str
    status: RunStatus
    returncode: int | None = None
    started_at: str  # ISO-8601 UTC
    ended_at: str | None = None
    command: list[str] = Field(default_factory=list)
    cwd: str | None = None
    # Terminal detail — the launch/failure/cancel reason, when there is one — so
    # a poller can diagnose a ``failed``/``cancelled`` run without the event log.
    message: str | None = None


class TrainerInfo(BaseModel):
    """Catalogue entry for GET /trainers."""

    id: TrainerId
    label: str
    files: list[str]
    notes: str
    # The runnable launcher script this trainer emits (relative to the forge out
    # dir), or None if it has none (OneTrainer is driven from its own UI). This
    # is the machine contract `argus-forge run` keys off — distinct from the
    # human-readable ``files`` list.
    entrypoint: str | None = None


# Models that make up the HTTP/CLI wire contract, in schema order.
WIRE_MODELS: tuple[type[BaseModel], ...] = (
    TargetProfile,
    ManifestRow,
    TrainingParams,
    ParamOverrides,
    SizeHint,
    DatasetInfo,
    InspectRequest,
    ForgeRequest,
    GeneratedFile,
    ForgeResult,
    RunRequest,
    RunEvent,
    RunState,
    TrainerInfo,
)


def wire_schema() -> dict:
    """Combined JSON Schema for forge's wire contract (all WIRE_MODELS)."""
    from pydantic.json_schema import models_json_schema

    _, schema = models_json_schema(
        [(m, "serialization") for m in WIRE_MODELS],
        title="argus-forge wire contract",
        ref_template="#/$defs/{model}",
    )
    return schema

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

from typing import Literal

from pydantic import BaseModel, Field

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

# Caption sidecar extension (argus-lens writes these next to the images).
CAPTION_EXT = ".txt"

# Env var holding a default path_map as "container=host[,container=host...]"
# (the compose file sets it from OUTPUT_DIR). Request-level maps win over it.
PATH_MAP_ENV = "FORGE_PATH_MAP"


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


class TrainerInfo(BaseModel):
    """Catalogue entry for GET /trainers."""

    id: TrainerId
    label: str
    files: list[str]
    notes: str


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

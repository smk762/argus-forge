"""Pydantic models — the forge stage's data contract.

argus-forge sits between argus-curator's export and an actual trainer run.
Its input contract is the curator handoff manifest (``manifest.jsonl``, one
row per selected image, each row stamped with ``manifest_version`` and the
shared :class:`TargetProfile`); its output contract is :class:`ForgeResult`
(the trainer-native files it rendered plus the resolved hyperparameters).

``TargetProfile`` / ``TargetCategory`` / ``TargetStyle`` are argus-cortex's
shared taxonomy (``argus_cortex.taxonomy``), re-exported here — the same
literals every suite stage uses, no local copy to drift.
"""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal

# Re-exported: the rest of forge (and consumers) import the taxonomy from here.
from argus_cortex.provenance import Provenance
from argus_cortex.taxonomy import (  # noqa: F401
    TARGET_CATEGORIES,
    TARGET_STYLES,
    TargetCategory,
    TargetProfile,
    TargetStyle,
)
from pydantic import BaseModel, Field, model_validator

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


def check_inside_export_root(field: str, value: str) -> None:
    """Refuse *value* unless it is a non-empty posix path relative to, and inside, the export root.

    Curator manifests name files under the export root by a relative posix path
    (``exported_path``, a speech row's ``transcript_path``) that forge joins onto
    the export dir it was given. An absolute path, ``..``, the root itself, or a
    Windows spelling (backslash, drive, UNC) would join to somewhere else.
    """
    if not value.strip():
        raise ValueError(f"{field} is empty")
    rel = PurePosixPath(value)
    win = PureWindowsPath(value)
    if (
        rel.is_absolute()
        or ".." in rel.parts
        # "." / "./" normalise to no parts: the export root itself, not a file in it.
        or not rel.parts
        # The curator writes posix paths; a backslash or drive/UNC prefix is a
        # Windows path that a native join would treat as traversal or another root.
        or "\\" in value
        or win.drive
        or win.is_absolute()
    ):
        raise ValueError(f"{field} {value!r} must be a relative path inside the export root")


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
# Comma-separated host names (no port) this server is reached by. A page served
# from the server's own host:port (e.g. /docs) may write only when that host is
# an IP literal, localhost, or listed here: argus-cortex's DNS-rebinding guard.
TRUSTED_HOSTS_ENV = "FORGE_TRUSTED_HOSTS"
# Directory the server keeps run records in (one JSON file per run), so a
# restart doesn't forget them. Unset: runs are kept in memory only.
STATE_DIR_ENV = "ARGUS_FORGE_STATE_DIR"
# Seconds a POST /run Idempotency-Key is remembered (default 24 h); a key whose
# run is still active is kept regardless. See argus_forge.server.idempotency.
IDEMPOTENCY_RETENTION_ENV = "ARGUS_FORGE_IDEMPOTENCY_RETENTION_S"

# Every env var the server reads. Tests clear these so a value exported in a
# developer's shell cannot decide whether the security tests pass.
SERVER_ENV_VARS: tuple[str, ...] = (
    PATH_MAP_ENV,
    ARGUS_ROOT_ENV,
    LEGACY_ROOT_ENV,
    READONLY_ENV,
    CORS_ORIGINS_ENV,
    TRUSTED_HOSTS_ENV,
    STATE_DIR_ENV,
    IDEMPOTENCY_RETENTION_ENV,
)


class ForgeError(RuntimeError):
    """A user-facing failure: bad input dir, unreadable manifest, bad request."""


class PoseLabels(BaseModel):
    """The primary person's coarse body-pose labels on a curator 2.7+ row (argus-puppet's rules).

    Plain strings, not literals: ``unknown`` is a real reading (the joints a rule
    needs were not confident), and a value a newer puppet adds must not make a
    manifest unreadable. Forge only counts these; it never re-derives them.
    """

    stance: str | None = None
    arms: str | None = None
    facing: str | None = None


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
    # Curator 2.7+: argus-puppet's body pose for the primary person. Absent on
    # older rows (read as null). ``pose_cluster`` is a *scan-level* cluster id
    # (0 the largest in the curator scan, not the export); null when the pose
    # was unusable, not checked, or the scan was not clustered. Forge only
    # counts them, for the pose-dominance warning (:mod:`argus_forge.pose`).
    pose_labels: PoseLabels | None = None
    pose_cluster: int | None = None

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
            check_inside_export_root("exported_path", self.exported_path)
        elif manifest_major(self.manifest_version) in MAJORS_REQUIRING_EXPORTED_PATH:
            raise ValueError(
                f"manifest_version {self.manifest_version} row has no exported_path "
                "— the manifest is malformed; re-export with argus-curator"
            )
        if self.target_profile.modality != "image":
            # Curator writes a speech export's rows to speech_manifest.jsonl, never
            # here; forge's image heuristics are keyed by image categories only.
            raise ValueError(
                f"target_profile.modality is {self.target_profile.modality!r}: manifest.jsonl holds image rows "
                "(a speech export's rows are in speech_manifest.jsonl)"
            )
        return self


# ---------------------------------------------------------------------------
# Speech: curator's speech_manifest.jsonl (argus-curator#187, forge#58)
# ---------------------------------------------------------------------------

# Major versions of curator's speech manifest this build understands. Curator
# stamps every SpeechManifestRow with SPEECH_MANIFEST_VERSION, which moves
# independently of the image MANIFEST_VERSION under the same rule (major =
# breaking, minor = additive). 1.x: every key present, null when empty.
SUPPORTED_SPEECH_MANIFEST_MAJORS: tuple[str, ...] = ("1",)

SplitPart = Literal["train", "held_out"]


class SpeechManifestRow(BaseModel):
    """One line of curator's ``speech_manifest.jsonl`` (``SpeechManifestRow`` 1.x).

    Unknown keys are ignored so a minor version with extra columns still parses;
    an unknown major is refused up front by
    :func:`argus_forge.speech.read_speech_manifest`. Curator writes a row only for
    a clip whose transfer and ``<stem>.txt`` transcript both succeeded.

    ``exported_path`` / ``transcript_path`` (relative to the export root) and
    their ``*_abs_path`` twins are curator's de-collided destinations, never to
    be re-derived from ``rel_path``. ``age`` is the corpus's label, never an
    estimate; forge carries it and decides nothing on it.
    """

    speech_manifest_version: str
    modality: Literal["speech"]
    rel_path: str
    abs_path: str
    exported_path: str
    exported_abs_path: str
    transcript_path: str
    transcript_abs_path: str
    target_profile: TargetProfile
    transcript: str
    duration_s: float = Field(ge=0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    speaker_id: str | None = None
    speaker_label: str | None = None
    speaker_cluster: str | None = None
    speaker_mismatch: bool = False
    accent: str | None = None
    language: str | None = None
    gender: str | None = None
    age: str | None = None
    score: float = 0.0
    similar_group: int = 0
    split: SplitPart | None = None
    corpus: str
    license: str
    provenance: Provenance
    consent_release_sha256: str | None = None
    consent_record_sha256: str | None = None

    @model_validator(mode="after")
    def _check_paths(self) -> SpeechManifestRow:
        """The paths forge joins onto the export dir stay inside it, and the transcript pairs with its clip.

        Curator writes ``<stem>.txt`` beside the exported clip; a transcript
        elsewhere, or under another stem, would pair text with the wrong audio.
        """
        check_inside_export_root("exported_path", self.exported_path)
        check_inside_export_root("transcript_path", self.transcript_path)
        clip, text = PurePosixPath(self.exported_path), PurePosixPath(self.transcript_path)
        if text != clip.with_suffix(CAPTION_EXT):
            raise ValueError(
                f"transcript_path {self.transcript_path!r} is not the <stem>{CAPTION_EXT} beside "
                f"exported_path {self.exported_path!r}"
            )
        if self.target_profile.modality != "speech":
            raise ValueError(f"target_profile.modality is {self.target_profile.modality!r}, not 'speech'")
        return self


class SpeechClip(BaseModel):
    """One training (or held-out) clip, resolved on disk: what a speech emitter reads.

    ``audio`` and ``transcript_file`` are the clip and its ``<stem>.txt`` under
    the export dir as forge sees it (curator's ``exported_path`` /
    ``transcript_path`` joined onto it). ``exported_abs_path`` is curator's own
    absolute path, kept for the record: under a container it names the file in
    curator's filesystem, not forge's. ``text`` is what the ``.txt`` holds on
    disk (an edit after export wins over the manifest's ``transcript``).
    """

    audio: str
    transcript_file: str
    exported_path: str
    exported_abs_path: str
    rel_path: str
    text: str
    duration_s: float
    sample_rate: int
    channels: int
    speaker: str | None = None
    speaker_label: str | None = None
    speaker_cluster: str | None = None
    accent: str | None = None
    language: str | None = None
    gender: str | None = None
    split: SplitPart | None = None


class SpeechSource(BaseModel):
    """Where a set of clips came from: one ``(corpus, license, provenance)`` and how many clips carry it."""

    corpus: str
    license: str
    provenance: Provenance
    consent_release_sha256: str | None = None
    consent_record_sha256: str | None = None
    clips: int = 0


class SpeechDataset(BaseModel):
    """A curator speech export read for training: the input a speech (TTS) emitter takes.

    ``clips`` are the rows to train on (``split`` ``train`` or null) whose clip
    and transcript are both on disk. ``held_out`` are the reference clips from a
    separate held-out export (or ``held_out`` rows found in this one), never
    trained on. ``sources`` carries corpus, license and provenance into the run's
    record.
    """

    export_dir: str
    held_out_dir: str | None = None
    speech_manifest_version: str
    target_profile: TargetProfile
    clips: list[SpeechClip] = Field(default_factory=list)
    held_out: list[SpeechClip] = Field(default_factory=list)
    # Rows whose clip or transcript is no longer on disk (left out of clips/held_out).
    missing_from_disk: int = 0
    # Transcripts whose .txt on disk differs from the manifest's transcript (the .txt wins).
    transcript_edits: int = 0
    sources: list[SpeechSource] = Field(default_factory=list)

    @property
    def hours(self) -> float:
        return sum(c.duration_s for c in self.clips) / 3600.0

    @property
    def speakers(self) -> list[str]:
        return sorted({c.speaker for c in self.clips if c.speaker is not None})

    @property
    def languages(self) -> list[str]:
        return sorted({c.language for c in self.clips if c.language is not None})


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
    # NB: the containment root for those sidecar *sources* is deliberately NOT a
    # field here. ``abs_path`` is manifest-supplied and on a server as untrusted
    # as the request itself, so the fence is a caller-side argument to
    # ``forge_config`` (the server passes its export root, the CLI passes None)
    # — a request that could name its own fence could also widen it to "/".
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


# The only keys a run's caller-supplied ``env`` may set: where the trainer
# checkout lives (each emitted train.sh reads exactly one of the first two) and
# which GPU / model cache it uses. Anything else, loader and interpreter hooks
# above all (``LD_*``, ``BASH_ENV``, ``ENV``, ``PATH``, ``PYTHON*``), could
# change what code runs, so ``runner.prepare_run`` refuses it. A CLI user who
# needs more sets it in their own shell, which the run inherits.
RUN_ENV_KEYS: frozenset[str] = frozenset(
    {"SD_SCRIPTS_DIR", "DIFFUSERS_SCRIPT", "CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_HUB_OFFLINE"}
)


class RunRequest(BaseModel):
    """POST /run — execute a forged trainer config by shelling out to its train.sh."""

    export_dir: str
    trainer: TrainerId
    # Extra environment for the trainer process — typically where its checkout
    # lives, e.g. {"SD_SCRIPTS_DIR": "/home/you/kohya-ss/sd-scripts"} for kohya
    # or {"DIFFUSERS_SCRIPT": "/home/you/.../train_...py"} for diffusers.
    # Keys outside RUN_ENV_KEYS are refused.
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


class RunArtifact(BaseModel):
    """The LoRA a succeeded run left behind (``RunState.lora``, GET /run/{id}/lora).

    Located from the forged config the run executed (kohya's ``output_dir`` +
    ``output_name``, diffusers' ``--output_dir``), so a caller never has to know
    a trainer's output layout, and hashed by forge once the trainer has exited.
    """

    # Absolute, as the trainer forge ran wrote it (the server's own view of the volume).
    path: str
    # *path* relative to the run's export_dir when it lies inside it (the usual
    # case: ``forge/kohya/output/<name>.safetensors``), so a caller that mounts
    # the shared volume elsewhere can still find it; None when it does not.
    export_path: str | None = None
    sha256: str
    size_bytes: int


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
    # The trained LoRA, set once a run has succeeded and its file was found and
    # hashed (before ``status`` turns ``succeeded``, so a poller that sees
    # ``succeeded`` sees this too). None for a dry run, or when the trainer
    # exited 0 but left no file where its config said (``message`` says where).
    lora: RunArtifact | None = None


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
    PoseLabels,
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
    RunArtifact,
    RunState,
    TrainerInfo,
)


def wire_schema() -> dict:
    """Combined JSON Schema for forge's wire contract (all WIRE_MODELS), via argus-cortex's suite-wide builder."""
    from argus_cortex.wire import wire_schema as build

    return build(WIRE_MODELS, title="argus-forge wire contract")

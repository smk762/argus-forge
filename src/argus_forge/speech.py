"""Read curator speech exports: ``speech_manifest.jsonl`` plus the clips and transcripts on disk.

A speech export is what argus-curator's POST /export writes for a speech scan
(argus-curator#187): the kept WAV/FLAC clips, a ``<stem>.txt`` transcript beside
each, and ``speech_manifest.jsonl`` (one :class:`SpeechManifestRow` per clip,
versioned by ``speech_manifest_version``). It writes no image ``manifest.jsonl``.

With a speaker-disjoint split, curator exports each part to its own folder. The
held-out part is a reference set for argus-proof: forge trains on the train part
and, given the held-out export too, refuses to train when any held-out speaker
(by ``speaker_id``, ``speaker_label`` or ``speaker_cluster``) is in training.

:func:`read_speech_export` returns a :class:`SpeechDataset`, the input a speech
emitter (the XTTS-v2 emitter, forge#56) takes; :class:`SpeechEmitter` is that
emitter's interface.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

import structlog
from pydantic import ValidationError

from argus_forge.manifest import SPEECH_MANIFEST_NAME
from argus_forge.models import (
    SUPPORTED_SPEECH_MANIFEST_MAJORS,
    ForgeError,
    GeneratedFile,
    SpeechClip,
    SpeechDataset,
    SpeechManifestRow,
    SpeechSource,
    manifest_major,
)

logger = structlog.get_logger()

# The speaker readings a held-out export must not share with training. Each is
# compared with the same field on the other side: a cluster id and a corpus label
# are different namespaces.
SPEAKER_FIELDS: tuple[str, ...] = ("speaker_id", "speaker_label", "speaker_cluster")


def is_speech_export(export_dir: Path) -> bool:
    """Whether *export_dir* is a curator speech export (it holds ``speech_manifest.jsonl``)."""
    return (export_dir / SPEECH_MANIFEST_NAME).is_file()


def read_speech_manifest(path: Path) -> list[SpeechManifestRow]:
    """Parse ``speech_manifest.jsonl``, refusing a manifest forge can't trust.

    Each failure names its line: a row whose major is not in
    :data:`SUPPORTED_SPEECH_MANIFEST_MAJORS`, a row that breaks the
    :class:`SpeechManifestRow` contract (a missing key, a path outside the export
    root, a transcript that isn't its clip's ``<stem>.txt``, a non-speech
    profile), and a file mixing majors.
    """
    rows: list[SpeechManifestRow] = []
    understood = ", ".join(f"{m}.x" for m in SUPPORTED_SPEECH_MANIFEST_MAJORS)
    first_major: str | None = None
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception as exc:
                raise ForgeError(f"{path.name}:{lineno}: unreadable speech manifest row: {exc}") from exc
            # The version is checked before the row's shape: a 2.x row may not fit
            # the 1.x model at all, and the honest error is the version, not a field.
            version = data.get("speech_manifest_version") if isinstance(data, dict) else None
            if not isinstance(version, str) or not version:
                raise ForgeError(f"{path.name}:{lineno}: row has no speech_manifest_version")
            major = manifest_major(version)
            if major not in SUPPORTED_SPEECH_MANIFEST_MAJORS:
                raise ForgeError(
                    f"{path.name}:{lineno}: speech_manifest_version {version} is not supported "
                    f"(this build understands {understood}) — upgrade argus-forge or re-export"
                )
            if first_major is None:
                first_major = major
            elif major != first_major:
                raise ForgeError(
                    f"{path.name}:{lineno}: speech_manifest_version {version} mixes major {major}.x with "
                    f"the file's earlier {first_major}.x rows — a manifest must be one version; re-export"
                )
            try:
                rows.append(SpeechManifestRow.model_validate(data))
            except ValidationError as exc:
                detail = "; ".join(
                    f"{'.'.join(str(p) for p in e['loc']) + ': ' if e['loc'] else ''}"
                    f"{e['msg'].removeprefix('Value error, ')}"
                    for e in exc.errors()
                )
                raise ForgeError(f"{path.name}:{lineno}: {detail}") from exc
    return rows


def _resolve(export_dir: Path, row: SpeechManifestRow) -> tuple[Path, Path] | None:
    """The row's clip and transcript under *export_dir*, or None if either is gone.

    Joined from curator's ``exported_path`` / ``transcript_path`` (validated as
    inside the export root), not from ``rel_path``. ``exported_abs_path`` is in
    curator's filesystem, which under a container is not forge's; when it does
    name a file forge can see, it must be the same file, or the manifest
    describes some other export and is refused.
    """
    audio = export_dir / row.exported_path
    text = export_dir / row.transcript_path
    if not (audio.is_file() and text.is_file()):
        return None
    theirs = Path(row.exported_abs_path)
    if theirs.is_file() and not os.path.samefile(theirs, audio):
        raise ForgeError(
            f"{SPEECH_MANIFEST_NAME}: exported_path {row.exported_path!r} and exported_abs_path name different "
            "files — the manifest does not belong to this export; re-export"
        )
    return audio, text


def _clip(row: SpeechManifestRow, audio: Path, text_file: Path) -> tuple[SpeechClip, bool]:
    """A :class:`SpeechClip` for *row*, and whether its ``.txt`` was edited since export."""
    text = text_file.read_text(encoding="utf-8").strip()
    clip = SpeechClip(
        audio=str(audio),
        transcript_file=str(text_file),
        exported_path=row.exported_path,
        exported_abs_path=row.exported_abs_path,
        rel_path=row.rel_path,
        text=text,
        duration_s=row.duration_s,
        sample_rate=row.sample_rate,
        channels=row.channels,
        speaker=row.speaker_id,
        speaker_label=row.speaker_label,
        speaker_cluster=row.speaker_cluster,
        accent=row.accent,
        language=row.language,
        gender=row.gender,
        split=row.split,
    )
    return clip, text != row.transcript.strip()


def _sources(rows: list[SpeechManifestRow]) -> list[SpeechSource]:
    """Distinct ``(corpus, license, provenance, consent hashes)`` among *rows*, with clip counts, in first-seen order."""
    seen: dict[str, SpeechSource] = {}
    for row in rows:
        source = SpeechSource(
            corpus=row.corpus,
            license=row.license,
            provenance=row.provenance,
            consent_release_sha256=row.consent_release_sha256,
            consent_record_sha256=row.consent_record_sha256,
        )
        key = source.model_dump_json(exclude={"clips"})
        seen.setdefault(key, source).clips += 1
    return list(seen.values())


def speaker_overlap(train: list[SpeechManifestRow], held_out: list[SpeechManifestRow]) -> dict[str, list[str]]:
    """Held-out speaker readings that also appear in *train*, per field (empty when the split is disjoint)."""
    overlap: dict[str, list[str]] = {}
    for field in SPEAKER_FIELDS:
        trained = {getattr(r, field) for r in train} - {None}
        shared = sorted({getattr(r, field) for r in held_out} & trained)
        if shared:
            overlap[field] = shared
    return overlap


def _read_rows(export_dir: Path) -> list[SpeechManifestRow]:
    if not export_dir.is_dir():
        raise ForgeError(f"not a directory: {export_dir}")
    manifest = export_dir / SPEECH_MANIFEST_NAME
    if not manifest.is_file():
        raise ForgeError(f"{export_dir} has no {SPEECH_MANIFEST_NAME}: not a curator speech export")
    rows = read_speech_manifest(manifest)
    if not rows:
        raise ForgeError(f"{manifest} has no rows: nothing to train on")
    return rows


def read_speech_export(export_dir: Path, held_out_dir: Path | None = None) -> SpeechDataset:
    """Read a curator speech export (and optionally its held-out part) into a :class:`SpeechDataset`.

    Training clips are the rows with ``split`` ``train`` or null; ``held_out``
    rows in the same export are kept apart as reference clips, never trained on.
    With *held_out_dir*, every row there is a reference clip, and a held-out
    speaker that also appears in training (by ``speaker_id``, ``speaker_label``
    or ``speaker_cluster``) refuses the read: a zero-shot evaluation on a voice
    the model trained on proves nothing.
    """
    rows = _read_rows(export_dir)
    train_rows = [r for r in rows if r.split != "held_out"]
    held_rows = [r for r in rows if r.split == "held_out"]
    held_pairs: list[tuple[Path, SpeechManifestRow]] = [(export_dir, r) for r in held_rows]

    if held_out_dir is not None:
        other = _read_rows(held_out_dir)
        stray = sum(1 for r in other if r.split == "train")
        if stray:
            raise ForgeError(
                f"{held_out_dir / SPEECH_MANIFEST_NAME}: {stray} row(s) are split 'train' — "
                "that is the training part, not the held-out one"
            )
        held_rows += other
        held_pairs += [(held_out_dir, r) for r in other]

    overlap = speaker_overlap(train_rows, held_rows)
    if overlap:
        detail = "; ".join(f"{field} {', '.join(values)}" for field, values in overlap.items())
        raise ForgeError(
            f"held-out speakers also appear in training ({detail}) — the split is not speaker-disjoint; "
            "re-export both parts from one curator split"
        )
    if not train_rows:
        raise ForgeError(f"{export_dir / SPEECH_MANIFEST_NAME} has no train rows: it is a held-out export")

    missing = edits = 0
    clips: list[SpeechClip] = []
    for row in train_rows:
        found = _resolve(export_dir, row)
        if found is None:
            missing += 1
            continue
        clip, edited = _clip(row, *found)
        edits += edited
        clips.append(clip)
    held_out: list[SpeechClip] = []
    for base, row in held_pairs:
        found = _resolve(base, row)
        if found is None:
            missing += 1
            continue
        clip, edited = _clip(row, *found)
        edits += edited
        held_out.append(clip)

    if missing:
        logger.warning("speech_clips_missing", export_dir=str(export_dir), missing=missing)
    dataset = SpeechDataset(
        export_dir=str(export_dir),
        held_out_dir=str(held_out_dir) if held_out_dir is not None else None,
        speech_manifest_version=rows[0].speech_manifest_version,
        target_profile=rows[0].target_profile.model_copy(),
        clips=clips,
        held_out=held_out,
        missing_from_disk=missing,
        transcript_edits=edits,
        sources=_sources(train_rows),
    )
    logger.debug(
        "inspected_speech_export",
        export_dir=str(export_dir),
        clips=len(clips),
        held_out=len(held_out),
        speakers=len(dataset.speakers),
        hours=round(dataset.hours, 3),
    )
    return dataset


class SpeechEmitter(Protocol):
    """What a speech (TTS fine-tune) emitter implements: a :class:`SpeechDataset` in, trainer files out.

    The XTTS-v2 emitter (forge#56) is the first. It writes its dataset (e.g. an
    LJSpeech-style ``metadata.csv`` from ``clip.audio``, ``clip.text``,
    ``clip.speaker`` and ``clip.language``) and config under *out_dir*,
    translating every path it emits through *path_map* as the image emitters do.
    """

    def emit(self, dataset: SpeechDataset, out_dir: Path, path_map: dict[str, str]) -> list[GeneratedFile]: ...

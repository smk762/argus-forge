"""Reading curator's speech export (speech_manifest.jsonl, SpeechManifestRow 1.x): fixture rows, no audio models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from argus_forge.cli import app
from argus_forge.manifest import inspect_export
from argus_forge.models import ForgeError, ManifestRow
from argus_forge.speech import SPEECH_MANIFEST_NAME, read_speech_export, read_speech_manifest

# A 44-byte WAV header with no samples. Forge never decodes audio; a real
# header keeps the fixture honest.
WAV_EMPTY = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"


def speech_row(export: Path, exported: str, **over: object) -> dict:
    """A row as curator's ``speech_export.write_manifest`` writes it: every key present, null when empty."""
    stem = exported.rsplit(".", 1)[0]
    row = {
        "speech_manifest_version": "1.0",
        "modality": "speech",
        "rel_path": f"wavs/{Path(exported).name}",
        "abs_path": f"/corpus/wavs/{Path(exported).name}",
        "exported_path": exported,
        "exported_abs_path": str(export / exported),
        "transcript_path": f"{stem}.txt",
        "transcript_abs_path": str(export / f"{stem}.txt"),
        "target_profile": {
            "modality": "speech",
            "target_category": "voice",
            "target_backend": "xtts-v2",
            "target_style": "photo",
            "checkpoint": None,
        },
        "transcript": f"transcript of {stem}",
        "duration_s": 4.5,
        "sample_rate": 22050,
        "channels": 1,
        "speaker_id": "s1",
        "speaker_label": "s1",
        "speaker_cluster": "0",
        "speaker_mismatch": False,
        "accent": "england",
        "language": "en",
        "gender": None,
        "age": None,
        "score": 0.8,
        "similar_group": 0,
        "split": "train",
        "corpus": "LJSpeech-1.1",
        "license": "public-domain",
        "provenance": {
            "kind": "stock",
            "source": "LJSpeech-1.1",
            "license": "public-domain",
            "subject": "",
            "release_sha256": None,
            "generator": None,
            "notes": "",
            "cleared": True,
        },
        "consent_release_sha256": None,
        "consent_record_sha256": None,
    }
    row.update(over)
    return row


def write_export(export: Path, rows: list[dict], *, files: bool = True) -> Path:
    export.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if files:
            clip = export / row["exported_path"]
            clip.parent.mkdir(parents=True, exist_ok=True)
            clip.write_bytes(WAV_EMPTY)
            (export / row["transcript_path"]).write_text(row["transcript"] + "\n", encoding="utf-8")
    (export / SPEECH_MANIFEST_NAME).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return export


def split_exports(tmp_path: Path, held_speaker: str = "s3", held_cluster: str = "2") -> tuple[Path, Path]:
    train = tmp_path / "train"
    rows = [
        speech_row(train, "s1_0.wav"),
        speech_row(train, "s1_1.wav"),
        speech_row(train, "s2_0.wav", speaker_id="s2", speaker_label="s2", speaker_cluster="1", accent="scotland"),
    ]
    write_export(train, rows)
    held = tmp_path / "held"
    write_export(
        held,
        [
            speech_row(
                held,
                "s3_0.wav",
                speaker_id=held_speaker,
                speaker_label=held_speaker,
                speaker_cluster=held_cluster,
                split="held_out",
            )
        ],
    )
    return train, held


def test_reads_rows_pairs_transcripts_and_carries_speaker_accent_language(tmp_path: Path) -> None:
    train, held = split_exports(tmp_path)
    ds = read_speech_export(train, held_out_dir=held)

    assert ds.speech_manifest_version == "1.0"
    assert ds.target_profile.modality == "speech" and ds.target_profile.target_backend == "xtts-v2"
    assert [Path(c.audio).name for c in ds.clips] == ["s1_0.wav", "s1_1.wav", "s2_0.wav"]
    for clip in ds.clips:
        assert Path(clip.transcript_file) == Path(clip.audio).with_suffix(".txt")
        assert clip.text == f"transcript of {Path(clip.audio).stem}"
    assert ds.speakers == ["s1", "s2"]
    assert ds.languages == ["en"]
    assert {c.accent for c in ds.clips} == {"england", "scotland"}
    assert ds.hours == pytest.approx(3 * 4.5 / 3600)
    assert [Path(c.audio).name for c in ds.held_out] == ["s3_0.wav"]
    assert ds.held_out[0].split == "held_out"


def test_paths_come_from_exported_path_never_rel_path(tmp_path: Path) -> None:
    # Curator de-collided two clips sharing a basename: rel_path can't say which file is which.
    export = tmp_path / "flat"
    rows = [
        speech_row(export, "a.wav", rel_path="x/a.wav", transcript="from x"),
        speech_row(export, "a-9fc3d2.wav", rel_path="y/a.wav", transcript="from y"),
    ]
    write_export(export, rows)
    ds = read_speech_export(export)
    assert {Path(c.audio).name: c.text for c in ds.clips} == {"a.wav": "from x", "a-9fc3d2.wav": "from y"}
    assert {c.rel_path for c in ds.clips} == {"x/a.wav", "y/a.wav"}


def test_held_out_rows_in_the_training_export_are_never_trained_on(tmp_path: Path) -> None:
    export = tmp_path / "both"
    write_export(
        export,
        [
            speech_row(export, "s1_0.wav"),
            speech_row(export, "s9_0.wav", speaker_id="s9", speaker_label="s9", speaker_cluster="9", split="held_out"),
            speech_row(export, "n_0.wav", split=None),  # no split: train on it
        ],
    )
    ds = read_speech_export(export)
    assert sorted(Path(c.audio).name for c in ds.clips) == ["n_0.wav", "s1_0.wav"]
    assert [Path(c.audio).name for c in ds.held_out] == ["s9_0.wav"]


@pytest.mark.parametrize(
    ("speaker", "cluster", "field"),
    [("s2", "7", "speaker_label"), ("s7", "1", "speaker_cluster")],
)
def test_held_out_speaker_in_training_is_refused(tmp_path: Path, speaker: str, cluster: str, field: str) -> None:
    train, held = split_exports(tmp_path, held_speaker=speaker, held_cluster=cluster)
    with pytest.raises(ForgeError, match=f"not speaker-disjoint.*{field}|{field}.*not speaker-disjoint"):
        read_speech_export(train, held_out_dir=held)


def test_a_held_out_export_holding_train_rows_is_refused(tmp_path: Path) -> None:
    train, _ = split_exports(tmp_path)
    with pytest.raises(ForgeError, match="split 'train'"):
        read_speech_export(train, held_out_dir=train)


def test_a_held_out_export_alone_has_nothing_to_train_on(tmp_path: Path) -> None:
    _, held = split_exports(tmp_path)
    with pytest.raises(ForgeError, match="no train rows"):
        read_speech_export(held)


def test_unknown_major_is_refused_before_the_row_shape(tmp_path: Path) -> None:
    export = tmp_path / "v2"
    # A 2.x row may drop 1.x keys; the error must name the version, not a missing field.
    row = {"speech_manifest_version": "2.0", "modality": "speech", "clip": "a.wav"}
    write_export(export, [row], files=False)
    with pytest.raises(
        ForgeError, match=r"speech_manifest_version 2\.0 is not supported \(this build understands 1\.x\)"
    ):
        read_speech_export(export)


def test_a_newer_minor_with_extra_keys_is_read(tmp_path: Path) -> None:
    export = tmp_path / "v1_4"
    write_export(export, [speech_row(export, "a.wav", speech_manifest_version="1.4", emotion="calm")])
    assert read_speech_export(export).speech_manifest_version == "1.4"


def test_mixed_majors_and_missing_version_are_refused(tmp_path: Path) -> None:
    export = tmp_path / "mixed"
    write_export(export, [speech_row(export, "a.wav"), speech_row(export, "b.wav", speech_manifest_version="3.1")])
    with pytest.raises(ForgeError, match=":2: speech_manifest_version 3.1"):
        read_speech_manifest(export / SPEECH_MANIFEST_NAME)
    row = speech_row(export, "a.wav")
    del row["speech_manifest_version"]
    write_export(export, [row])
    with pytest.raises(ForgeError, match="no speech_manifest_version"):
        read_speech_manifest(export / SPEECH_MANIFEST_NAME)


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"exported_path": "../escape.wav", "transcript_path": "../escape.txt"}, "inside the export root"),
        ({"exported_path": "/abs/a.wav", "transcript_path": "/abs/a.txt"}, "inside the export root"),
        ({"transcript_path": "other.txt"}, r"not the <stem>\.txt beside"),
        ({"target_profile": {"modality": "image", "target_category": "identity"}}, "not 'speech'"),
        ({"split": "validation"}, "split"),
    ],
)
def test_rows_breaking_the_contract_are_refused(tmp_path: Path, over: dict, match: str) -> None:
    export = tmp_path / "bad"
    write_export(export, [speech_row(export, "a.wav", **over)], files=False)
    with pytest.raises(ForgeError, match=match):
        read_speech_manifest(export / SPEECH_MANIFEST_NAME)


def test_missing_clip_or_transcript_is_counted_not_trained(tmp_path: Path) -> None:
    export = tmp_path / "gone"
    write_export(export, [speech_row(export, "a.wav"), speech_row(export, "b.wav")])
    (export / "b.txt").unlink()
    ds = read_speech_export(export)
    assert [Path(c.audio).name for c in ds.clips] == ["a.wav"]
    assert ds.missing_from_disk == 1


def test_an_edited_transcript_wins_and_is_counted(tmp_path: Path) -> None:
    export = tmp_path / "edited"
    write_export(export, [speech_row(export, "a.wav")])
    (export / "a.txt").write_text("fixed by hand\n", encoding="utf-8")
    ds = read_speech_export(export)
    assert ds.clips[0].text == "fixed by hand"
    assert ds.transcript_edits == 1


def test_exported_abs_path_naming_another_file_is_refused(tmp_path: Path) -> None:
    export = tmp_path / "wrong"
    other = tmp_path / "elsewhere.wav"
    other.write_bytes(WAV_EMPTY)
    write_export(export, [speech_row(export, "a.wav", exported_abs_path=str(other))])
    with pytest.raises(ForgeError, match="name different files"):
        read_speech_export(export)


def test_exported_abs_path_from_curators_container_is_carried_not_followed(tmp_path: Path) -> None:
    export = tmp_path / "mounted"
    write_export(export, [speech_row(export, "a.wav", exported_abs_path="/data/exports/voice/a.wav")])
    clip = read_speech_export(export).clips[0]
    assert clip.audio == str(export / "a.wav")
    assert clip.exported_abs_path == "/data/exports/voice/a.wav"


def test_corpus_license_and_provenance_are_carried(tmp_path: Path) -> None:
    export = tmp_path / "prov"
    sha = "a" * 64
    consented = {
        "kind": "consented",
        # What curator's speech export writes for a voice under a release (speech_export.py).
        "source": "recorded under the subject's voice release",
        "license": "the subject's signed likeness release (release_sha256)",
        "subject": f"consent:{sha}",
        "release_sha256": sha,
        "cleared": True,
    }
    write_export(
        export,
        [
            speech_row(export, "a.wav"),
            speech_row(export, "b.wav"),
            speech_row(
                export,
                "c.wav",
                corpus="own recordings",
                license="",
                provenance=consented,
                consent_release_sha256=sha,
                consent_record_sha256="b" * 64,
            ),
        ],
    )
    sources = read_speech_export(export).sources
    assert [(s.corpus, s.provenance.kind, s.clips) for s in sources] == [
        ("LJSpeech-1.1", "stock", 2),
        ("own recordings", "consented", 1),
    ]
    assert sources[1].provenance.subject == f"consent:{sha}"
    assert sources[1].consent_record_sha256 == "b" * 64


def test_image_inspect_refuses_a_speech_export_instead_of_reading_it_empty(tmp_path: Path) -> None:
    export = tmp_path / "speech"
    write_export(export, [speech_row(export, "a.wav")])
    with pytest.raises(ForgeError, match="speech export"):
        inspect_export(export)


def test_an_image_manifest_row_with_a_speech_profile_is_refused() -> None:
    with pytest.raises(ValueError, match="speech_manifest.jsonl"):
        ManifestRow(
            manifest_version="2.0",
            rel_path="a.png",
            abs_path="/a.png",
            exported_path="a.png",
            target_profile={"modality": "speech"},
        )


def test_cli_inspect_summarises_a_speech_export(tmp_path: Path) -> None:
    train, held = split_exports(tmp_path)
    result = CliRunner().invoke(app, ["inspect", str(train), "--held-out", str(held)])
    assert result.exit_code == 0, result.output
    assert "3 clips" in result.output and "2 speakers" in result.output
    assert "held out: 1 clips" in result.output
    assert "LJSpeech-1.1" in result.output

    overlap_train, overlap_held = split_exports(tmp_path / "overlap", held_speaker="s1", held_cluster="0")
    result = CliRunner().invoke(app, ["inspect", str(overlap_train), "--held-out", str(overlap_held)])
    assert result.exit_code == 1
    assert "not speaker-disjoint" in result.output

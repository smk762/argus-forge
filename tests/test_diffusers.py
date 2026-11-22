"""diffusers.lora_file: where a forged (or hand-edited) train.sh writes its LoRA."""

from __future__ import annotations

from pathlib import Path

from argus_forge.emitters import diffusers


def _script(tmp_path: Path, body: str) -> Path:
    (tmp_path / "train.sh").write_text(body, encoding="utf-8")
    return tmp_path


def test_output_dir_on_a_continued_line_is_found(tmp_path: Path) -> None:
    """A hand-edited script that adds a flag after --output_dir leaves that flag on a ` \\` line."""
    out = _script(
        tmp_path,
        'accelerate launch "$SCRIPT" \\\n  --output_dir=/runs/new \\\n  --seed=7\n',
    )
    assert diffusers.lora_file(out) == Path("/runs/new") / diffusers.LORA_FILENAME


def test_last_output_dir_wins_across_continued_lines(tmp_path: Path) -> None:
    out = _script(
        tmp_path,
        "accelerate launch train.py \\\n  --output_dir=/runs/old \\\n  --output_dir=/runs/new \\\n  --seed=7\n",
    )
    assert diffusers.lora_file(out) == Path("/runs/new") / diffusers.LORA_FILENAME

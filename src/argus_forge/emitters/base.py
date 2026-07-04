"""Shared emitter context and small rendering helpers."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from argus_forge.models import GeneratedFile, TargetCategory, TargetProfile, TrainingParams


@dataclass
class EmitContext:
    """Everything an emitter needs to render trainer files.

    Emitters are pure: they return :class:`GeneratedFile`s (``name`` relative
    to the export dir) and may append to ``warnings``; the core writes files.
    """

    export_dir: Path
    out_rel: str  # e.g. "forge/kohya"
    params: TrainingParams
    profile: TargetProfile
    base_model: str
    trigger: str
    output_name: str
    images: list[Path]
    warnings: list[str] = field(default_factory=list)

    @property
    def category(self) -> TargetCategory:
        return self.profile.target_category

    def out(self, name: str) -> str:
        """Path of an output file relative to the export dir."""
        return f"{self.out_rel}/{name}"

    def abs_out(self, name: str) -> str:
        """Absolute path of an output file (for cross-references inside configs)."""
        return str(self.export_dir / self.out_rel / name)

    def file(self, name: str, content: str) -> GeneratedFile:
        return GeneratedFile(name=self.out(name), content=content)

    def caption_fallback(self) -> str:
        """Prompt used for images that have no ``.txt`` sidecar."""
        noun = "an illustration" if self.profile.target_style == "anime" else "a photo"
        return f"{noun} of {self.trigger}"

    def steps_comment(self) -> str:
        p = self.params
        return (
            f"{p.images} images x {p.repeats} repeats x {p.epochs} epochs "
            f"= {p.total_steps} samples ({p.optimizer_steps} optimizer steps @ batch {p.batch_size})"
        )


def toml_value(value: object) -> str:
    """Render a scalar (or flat list) as TOML.

    TOML shares JSON's syntax for basic strings, booleans, ints, floats and
    flat arrays, so ``json.dumps`` produces valid TOML for everything forge
    writes — no extra dependency needed.
    """
    if isinstance(value, bool | int | float | str):
        return json.dumps(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot render {type(value).__name__} as TOML")


def toml_lines(pairs: list[tuple[str, object] | str]) -> str:
    """Render ``key = value`` lines; bare strings pass through (comments, headers)."""
    out: list[str] = []
    for item in pairs:
        if isinstance(item, str):
            out.append(item)
        else:
            key, value = item
            out.append(f"{key} = {toml_value(value)}")
    return "\n".join(out) + "\n"


def sh(value: str) -> str:
    """Shell-quote a value for generated ``train.sh`` scripts."""
    return shlex.quote(value)

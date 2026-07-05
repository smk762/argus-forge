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
    # Prefix rewrites applied to every absolute path rendered into configs
    # (container path -> host path), longest prefix wins. See ForgeRequest.path_map.
    path_map: dict[str, str] = field(default_factory=dict)

    @property
    def category(self) -> TargetCategory:
        return self.profile.target_category

    def out(self, name: str) -> str:
        """Path of an output file relative to the export dir."""
        return f"{self.out_rel}/{name}"

    def abs_out(self, name: str) -> str:
        """Absolute path of an output file (for cross-references inside configs)."""
        return self.mapped(self.export_dir / self.out_rel / name)

    def mapped(self, path: Path | str) -> str:
        """*path* with ``path_map`` prefix rewrites applied (longest match wins).

        Configs are often forged inside a container but run on the host; this
        is where ``/data/out/...`` becomes ``/home/you/argus/out/...``.
        """
        s = str(path)
        for src in sorted(self.path_map, key=len, reverse=True):
            prefix = src.rstrip("/")
            if prefix and (s == prefix or s.startswith(prefix + "/")):
                return self.path_map[src].rstrip("/") + s[len(prefix) :]
        return s

    def path_note(self) -> str:
        """README blurb explaining whether config paths were remapped."""
        if self.path_map:
            pairs = ", ".join(f"`{src}` -> `{dst}`" for src, dst in sorted(self.path_map.items()))
            return f"- Absolute paths in these files were remapped for the host: {pairs}."
        return (
            "- Absolute paths in these files are as seen by the process that ran forge. "
            "If that was a container (e.g. the compose stack), they will not exist on the "
            "host — re-forge with `path_map` (or set `FORGE_PATH_MAP=container=host`) to "
            "rewrite them."
        )

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

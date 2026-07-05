"""Shared emitter context and small rendering helpers."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from argus_forge.models import PATH_MAP_ENV, GeneratedFile, TargetCategory, TargetProfile, TrainingParams


def map_path(path: Path | str, path_map: dict[str, str]) -> str:
    """*path* with the longest matching ``path_map`` prefix rewritten.

    Keys are expected pre-normalized (no trailing slash — see
    :func:`argus_forge.core.resolve_path_map`); a prefix only matches at a
    path-component boundary, so ``/data/out`` never rewrites ``/data/output``.
    """
    s = str(path)
    for src in sorted(path_map, key=len, reverse=True):
        if s == src or s.startswith(src + "/"):
            dst = path_map[src]
            mapped = (dst if dst != "/" else "") + s[len(src) :]
            return mapped or "/"
    return s


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
    # (container path -> host path), longest prefix wins. Keys/values arrive
    # normalized from core.resolve_path_map. See ForgeRequest.path_map.
    path_map: dict[str, str] = field(default_factory=dict)
    # How many mapped() calls actually rewrote a path — path_note() uses this
    # to avoid claiming a remap happened when no prefix ever matched.
    map_hits: int = field(default=0, init=False)

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
        out = map_path(path, self.path_map)
        if out != str(path):
            self.map_hits += 1
        return out

    def path_note(self) -> str:
        """README blurb stating whether config paths were actually remapped.

        Emitters must call this after rendering everything that goes through
        :meth:`mapped`, so ``map_hits`` reflects the whole artifact set.
        """
        if self.path_map and self.map_hits:
            pairs = ", ".join(f"`{src}` -> `{dst}`" for src, dst in sorted(self.path_map.items()))
            return f"- Absolute paths in these files were remapped for the host: {pairs}."
        if self.path_map:
            note = (
                "a path map was configured but no rendered path matched its prefixes — "
                "these files keep their original paths; check the map against the export dir"
            )
            self.warnings.append(f"path_map: {note}")
            return f"- WARNING: {note}."
        return (
            "- Absolute paths in these files are as seen by the process that ran forge. "
            "If that was a container (e.g. the compose stack), they will not exist on the "
            f"host — re-forge with `path_map` (or set `{PATH_MAP_ENV}=container=host`) to "
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
    writes — no extra dependency needed. ``ensure_ascii=False`` matters:
    JSON's ``\\ud83d\\udcc1`` surrogate-pair escapes for non-BMP characters
    (emoji in a mapped host path, say) are invalid TOML, while the raw UTF-8
    characters are fine in both.
    """
    if isinstance(value, bool | int | float | str):
        return json.dumps(value, ensure_ascii=False)
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

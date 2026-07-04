"""SDXL LoRA hyperparameter heuristics, ported from the curate UI.

This is a line-for-line port of ``suggestTrainingParams`` /
``DATASET_SIZE_GUIDE`` in argus-studio's
``frontend/src/components/curator/types.ts`` — the numbers the /curate
SelectionInsights panel shows are the numbers forge writes into configs.
Keep the two in lockstep: repeats/epochs are solved to land near a
category-appropriate total step count so small sets train longer per image
and large sets don't overcook.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from argus_forge.models import ParamOverrides, SizeHint, TargetCategory, TrainingParams

EPOCHS = 10
BATCH_SIZE = 2
RESOLUTION = 1024
UNET_LR = 1e-4
TEXT_ENCODER_LR = 5e-5
OPTIMIZER = "AdamW8bit"
SCHEDULER = "cosine"
PRECISION = "bf16"


@dataclass(frozen=True)
class CategoryBias:
    target_steps: int
    dim: int
    alpha: int


TRAINING_BIAS: dict[TargetCategory, CategoryBias] = {
    "identity": CategoryBias(target_steps=1500, dim=16, alpha=8),
    "wardrobe": CategoryBias(target_steps=1600, dim=16, alpha=8),
    "pose_composition": CategoryBias(target_steps=1800, dim=32, alpha=16),
    "setting": CategoryBias(target_steps=2000, dim=32, alpha=16),
}


@dataclass(frozen=True)
class SizeGuide:
    ideal: str
    low: int  # below this = too few
    hi: int  # above this = getting large


DATASET_SIZE_GUIDE: dict[TargetCategory, SizeGuide] = {
    "identity": SizeGuide(ideal="15–30", low=12, hi=50),
    "wardrobe": SizeGuide(ideal="20–40", low=15, hi=60),
    "pose_composition": SizeGuide(ideal="20–40", low=15, hi=60),
    "setting": SizeGuide(ideal="25–50", low=15, hi=80),
}

CATEGORY_LABELS: dict[TargetCategory, str] = {
    "identity": "Identity",
    "wardrobe": "Wardrobe",
    "pose_composition": "Pose / Composition",
    "setting": "Setting",
}


def _js_round(x: float) -> int:
    """JavaScript ``Math.round`` (half away from zero for positives) — Python's
    banker's rounding would drift from the UI on exact halves."""
    return math.floor(x + 0.5)


def _with_derived_steps(params: TrainingParams) -> TrainingParams:
    total = params.images * params.repeats * params.epochs
    return params.model_copy(
        update={
            "total_steps": total,
            "optimizer_steps": math.ceil(total / max(1, params.batch_size)),
        }
    )


def suggest_training_params(count: int, category: TargetCategory) -> TrainingParams:
    """Suggested kohya-style SDXL LoRA params for a set of *count* images."""
    bias = TRAINING_BIAS[category]
    n = max(1, count)
    repeats = max(1, _js_round(bias.target_steps / (n * EPOCHS)))
    return _with_derived_steps(
        TrainingParams(
            images=count,
            repeats=repeats,
            epochs=EPOCHS,
            total_steps=0,  # derived
            optimizer_steps=0,  # derived
            network_dim=bias.dim,
            network_alpha=bias.alpha,
            unet_lr=UNET_LR,
            text_encoder_lr=TEXT_ENCODER_LR,
            optimizer=OPTIMIZER,
            scheduler=SCHEDULER,
            resolution=RESOLUTION,
            batch_size=BATCH_SIZE,
            precision=PRECISION,
        )
    )


def apply_overrides(params: TrainingParams, overrides: ParamOverrides) -> TrainingParams:
    """Apply user overrides, then recompute the derived step counts."""
    updates = overrides.model_dump(exclude_none=True)
    if not updates:
        return params
    return _with_derived_steps(params.model_copy(update=updates))


def dataset_size_status(count: int, category: TargetCategory) -> SizeHint:
    """Dataset-size guidance string, mirroring the UI's ``datasetSizeStatus``."""
    g = DATASET_SIZE_GUIDE[category]
    label = CATEGORY_LABELS[category]
    if count == 0:
        return SizeHint(tone="empty", text=f"Aim for ~{g.ideal} sharp, varied images for an SDXL {label} LoRA.")
    if count < g.low:
        return SizeHint(
            tone="low",
            text=f"{count} selected — light for SDXL; ~{g.ideal} usually trains a stronger, more flexible LoRA.",
        )
    if count > g.hi:
        return SizeHint(
            tone="high",
            text=f"{count} selected — more than needed. Trimming to your best ~{g.ideal} keeps the concept clean.",
        )
    return SizeHint(tone="good", text=f"{count} selected — in the ~{g.ideal} sweet spot for an SDXL {label} LoRA.")

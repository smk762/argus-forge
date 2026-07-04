"""Trainer emitters — one module per supported trainer."""

from __future__ import annotations

from collections.abc import Callable

from argus_forge.emitters import diffusers, kohya, onetrainer
from argus_forge.emitters.base import EmitContext
from argus_forge.models import GeneratedFile, TrainerId, TrainerInfo

Emitter = Callable[[EmitContext], list[GeneratedFile]]

EMITTERS: dict[TrainerId, Emitter] = {
    "kohya": kohya.emit,
    "onetrainer": onetrainer.emit,
    "diffusers": diffusers.emit,
}

TRAINER_INFO: dict[TrainerId, TrainerInfo] = {
    "kohya": TrainerInfo(
        id="kohya",
        label="kohya-ss / sd-scripts",
        files=["dataset.toml", "config.toml", "train.sh", "README.md"],
        notes="Native two-file layout for sdxl_train_network.py (--dataset_config + --config_file).",
    ),
    "onetrainer": TrainerInfo(
        id="onetrainer",
        label="OneTrainer",
        files=["concepts.json", "config.json", "README.md"],
        notes="Concepts + a partial config to load in the OneTrainer UI (missing keys keep defaults).",
    ),
    "diffusers": TrainerInfo(
        id="diffusers",
        label="diffusers (train_text_to_image_lora_sdxl.py)",
        files=["metadata.jsonl (dataset root)", "train.sh", "README.md"],
        notes="HF imagefolder metadata from .txt sidecars + an accelerate launch script.",
    ),
}

__all__ = ["EMITTERS", "TRAINER_INFO", "EmitContext", "Emitter"]

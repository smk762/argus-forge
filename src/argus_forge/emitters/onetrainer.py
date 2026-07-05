"""OneTrainer emitter.

Emits ``concepts.json`` (the dataset side: path, repeats-style balancing,
``.txt`` sidecar prompts) and a *partial* ``config.json`` carrying the forged
hyperparameters. OneTrainer merges loaded configs over its defaults, so the
partial file acts as a starting point — load it in the UI and review before
training; its full config schema moves faster than the keys used here.
"""

from __future__ import annotations

import json

from argus_forge.emitters.base import EmitContext
from argus_forge.models import GeneratedFile

_MODEL_TYPES = {
    "sdxl": "STABLE_DIFFUSION_XL_10_BASE",
}


def emit(ctx: EmitContext) -> list[GeneratedFile]:
    p = ctx.params
    backend = (ctx.profile.target_backend or "sdxl").lower()
    model_type = _MODEL_TYPES.get(backend)
    if model_type is None:
        model_type = _MODEL_TYPES["sdxl"]
        ctx.warnings.append(
            f"onetrainer: no model_type mapping for backend {backend!r} — wrote {model_type}; adjust in the UI"
        )

    concepts = [
        {
            "name": ctx.trigger,
            "path": ctx.mapped(ctx.export_dir),
            "seed": -1,
            "enabled": True,
            "include_subdirectories": True,
            "image_variations": 1,
            "text_variations": 1,
            # REPEATS balancing == kohya num_repeats.
            "balancing": float(p.repeats),
            "balancing_strategy": "REPEATS",
            "loss_weight": 1.0,
            "text": {
                # "sample": read the prompt from the image's .txt sidecar.
                "prompt_source": "sample",
                "prompt_path": "",
                "enable_tag_shuffling": False,
                "tag_delimiter": ",",
                "keep_tags_count": 1,
            },
        }
    ]

    config = {
        "training_method": "LORA",
        "model_type": model_type,
        "base_model_name": ctx.base_model,
        "concept_file_name": ctx.abs_out("concepts.json"),
        "workspace_dir": ctx.abs_out("workspace"),
        "epochs": p.epochs,
        "batch_size": p.batch_size,
        "learning_rate": p.unet_lr,
        "learning_rate_scheduler": p.scheduler.upper(),
        "resolution": str(p.resolution),
        "train_dtype": "BFLOAT_16" if p.precision == "bf16" else "FLOAT_16",
        "lora_rank": p.network_dim,
        "lora_alpha": float(p.network_alpha),
        "output_model_format": "SAFETENSORS",
        "output_model_destination": ctx.abs_out(f"output/{ctx.output_name}.safetensors"),
    }

    readme = f"""# OneTrainer — forged by argus-forge

{ctx.steps_comment()}.

| File | Purpose |
| ---- | ------- |
| `concepts.json` | one concept: `{ctx.mapped(ctx.export_dir)}`, {p.repeats}x REPEATS balancing, prompts from `.txt` sidecars |
| `config.json` | **partial** training config (LoRA, {p.epochs} epochs, rank {p.network_dim}/{p.network_alpha}, lr {p.unet_lr}) |

Load `config.json` in OneTrainer (File -> Load Config); anything not set here
keeps OneTrainer's defaults. It's a starting point — review before training,
OneTrainer's config schema evolves faster than this exporter.

Images without a `.txt` sidecar get no prompt under `prompt_source: sample`;
either caption them with argus-lens first or switch the concept's prompt
source to a single prompt like `{ctx.caption_fallback()}`.

{ctx.path_note()}
"""

    return [
        ctx.file("concepts.json", json.dumps(concepts, indent=2) + "\n"),
        ctx.file("config.json", json.dumps(config, indent=2) + "\n"),
        ctx.file("README.md", readme),
    ]

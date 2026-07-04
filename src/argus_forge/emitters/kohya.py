"""kohya-ss / sd-scripts emitter.

Emits the two-file layout ``sdxl_train_network.py`` consumes natively:

- ``dataset.toml`` — dataset/bucketing config (``--dataset_config``)
- ``config.toml``  — training arguments (``--config_file``)
- ``train.sh``     — the accelerate launch line wiring both together
"""

from __future__ import annotations

import math

from argus_forge.emitters.base import EmitContext, toml_lines
from argus_forge.models import GeneratedFile


def emit(ctx: EmitContext) -> list[GeneratedFile]:
    p = ctx.params
    warmup = math.floor(0.05 * p.optimizer_steps)

    dataset_toml = toml_lines(
        [
            "# argus-forge dataset config for kohya sd-scripts (pass via --dataset_config)",
            f"# {ctx.steps_comment()}",
            "",
            "[general]",
            ("enable_bucket", True),
            ("caption_extension", ".txt"),
            ("shuffle_caption", False),
            ("keep_tokens", 0),
            "",
            "[[datasets]]",
            ("resolution", p.resolution),
            ("batch_size", p.batch_size),
            ("min_bucket_reso", 256),
            ("max_bucket_reso", 2048),
            ("bucket_reso_steps", 64),
            "",
            "[[datasets.subsets]]",
            ("image_dir", str(ctx.export_dir)),
            ("num_repeats", p.repeats),
            "# Fallback caption for images without a .txt sidecar.",
            ("class_tokens", ctx.trigger),
        ]
    )

    config_toml = toml_lines(
        [
            "# argus-forge training config for kohya sd-scripts (pass via --config_file)",
            f"# Seeded from argus-curator selection insights ({ctx.category}, {p.images} images).",
            "# Starting points, not gospel — watch samples and stop early if it overfits.",
            "",
            ("pretrained_model_name_or_path", ctx.base_model),
            ("output_dir", ctx.abs_out("output")),
            ("output_name", ctx.output_name),
            ("save_model_as", "safetensors"),
            ("save_every_n_epochs", 1),
            ("save_precision", p.precision),
            "",
            ("max_train_epochs", p.epochs),
            ("train_batch_size", p.batch_size),
            ("seed", 42),
            ("mixed_precision", p.precision),
            "",
            ("network_module", "networks.lora"),
            ("network_dim", p.network_dim),
            ("network_alpha", p.network_alpha),
            "",
            ("learning_rate", p.unet_lr),
            ("unet_lr", p.unet_lr),
            ("text_encoder_lr", p.text_encoder_lr),
            ("optimizer_type", p.optimizer),
            ("lr_scheduler", p.scheduler),
            ("lr_warmup_steps", warmup),
            ("min_snr_gamma", 5),
            "",
            ("gradient_checkpointing", True),
            ("cache_latents", True),
            ("sdpa", True),
            "# SDXL's fp16 VAE is numerically unstable; keep it in fp32.",
            ("no_half_vae", True),
            ("logging_dir", ctx.abs_out("logs")),
        ]
    )

    train_sh = """#!/usr/bin/env bash
# argus-forge launcher for kohya-ss/sd-scripts (SDXL LoRA).
# Run from a sd-scripts checkout, or point SD_SCRIPTS_DIR at one.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SD_SCRIPTS_DIR:-.}"
accelerate launch --num_cpu_threads_per_process=2 sdxl_train_network.py \\
  --config_file "$HERE/config.toml" \\
  --dataset_config "$HERE/dataset.toml"
"""

    readme = f"""# kohya sd-scripts — forged by argus-forge

{ctx.steps_comment()}.

| File | Purpose |
| ---- | ------- |
| `dataset.toml` | dataset + bucketing (`--dataset_config`) |
| `config.toml` | training args (`--config_file`) |
| `train.sh` | `accelerate launch sdxl_train_network.py` wiring both |

- Images (and `.txt` caption sidecars) are read from `{ctx.export_dir}`.
- Images without a sidecar fall back to `class_tokens = {ctx.trigger!r}`.
- Base model: `{ctx.base_model}` — edit `pretrained_model_name_or_path` to
  point at a local checkpoint if you don't want the HF download.
- The LoRA lands in `output/` as `{ctx.output_name}.safetensors`.

```bash
SD_SCRIPTS_DIR=~/kohya-ss/sd-scripts bash train.sh
```
"""

    return [
        ctx.file("dataset.toml", dataset_toml),
        ctx.file("config.toml", config_toml),
        ctx.file("train.sh", train_sh),
        ctx.file("README.md", readme),
    ]

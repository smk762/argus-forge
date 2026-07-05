"""diffusers emitter.

diffusers has no config-file format — its native interface is the example
script plus flags. Forge emits:

- ``metadata.jsonl`` at the *dataset root* (the HF ``imagefolder`` convention:
  ``file_name`` + ``text`` per image, text from the ``.txt`` sidecar when
  present), and
- ``forge/diffusers/train.sh`` launching ``train_text_to_image_lora_sdxl.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from argus_forge.emitters.base import EmitContext, sh
from argus_forge.manifest import caption_path
from argus_forge.models import GeneratedFile


def _caption_for(ctx: EmitContext, image_abs: str) -> str:
    sidecar = caption_path(Path(image_abs))
    if sidecar.is_file():
        text = " ".join(sidecar.read_text(encoding="utf-8").split())
        if text:
            return text
    return ctx.caption_fallback()


def emit(ctx: EmitContext) -> list[GeneratedFile]:
    p = ctx.params
    warmup = int(0.05 * p.optimizer_steps)
    # Checkpoint roughly once per epoch.
    checkpointing = max(1, p.optimizer_steps // max(1, p.epochs))

    rows = []
    for img in ctx.images:
        rel = img.relative_to(ctx.export_dir).as_posix()
        rows.append(json.dumps({"file_name": rel, "text": _caption_for(ctx, str(img))}))
    metadata = "\n".join(rows) + "\n"

    # Horizontal flips are free variety for scenes/outfits but hurt identity
    # sets (faces are asymmetric), so the flag is category-aware.
    flip_flag = "" if ctx.category == "identity" else "  --random_flip \\\n"

    train_sh = f"""#!/usr/bin/env bash
# argus-forge launcher for the diffusers SDXL LoRA example script.
# Requires: pip install diffusers accelerate datasets transformers peft
# Script: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py
# {ctx.steps_comment()}
set -euo pipefail
SCRIPT="${{DIFFUSERS_SCRIPT:-train_text_to_image_lora_sdxl.py}}"
accelerate launch "$SCRIPT" \\
  --pretrained_model_name_or_path={sh(ctx.base_model)} \\
  --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \\
  --train_data_dir={sh(ctx.mapped(ctx.export_dir))} \\
  --caption_column=text \\
  --resolution={p.resolution} \\
{flip_flag}  --train_batch_size={p.batch_size} \\
  --max_train_steps={p.optimizer_steps} \\
  --learning_rate={p.unet_lr} \\
  --lr_scheduler={sh(p.scheduler)} \\
  --lr_warmup_steps={warmup} \\
  --rank={p.network_dim} \\
  --mixed_precision={sh(p.precision)} \\
  --checkpointing_steps={checkpointing} \\
  --seed=42 \\
  --output_dir={sh(ctx.abs_out("output"))}
"""

    readme = f"""# diffusers — forged by argus-forge

{ctx.steps_comment()}.

| File | Purpose |
| ---- | ------- |
| `../../metadata.jsonl` | HF `imagefolder` captions at the dataset root (`file_name` + `text`) |
| `train.sh` | `accelerate launch train_text_to_image_lora_sdxl.py` with the forged flags |

- Captions come from each image's `.txt` sidecar; images without one fall
  back to `{ctx.caption_fallback()}`. Re-run forge after captioning to refresh.
- diffusers counts optimizer steps, so `--max_train_steps={p.optimizer_steps}`
  (= {p.total_steps} samples / batch {p.batch_size}).
- The fp16-fix VAE is pinned because SDXL's stock VAE is unstable in half precision.
{ctx.path_note()}

```bash
DIFFUSERS_SCRIPT=~/diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py bash train.sh
```
"""

    return [
        # Root-level: the imagefolder loader requires metadata.jsonl beside the images.
        GeneratedFile(name="metadata.jsonl", content=metadata),
        ctx.file("train.sh", train_sh),
        ctx.file("README.md", readme),
    ]

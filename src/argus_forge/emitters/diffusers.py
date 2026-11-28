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
import os
import re
import shlex
from pathlib import Path

from argus_forge.emitters.base import EmitContext, sh
from argus_forge.manifest import caption_path
from argus_forge.models import ForgeError, GeneratedFile

# The file train_text_to_image_lora_sdxl.py saves into --output_dir
# (StableDiffusionXLPipeline.save_lora_weights' default weight_name).
LORA_FILENAME = "pytorch_lora_weights.safetensors"
_OUTPUT_FLAG = "--output_dir"
_CONTINUATION = re.compile(r"\\\r?\n")
# A word that moves the script out of the directory it was launched in.
_CHDIR = re.compile(r"(?:.*[(;&|])?(?:cd|pushd)")
# A value only the shell can resolve: an expansion anywhere, or a leading tilde.
_SHELL_EXPANSION = re.compile(r"~|.*[$`]")


def lora_file(out_dir: Path) -> Path:
    """Where the run of the ``train.sh`` forged in *out_dir* writes its LoRA.

    Read from the script's ``--output_dir=PATH`` or ``--output_dir PATH`` flag
    (the last one wins, as in argparse), so a hand-edited script is followed:
    that file is what ran. A relative path is anchored to *out_dir*, the run's
    working directory, and ".." is collapsed either way. It is returned
    relative, for the caller to refuse, when the value needs the shell to expand
    it (absolute-looking or not), or is relative and the script changes
    directory: forge cannot tell where that lands.
    """
    script = out_dir / "train.sh"
    try:
        text = script.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ForgeError(f"cannot read {script}: {exc}") from exc
    found: str | None = None
    changes_dir = False
    # Join backslash-continued lines first: the launch command spans many of
    # them, and shlex refuses a line ending in a lone backslash.
    for line in _CONTINUATION.sub(" ", text).splitlines():
        try:
            words = shlex.split(line, comments=True)
        except ValueError:
            continue
        for i, word in enumerate(words):
            if _CHDIR.fullmatch(word):
                changes_dir = True
            elif word.startswith(f"{_OUTPUT_FLAG}="):
                found = word[len(_OUTPUT_FLAG) + 1 :]
            elif word == _OUTPUT_FLAG and i + 1 < len(words) and not words[i + 1].startswith("-"):
                found = words[i + 1]
    if not found:
        raise ForgeError(f"{script} passes no {_OUTPUT_FLAG}")
    if _SHELL_EXPANSION.match(found):
        # Relative even when it starts with "/": the expansion decides where it
        # lands, so the caller must refuse it rather than open the literal path.
        return Path(found.lstrip("/")) / LORA_FILENAME
    path = Path(found) / LORA_FILENAME
    if changes_dir and not path.is_absolute():
        return path
    # Collapse "..", absolute or not, so a path leaving the export never looks
    # like one inside it.
    return Path(os.path.normpath(path if path.is_absolute() else out_dir / path))


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

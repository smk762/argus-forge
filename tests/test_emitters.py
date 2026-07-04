from __future__ import annotations

import json
import os
import tomllib

import pytest
from conftest import ExportFactory

from argus_forge.core import forge_config, slugify
from argus_forge.models import ForgeError, ForgeRequest, ParamOverrides


def test_slugify() -> None:
    assert slugify("My Set!") == "my_set"
    assert slugify("--") == "dataset"


def test_kohya_emits_valid_toml(export_factory: ExportFactory) -> None:
    export = export_factory(n=27, captions=27)
    result = forge_config(ForgeRequest(export_dir=str(export), trainer="kohya"))

    assert result.out_dir == str(export / "forge" / "kohya")
    dataset = tomllib.loads((export / "forge/kohya/dataset.toml").read_text())
    assert dataset["datasets"][0]["subsets"][0]["image_dir"] == str(export)
    assert dataset["datasets"][0]["subsets"][0]["num_repeats"] == 6
    assert dataset["datasets"][0]["subsets"][0]["class_tokens"] == "myset"
    assert dataset["datasets"][0]["resolution"] == 1024
    assert dataset["general"]["caption_extension"] == ".txt"

    config = tomllib.loads((export / "forge/kohya/config.toml").read_text())
    assert config["pretrained_model_name_or_path"] == "stabilityai/stable-diffusion-xl-base-1.0"
    assert config["network_dim"] == 16
    assert config["network_alpha"] == 8
    assert config["max_train_epochs"] == 10
    assert config["unet_lr"] == pytest.approx(1e-4)
    assert config["output_name"] == "myset-lora"

    train_sh = export / "forge/kohya/train.sh"
    assert os.access(train_sh, os.X_OK)
    assert "sdxl_train_network.py" in train_sh.read_text()


def test_kohya_uses_manifest_checkpoint(export_factory: ExportFactory) -> None:
    export = export_factory(n=10, checkpoint="/models/juggernaut-xl.safetensors")
    result = forge_config(ForgeRequest(export_dir=str(export), trainer="kohya"))
    assert result.base_model == "/models/juggernaut-xl.safetensors"


def test_overrides_flow_into_config(export_factory: ExportFactory) -> None:
    export = export_factory(n=10)
    result = forge_config(
        ForgeRequest(
            export_dir=str(export),
            trainer="kohya",
            trigger="zxq person",
            output_name="custom",
            overrides=ParamOverrides(network_dim=64, network_alpha=32, epochs=4),
        )
    )
    config = tomllib.loads((export / "forge/kohya/config.toml").read_text())
    assert config["network_dim"] == 64
    assert config["max_train_epochs"] == 4
    assert result.params.total_steps == 10 * result.params.repeats * 4
    dataset = tomllib.loads((export / "forge/kohya/dataset.toml").read_text())
    assert dataset["datasets"][0]["subsets"][0]["class_tokens"] == "zxq person"
    assert result.output_name == "custom"


def test_diffusers_metadata_and_script(export_factory: ExportFactory) -> None:
    export = export_factory(n=6, captions=2, category="identity")
    result = forge_config(ForgeRequest(export_dir=str(export), trainer="diffusers"))

    rows = [json.loads(line) for line in (export / "metadata.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert rows[0]["text"] == "caption 0"
    assert rows[5]["text"] == "a photo of myset"  # fallback for uncaptioned images
    assert all(not r["file_name"].startswith("/") for r in rows)

    script = (export / "forge/diffusers/train.sh").read_text()
    assert f"--max_train_steps={result.params.optimizer_steps}" in script
    assert "--random_flip" not in script  # identity sets keep faces un-mirrored


def test_diffusers_random_flip_for_non_identity(export_factory: ExportFactory) -> None:
    export = export_factory(n=6, category="setting")
    forge_config(ForgeRequest(export_dir=str(export), trainer="diffusers"))
    assert "--random_flip" in (export / "forge/diffusers/train.sh").read_text()


def test_diffusers_warns_on_existing_metadata(export_factory: ExportFactory) -> None:
    export = export_factory(n=3)
    (export / "metadata.jsonl").write_text('{"file_name": "x.png", "text": "old"}\n')
    result = forge_config(ForgeRequest(export_dir=str(export), trainer="diffusers"))
    assert any("metadata.jsonl" in w for w in result.warnings)


def test_onetrainer_files(export_factory: ExportFactory) -> None:
    export = export_factory(n=12)
    result = forge_config(ForgeRequest(export_dir=str(export), trainer="onetrainer"))

    concepts = json.loads((export / "forge/onetrainer/concepts.json").read_text())
    assert concepts[0]["path"] == str(export)
    assert concepts[0]["balancing"] == float(result.params.repeats)
    assert concepts[0]["balancing_strategy"] == "REPEATS"

    config = json.loads((export / "forge/onetrainer/config.json").read_text())
    assert config["training_method"] == "LORA"
    assert config["model_type"] == "STABLE_DIFFUSION_XL_10_BASE"
    assert config["lora_rank"] == 16
    assert config["concept_file_name"].endswith("concepts.json")


def test_collect_captions_from_sources(export_factory: ExportFactory) -> None:
    export = export_factory(n=5, captions=0, source_captions=3)
    result = forge_config(ForgeRequest(export_dir=str(export), trainer="kohya"))
    assert result.captions_collected == 3
    assert result.dataset.caption_count == 3
    assert (export / "img_000.txt").read_text() == "source caption 0"


def test_collect_captions_never_overwrites(export_factory: ExportFactory) -> None:
    export = export_factory(n=3, captions=1, source_captions=3)
    forge_config(ForgeRequest(export_dir=str(export), trainer="kohya"))
    assert (export / "img_000.txt").read_text() == "caption 0"  # export-side sidecar wins


def test_dry_run_writes_nothing(export_factory: ExportFactory) -> None:
    export = export_factory(n=5, source_captions=2)
    result = forge_config(ForgeRequest(export_dir=str(export), trainer="kohya", dry_run=True))
    assert not (export / "forge").exists()
    assert not (export / "img_000.txt").exists()  # collection skipped too
    assert all(f.path is None and f.content for f in result.files)
    assert any("dry run" in w for w in result.warnings)


def test_no_images_is_an_error(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ForgeError, match="no images"):
        forge_config(ForgeRequest(export_dir=str(empty), trainer="kohya"))


def test_uncaptioned_set_warns(export_factory: ExportFactory) -> None:
    export = export_factory(n=5)
    result = forge_config(ForgeRequest(export_dir=str(export), trainer="kohya"))
    assert any("no .txt captions" in w for w in result.warnings)

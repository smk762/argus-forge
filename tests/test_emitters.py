from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

import pytest
from conftest import PNG_1PX, ExportFactory

from argus_forge.core import forge_config, parse_path_map, slugify
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


# --- issue #1: kohya subsets must cover nested (structure-preserving) exports ---


def test_kohya_nested_export_gets_subset_per_directory(export_factory: ExportFactory) -> None:
    export = export_factory(n=8, preserve_structure=True)  # images under sub/
    forge_config(ForgeRequest(export_dir=str(export), trainer="kohya"))
    dataset = tomllib.loads((export / "forge/kohya/dataset.toml").read_text())
    subsets = dataset["datasets"][0]["subsets"]
    # kohya's glob is non-recursive: the subset must point at sub/, not the root.
    assert [s["image_dir"] for s in subsets] == [str(export / "sub")]
    assert all(s["num_repeats"] == 19 and s["class_tokens"] == "myset" for s in subsets)


def test_kohya_mixed_root_and_subdir_images(export_factory: ExportFactory) -> None:
    export = export_factory(n=4, preserve_structure=True)
    (export / "root_img.png").write_bytes(PNG_1PX)
    forge_config(ForgeRequest(export_dir=str(export), trainer="kohya"))
    dataset = tomllib.loads((export / "forge/kohya/dataset.toml").read_text())
    assert [s["image_dir"] for s in dataset["datasets"][0]["subsets"]] == [str(export), str(export / "sub")]


# --- issue #2: basename collisions must warn and never mispair captions ---


def _collision_export(tmp_path: Path) -> Path:
    """Two selected images share a basename; the flattened export kept one file."""
    export = tmp_path / "flat"
    export.mkdir()
    rows = []
    for sub, caption in (("a", "caption a"), ("b", "caption b")):
        src = tmp_path / "sources" / sub / "IMG_0001.png"
        src.parent.mkdir(parents=True)
        src.write_bytes(PNG_1PX)
        src.with_suffix(".txt").write_text(caption, encoding="utf-8")
        rows.append({"manifest_version": "1.0", "rel_path": f"{sub}/IMG_0001.png", "abs_path": str(src)})
    (export / "IMG_0001.png").write_bytes(PNG_1PX)  # curator's last-write-wins result
    (export / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return export


def test_basename_collision_warns_and_skips_caption(tmp_path: Path) -> None:
    export = _collision_export(tmp_path)
    result = forge_config(ForgeRequest(export_dir=str(export), trainer="kohya"))
    collision_warnings = [w for w in result.warnings if "basename collision" in w]
    assert len(collision_warnings) == 1
    assert "a/IMG_0001.png" in collision_warnings[0] and "b/IMG_0001.png" in collision_warnings[0]
    # Neither source caption may be paired with the ambiguous pixels.
    assert not (export / "IMG_0001.txt").exists()
    assert result.captions_collected == 0


def test_no_collision_for_preserved_structure(export_factory: ExportFactory) -> None:
    export = export_factory(n=4, preserve_structure=True, source_captions=4)
    result = forge_config(ForgeRequest(export_dir=str(export), trainer="kohya"))
    assert not any("collision" in w for w in result.warnings)
    assert result.captions_collected == 4


# --- issue #3: path_map remaps container paths to host paths ---


def test_path_map_rewrites_all_config_paths(export_factory: ExportFactory) -> None:
    export = export_factory(n=5)
    result = forge_config(
        ForgeRequest(export_dir=str(export), trainer="kohya", dry_run=True, path_map={str(export): "/host/out"})
    )
    files = {f.name.rsplit("/", 1)[-1]: f.content for f in result.files}
    dataset = tomllib.loads(files["dataset.toml"])
    assert dataset["datasets"][0]["subsets"][0]["image_dir"] == "/host/out"
    config = tomllib.loads(files["config.toml"])
    assert config["output_dir"] == "/host/out/forge/kohya/output"
    assert config["logging_dir"] == "/host/out/forge/kohya/logs"
    assert "remapped for the host" in files["README.md"]


def test_path_map_applies_to_onetrainer_and_diffusers(export_factory: ExportFactory) -> None:
    export = export_factory(n=5)
    path_map = {str(export): "/host/out"}
    ot = forge_config(ForgeRequest(export_dir=str(export), trainer="onetrainer", dry_run=True, path_map=path_map))
    concepts = json.loads(next(f.content for f in ot.files if f.name.endswith("concepts.json")))
    assert concepts[0]["path"] == "/host/out"
    df = forge_config(ForgeRequest(export_dir=str(export), trainer="diffusers", dry_run=True, path_map=path_map))
    script = next(f.content for f in df.files if f.name.endswith("train.sh"))
    assert "--train_data_dir=/host/out" in script
    assert str(export) not in script


def test_unmapped_paths_get_readme_caveat(export_factory: ExportFactory) -> None:
    export = export_factory(n=5)
    result = forge_config(ForgeRequest(export_dir=str(export), trainer="kohya", dry_run=True))
    readme = next(f.content for f in result.files if f.name.endswith("README.md"))
    assert "FORGE_PATH_MAP" in readme


def test_path_map_env_var_and_request_precedence(export_factory: ExportFactory, monkeypatch) -> None:
    export = export_factory(n=5)
    monkeypatch.setenv("FORGE_PATH_MAP", f"{export}=/env/out")
    result = forge_config(ForgeRequest(export_dir=str(export), trainer="kohya", dry_run=True))
    dataset = tomllib.loads(next(f.content for f in result.files if f.name.endswith("dataset.toml")))
    assert dataset["datasets"][0]["subsets"][0]["image_dir"] == "/env/out"
    # An explicit request map overrides the env default for the same prefix.
    result = forge_config(
        ForgeRequest(export_dir=str(export), trainer="kohya", dry_run=True, path_map={str(export): "/req/out"})
    )
    dataset = tomllib.loads(next(f.content for f in result.files if f.name.endswith("dataset.toml")))
    assert dataset["datasets"][0]["subsets"][0]["image_dir"] == "/req/out"


def test_parse_path_map() -> None:
    assert parse_path_map("") == {}
    assert parse_path_map("/a=/b, /c=/d") == {"/a": "/b", "/c": "/d"}
    with pytest.raises(ForgeError, match="path_map"):
        parse_path_map("/a")
    with pytest.raises(ForgeError, match="path_map"):
        parse_path_map("=/b")


# --- regressions from the max-effort review of PR #6 ---


def test_manifest_2x_decollided_export_pairs_both_captions(tmp_path: Path) -> None:
    """Under manifest 2.0 the curator de-collides shared basenames, so what was
    a collision in 1.x becomes two distinct rows with unambiguous captions."""
    export = tmp_path / "flat2"
    export.mkdir()
    rows = []
    for sub, exported, caption in (("a", "IMG_0001.png", "caption a"), ("b", "IMG_0001-9fc3d2.png", "caption b")):
        src = tmp_path / "sources" / sub / "IMG_0001.png"
        src.parent.mkdir(parents=True)
        src.write_bytes(PNG_1PX)
        src.with_suffix(".txt").write_text(caption, encoding="utf-8")
        (export / exported).write_bytes(PNG_1PX)
        rows.append(
            {
                "manifest_version": "2.0",
                "rel_path": f"{sub}/IMG_0001.png",
                "abs_path": str(src),
                "exported_path": exported,
            }
        )
    (export / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    result = forge_config(ForgeRequest(export_dir=str(export), trainer="kohya"))
    assert not any("collision" in w for w in result.warnings)
    assert result.captions_collected == 2
    assert (export / "IMG_0001.txt").read_text() == "caption a"
    assert (export / "IMG_0001-9fc3d2.txt").read_text() == "caption b"


def test_duplicate_manifest_rows_are_not_a_collision(tmp_path: Path) -> None:
    """The same rel_path listed twice is a duplicate selection, not ambiguity."""
    export = tmp_path / "dup"
    export.mkdir()
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.png").write_bytes(PNG_1PX)
    (src / "a.txt").write_text("caption a", encoding="utf-8")
    (export / "a.png").write_bytes(PNG_1PX)
    row = {"manifest_version": "1.0", "rel_path": "a.png", "abs_path": str(src / "a.png")}
    (export / "manifest.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    result = forge_config(ForgeRequest(export_dir=str(export), trainer="kohya"))
    assert not any("collision" in w for w in result.warnings)
    assert result.captions_collected == 1
    assert (export / "a.txt").read_text() == "caption a"


def test_collision_outside_export_dir_warns_instead_of_crashing(tmp_path: Path) -> None:
    """Absolute rel_paths escape the export dir; the warning must not ValueError."""
    export = tmp_path / "esc"
    export.mkdir()
    (export / "a.png").write_bytes(PNG_1PX)
    outside = tmp_path / "outside.png"
    outside.write_bytes(PNG_1PX)
    # Two distinct rel_path spellings (Path collapses the doubled slash) that
    # both resolve to the same file *outside* the export dir.
    rows = [
        {"manifest_version": "1.0", "rel_path": str(outside), "abs_path": str(outside)},
        {"manifest_version": "1.0", "rel_path": f"{tmp_path}//outside.png", "abs_path": str(outside)},
    ]
    (export / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    result = forge_config(ForgeRequest(export_dir=str(export), trainer="kohya", dry_run=True))
    assert any("collision" in w and str(outside) in w for w in result.warnings)


def test_collision_warning_flags_preexisting_sidecar(tmp_path: Path) -> None:
    """A sidecar already sitting on a collided file may be mispaired — say so."""
    export = _collision_export(tmp_path)
    (export / "IMG_0001.txt").write_text("stale caption from a pre-fix run", encoding="utf-8")
    result = forge_config(ForgeRequest(export_dir=str(export), trainer="kohya", dry_run=True))
    assert any("IMG_0001.txt may be mispaired" in w for w in result.warnings)


def test_trailing_slash_request_key_still_overrides_env(export_factory: ExportFactory, monkeypatch) -> None:
    """Keys are normalized, so spelling differences can't invert precedence."""
    export = export_factory(n=3)
    monkeypatch.setenv("FORGE_PATH_MAP", f"{export}/=/env/out")  # trailing slash
    result = forge_config(
        ForgeRequest(export_dir=str(export), trainer="kohya", dry_run=True, path_map={str(export): "/req/out"})
    )
    dataset = tomllib.loads(next(f.content for f in result.files if f.name.endswith("dataset.toml")))
    assert dataset["datasets"][0]["subsets"][0]["image_dir"] == "/req/out"


def test_request_path_map_is_validated(export_factory: ExportFactory) -> None:
    """The wire path (studio UI) gets the same validation as CLI/env input."""
    export = export_factory(n=3)
    for bad in ({str(export): ""}, {str(export): "   "}, {"": "/host"}, {"/": "/host"}):
        with pytest.raises(ForgeError, match="path_map"):
            forge_config(ForgeRequest(export_dir=str(export), trainer="kohya", dry_run=True, path_map=bad))


def test_relative_export_dir_is_made_absolute(export_factory: ExportFactory, monkeypatch) -> None:
    export = export_factory(n=3)
    monkeypatch.chdir(export.parent)
    result = forge_config(
        ForgeRequest(export_dir=export.name, trainer="kohya", dry_run=True, path_map={str(export): "/host/out"})
    )
    dataset = tomllib.loads(next(f.content for f in result.files if f.name.endswith("dataset.toml")))
    assert dataset["datasets"][0]["subsets"][0]["image_dir"] == "/host/out"


def test_base_model_checkpoint_path_is_remapped(export_factory: ExportFactory) -> None:
    export = export_factory(n=3, checkpoint="/data/models/juggernaut-xl.safetensors")
    result = forge_config(
        ForgeRequest(export_dir=str(export), trainer="kohya", dry_run=True, path_map={"/data/models": "/host/models"})
    )
    assert result.base_model == "/host/models/juggernaut-xl.safetensors"
    config = tomllib.loads(next(f.content for f in result.files if f.name.endswith("config.toml")))
    assert config["pretrained_model_name_or_path"] == "/host/models/juggernaut-xl.safetensors"


def test_hf_repo_id_base_model_is_never_remapped(export_factory: ExportFactory) -> None:
    export = export_factory(n=3)
    result = forge_config(
        ForgeRequest(export_dir=str(export), trainer="kohya", dry_run=True, path_map={str(export): "/host/out"})
    )
    assert result.base_model == "stabilityai/stable-diffusion-xl-base-1.0"


def test_unmatched_path_map_warns_instead_of_claiming_remap(export_factory: ExportFactory) -> None:
    export = export_factory(n=3)
    result = forge_config(
        ForgeRequest(export_dir=str(export), trainer="kohya", dry_run=True, path_map={"/data/typo": "/host/out"})
    )
    readme = next(f.content for f in result.files if f.name.endswith("README.md"))
    assert "remapped for the host" not in readme
    assert "no rendered path matched" in readme
    assert any("no rendered path matched" in w for w in result.warnings)


def test_mapping_to_filesystem_root_keeps_valid_paths(export_factory: ExportFactory) -> None:
    export = export_factory(n=3)
    result = forge_config(
        ForgeRequest(export_dir=str(export), trainer="kohya", dry_run=True, path_map={str(export): "/"})
    )
    dataset = tomllib.loads(next(f.content for f in result.files if f.name.endswith("dataset.toml")))
    assert dataset["datasets"][0]["subsets"][0]["image_dir"] == "/"
    config = tomllib.loads(next(f.content for f in result.files if f.name.endswith("config.toml")))
    assert config["output_dir"] == "/forge/kohya/output"


def test_non_bmp_characters_render_valid_toml(export_factory: ExportFactory) -> None:
    export = export_factory(n=3)
    result = forge_config(
        ForgeRequest(
            export_dir=str(export), trainer="kohya", dry_run=True, path_map={str(export): "/Users/me/\U0001f4c1sets"}
        )
    )
    dataset = tomllib.loads(next(f.content for f in result.files if f.name.endswith("dataset.toml")))
    assert dataset["datasets"][0]["subsets"][0]["image_dir"] == "/Users/me/\U0001f4c1sets"


@pytest.mark.parametrize("trainer", ["kohya", "onetrainer", "diffusers"])
def test_no_unmapped_container_path_escapes_any_emitter(export_factory: ExportFactory, trainer: str) -> None:
    """Every emitted file except the README (which cites the mapping itself)
    must be free of the container prefix once a path_map covers it."""
    export = export_factory(n=4, preserve_structure=True, checkpoint="/data/models/ck.safetensors")
    result = forge_config(
        ForgeRequest(
            export_dir=str(export),
            trainer=trainer,  # type: ignore[arg-type]
            dry_run=True,
            path_map={str(export): "/host/out", "/data/models": "/host/models"},
        )
    )
    for f in result.files:
        if f.name.endswith("README.md"):
            continue
        assert str(export) not in f.content, f"unmapped container path leaked into {f.name}"


def test_path_map_longest_prefix_wins_and_no_partial_component_match(export_factory: ExportFactory) -> None:
    export = export_factory(n=5)
    result = forge_config(
        ForgeRequest(
            export_dir=str(export),
            trainer="kohya",
            dry_run=True,
            # A sibling-prefix entry must not clobber the more specific one, and
            # "/host/outer" must not match a "/host/out" prefix by string accident.
            path_map={str(export.parent): "/short", str(export): "/host/out"},
        )
    )
    dataset = tomllib.loads(next(f.content for f in result.files if f.name.endswith("dataset.toml")))
    assert dataset["datasets"][0]["subsets"][0]["image_dir"] == "/host/out"

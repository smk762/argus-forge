"""The suite settings file (argus_cortex.settings) and the suite's /health contract.

Forge reads shared settings through cortex but never copies the suite file into
``os.environ``: the trainer inherits the server's environment, and the file is
where the suite's tokens live.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from argus_cortex import settings
from argus_cortex.service import ServiceHealth
from fastapi.testclient import TestClient

from argus_forge import gpu
from argus_forge.models import READONLY_ENV, RunRequest
from argus_forge.runner import child_env
from argus_forge.server.app import create_app, env_readonly

SRC = Path(__file__).resolve().parent.parent / "src" / "argus_forge"

# Made-up names and values: a real variable may be exported in the shell.
FAKE_TOKEN = "not-a-real-token"


def _suite_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> Path:
    path = tmp_path / "suite.env"
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)
    monkeypatch.setenv("ARGUS_ENV_FILE", str(path))
    return path


def test_health_is_service_health_and_keeps_forges_fields(tmp_path: Path) -> None:
    with TestClient(create_app(export_root=str(tmp_path))) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        health = ServiceHealth.model_validate(body)
        assert (health.service, health.status, health.read_only) == ("argus-forge", "ok", False)
        assert health.capability_names == ("training",)
        from argus_forge import __version__

        assert body["version"] == __version__
        # Forge's own fields, as clients read them.
        assert body["export_root"] == str(tmp_path.resolve())
        assert body["training"] == "enabled"
        assert client.get("/health?checks=0").status_code == 200


def test_readonly_health_says_so_in_both_forms(tmp_path: Path) -> None:
    with TestClient(create_app(export_root=str(tmp_path), allow_run=False)) as client:
        body = client.get("/health").json()
        health = ServiceHealth.model_validate(body)
        assert health.read_only is True
        assert health.capability_names == ()
        assert body["training"] == "disabled"
        assert body["status"] == "ok"


def test_trainer_gets_nothing_from_the_suite_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The broker token in the suite file reaches cortex's coordinator but never the trainer's environment."""
    _suite_file(
        tmp_path,
        monkeypatch,
        f"ARGUS_GPU_BROKER_TOKEN={FAKE_TOKEN}\nFORGE_TEST_SUITE_ONLY={FAKE_TOKEN}\nHF_TOKEN={FAKE_TOKEN}\n",
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert settings.get("ARGUS_GPU_BROKER_TOKEN") == FAKE_TOKEN  # the file is read by this process
    create_app(export_root=str(tmp_path))  # building the app must not export the file either
    env = child_env(RunRequest(export_dir=str(tmp_path), trainer="kohya"))
    assert "ARGUS_GPU_BROKER_TOKEN" not in env
    assert "FORGE_TEST_SUITE_ONLY" not in env
    assert "HF_TOKEN" not in env
    assert FAKE_TOKEN not in env.values()


def test_forge_never_applies_the_suite_file_to_the_environment() -> None:
    """``settings.apply()`` copies every suite-file value, tokens included, into ``os.environ``, which the
    trainer inherits (less only :data:`~argus_forge.runner.SECRET_ENV_KEYS`). Forge must not call it."""
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and node.attr == "apply":
                offenders.append(f"{path.relative_to(SRC.parent)}:{node.lineno}")
            if isinstance(node, ast.ImportFrom) and any(a.name == "apply" for a in node.names):
                offenders.append(f"{path.relative_to(SRC.parent)}:{node.lineno}")
    assert not offenders, "forge calls an apply(): the suite file must never reach os.environ\n" + "\n".join(offenders)


@pytest.mark.parametrize(("value", "readonly"), [("1", True), ("y", True), ("enabled", True), ("0", False)])
def test_readonly_typo_in_the_suite_file_fails_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str, readonly: bool
) -> None:
    """A set-but-unrecognised ``ARGUS_FORGE_READONLY`` keeps the guard on wherever it was set."""
    _suite_file(tmp_path, monkeypatch, f"{READONLY_ENV}={value}\n")
    assert env_readonly() is readonly


def test_the_environment_wins_over_the_suite_file_for_readonly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _suite_file(tmp_path, monkeypatch, f"{READONLY_ENV}=0\n")
    monkeypatch.setenv(READONLY_ENV, "1")
    assert env_readonly() is True


def test_gpu_device_is_read_through_the_suite_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Lease:
        device = None

    _suite_file(tmp_path, monkeypatch, f"{gpu.DEVICE_ENV}=1\n")
    assert gpu._device(_Lease()) == 1  # type: ignore[arg-type]
    monkeypatch.setenv(gpu.DEVICE_ENV, "2")
    assert gpu._device(_Lease()) == 2  # type: ignore[arg-type]

from __future__ import annotations

import pytest
from conftest import ExportFactory
from fastapi.testclient import TestClient

from argus_forge.server import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(cors=True))


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["service"] == "argus-forge"


def test_trainers(client: TestClient) -> None:
    body = client.get("/trainers").json()
    assert [t["id"] for t in body] == ["kohya", "onetrainer", "diffusers"]


def test_inspect(client: TestClient, export_factory: ExportFactory) -> None:
    export = export_factory(n=27, captions=4)
    body = client.post("/inspect", json={"export_dir": str(export)}).json()
    assert body["image_count"] == 27
    assert body["caption_count"] == 4
    assert body["suggested"]["repeats"] == 6
    assert body["size_hint"]["tone"] == "good"


def test_inspect_bad_dir(client: TestClient) -> None:
    resp = client.post("/inspect", json={"export_dir": "/nope/missing"})
    assert resp.status_code == 400
    assert "not a directory" in resp.json()["detail"]


def test_config_dry_run(client: TestClient, export_factory: ExportFactory) -> None:
    export = export_factory(n=10)
    resp = client.post(
        "/config",
        json={"export_dir": str(export), "trainer": "kohya", "dry_run": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["trainer"] == "kohya"
    assert {f["name"] for f in body["files"]} >= {"forge/kohya/dataset.toml", "forge/kohya/config.toml"}
    assert all(f["path"] is None for f in body["files"])
    assert not (export / "forge").exists()


def test_config_writes(client: TestClient, export_factory: ExportFactory) -> None:
    export = export_factory(n=10)
    resp = client.post(
        "/config",
        json={
            "export_dir": str(export),
            "trainer": "diffusers",
            "trigger": "zxq",
            "overrides": {"batch_size": 1},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["params"]["batch_size"] == 1
    assert (export / "metadata.jsonl").exists()
    assert (export / "forge/diffusers/train.sh").exists()


def test_config_invalid_trainer_is_422(client: TestClient, export_factory: ExportFactory) -> None:
    export = export_factory(n=3)
    resp = client.post("/config", json={"export_dir": str(export), "trainer": "dreambooth9000"})
    assert resp.status_code == 422  # pydantic literal validation

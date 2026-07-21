from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from conftest import ExportFactory, forge_stub
from fastapi.testclient import TestClient

from argus_forge.server import create_app


def _wait_terminal(client: TestClient, run_id: str, timeout: float = 15.0) -> dict:
    """Poll GET /run/{id} until the run leaves 'running' (it executes on a
    background job, so it finishes without us consuming the stream)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/run/{run_id}")
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} still running after {timeout}s")


@pytest.fixture
def client() -> Iterator[TestClient]:
    # Context-manager form: one persistent event loop across requests (so a
    # background run job and a later cancel/status request share it) + lifespan
    # startup/shutdown (the shutdown hook cancels any run still in flight).
    with TestClient(create_app(cors=True)) as c:
        yield c


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


def test_run_starts_and_returns_run_id(client: TestClient, tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "echo hi\n")
    resp = client.post("/run", json={"export_dir": str(export), "trainer": "kohya"})
    assert resp.status_code == 202  # accepted; runs in the background
    body = resp.json()
    assert body["run_id"] and body["trainer"] == "kohya"
    assert body["status"] in {"running", "succeeded"}


def test_run_missing_config_is_400(client: TestClient, tmp_path: Path) -> None:
    export = tmp_path / "exp"
    export.mkdir()
    resp = client.post("/run", json={"export_dir": str(export), "trainer": "kohya"})
    assert resp.status_code == 400
    assert "no forged config" in resp.json()["detail"]


def test_run_unrunnable_trainer_is_400(client: TestClient, tmp_path: Path) -> None:
    export = tmp_path / "exp"
    export.mkdir()
    resp = client.post("/run", json={"export_dir": str(export), "trainer": "onetrainer"})
    assert resp.status_code == 400
    assert "no launcher" in resp.json()["detail"]


def test_run_blocked_env_is_400(client: TestClient, tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "echo hi\n")
    resp = client.post("/run", json={"export_dir": str(export), "trainer": "kohya", "env": {"LD_PRELOAD": "/x.so"}})
    assert resp.status_code == 400
    assert "LD_PRELOAD" in resp.json()["detail"]


# --- job registry (#13): runs outlive the connection ---


def test_run_executes_without_a_consumer(client: TestClient, tmp_path: Path) -> None:
    """The run executes on a background job even though nobody ever attaches to
    its stream — proving it doesn't depend on (and so survives) a connection."""
    export = forge_stub(tmp_path, "kohya", "echo a\nsleep 0.3\necho b\n")
    run_id = client.post("/run", json={"export_dir": str(export), "trainer": "kohya"}).json()["run_id"]
    final = _wait_terminal(client, run_id)
    assert final["status"] == "succeeded" and final["returncode"] == 0
    assert final["started_at"] and final["ended_at"]


def test_dry_run_reaches_terminal_status(client: TestClient, tmp_path: Path) -> None:
    """A dry run yields only `start` and executes nothing, but must still reach a
    terminal status — a poller of the argus-proof join must not hang forever."""
    export = forge_stub(tmp_path, "kohya", "echo hi\n")
    run_id = client.post("/run", json={"export_dir": str(export), "trainer": "kohya", "dry_run": True}).json()["run_id"]
    final = _wait_terminal(client, run_id)
    assert final["status"] == "succeeded"
    assert final["ended_at"]


def test_run_stream_reconnect_replays_backlog_and_terminal(client: TestClient, tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "echo one\necho two\n")
    run_id = client.post("/run", json={"export_dir": str(export), "trainer": "kohya"}).json()["run_id"]
    _wait_terminal(client, run_id)
    # Attach after the run finished: the buffered backlog (incl. the terminal
    # event) replays, tagged with the run id.
    replay = client.get(f"/run/{run_id}/stream")
    assert replay.status_code == 200
    assert replay.headers["content-type"].startswith("application/x-ndjson")
    assert replay.headers["x-training-run-id"] == run_id
    events = [json.loads(line) for line in replay.text.splitlines() if line]
    assert any(e["type"] == "log" and e["message"] == "two" for e in events)
    assert events[-1]["type"] == "exit" and events[-1]["returncode"] == 0
    assert all(e["run_id"] == run_id for e in events)


def test_run_cancel_stops_it(client: TestClient, tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "echo up\nsleep 30\n")
    run_id = client.post("/run", json={"export_dir": str(export), "trainer": "kohya"}).json()["run_id"]
    time.sleep(0.4)  # let the subprocess actually launch before cancelling
    assert client.get(f"/run/{run_id}").json()["status"] == "running"
    cancel = client.post(f"/run/{run_id}/cancel")
    assert cancel.status_code == 200
    assert cancel.json()["status"] == "cancelled"
    assert client.get(f"/run/{run_id}").json()["status"] == "cancelled"


def test_runs_lists_tracked_runs(client: TestClient, tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "echo hi\n")
    run_id = client.post("/run", json={"export_dir": str(export), "trainer": "kohya"}).json()["run_id"]
    _wait_terminal(client, run_id)
    listed = client.get("/runs").json()
    assert run_id in {r["run_id"] for r in listed}


def test_unknown_run_is_404(client: TestClient) -> None:
    assert client.get("/run/nope").status_code == 404
    assert client.get("/run/nope/stream").status_code == 404
    assert client.post("/run/nope/cancel").status_code == 404

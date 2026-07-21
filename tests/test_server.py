from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from conftest import ExportFactory, forge_stub
from fastapi.testclient import TestClient

from argus_forge.manifest import resolve_export_dir
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
def client(tmp_path: Path) -> Iterator[TestClient]:
    # Context-manager form: one persistent event loop across requests (so a
    # background run job and a later cancel/status request share it) + lifespan
    # startup/shutdown (the shutdown hook cancels any run still in flight).
    #
    # Rooted at tmp_path, where export_factory and forge_stub both build, so the
    # containment fence is on for every test rather than something only the
    # containment tests see.
    with TestClient(create_app(cors=True, export_root=str(tmp_path))) as c:
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


def test_inspect_outside_the_root_is_refused(client: TestClient) -> None:
    """Containment is checked before existence — an outside path is refused for
    being outside, never probed to see whether it happens to exist."""
    resp = client.post("/inspect", json={"export_dir": "/nope/missing"})
    assert resp.status_code == 400
    assert "escapes the export root" in resp.json()["detail"]


def test_inspect_missing_dir_inside_the_root(client: TestClient) -> None:
    resp = client.post("/inspect", json={"export_dir": "no-such-export"})
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
    # The replayed stream ends with a terminal `cancelled` event, distinct from
    # `error` — a consumer must be able to tell a user cancel from a failure.
    events = [json.loads(line) for line in client.get(f"/run/{run_id}/stream").text.splitlines() if line]
    assert events[-1]["type"] == "cancelled"
    assert not any(e["type"] == "error" for e in events)


def test_runs_lists_tracked_runs(client: TestClient, tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "echo hi\n")
    run_id = client.post("/run", json={"export_dir": str(export), "trainer": "kohya"}).json()["run_id"]
    _wait_terminal(client, run_id)
    listed = client.get("/runs").json()
    assert run_id in {r["run_id"] for r in listed}


def test_failed_launch_reports_the_reason(client: TestClient, tmp_path: Path) -> None:
    """A run that can't be launched (NUL in an env value) surfaces why on the
    polled RunState, not only in the event stream."""
    export = forge_stub(tmp_path, "kohya", "echo hi\n")
    run_id = client.post("/run", json={"export_dir": str(export), "trainer": "kohya", "env": {"X": "a\x00b"}}).json()[
        "run_id"
    ]
    final = _wait_terminal(client, run_id)
    assert final["status"] == "failed"
    assert final["message"] and "failed to launch" in final["message"]


def test_run_state_export_dir_is_resolved(client: TestClient, tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "echo hi\n")
    body = client.post("/run", json={"export_dir": str(export), "trainer": "kohya"}).json()
    # The shared resolution policy (absolute, but not symlink-resolved).
    assert body["export_dir"] == str(resolve_export_dir(str(export)))


def test_cors_exposes_the_run_id_header(client: TestClient, tmp_path: Path) -> None:
    """Cross-origin JS can only read X-Training-Run-Id if it's explicitly exposed."""
    export = forge_stub(tmp_path, "kohya", "echo hi\n")
    resp = client.post(
        "/run",
        json={"export_dir": str(export), "trainer": "kohya"},
        headers={"Origin": "http://ui.local"},
    )
    exposed = resp.headers.get("access-control-expose-headers", "").lower()
    assert "x-training-run-id" in exposed


def test_unknown_run_is_404(client: TestClient) -> None:
    assert client.get("/run/nope").status_code == 404
    assert client.get("/run/nope/stream").status_code == 404
    assert client.post("/run/nope/cancel").status_code == 404


# --- demo-safe mode (#16): render configs, never train ---


@pytest.fixture
def readonly_client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(cors=True, export_root=str(tmp_path), allow_run=False)) as c:
        yield c


def test_readonly_health_advertises_disabled_training(readonly_client: TestClient) -> None:
    """The frontend disables its train button from /health, not from a 403."""
    assert readonly_client.get("/health").json()["training"] == "disabled"


def test_health_advertises_enabled_training_by_default(client: TestClient) -> None:
    assert client.get("/health").json()["training"] == "enabled"


def test_readonly_refuses_run(readonly_client: TestClient, tmp_path: Path) -> None:
    """A run that would otherwise succeed is refused — the gate is the mode,
    not the request."""
    export = forge_stub(tmp_path, "kohya", "echo hi\n")
    resp = readonly_client.post("/run", json={"export_dir": str(export), "trainer": "kohya"})
    assert resp.status_code == 403
    assert "training is disabled" in resp.json()["detail"]
    # Nothing was started.
    assert readonly_client.get("/runs").json() == []


def test_readonly_refuses_run_before_validating(readonly_client: TestClient, tmp_path: Path) -> None:
    """An invalid request gets the same 403, not a 400 implying it could work."""
    export = tmp_path / "exp"
    export.mkdir()
    resp = readonly_client.post("/run", json={"export_dir": str(export), "trainer": "kohya"})
    assert resp.status_code == 403


def test_readonly_still_renders_configs(readonly_client: TestClient, export_factory: ExportFactory) -> None:
    """The demoable half of forge is untouched."""
    export = export_factory(n=10)
    resp = readonly_client.post(
        "/config",
        json={"export_dir": str(export), "trainer": "kohya", "dry_run": True},
    )
    assert resp.status_code == 200
    assert resp.json()["files"]


def test_readonly_inspect_still_works(readonly_client: TestClient, export_factory: ExportFactory) -> None:
    export = export_factory(n=27, captions=4)
    assert readonly_client.post("/inspect", json={"export_dir": str(export)}).json()["image_count"] == 27


# --- containment: a request's export_dir is untrusted ---


def test_traversal_escape_is_refused(client: TestClient) -> None:
    resp = client.post("/inspect", json={"export_dir": "../../etc"})
    assert resp.status_code == 400
    assert "escapes the export root" in resp.json()["detail"]


def test_absolute_path_outside_the_root_is_refused(client: TestClient, tmp_path: Path) -> None:
    """A sibling of the root is not under it, however similar the prefix."""
    outside = tmp_path.parent / f"{tmp_path.name}-evil"
    outside.mkdir(exist_ok=True)
    resp = client.post("/inspect", json={"export_dir": str(outside)})
    assert resp.status_code == 400
    assert "escapes the export root" in resp.json()["detail"]


def test_absolute_path_inside_the_root_is_allowed(client: TestClient, export_factory: ExportFactory) -> None:
    """The studio UI echoes back the absolute export_dir forge reported, so an
    in-root absolute path has to keep working."""
    export = export_factory(n=5)
    assert client.post("/inspect", json={"export_dir": str(export)}).status_code == 200


def test_relative_path_resolves_under_the_root(client: TestClient, export_factory: ExportFactory) -> None:
    export = export_factory(n=5)
    resp = client.post("/inspect", json={"export_dir": export.name})
    assert resp.status_code == 200
    assert resp.json()["image_count"] == 5


def test_config_outside_the_root_writes_nothing(client: TestClient, tmp_path: Path) -> None:
    """The point of the fence: /config must not forge a tree into a directory
    the operator never offered."""
    outside = tmp_path.parent / f"{tmp_path.name}-target"
    outside.mkdir(exist_ok=True)
    resp = client.post("/config", json={"export_dir": str(outside), "trainer": "kohya"})
    assert resp.status_code == 400
    assert not (outside / "forge").exists()


def test_run_outside_the_root_is_refused(client: TestClient, tmp_path: Path) -> None:
    outside = forge_stub(tmp_path.parent / f"{tmp_path.name}-run", "kohya", "echo hi\n")
    resp = client.post("/run", json={"export_dir": str(outside), "trainer": "kohya"})
    assert resp.status_code == 400
    assert "escapes the export root" in resp.json()["detail"]


def test_malformed_path_is_400_not_500(client: TestClient) -> None:
    """An embedded NUL raises out of resolve(); it must not surface as a 500."""
    resp = client.post("/inspect", json={"export_dir": "a\x00b"})
    assert resp.status_code == 400


def test_no_export_root_refuses_everything() -> None:
    """Unconfigured means closed, not wide open."""
    with TestClient(create_app(cors=True)) as c:
        for path, body in (
            ("/inspect", {"export_dir": "/tmp"}),
            ("/config", {"export_dir": "/tmp", "trainer": "kohya"}),
            ("/run", {"export_dir": "/tmp", "trainer": "kohya"}),
        ):
            resp = c.post(path, json=body)
            assert resp.status_code == 400, path
            assert "no export root configured" in resp.json()["detail"]
        assert c.get("/health").json()["export_root"] is None


def test_health_reports_the_export_root(client: TestClient, tmp_path: Path) -> None:
    assert client.get("/health").json()["export_root"] == str(tmp_path.resolve())


# --- CORS is an allow-list, and not a write boundary ---


def test_cors_default_is_the_localhost_frontend_not_a_wildcard(client: TestClient) -> None:
    """A bare --cors must not reflect any origin back with credentials."""
    resp = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in resp.headers

    allowed = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert allowed.headers.get("access-control-allow-credentials") == "true"


def test_named_origin_is_allowed_and_may_write(tmp_path: Path) -> None:
    with TestClient(create_app(cors_origins=["http://studio.local"], export_root=str(tmp_path))) as c:
        resp = c.post(
            "/inspect",
            json={"export_dir": "nope"},
            headers={"Origin": "http://studio.local"},
        )
        # Reached the handler (400 for the missing dir), not refused by the guard.
        assert resp.status_code == 400
        assert "not a directory" in resp.json()["detail"]


def test_wildcard_is_credential_less(tmp_path: Path) -> None:
    """--cors-any grants anonymous reads from anywhere, never credentialed ones."""
    with TestClient(create_app(cors_allow_any=True, export_root=str(tmp_path))) as c:
        resp = c.get("/health", headers={"Origin": "http://evil.example"})
        assert resp.headers["access-control-allow-origin"] == "*"
        assert "access-control-allow-credentials" not in resp.headers


def test_literal_star_in_the_allow_list_takes_the_wildcard_path(tmp_path: Path) -> None:
    """A literal "*" means --cors-any, not credentialed origin reflection."""
    with TestClient(create_app(cors_origins=["*"], export_root=str(tmp_path))) as c:
        resp = c.get("/health", headers={"Origin": "http://evil.example"})
        assert resp.headers["access-control-allow-origin"] == "*"
        assert "access-control-allow-credentials" not in resp.headers


def test_cross_site_write_is_refused(client: TestClient, export_factory: ExportFactory) -> None:
    """CORS is not a write boundary: a page the user visits can POST here with
    no preflight, so unsafe methods are gated on Origin itself."""
    export = export_factory(n=5)
    resp = client.post(
        "/config",
        json={"export_dir": str(export), "trainer": "kohya"},
        headers={"Origin": "http://evil.example"},
    )
    assert resp.status_code == 403
    assert "cross-site POST" in resp.json()["detail"]
    assert not (export / "forge").exists()


def test_wildcard_still_refuses_cross_site_writes(tmp_path: Path, export_factory: ExportFactory) -> None:
    """A public demo must not double as a way to make this host forge configs."""
    export = export_factory(n=5)
    with TestClient(create_app(cors_allow_any=True, export_root=str(tmp_path))) as c:
        resp = c.post(
            "/config",
            json={"export_dir": str(export), "trainer": "kohya"},
            headers={"Origin": "http://evil.example"},
        )
        assert resp.status_code == 403


def test_non_browser_clients_still_write(client: TestClient, export_factory: ExportFactory) -> None:
    """curl and the studio server-side never send Origin; they must not be gated."""
    export = export_factory(n=5)
    resp = client.post("/config", json={"export_dir": str(export), "trainer": "kohya", "dry_run": True})
    assert resp.status_code == 200

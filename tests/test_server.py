from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from conftest import ExportFactory, forge_stub
from fastapi.testclient import TestClient

from argus_forge.manifest import resolve_export_dir
from argus_forge.models import CORS_ORIGINS_ENV, READONLY_ENV
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


@pytest.mark.parametrize("key", ["LD_PRELOAD", "BASH_ENV", "ENV", "LD_AUDIT", "PATH"])
def test_run_blocked_env_is_400(client: TestClient, tmp_path: Path, key: str) -> None:
    """Every key that redirects *what* runs, not just where. PATH belongs here:
    the command is the bare name "bash", so a caller-supplied PATH pointing at a
    directory holding its own ./bash replaces the forged script entirely."""
    export = forge_stub(tmp_path, "kohya", "echo hi\n")
    resp = client.post("/run", json={"export_dir": str(export), "trainer": "kohya", "env": {key: "/x"}})
    assert resp.status_code == 400
    assert key in resp.json()["detail"]


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
    """Cross-origin JS can only read X-Training-Run-Id if it's explicitly exposed.

    The origin must be one the write guard trusts: from an untrusted origin the
    POST is refused before the handler runs, and asserting on the refusal's
    headers would prove nothing about the success path.
    """
    export = forge_stub(tmp_path, "kohya", "echo hi\n")
    resp = client.post(
        "/run",
        json={"export_dir": str(export), "trainer": "kohya"},
        headers={"Origin": "http://localhost:3000"},
    )
    assert resp.status_code == 202
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


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="empty"),
        pytest.param({"trainer": "kohya"}, id="no-export-dir"),
        pytest.param({"export_dir": "x", "trainer": "bogus"}, id="unknown-trainer"),
    ],
)
def test_readonly_refuses_run_before_validating(readonly_client: TestClient, body: dict) -> None:
    """A schema-invalid request gets the same 403, not a 422 implying it could work.

    The guard is middleware precisely so it lands ahead of FastAPI's body
    validation; an in-handler check cannot, since the 422 is raised first.
    """
    assert readonly_client.post("/run", json=body).status_code == 403


def test_readonly_refuses_run_cancel(readonly_client: TestClient) -> None:
    """The fence is path-based, so every /run route is covered — not just POST /run."""
    resp = readonly_client.post("/run/anything/cancel")
    assert resp.status_code == 403
    assert "training is disabled" in resp.json()["detail"]


def test_readonly_still_renders_configs(readonly_client: TestClient, export_factory: ExportFactory) -> None:
    """The demoable half of forge is untouched."""
    export = export_factory(n=10)
    resp = readonly_client.post(
        "/config",
        json={"export_dir": str(export), "trainer": "kohya", "dry_run": True},
    )
    assert resp.status_code == 200
    assert resp.json()["files"]


def test_readonly_config_is_forced_to_dry_run(readonly_client: TestClient, export_factory: ExportFactory) -> None:
    """A demo host is unauthenticated by assumption, so /config never writes.

    Without this, an anonymous curl overwrites the curator's metadata.jsonl and
    plants an executable train.sh on the shared volume.
    """
    export = export_factory(n=10)
    resp = readonly_client.post("/config", json={"export_dir": str(export), "trainer": "kohya"})
    assert resp.status_code == 200
    assert resp.json()["files"]  # still rendered...
    assert not (export / "forge").exists()  # ...but nothing on disk
    assert all(f["path"] is None for f in resp.json()["files"])


def test_readonly_mode_comes_from_the_env(tmp_path: Path) -> None:
    """create_app honours ARGUS_FORGE_READONLY itself, so any ASGI entry point
    (uvicorn --factory, an embedding) gets the same fence as `serve`."""
    os.environ[READONLY_ENV] = "1"
    try:
        with TestClient(create_app(export_root=str(tmp_path))) as c:
            assert c.get("/health").json()["training"] == "disabled"
            assert c.post("/run", json={}).status_code == 403
    finally:
        del os.environ[READONLY_ENV]


def test_unrecognised_readonly_value_fails_safe_not_fatal(tmp_path: Path) -> None:
    """A typo must warn and leave the guard ON, and must not kill the process.

    ARGUS_FORGE_READONLY is a *protection* flag, so the unrecognised case has to
    fail safe: an operator who writes `=y` or `=enabled` is asking for the guard,
    and treating that as "off" would silently enable training and /config writes
    on a host that is unauthenticated and public by assumption. Not fatal either
    — under compose's `restart: unless-stopped` a hard exit is a crash loop.
    """
    os.environ[READONLY_ENV] = "enabled"
    try:
        with TestClient(create_app(export_root=str(tmp_path))) as c:
            assert c.get("/health").json()["training"] == "disabled"
            assert c.post("/run", json={}).status_code == 403
    finally:
        del os.environ[READONLY_ENV]


def test_explicitly_falsy_readonly_allows_runs(tmp_path: Path) -> None:
    """Failing safe on a typo must not make the documented off-switch unreachable."""
    for value in ("0", "false", "no", "off"):
        os.environ[READONLY_ENV] = value
        try:
            with TestClient(create_app(export_root=str(tmp_path))) as c:
                assert c.get("/health").json()["training"] == "enabled", value
        finally:
            del os.environ[READONLY_ENV]


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


@pytest.mark.parametrize("route", ["/inspect", "/config", "/run"])
def test_symlink_loop_is_400_not_500(client: TestClient, tmp_path: Path, route: str) -> None:
    """pathlib raises RuntimeError (NOT OSError) for a symlink loop on every
    Python this package supports, so catching OSError alone lets it escape as an
    unhandled 500 — the exact thing the malformed-path guard exists to prevent."""
    (tmp_path / "loop").symlink_to(tmp_path / "loop")
    resp = client.post(route, json={"export_dir": "loop", "trainer": "kohya"})
    assert resp.status_code == 400
    assert "malformed path" in resp.json()["detail"]


def test_health_survives_an_unresolvable_root(tmp_path: Path) -> None:
    """A liveness probe must not 500 because the export root is misconfigured."""
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    with TestClient(create_app(export_root=str(loop))) as c:
        health = c.get("/health")
        assert health.status_code == 200
        assert health.json()["export_root"] is None
        assert c.post("/inspect", json={"export_dir": "x"}).status_code == 400


@pytest.mark.parametrize("blank", ["", ".", "./"])
def test_export_dir_may_not_be_the_export_root(client: TestClient, export_factory: ExportFactory, blank: str) -> None:
    """A UI that posts before the user picks a directory must not silently be
    handed the whole shared volume: that merges every sibling export into one
    dataset and writes a forge/ tree at the root."""
    export_factory(n=3, name="setA")
    resp = client.post("/inspect", json={"export_dir": blank})
    assert resp.status_code == 400
    assert "not the root itself" in resp.json()["detail"]


def test_symlink_out_of_the_root_is_refused(client: TestClient, tmp_path: Path) -> None:
    """Containment is decided on the resolved path, so a symlink is not a way out."""
    outside = tmp_path.parent / f"{tmp_path.name}-escape"
    outside.mkdir(exist_ok=True)
    (tmp_path / "bridge").symlink_to(outside)
    resp = client.post("/inspect", json={"export_dir": "bridge"})
    assert resp.status_code == 400
    assert "escapes the export root" in resp.json()["detail"]


def test_symlinked_root_keeps_the_requested_spelling(tmp_path: Path, export_factory: ExportFactory) -> None:
    """/data/out -> /mnt/big/out is an ordinary bind/NAS shape. The fence has to
    resolve it to *check* containment, but must hand downstream the spelling the
    caller used — path_map prefixes are matched against that, and dereferencing
    it silently stops every rewrite (README "Container <-> host paths")."""
    export_factory(n=5, name="myset")
    link = tmp_path.parent / f"{tmp_path.name}-link"
    if link.is_symlink():
        link.unlink()
    link.symlink_to(tmp_path)
    with TestClient(create_app(export_root=str(link))) as c:
        body = c.post("/inspect", json={"export_dir": "myset"}).json()
        assert body["export_dir"] == str(link / "myset")


def test_caption_sources_outside_the_root_are_refused(client: TestClient, tmp_path: Path) -> None:
    """The manifest's abs_path is as untrusted as the request. Without a fence,
    any readable .txt on the host is copied into the shared volume and — for
    trainers that inline captions — echoed straight back in the response."""
    secret = tmp_path.parent / f"{tmp_path.name}-secret"
    secret.mkdir(exist_ok=True)
    (secret / "id_rsa.png").write_bytes(b"x")
    (secret / "id_rsa.txt").write_text("BEGIN PRIVATE KEY hunter2", encoding="utf-8")

    export = tmp_path / "myset"
    export.mkdir()
    (export / "a.png").write_bytes(b"x")
    row = {
        "manifest_version": "2.0",
        "rel_path": "a.png",
        "abs_path": str(secret / "id_rsa.png"),
        "exported_path": "a.png",
    }
    (export / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    body = client.post("/config", json={"export_dir": "myset", "trainer": "diffusers"}).json()
    assert not (export / "a.txt").exists()
    assert "hunter2" not in json.dumps(body)
    assert any("outside the export root" in w for w in body["warnings"])


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


def test_wildcard_keeps_named_origins_writable(tmp_path: Path, export_factory: ExportFactory) -> None:
    """The wildcard grants anonymous reads; it must not silently revoke a write
    grant the operator gave explicitly. Otherwise a public demo cannot drive its
    own frontend, since every useful forge endpoint (/inspect, /config) is a POST."""
    export = export_factory(n=5)
    with TestClient(
        create_app(cors_allow_any=True, cors_origins=["https://demo.example"], export_root=str(tmp_path))
    ) as c:
        anyone = c.get("/health", headers={"Origin": "http://evil.example"})
        assert anyone.headers["access-control-allow-origin"] == "*"

        mine = c.post(
            "/config",
            json={"export_dir": str(export), "trainer": "kohya", "dry_run": True},
            headers={"Origin": "https://demo.example"},
        )
        assert mine.status_code == 200

        theirs = c.post(
            "/config",
            json={"export_dir": str(export), "trainer": "kohya", "dry_run": True},
            headers={"Origin": "http://evil.example"},
        )
        assert theirs.status_code == 403


def test_cors_origin_augments_rather_than_replaces_localhost(tmp_path: Path, export_factory: ExportFactory) -> None:
    """README: "A bare --cors allows the localhost:3000 dev frontend. Name other
    origins with --cors-origin." Other, not instead of — the shipped image CMD
    passes --cors, so replacing would kill the studio dev frontend."""
    export = export_factory(n=5)
    with TestClient(create_app(cors=True, cors_origins=["https://studio.example"], export_root=str(tmp_path))) as c:
        for origin in ("http://localhost:3000", "https://studio.example"):
            assert c.get("/health", headers={"Origin": origin}).headers["access-control-allow-origin"] == origin
            resp = c.post(
                "/config",
                json={"export_dir": str(export), "trainer": "kohya", "dry_run": True},
                headers={"Origin": origin},
            )
            assert resp.status_code == 200, origin


@pytest.mark.parametrize("spelling", ["https://studio.example/", " https://studio.example ", "https://studio.example"])
def test_allow_list_entries_are_normalized(tmp_path: Path, spelling: str) -> None:
    """A trailing slash or stray whitespace is a common way to write an entry
    that can never match the Origin header a browser actually sends."""
    with TestClient(create_app(cors_origins=[spelling], export_root=str(tmp_path))) as c:
        resp = c.get("/health", headers={"Origin": "https://studio.example"})
        assert resp.headers.get("access-control-allow-origin") == "https://studio.example"


def test_env_origins_are_honoured(tmp_path: Path) -> None:
    os.environ[CORS_ORIGINS_ENV] = "https://a.example, https://b.example"
    try:
        with TestClient(create_app(export_root=str(tmp_path))) as c:
            for origin in ("https://a.example", "https://b.example"):
                assert c.get("/health", headers={"Origin": origin}).headers["access-control-allow-origin"] == origin
    finally:
        del os.environ[CORS_ORIGINS_ENV]


def test_non_browser_clients_still_write(client: TestClient, export_factory: ExportFactory) -> None:
    """curl and the studio server-side never send Origin; they must not be gated."""
    export = export_factory(n=5)
    resp = client.post("/config", json={"export_dir": str(export), "trainer": "kohya", "dry_run": True})
    assert resp.status_code == 200


# --- regressions from the PR review ---


@pytest.mark.parametrize("route", ["/inspect", "/config"])
def test_symlink_plus_dotdot_cannot_escape_the_root(client: TestClient, tmp_path: Path, route: str) -> None:
    """`..` after an in-root symlink must not escape the fence.

    The check and the returned path have to denote the same file. Resolving the
    raw candidate validates a *different* one: `resolve()` cancels `..` against
    the symlink's target while `os.path.abspath` cancels it lexically, so with
    `d -> <root>/x/y` the request below resolved to `<root>/evil` (inside, so it
    passed) yet abspath'd to `<root>/../evil` — and that escaped path was what
    every endpoint then read, wrote and executed.
    """
    (tmp_path / "x" / "y").mkdir(parents=True)
    (tmp_path / "d").symlink_to(tmp_path / "x" / "y")
    outside = tmp_path.parent / f"{tmp_path.name}-escape"
    outside.mkdir(exist_ok=True)

    resp = client.post(route, json={"export_dir": "d/../../evil", "trainer": "kohya"})
    assert resp.status_code == 400
    assert "escapes the export root" in resp.json()["detail"]
    assert not (tmp_path.parent / "evil").exists()  # nothing forged outside the root


def test_symlink_into_the_root_still_works(client: TestClient, tmp_path: Path, export_factory: ExportFactory) -> None:
    """Tightening the traversal check must not break an ordinary in-root symlink."""
    export_factory(n=5, name="realset")
    (tmp_path / "latest").symlink_to(tmp_path / "realset")
    resp = client.post("/inspect", json={"export_dir": "latest"})
    assert resp.status_code == 200
    assert resp.json()["image_count"] == 5


def test_health_does_not_advertise_an_unmounted_root(tmp_path: Path) -> None:
    """The published image's default root with no volume mounted: resolve()
    succeeds (it is non-strict), so health used to answer "ok" with a root while
    every request 400'd on "not a directory" — a probe and a frontend both read
    the service as usable when nothing was."""
    missing = tmp_path / "not-mounted"
    with TestClient(create_app(export_root=str(missing))) as c:
        health = c.get("/health")
        assert health.status_code == 200
        assert health.json()["export_root"] is None
        resp = c.post("/inspect", json={"export_dir": "x"})
        assert resp.status_code == 400
        assert "not a directory" in resp.json()["detail"]


def test_health_reports_the_undereferenced_spelling(tmp_path: Path, export_factory: ExportFactory) -> None:
    """/health must report the same spelling /inspect returns, since path_map
    prefixes are matched against it — the realpath of a symlinked root would make
    a client that echoes the value back silently lose every rewrite."""
    export_factory(n=3, name="myset")
    link = tmp_path.parent / f"{tmp_path.name}-healthlink"
    if link.is_symlink():
        link.unlink()
    link.symlink_to(tmp_path)
    with TestClient(create_app(export_root=str(link))) as c:
        assert c.get("/health").json()["export_root"] == str(link)


def test_readonly_config_warns_that_it_was_forced_to_dry_run(
    readonly_client: TestClient, export_factory: ExportFactory
) -> None:
    """A caller that asked for a real write must learn from the body that it did
    not happen, not have to infer it from every file's path being null."""
    export = export_factory(n=5)
    body = readonly_client.post("/config", json={"export_dir": str(export), "trainer": "kohya"}).json()
    assert any("demo-safe" in w for w in body["warnings"])
    assert not (export / "forge").exists()


def test_asking_for_a_dry_run_is_not_reported_as_forced(
    readonly_client: TestClient, export_factory: ExportFactory
) -> None:
    """The warning is about a *taken away* write, so it must not fire for a
    caller who asked for a dry run in the first place."""
    export = export_factory(n=5)
    body = readonly_client.post("/config", json={"export_dir": str(export), "trainer": "kohya", "dry_run": True}).json()
    assert not any("demo-safe" in w for w in body["warnings"])


def test_named_origin_does_not_drop_the_localhost_defaults(tmp_path: Path, export_factory: ExportFactory) -> None:
    """--cors-origin *adds* to the localhost dev frontend rather than replacing
    it (README), so naming a production origin must not cost you the studio
    frontend you were already developing against — for reads or for writes."""
    export = export_factory(n=5)
    with TestClient(create_app(cors_origins=["https://prod.example"], export_root=str(tmp_path))) as c:
        for origin in ("http://localhost:3000", "https://prod.example"):
            assert c.get("/health", headers={"Origin": origin}).headers["access-control-allow-origin"] == origin
            resp = c.post(
                "/config",
                json={"export_dir": str(export), "trainer": "kohya", "dry_run": True},
                headers={"Origin": origin},
            )
            assert resp.status_code == 200, origin


def test_empty_manifest_abs_path_is_not_a_500(client: TestClient, tmp_path: Path) -> None:
    """abs_path is manifest-supplied and only exported_path is validated, so it
    can be empty — which names no file and made with_suffix() raise straight
    through the /config catch-all as a 500."""
    export = tmp_path / "emptyabs"
    export.mkdir()
    (export / "a.png").write_bytes(b"x")
    row = {"manifest_version": "2.0", "rel_path": "a.png", "abs_path": "", "exported_path": "a.png"}
    (export / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    resp = client.post("/config", json={"export_dir": "emptyabs", "trainer": "kohya", "dry_run": True})
    assert resp.status_code == 200
    assert resp.json()["files"]


def test_caption_source_root_is_not_a_request_field() -> None:
    """The caption fence is a server-side argument, never wire input: a request
    that could name its own containment root could also widen it to "/"."""
    from argus_forge.models import ForgeRequest

    assert "caption_source_root" not in ForgeRequest.model_fields
    req = ForgeRequest(export_dir="/x", trainer="kohya", caption_source_root="/")
    assert not hasattr(req, "caption_source_root")

"""``Idempotency-Key`` on ``POST /run``: a retried request never starts a second run.

Every run here is a trivial bash stub (``forge_stub``); nothing trains and no GPU is touched.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from conftest import forge_stub
from fastapi.testclient import TestClient

from argus_forge.models import RunRequest
from argus_forge.server import app as app_module
from argus_forge.server import create_app, jobs
from argus_forge.server.idempotency import (
    DEFAULT_RETENTION_S,
    IdempotencyStore,
    env_retention_s,
    key_hash,
    request_fingerprint,
)

KEY = "Idempotency-Key"
REPLAYED = "idempotent-replayed"


def _wait(client: TestClient, run_id: str) -> dict:
    for _ in range(300):
        body = client.get(f"/run/{run_id}").json()
        if body["status"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError("run never finished")


def _body(export: Path, **extra: object) -> dict:
    return {"export_dir": str(export), "trainer": "kohya", **extra}


def test_without_the_header_every_post_starts_a_run(tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "exit 0\n")
    with TestClient(create_app(export_root=str(tmp_path))) as client:
        first = client.post("/run", json=_body(export))
        _wait(client, first.json()["run_id"])  # one run at a time (test_one_active_run)
        second = client.post("/run", json=_body(export))
        assert first.status_code == second.status_code == 202
        assert first.json()["run_id"] != second.json()["run_id"]
        assert REPLAYED not in first.headers and REPLAYED not in second.headers
        assert len(client.get("/runs").json()) == 2


def test_a_replay_returns_the_same_run_and_says_so(tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "echo up\nsleep 30\n")
    with TestClient(create_app(export_root=str(tmp_path))) as client:
        first = client.post("/run", json=_body(export), headers={KEY: "k-1"})
        assert first.status_code == 202 and REPLAYED not in first.headers
        run_id = first.json()["run_id"]
        # Same body, different spelling: whitespace and key order do not matter.
        raw = json.dumps({"trainer": "kohya", "export_dir": str(export)}, indent=4)
        again = client.post("/run", content=raw, headers={KEY: "k-1", "Content-Type": "application/json"})
        assert again.status_code == 202
        assert again.headers[REPLAYED] == "true"
        assert again.json()["run_id"] == run_id
        assert again.json().keys() == first.json().keys()
        assert [r["run_id"] for r in client.get("/runs").json()] == [run_id]
        client.post(f"/run/{run_id}/cancel")


@pytest.mark.parametrize(
    ("script", "end", "cancel"),
    [("exit 0\n", "succeeded", False), ("exit 3\n", "failed", False), ("sleep 30\n", "cancelled", True)],
)
def test_a_replay_after_the_run_ended_returns_it_as_it_ended(
    tmp_path: Path, script: str, end: str, cancel: bool
) -> None:
    export = forge_stub(tmp_path, "kohya", script)
    with TestClient(create_app(export_root=str(tmp_path))) as client:
        run_id = client.post("/run", json=_body(export), headers={KEY: "k-end"}).json()["run_id"]
        if cancel:
            client.post(f"/run/{run_id}/cancel")
        assert _wait(client, run_id)["status"] == end
        replay = client.post("/run", json=_body(export), headers={KEY: "k-end"})
        assert replay.status_code == 202 and replay.headers[REPLAYED] == "true"
        assert replay.json()["run_id"] == run_id and replay.json()["status"] == end
        assert len(client.get("/runs").json()) == 1


def test_the_same_key_with_a_different_body_is_409_and_starts_nothing(tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "exit 0\n")
    with TestClient(create_app(export_root=str(tmp_path))) as client:
        run_id = client.post("/run", json=_body(export), headers={KEY: "k-2"}).json()["run_id"]
        clash = client.post("/run", json=_body(export, dry_run=True), headers={KEY: "k-2"})
        assert clash.status_code == 409
        assert "different request" in clash.json()["detail"]
        assert [r["run_id"] for r in client.get("/runs").json()] == [run_id]


async def test_concurrent_requests_with_one_new_key_start_exactly_one_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    real = app_module.prepare_run

    def slow_prepare(req: RunRequest) -> tuple[list[str], str]:
        nonlocal calls
        calls += 1
        time.sleep(0.2)  # widen the check-then-create window
        return real(req)

    monkeypatch.setattr(app_module, "prepare_run", slow_prepare)
    export = forge_stub(tmp_path, "kohya", "exit 0\n")
    app = create_app(export_root=str(tmp_path), state_dir=str(tmp_path / "state"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://forge") as client:
        responses = await asyncio.gather(
            *(client.post("/run", json=_body(export), headers={KEY: "race"}) for _ in range(6))
        )
        assert all(r.status_code == 202 for r in responses)
        assert len({r.json()["run_id"] for r in responses}) == 1
        assert sum(REPLAYED not in r.headers for r in responses) == 1
        assert calls == 1
        runs = (await client.get("/runs")).json()
        assert len(runs) == 1
        for _ in range(200):  # let the stub finish before the loop closes
            if (await client.get(f"/run/{runs[0]['run_id']}")).json()["status"] != "running":
                break
            await asyncio.sleep(0.05)
    assert len(list((tmp_path / "state/runs").glob("*.json"))) == 1


def test_a_key_survives_a_restart_with_a_state_dir(tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "exit 0\n")
    state = str(tmp_path / "state")
    with TestClient(create_app(export_root=str(tmp_path), state_dir=state)) as client:
        run_id = client.post("/run", json=_body(export), headers={KEY: "k-restart"}).json()["run_id"]
        _wait(client, run_id)
    stored = list((tmp_path / "state/idempotency").glob("*.json"))
    assert [p.stem for p in stored] == [key_hash("k-restart")]
    assert "k-restart" not in stored[0].read_text()  # only the key's hash is written
    with TestClient(create_app(export_root=str(tmp_path), state_dir=state)) as client:
        replay = client.post("/run", json=_body(export), headers={KEY: "k-restart"})
        assert replay.status_code == 202 and replay.headers[REPLAYED] == "true"
        assert replay.json()["run_id"] == run_id
        assert client.post("/run", json=_body(export, dry_run=True), headers={KEY: "k-restart"}).status_code == 409
        assert len(client.get("/runs").json()) == 1


def test_without_a_state_dir_keys_are_in_memory(tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "exit 0\n")
    with TestClient(create_app(export_root=str(tmp_path))) as client:
        run_id = client.post("/run", json=_body(export), headers={KEY: "k-mem"}).json()["run_id"]
        assert client.post("/run", json=_body(export), headers={KEY: "k-mem"}).json()["run_id"] == run_id
    assert not list(tmp_path.rglob("idempotency"))


def test_an_expired_key_is_pruned_but_never_while_its_run_is_active(tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "echo up\nsleep 30\n")
    state = tmp_path / "state"
    app = create_app(export_root=str(tmp_path), state_dir=str(state), idempotency_retention_s=0)
    with TestClient(app) as client:
        run_id = client.post("/run", json=_body(export), headers={KEY: "k-old"}).json()["run_id"]
        time.sleep(0.05)
        # Past retention, but the run is still going: the key holds.
        replay = client.post("/run", json=_body(export), headers={KEY: "k-old"})
        assert replay.headers[REPLAYED] == "true" and replay.json()["run_id"] == run_id
        assert client.post("/run", json=_body(export, dry_run=True), headers={KEY: "k-old"}).status_code == 409
        assert (state / "idempotency" / f"{key_hash('k-old')}.json").exists()
        client.post(f"/run/{run_id}/cancel")
        assert _wait(client, run_id)["status"] == "cancelled"
        # Run over and past retention: the next keyed request prunes it, so the key is free.
        fresh = client.post("/run", json=_body(export, dry_run=True), headers={KEY: "k-old"})
        assert fresh.status_code == 202 and REPLAYED not in fresh.headers
        assert fresh.json()["run_id"] != run_id
        _wait(client, fresh.json()["run_id"])


async def test_prune_drops_only_expired_records_whose_run_is_over(tmp_path: Path) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    store = IdempotencyStore(tmp_path, retention_s=3600, clock=lambda: now)
    for name, run_id in [("old-done", "a" * 32), ("old-active", "b" * 32), ("new-done", "c" * 32)]:
        rec = store.new_record(key_hash(name), run_id, "f")
        if name.startswith("old"):
            rec.created_at = (now - timedelta(hours=2)).isoformat()
        store.put(rec)
        store.save(rec)
    await store.prune(lambda run_id: run_id == "b" * 32)
    kept = {key_hash("old-active"), key_hash("new-done")}
    assert set(store._records) == kept
    assert {p.stem for p in (tmp_path / "idempotency").glob("*.json")} == kept
    # And a fresh store loads exactly what is left.
    assert set(IdempotencyStore(tmp_path, clock=lambda: now)._records) == kept


def test_a_key_whose_run_was_evicted_is_409_not_a_second_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs, "MAX_FINISHED_JOBS", 1)
    export = forge_stub(tmp_path, "kohya", "exit 0\n")
    with TestClient(create_app(export_root=str(tmp_path))) as client:
        first = client.post("/run", json=_body(export), headers={KEY: "k-evict"}).json()["run_id"]
        _wait(client, first)
        for _ in range(2):
            _wait(client, client.post("/run", json=_body(export)).json()["run_id"])
        assert client.get(f"/run/{first}").status_code == 404
        before = client.get("/runs").json()
        resp = client.post("/run", json=_body(export), headers={KEY: "k-evict"})
        assert resp.status_code == 409 and "no longer retains" in resp.json()["detail"]
        assert client.get("/runs").json() == before


@pytest.mark.parametrize(
    "value",
    [b"", b"x" * 201, "secret-été".encode(), b"secret\x01ctl"],
    ids=["empty", "too-long", "non-ascii", "control"],
)
def test_an_invalid_key_is_422_without_echo_and_starts_nothing(tmp_path: Path, value: bytes) -> None:
    export = forge_stub(tmp_path, "kohya", "exit 0\n")
    with TestClient(create_app(export_root=str(tmp_path))) as client:
        resp = client.post("/run", json=_body(export), headers={KEY: value})
        assert resp.status_code == 422
        assert "printable ASCII" in resp.json()["detail"]
        assert b"secret" not in resp.content and b"xxxx" not in resp.content
        assert client.get("/runs").json() == []


def test_a_200_char_key_is_accepted(tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "exit 0\n")
    with TestClient(create_app(export_root=str(tmp_path))) as client:
        key = "~ " + "k" * 198
        run_id = client.post("/run", json=_body(export), headers={KEY: key}).json()["run_id"]
        assert client.post("/run", json=_body(export), headers={KEY: key}).json()["run_id"] == run_id


def test_a_request_that_fails_validation_records_no_key(tmp_path: Path) -> None:
    export = tmp_path / "exp"
    export.mkdir()
    with TestClient(create_app(export_root=str(tmp_path))) as client:
        assert client.post("/run", json=_body(export), headers={KEY: "k-400"}).status_code == 400
        forge_stub(tmp_path, "kohya", "exit 0\n")  # the operator fixes the config; the retry runs
        retry = client.post("/run", json=_body(export), headers={KEY: "k-400"})
        assert retry.status_code == 202 and REPLAYED not in retry.headers


def test_the_fingerprint_ignores_spelling_but_not_content() -> None:
    a = RunRequest.model_validate_json('{"export_dir": "/x", "trainer": "kohya"}')
    b = RunRequest.model_validate_json('{ "trainer":"kohya","env":{},  "export_dir":"/x","dry_run":false }')
    assert request_fingerprint(a) == request_fingerprint(b)
    assert request_fingerprint(a) != request_fingerprint(a.model_copy(update={"dry_run": True}))


def test_retention_comes_from_the_env_and_a_bad_value_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    assert env_retention_s() == DEFAULT_RETENTION_S == 86400
    monkeypatch.setenv("ARGUS_FORGE_IDEMPOTENCY_RETENTION_S", "60")
    assert env_retention_s() == 60
    for bad in ("soon", "-1", "nan", "inf"):
        monkeypatch.setenv("ARGUS_FORGE_IDEMPOTENCY_RETENTION_S", bad)
        assert env_retention_s() == DEFAULT_RETENTION_S


def test_cors_exposes_the_replay_header(tmp_path: Path) -> None:
    with TestClient(create_app(cors=True, export_root=str(tmp_path))) as client:
        resp = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert "idempotent-replayed" in resp.headers["access-control-expose-headers"].lower()

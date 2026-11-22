"""One active run at a time: a second ``POST /run`` is a 409 while a run is queued or running.

Every run here is a trivial bash stub (``forge_stub``) or a fake GPU coordinator that never grants;
nothing trains and no GPU is touched.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import get_args

import httpx
import pytest
from argus_cortex.gpu import GpuLease
from conftest import forge_stub
from fastapi.testclient import TestClient

from argus_forge import gpu
from argus_forge.models import (
    ACTIVE_RUN_STATUSES,
    RUN_ACTIVE_CODE,
    TERMINAL_RUN_STATUSES,
    RunRequest,
    RunState,
    RunStatus,
    run_is_active,
)
from argus_forge.server import app as app_module
from argus_forge.server import create_app
from argus_forge.server.app import DEFAULT_MAX_ACTIVE_RUNS, env_max_active_runs
from argus_forge.server.jobs import Job

KEY = "Idempotency-Key"
REPLAYED = "idempotent-replayed"


def _body(export: Path, **extra: object) -> dict:
    return {"export_dir": str(export), "trainer": "kohya", **extra}


def _wait(client: TestClient, run_id: str) -> dict:
    for _ in range(300):
        body = client.get(f"/run/{run_id}").json()
        if body["status"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError("run never finished")


def _assert_run_active(resp: httpx.Response, run_id: str, status: str = "running") -> None:
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == RUN_ACTIVE_CODE == "run_active"
    assert body["active_run"] == {"run_id": run_id, "status": status}
    assert f"run {run_id}" in body["detail"] and "already active" in body["detail"]


def test_active_and_terminal_statuses_partition_run_status() -> None:
    statuses = set(get_args(RunStatus))
    assert statuses == ACTIVE_RUN_STATUSES | TERMINAL_RUN_STATUSES
    assert not ACTIVE_RUN_STATUSES & TERMINAL_RUN_STATUSES
    assert [s for s in sorted(statuses) if run_is_active(s)] == ["running"]


def test_a_second_run_is_409_while_one_is_running_and_creates_nothing(tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "echo up\nsleep 30\n")
    with TestClient(create_app(export_root=str(tmp_path))) as client:
        first = client.post("/run", json=_body(export))
        assert first.status_code == 202
        run_id = first.json()["run_id"]
        for body in (_body(export), _body(export, env={"SECRET_ISH": "do-not-echo"})):
            second = client.post("/run", json=body)
            _assert_run_active(second, run_id)
            assert b"do-not-echo" not in second.content and str(export).encode() not in second.content
        assert [r["run_id"] for r in client.get("/runs").json()] == [run_id]
        client.post(f"/run/{run_id}/cancel")
        assert _wait(client, run_id)["status"] == "cancelled"
        # Once nothing is active, the next request starts a run.
        third = client.post("/run", json=_body(export))
        assert third.status_code == 202 and third.json()["run_id"] != run_id
        client.post(f"/run/{third.json()['run_id']}/cancel")


class _NeverGrants:
    """A GPU coordinator whose lease never comes: every run stays queued until cancelled."""

    name = "never"

    def __init__(self) -> None:
        self.waiting = 0

    @contextlib.asynccontextmanager
    async def alease(self, **_kwargs) -> AsyncIterator[GpuLease]:
        self.waiting += 1
        await asyncio.sleep(3600)
        yield GpuLease(via="broker")  # pragma: no cover - never reached


def test_a_run_queued_for_its_gpu_lease_is_active(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    coord = _NeverGrants()
    monkeypatch.setattr(gpu, "_cached_coordinator", lambda: coord)
    export = forge_stub(tmp_path, "kohya", "echo trained > ran.txt\n")
    with TestClient(create_app(export_root=str(tmp_path))) as client:
        run_id = client.post("/run", json=_body(export)).json()["run_id"]
        for _ in range(100):
            if coord.waiting:
                break
            time.sleep(0.02)
        assert coord.waiting == 1  # queued: waiting for its lease, no trainer started
        resp = client.post("/run", json=_body(export))
        _assert_run_active(resp, run_id)
        assert "queued or running" in resp.json()["detail"]
        assert coord.waiting == 1 and len(client.get("/runs").json()) == 1
        client.post(f"/run/{run_id}/cancel")
        assert _wait(client, run_id)["status"] == "cancelled"
        assert not (export / "forge/kohya/ran.txt").exists()


async def test_concurrent_requests_with_different_keys_or_none_start_exactly_one_run(
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
    export = forge_stub(tmp_path, "kohya", "echo up\nsleep 30\n")
    app = create_app(export_root=str(tmp_path), state_dir=str(tmp_path / "state"))
    headers = [{KEY: f"k-{i}"} for i in range(4)] + [{}, {}]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://forge") as client:
        responses = await asyncio.gather(*(client.post("/run", json=_body(export), headers=h) for h in headers))
        runs = (await client.get("/runs")).json()
        for run in runs:  # stop whatever started (more than one if the critical section were broken)
            await client.post(f"/run/{run['run_id']}/cancel")
    started = [r for r in responses if r.status_code == 202]
    assert len(started) == 1, [r.status_code for r in responses]
    run_id = started[0].json()["run_id"]
    for resp in responses:
        if resp is not started[0]:
            _assert_run_active(resp, run_id)
    assert calls == 1  # the refused requests never reached validation
    assert [r["run_id"] for r in runs] == [run_id]
    assert len(list((tmp_path / "state/runs").glob("*.json"))) == 1
    # Only the request that started the run recorded its key (none, if it was an unkeyed one).
    assert len(list((tmp_path / "state/idempotency").glob("*.json"))) <= 1


def test_a_replay_of_the_active_runs_own_key_is_202_not_409(tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "echo up\nsleep 30\n")
    with TestClient(create_app(export_root=str(tmp_path))) as client:
        run_id = client.post("/run", json=_body(export), headers={KEY: "mine"}).json()["run_id"]
        replay = client.post("/run", json=_body(export), headers={KEY: "mine"})
        assert replay.status_code == 202 and replay.headers[REPLAYED] == "true"
        assert replay.json()["run_id"] == run_id and replay.json()["status"] == "running"
        # The same key with a different body is still the idempotency 409 (no code), not run_active.
        clash = client.post("/run", json=_body(export, env={"SEED": "2"}), headers={KEY: "mine"})
        assert clash.status_code == 409
        assert "different request" in clash.json()["detail"] and "code" not in clash.json()
        # Another key is refused because a run is active.
        _assert_run_active(client.post("/run", json=_body(export), headers={KEY: "theirs"}), run_id)
        assert len(client.get("/runs").json()) == 1
        client.post(f"/run/{run_id}/cancel")


def test_a_refused_request_records_no_key_so_its_retry_starts_a_run(tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "echo up\nsleep 30\n")
    state = tmp_path / "state"
    with TestClient(create_app(export_root=str(tmp_path), state_dir=str(state))) as client:
        blocker = client.post("/run", json=_body(export)).json()["run_id"]
        _assert_run_active(client.post("/run", json=_body(export), headers={KEY: "later"}), blocker)
        assert not list((state / "idempotency").glob("*.json"))
        client.post(f"/run/{blocker}/cancel")
        _wait(client, blocker)
        retry = client.post("/run", json=_body(export), headers={KEY: "later"})
        assert retry.status_code == 202 and REPLAYED not in retry.headers
        assert retry.json()["run_id"] != blocker
        client.post(f"/run/{retry.json()['run_id']}/cancel")


def test_a_run_in_flight_at_a_crash_does_not_block_new_runs_after_a_restart(tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "exit 0\n")
    state = tmp_path / "state"
    # What a crash leaves behind: a record still saying `running`, nothing driving it.
    stale = RunState(
        run_id="a" * 32,
        trainer="kohya",
        export_dir=str(export),
        status="running",
        started_at="2026-01-01T00:00:00+00:00",
    )
    (state / "runs").mkdir(parents=True)
    (state / "runs" / f"{stale.run_id}.json").write_text(stale.model_dump_json(), encoding="utf-8")
    with TestClient(create_app(export_root=str(tmp_path), state_dir=str(state))) as client:
        assert client.get(f"/run/{stale.run_id}").json()["status"] == "failed"
        resp = client.post("/run", json=_body(export))
        assert resp.status_code == 202
        _wait(client, resp.json()["run_id"])


def test_an_active_run_survives_a_graceful_restart_as_cancelled_and_blocks_nothing(tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "echo up\nsleep 30\n")
    state = str(tmp_path / "state")
    with TestClient(create_app(export_root=str(tmp_path), state_dir=state)) as client:
        run_id = client.post("/run", json=_body(export)).json()["run_id"]
    # Shutdown cancelled it, so it is saved terminal and the next server starts a run.
    with TestClient(create_app(export_root=str(tmp_path), state_dir=state)) as client:
        assert client.get(f"/run/{run_id}").json()["status"] == "cancelled"
        resp = client.post("/run", json=_body(export))
        assert resp.status_code == 202
        client.post(f"/run/{resp.json()['run_id']}/cancel")


def test_a_dry_run_is_exempt_both_ways(tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "echo up\nsleep 30\n")
    with TestClient(create_app(export_root=str(tmp_path))) as client:
        run_id = client.post("/run", json=_body(export)).json()["run_id"]
        dry = client.post("/run", json=_body(export, dry_run=True))
        assert dry.status_code == 202
        assert _wait(client, dry.json()["run_id"])["status"] == "succeeded"
        client.post(f"/run/{run_id}/cancel")
    # A dry run that has not ended yet holds no slot: it takes no lease and starts no process.
    dry_job = Job("b" * 32, RunRequest(export_dir=str(export), trainer="kohya", dry_run=True), ["bash"], str(export))
    real_job = Job("c" * 32, RunRequest(export_dir=str(export), trainer="kohya"), ["bash"], str(export))
    assert dry_job.active and not dry_job.holds_a_slot
    assert real_job.active and real_job.holds_a_slot


def test_max_active_runs_raises_the_limit(tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "echo up\nsleep 30\n")
    with TestClient(create_app(export_root=str(tmp_path), max_active_runs=2)) as client:
        ids = [client.post("/run", json=_body(export)).json()["run_id"] for _ in range(2)]
        third = client.post("/run", json=_body(export))
        _assert_run_active(third, ids[0])  # names the oldest active run
        assert third.json()["detail"].startswith("2 runs are already active")
        for run_id in ids:
            client.post(f"/run/{run_id}/cancel")
    with pytest.raises(ValueError):
        create_app(export_root=str(tmp_path), max_active_runs=0)


def test_max_active_runs_comes_from_the_env_and_a_bad_value_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    assert env_max_active_runs() == DEFAULT_MAX_ACTIVE_RUNS == 1
    monkeypatch.setenv("ARGUS_FORGE_MAX_ACTIVE_RUNS", "3")
    assert env_max_active_runs() == 3
    for bad in ("two", "0", "-1", "1.5"):
        monkeypatch.setenv("ARGUS_FORGE_MAX_ACTIVE_RUNS", bad)
        assert env_max_active_runs() == DEFAULT_MAX_ACTIVE_RUNS


def test_openapi_documents_the_run_active_409(tmp_path: Path) -> None:
    with TestClient(create_app(export_root=str(tmp_path))) as client:
        spec = client.get("/openapi.json").json()
    conflict = spec["paths"]["/run"]["post"]["responses"]["409"]
    assert conflict["content"]["application/json"]["schema"]["$ref"].endswith("/RunActiveError")
    assert spec["components"]["schemas"]["RunActiveError"]["properties"]["code"]["const"] == "run_active"

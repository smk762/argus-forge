"""A run's LoRA (RunState.lora, GET /run/{id}/lora) and run records that survive a restart (#52)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

import pytest
from conftest import ExportFactory, forge_stub
from fastapi.testclient import TestClient

from argus_forge.core import forge_config
from argus_forge.emitters import LORA_LOCATORS
from argus_forge.manifest import FORGE_DIR_NAME
from argus_forge.models import ForgeError, ForgeRequest, RunRequest, RunState
from argus_forge.runner import RUNNABLE_TRAINERS, lora_artifact, prepare_run
from argus_forge.server import create_app
from argus_forge.server.jobs import INTERRUPTED_MESSAGE, Job, JobRegistry, RunStore

LORA_BYTES = b"not really a safetensors file\n" * 1000


def _kohya_run(tmp_path: Path, *, write_lora: bool = True, name: str = "me-lora") -> Path:
    """A forged kohya export whose train.sh writes a LoRA where config.toml says."""
    out_dir = tmp_path / "exp" / FORGE_DIR_NAME / "kohya" / "output"
    body = f"mkdir -p '{out_dir}'\nprintf %s '{LORA_BYTES.decode()}' > '{out_dir / name}.safetensors'\n"
    export = forge_stub(tmp_path, "kohya", body if write_lora else "echo trained nothing\n")
    (export / FORGE_DIR_NAME / "kohya" / "config.toml").write_text(
        f'output_dir = "{out_dir}"\noutput_name = "{name}"\n', encoding="utf-8"
    )
    return export


# --- where each trainer puts its LoRA ---


def test_every_runnable_trainer_has_a_lora_locator() -> None:
    assert set(LORA_LOCATORS) == set(RUNNABLE_TRAINERS)


@pytest.mark.parametrize(
    ("trainer", "expected"),
    [
        ("kohya", "forge/kohya/output/set-lora.safetensors"),
        ("diffusers", "forge/diffusers/output/pytorch_lora_weights.safetensors"),
    ],
)
def test_locator_reads_what_forge_rendered(export_factory: ExportFactory, trainer: str, expected: str) -> None:
    """The locators parse the real emitted files, so they can't drift from the emitters."""
    export = export_factory(n=5)
    forge_config(ForgeRequest(export_dir=str(export), trainer=trainer, output_name="set-lora"))
    found = LORA_LOCATORS[trainer](export / FORGE_DIR_NAME / trainer)
    assert found == export / expected


def test_kohya_locator_follows_a_hand_edited_config(tmp_path: Path) -> None:
    out = tmp_path / "forge" / "kohya"
    out.mkdir(parents=True)
    (out / "config.toml").write_text('output_dir = "/elsewhere"\noutput_name = "mine"\n', encoding="utf-8")
    assert LORA_LOCATORS["kohya"](out) == Path("/elsewhere/mine.safetensors")


@pytest.mark.parametrize("config", ["", 'output_dir = "/x"\n', "not = [toml"])
def test_kohya_locator_refuses_a_config_that_names_no_output(tmp_path: Path, config: str) -> None:
    (tmp_path / "config.toml").write_text(config, encoding="utf-8")
    with pytest.raises(ForgeError):
        LORA_LOCATORS["kohya"](tmp_path)


def test_diffusers_locator_takes_the_last_output_dir(tmp_path: Path) -> None:
    (tmp_path / "train.sh").write_text(
        "# --output_dir=/commented\naccelerate launch x.py \\\n  --output_dir=/first \\\n  --output_dir='/a b'\n",
        encoding="utf-8",
    )
    assert LORA_LOCATORS["diffusers"](tmp_path) == Path("/a b/pytorch_lora_weights.safetensors")


# --- finding and hashing it ---


def test_lora_artifact_hashes_the_file(tmp_path: Path) -> None:
    export = _kohya_run(tmp_path)
    lora = export / "forge/kohya/output/me-lora.safetensors"
    lora.parent.mkdir(parents=True)
    lora.write_bytes(LORA_BYTES)
    art = lora_artifact(RunRequest(export_dir=str(export), trainer="kohya"))
    assert art.path == str(lora)
    assert art.export_path == "forge/kohya/output/me-lora.safetensors"
    assert art.sha256 == hashlib.sha256(LORA_BYTES).hexdigest()
    assert art.size_bytes == len(LORA_BYTES)


def test_lora_artifact_outside_the_export_has_no_export_path(tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "true\n")
    elsewhere = tmp_path / "loras"
    elsewhere.mkdir()
    (elsewhere / "x.safetensors").write_bytes(b"x")
    (export / "forge/kohya/config.toml").write_text(f'output_dir = "{elsewhere}"\noutput_name = "x"\n')
    art = lora_artifact(RunRequest(export_dir=str(export), trainer="kohya"))
    assert art.export_path is None and art.path == str(elsewhere / "x.safetensors")


def test_lora_artifact_missing_file_says_where_it_looked(tmp_path: Path) -> None:
    export = _kohya_run(tmp_path)
    with pytest.raises(ForgeError, match="me-lora.safetensors"):
        lora_artifact(RunRequest(export_dir=str(export), trainer="kohya"))


# --- the job: lora set before `succeeded` ---


def _job(export: Path, store: RunStore | None = None, run_id: str = "a" * 32) -> Job:
    req = RunRequest(export_dir=str(export), trainer="kohya")
    command, cwd = prepare_run(req)
    return Job(run_id, req, command, cwd, store)


async def test_a_succeeded_run_carries_its_lora(tmp_path: Path) -> None:
    job = _job(_kohya_run(tmp_path))
    job._task = asyncio.create_task(job._drive())
    await asyncio.wait_for(job._task, timeout=10)
    state = job.state()
    assert state.status == "succeeded"
    assert state.lora is not None and state.lora.sha256 == hashlib.sha256(LORA_BYTES).hexdigest()


async def test_lora_is_set_before_the_run_turns_succeeded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A poller reads status and lora from one RunState: `succeeded` must never come without its lora."""
    from argus_forge.server import jobs

    job = _job(_kohya_run(tmp_path))
    seen: list[str] = []
    real = jobs.lora_artifact

    def spy(req: RunRequest):  # noqa: ANN202 - test spy
        seen.append(job.status)
        return real(req)

    monkeypatch.setattr(jobs, "lora_artifact", spy)
    job._task = asyncio.create_task(job._drive())
    await asyncio.wait_for(job._task, timeout=10)
    assert seen == ["running"]
    assert job.status == "succeeded" and job.lora is not None


async def test_a_run_that_left_no_lora_still_succeeds_and_says_so(tmp_path: Path) -> None:
    job = _job(_kohya_run(tmp_path, write_lora=False))
    job._task = asyncio.create_task(job._drive())
    await asyncio.wait_for(job._task, timeout=10)
    assert job.status == "succeeded" and job.lora is None
    assert job.message and "me-lora.safetensors" in job.message


async def test_a_failed_run_is_not_hashed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from argus_forge.server import jobs

    monkeypatch.setattr(jobs, "lora_artifact", lambda req: pytest.fail("hashed a failed run"))
    job = _job(forge_stub(tmp_path, "kohya", "exit 2\n"))
    job._task = asyncio.create_task(job._drive())
    await asyncio.wait_for(job._task, timeout=10)
    assert job.status == "failed" and job.lora is None


# --- persistence ---


async def test_runs_survive_a_restart(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    registry = JobRegistry(state_dir)
    export = _kohya_run(tmp_path)
    req = RunRequest(export_dir=str(export), trainer="kohya")
    job = registry.start(req, *prepare_run(req))
    assert job._task is not None
    await asyncio.wait_for(job._task, timeout=10)
    before = job.state()
    assert before.status == "succeeded" and before.lora is not None

    restored = JobRegistry(state_dir).get(job.run_id)
    assert restored is not None
    assert restored.state() == before
    events = [ev async for ev in restored.subscribe()]
    assert [ev.type for ev in events] == ["start", "exit"]
    assert events[0].command == before.command and events[1].returncode == 0


async def test_a_run_in_flight_at_a_crash_is_restored_failed(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state")
    job = _job(forge_stub(tmp_path, "kohya", "sleep 30\n"), store)
    job.save()  # what JobRegistry.start writes; the "crash" is never driving it

    restored = JobRegistry(tmp_path / "state").get(job.run_id)
    assert restored is not None and restored.finished
    assert restored.status == "failed" and restored.message == INTERRUPTED_MESSAGE
    assert [ev.type for ev in [ev async for ev in restored.subscribe()]] == ["start", "error"]
    # ...and saved that way, so the next restart doesn't have to correct it again.
    saved = RunState.model_validate_json((tmp_path / "state/runs" / f"{job.run_id}.json").read_bytes())
    assert saved.status == "failed"


async def test_a_cancelled_run_is_restored_cancelled(tmp_path: Path) -> None:
    registry = JobRegistry(tmp_path / "state")
    req = RunRequest(export_dir=str(forge_stub(tmp_path, "kohya", "sleep 30\n")), trainer="kohya")
    job = registry.start(req, *prepare_run(req))
    await asyncio.sleep(0.2)
    await registry.shutdown()  # a graceful stop cancels, then saves
    restored = JobRegistry(tmp_path / "state").get(job.run_id)
    assert restored is not None and restored.status == "cancelled"


def test_foreign_and_broken_records_are_skipped(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "notes.json").write_text("{}")
    (runs / ("b" * 32 + ".json")).write_text("{not json")
    other = RunState(run_id="c" * 32, trainer="kohya", export_dir="/x", status="failed", started_at="t")
    (runs / ("d" * 32 + ".json")).write_text(other.model_dump_json())  # id doesn't match its file
    assert RunStore(tmp_path).load() == []


async def test_eviction_deletes_the_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from argus_forge.server import jobs

    monkeypatch.setattr(jobs, "MAX_FINISHED_JOBS", 1)
    registry = JobRegistry(tmp_path / "state")
    ids = []
    for i in range(2):
        req = RunRequest(export_dir=str(forge_stub(tmp_path / str(i), "kohya", "true\n")), trainer="kohya")
        job = registry.start(req, *prepare_run(req))
        assert job._task is not None
        await asyncio.wait_for(job._task, timeout=10)
        ids.append(job.run_id)
    registry._evict()
    left = sorted(p.stem for p in (tmp_path / "state/runs").glob("*.json"))
    assert left == [ids[1]]


# --- the HTTP route ---


def _wait(client: TestClient, run_id: str) -> dict:
    for _ in range(300):
        body = client.get(f"/run/{run_id}").json()
        if body["status"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError("run never finished")


def test_lora_route(tmp_path: Path) -> None:
    export = _kohya_run(tmp_path)
    with TestClient(create_app(export_root=str(tmp_path), state_dir=str(tmp_path / "state"))) as client:
        run_id = client.post("/run", json={"export_dir": str(export), "trainer": "kohya"}).json()["run_id"]
        state = _wait(client, run_id)
        resp = client.get(f"/run/{run_id}/lora")
        assert resp.status_code == 200
        assert resp.json() == state["lora"]
        assert resp.json()["export_path"] == "forge/kohya/output/me-lora.safetensors"
        assert client.get("/run/nope/lora").status_code == 404
    # A new app on the same state dir still answers for it.
    with TestClient(create_app(export_root=str(tmp_path), state_dir=str(tmp_path / "state"))) as client:
        assert client.get(f"/run/{run_id}/lora").json() == state["lora"]


def test_lora_route_refuses_a_run_that_did_not_succeed(tmp_path: Path) -> None:
    export = forge_stub(tmp_path, "kohya", "exit 1\n")
    with TestClient(create_app(export_root=str(tmp_path))) as client:
        run_id = client.post("/run", json={"export_dir": str(export), "trainer": "kohya"}).json()["run_id"]
        _wait(client, run_id)
        resp = client.get(f"/run/{run_id}/lora")
        assert resp.status_code == 409 and "failed" in resp.json()["detail"]


def test_lora_route_404s_a_run_that_left_no_file(tmp_path: Path) -> None:
    export = _kohya_run(tmp_path, write_lora=False)
    with TestClient(create_app(export_root=str(tmp_path))) as client:
        run_id = client.post("/run", json={"export_dir": str(export), "trainer": "kohya"}).json()["run_id"]
        assert _wait(client, run_id)["status"] == "succeeded"
        resp = client.get(f"/run/{run_id}/lora")
        assert resp.status_code == 404 and "me-lora.safetensors" in resp.json()["detail"]


def test_state_dir_comes_from_the_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = RunStore(tmp_path / "state")
    state = RunState(run_id="e" * 32, trainer="kohya", export_dir="/x", status="failed", started_at="t", message="m")
    store.save(state)
    monkeypatch.setenv("ARGUS_FORGE_STATE_DIR", str(tmp_path / "state"))
    with TestClient(create_app(export_root=str(tmp_path))) as client:
        assert [r["run_id"] for r in client.get("/runs").json()] == [state.run_id]
        assert json.loads(client.get(f"/run/{state.run_id}").content)["message"] == "m"

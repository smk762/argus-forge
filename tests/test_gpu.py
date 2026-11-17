"""The argus-iris GPU lease a run takes (#41): the tenant manifest, the lease's shape, and its lifecycle."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from argus_cortex.gpu import GpuLease, GpuLeaseTimeout, NullCoordinator
from argus_cortex.gpu_broker import BrokerCoordinator
from argus_cortex.gpu_wire import load_tenant_manifest
from conftest import forge_stub

from argus_forge import gpu
from argus_forge.models import RunEvent, RunRequest
from argus_forge.runner import astream_run

MANIFEST = Path(__file__).resolve().parent.parent / "gpu-tenant.yaml"


class _Recording:
    """A coordinator that records each lease's arguments and how its body ended."""

    name = "rec"

    def __init__(self, *, refuse: BaseException | None = None) -> None:
        self.calls: list[dict] = []
        self.exits: list[type[BaseException] | None] = []
        self.refuse = refuse

    @contextlib.asynccontextmanager
    async def alease(self, **kwargs) -> AsyncIterator[GpuLease]:
        self.calls.append(kwargs)
        if self.refuse is not None:
            raise self.refuse
        try:
            yield GpuLease(via="broker", node="morpheus")
        except BaseException as exc:
            self.exits.append(type(exc))
            raise
        self.exits.append(None)


async def _collect(req: RunRequest, **kw) -> list[RunEvent]:
    return [ev async for ev in astream_run(req, **kw)]


def test_manifest_validates_and_matches_the_lease() -> None:
    """gpu-tenant.yaml validates and declares the kind the runner leases, with the same terms."""
    manifest = load_tenant_manifest(MANIFEST)
    assert manifest.tenant == gpu.TENANT
    assert manifest.residents == ()
    kind = manifest.kind_for(gpu.LEASE_KIND_TRAIN)
    assert kind is not None
    assert kind.priority == gpu.LEASE_PRIORITY
    assert kind.preemptible is False
    assert kind.constraints.node == gpu.LEASE_NODE
    assert kind.vram_mb == gpu.DEFAULT_TRAIN_VRAM_MB


def test_broker_coordinator_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """ARGUS_GPU_COORDINATOR=broker builds cortex's broker client, the tenant read from the manifest."""
    monkeypatch.setenv("ARGUS_GPU_COORDINATOR", "broker")
    monkeypatch.setenv("ARGUS_GPU_BROKER_URL", "http://broker.invalid:8120")
    monkeypatch.setenv("ARGUS_GPU_TENANT_MANIFEST", str(MANIFEST))
    coord = gpu.coordinator()
    assert isinstance(coord, BrokerCoordinator)
    assert coord.tenant == gpu.TENANT
    assert coord.manifest is not None and coord.manifest.tenant == gpu.TENANT
    assert gpu.coordinator() is coord  # built once, so the manifest registers once


def test_unset_env_means_no_gating() -> None:
    assert isinstance(gpu.coordinator(), NullCoordinator)


@pytest.mark.parametrize(
    ("raw", "want"), [("", gpu.DEFAULT_TRAIN_VRAM_MB), ("10000", 10000), ("lots", 22000), ("-1", 22000)]
)
def test_vram_override(monkeypatch: pytest.MonkeyPatch, raw: str, want: int) -> None:
    monkeypatch.setenv(gpu.VRAM_ENV, raw)
    assert gpu.train_vram_mb() == want


async def test_run_takes_an_exclusive_background_lease(tmp_path: Path) -> None:
    """The lease names the train kind, background priority, not preemptible, pinned to morpheus, whole card."""
    coord = _Recording()
    export = forge_stub(tmp_path, "kohya", "echo hi\n")
    events = await _collect(RunRequest(export_dir=str(export), trainer="kohya"), run_id="r1", coordinator=coord)

    (call,) = coord.calls
    assert call["kind"] == "train"
    assert call["priority"] == "background"
    assert call["preemptible"] is False
    assert call["constraints"].node == "morpheus"
    assert call["min_vram_mb"] == gpu.DEFAULT_TRAIN_VRAM_MB
    assert "r1" in call["caller"]
    logs = [e.message for e in events if e.type == "log"]
    assert logs[0].startswith("[argus-forge] waiting for a GPU lease")
    assert logs[1] == "[argus-forge] GPU lease granted via broker on morpheus"
    assert logs[2] == "hi"
    assert events[-1].type == "exit" and events[-1].returncode == 0
    assert coord.exits == [None]


async def test_lease_is_released_before_the_terminal_event(tmp_path: Path) -> None:
    """Whoever acts on `exit` (an argus-proof eval) must find the GPU already free."""
    coord = _Recording()
    export = forge_stub(tmp_path, "kohya", "true\n")
    async for ev in astream_run(RunRequest(export_dir=str(export), trainer="kohya"), coordinator=coord):
        if ev.type == "exit":
            assert coord.exits == [None]


async def test_failed_run_is_reported_to_the_coordinator(tmp_path: Path) -> None:
    """A non-zero exit reaches the lease as an exception, so the broker records `failed`, not `ok`."""
    coord = _Recording()
    export = forge_stub(tmp_path, "kohya", "exit 3\n")
    events = await _collect(RunRequest(export_dir=str(export), trainer="kohya"), coordinator=coord)
    assert events[-1].type == "exit" and events[-1].returncode == 3
    assert coord.exits[0] is not None and coord.exits[0] is not GeneratorExit


async def test_lease_not_granted_is_an_error_and_nothing_runs(tmp_path: Path) -> None:
    sentinel = tmp_path / "ran"
    coord = _Recording(refuse=GpuLeaseTimeout("no room within 300s"))
    export = forge_stub(tmp_path, "kohya", f'touch "{sentinel}"\n')
    events = await _collect(RunRequest(export_dir=str(export), trainer="kohya"), coordinator=coord)
    assert events[-1].type == "error"
    assert events[-1].message == "GPU lease not granted: no room within 300s"
    assert not sentinel.exists()


async def test_dry_run_takes_no_lease(tmp_path: Path) -> None:
    coord = _Recording()
    export = forge_stub(tmp_path, "kohya", "true\n")
    await _collect(RunRequest(export_dir=str(export), trainer="kohya", dry_run=True), coordinator=coord)
    assert coord.calls == []


async def test_peak_vram_is_the_drop_below_the_launch_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    readings = iter([20000, 15000, 9000, 12000])
    monkeypatch.setattr(gpu, "free_vram_mb", lambda _device=0: next(readings, 12000))
    async with gpu.lease_training(NullCoordinator(), "r", sample_s=0.01) as lease:
        for _ in range(20):
            if lease.peak_vram_mb == 11000:
                break
            await asyncio.sleep(0.01)
    assert lease.peak_vram_mb == 11000


def _broker(log: list[tuple[str, dict]], *, lease_route: bool = True) -> BrokerCoordinator:
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        log.append((request.url.path, body))
        if not lease_route:
            return httpx.Response(404)
        if request.url.path == "/tenants":
            return httpx.Response(200, json={})
        if request.url.path == "/leases":
            return httpx.Response(200, json={"lease_id": "L1", "state": "granted", "node": "morpheus", "ttl_s": 60})
        return httpx.Response(200, json={"state": "granted", "ttl_s": 60})

    return BrokerCoordinator(
        "http://iris.test",
        tenant=gpu.TENANT,
        manifest=load_tenant_manifest(MANIFEST),
        async_transport=httpx.MockTransport(handle),
        transport=httpx.MockTransport(handle),
    )


@pytest.mark.parametrize(
    ("body", "outcome"), [("true\n", "ok"), ("exit 2\n", "failed"), ("echo 'CUDA out of memory'\nexit 1\n", "oom")]
)
async def test_through_cortex_broker_client(tmp_path: Path, body: str, outcome: str) -> None:
    """End to end with the real broker client: register, lease, then release with the run's outcome."""
    log: list[tuple[str, dict]] = []
    export = forge_stub(tmp_path, "kohya", body)
    events = await _collect(RunRequest(export_dir=str(export), trainer="kohya"), coordinator=_broker(log))

    paths = [p for p, _ in log]
    assert paths[0] == "/tenants" and paths[1] == "/leases" and paths[-1] == "/leases/L1/release"
    lease_req = log[1][1]
    assert lease_req["tenant"] == "argus-forge" and lease_req["kind"] == "train"
    assert lease_req["priority"] == "background" and lease_req["constraints"]["node"] == "morpheus"
    assert log[-1][1]["outcome"] == outcome
    assert events[-1].type == "exit"


async def test_broker_without_lease_routes_falls_back(tmp_path: Path) -> None:
    """iris is observe-only today: /leases 404s, so the run proceeds on the fallback coordinator."""
    log: list[tuple[str, dict]] = []
    export = forge_stub(tmp_path, "kohya", "echo trained\n")
    events = await _collect(
        RunRequest(export_dir=str(export), trainer="kohya"), coordinator=_broker(log, lease_route=False)
    )
    assert "[argus-forge] GPU lease granted via fallback" in [e.message for e in events]
    assert events[-1].type == "exit" and events[-1].returncode == 0

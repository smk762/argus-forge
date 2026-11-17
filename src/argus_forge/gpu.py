"""Training's GPU lease, through argus-cortex's coordinator (argus-iris, the suite's GPU broker).

A run takes one lease for the whole of ``train.sh``: kind ``train``, ``background``
priority, not preemptible, pinned to morpheus, and sized to fill the card so no
other tenant is admitted beside it (an exclusive whole-card lease). The kind is
declared in the committed ``gpu-tenant.yaml``; ``tests/test_gpu.py`` keeps the
constants here and the manifest in step.

Which coordinator is the environment's (:func:`argus_cortex.gpu.coordinator_from_env`):
unset ``ARGUS_GPU_COORDINATOR`` means ``none``, so a run starts at once exactly as
it did before the broker existed. With ``broker``, iris decides when the run may
start (the operator's schedule window for ``argus-forge`` sets its class); when iris
is unreachable the lease comes from ``ARGUS_GPU_BROKER_FALLBACK`` instead.

Peak VRAM is sampled through NVML (``argus-cortex[gpu]``) as the drop in the card's
free memory below what was free at launch; the lease is whole-card, so that drop is
the trainer's. Without NVML nothing is reported and the broker keeps its own estimate.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import os
from collections.abc import AsyncIterator

import structlog
from argus_cortex.gpu import GpuCoordinator, GpuLease, coordinator_from_env, free_vram_mb
from argus_cortex.gpu_wire import GpuConstraints

logger = structlog.get_logger()

TENANT = "argus-forge"
LEASE_KIND_TRAIN = "train"
LEASE_PRIORITY = "background"
LEASE_NODE = "morpheus"
# Whole card on morpheus's RTX 3090 (24576 MiB) less iris's per-node headroom and
# the driver's own use: large enough that nothing else the suite declares fits
# beside it, small enough that iris (and the local `lease` fallback, which waits
# for this much free) can actually admit it.
DEFAULT_TRAIN_VRAM_MB = 22000
VRAM_ENV = "FORGE_GPU_VRAM_MB"
DEVICE_ENV = "ARGUS_GPU_DEVICE"
PEAK_SAMPLE_S = 5.0


class LeaseNotGranted(RuntimeError):
    """The coordinator would not grant the run a GPU (timed out, refused, or the broker rejected the request)."""


def train_vram_mb() -> int:
    """The lease's footprint: ``FORGE_GPU_VRAM_MB`` if set to a positive integer, else the whole-card default."""
    raw = os.environ.get(VRAM_ENV, "").strip()
    if not raw:
        return DEFAULT_TRAIN_VRAM_MB
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        logger.warning("gpu_vram_env_ignored", env=VRAM_ENV, value=raw, default=DEFAULT_TRAIN_VRAM_MB)
        return DEFAULT_TRAIN_VRAM_MB
    return value


@functools.cache
def coordinator() -> GpuCoordinator:
    """The process's coordinator, built once so the broker registers the tenant manifest once."""
    return coordinator_from_env()


def _lease_errors() -> tuple[type[BaseException], ...]:
    from argus_cortex.gpu import GpuLeaseRefused, GpuLeaseTimeout

    errors: list[type[BaseException]] = [GpuLeaseTimeout, GpuLeaseRefused]
    with contextlib.suppress(ImportError):
        from argus_cortex.gpu_broker import BrokerError

        errors.append(BrokerError)
    return tuple(errors)


def _device(lease: GpuLease) -> int:
    if lease.device is not None:
        return lease.device
    try:
        return int(os.environ.get(DEVICE_ENV, "") or 0)
    except ValueError:
        return 0


async def _sample_peak(lease: GpuLease, device: int, baseline: int, interval_s: float) -> None:
    """Report the largest drop in free VRAM below *baseline* until cancelled."""
    while True:
        await asyncio.sleep(interval_s)
        free = await asyncio.to_thread(free_vram_mb, device)
        if free is not None:
            lease.report_peak_vram(max(0, baseline - free))


@contextlib.asynccontextmanager
async def lease_training(
    coord: GpuCoordinator, run_id: str, *, sample_s: float | None = None
) -> AsyncIterator[GpuLease]:
    """Hold the training lease for the body; raise :class:`LeaseNotGranted` if it is not granted.

    An exception from the body reaches the coordinator, which reports the run's
    outcome (``failed``, ``oom``, ``cancelled``) to the broker on release.
    """
    try:
        cm = coord.alease(
            caller=f"argus-forge run {run_id}",
            min_vram_mb=train_vram_mb(),
            kind=LEASE_KIND_TRAIN,
            priority=LEASE_PRIORITY,
            preemptible=False,
            constraints=GpuConstraints(node=LEASE_NODE),
        )
        lease = await cm.__aenter__()
    except _lease_errors() as exc:
        raise LeaseNotGranted(str(exc) or type(exc).__name__) from exc

    sampler: asyncio.Task[None] | None = None
    try:
        device = _device(lease)
        baseline = await asyncio.to_thread(free_vram_mb, device)
        if baseline is not None:
            sampler = asyncio.create_task(_sample_peak(lease, device, baseline, sample_s or PEAK_SAMPLE_S))
        yield lease
    except BaseException as exc:
        await _stop(sampler)
        if not await cm.__aexit__(type(exc), exc, exc.__traceback__):
            raise
    else:
        await _stop(sampler)
        await cm.__aexit__(None, None, None)


async def _stop(task: asyncio.Task[None] | None) -> None:
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task

"""``Idempotency-Key`` on ``POST /run``: a retried request must not start a second run.

A training run holds the GPU for hours, so a client that retries ``POST /run``
after a dropped connection (the run started, the 202 never arrived) must get the
run it already started rather than a duplicate. A caller opts in by sending an
``Idempotency-Key`` header; without one, ``POST /run`` behaves exactly as before.

The server keeps, per key, the run it started and a fingerprint of the request
(sha256 over the validated :class:`~argus_forge.models.RunRequest` dumped with
sorted keys, so whitespace or key order in the JSON body doesn't matter):

* same key, same fingerprint -> the existing run, whatever its state now;
* same key, different fingerprint -> refused (409), no new run;
* a new key -> a new run, recorded under the key.

Keys are **global to the server**: forge has no user identity beyond its shared
write guards, so two clients that pick the same key share it. Use a random key
(a UUID) per intended run.

Records live in ``<state_dir>/idempotency/<sha256(key)>.json`` (atomic writes,
like :class:`~argus_forge.server.jobs.RunStore`), so a replay after a restart
still finds its run; without a state dir they are kept in memory. The raw key is
never written: only its sha256. Records older than the retention window
(``ARGUS_FORGE_IDEMPOTENCY_RETENTION_S``, default 24 h) are pruned lazily on the
next keyed ``POST /run``, except one whose run is still active, which is kept
for as long as the run lasts.

Like :mod:`argus_forge.server.jobs`, this needs nothing from the ``server``
extra: stdlib, structlog and pydantic only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import structlog
from pydantic import BaseModel, ValidationError

from argus_forge.models import IDEMPOTENCY_RETENTION_ENV, RunRequest

logger = structlog.get_logger()

IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
# Set to "true" on a response that returns a run an earlier request started.
REPLAYED_HEADER = "Idempotent-Replayed"
# 1-200 printable ASCII characters (space through tilde).
MAX_KEY_LENGTH = 200
_VALID_KEY = re.compile(rf"[\x20-\x7e]{{1,{MAX_KEY_LENGTH}}}")
# Fixed text: the offending value is never echoed back.
INVALID_KEY_DETAIL = f"{IDEMPOTENCY_KEY_HEADER} must be 1-{MAX_KEY_LENGTH} printable ASCII characters"

DEFAULT_RETENTION_S = 24 * 60 * 60.0

# A record file is named by the key's sha256; anything else in the dir is not ours.
_KEY_HASH = re.compile(r"[0-9a-f]{64}")


def valid_key(key: str) -> bool:
    """Whether *key* is an acceptable ``Idempotency-Key`` value."""
    return _VALID_KEY.fullmatch(key) is not None


def key_hash(key: str) -> str:
    """The sha256 a key is stored and looked up under (the raw key is never persisted)."""
    return hashlib.sha256(key.encode("ascii")).hexdigest()


def request_fingerprint(req: RunRequest) -> str:
    """sha256 over the validated request, canonically dumped (sorted keys, no whitespace)."""
    canonical = json.dumps(req.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def env_retention_s() -> float:
    """The retention window from ``ARGUS_FORGE_IDEMPOTENCY_RETENTION_S``, in seconds.

    Unset, unparseable or negative falls back to the default (with a warning for
    the latter two): never fatal, like the other deployment knobs.
    """
    raw = os.environ.get(IDEMPOTENCY_RETENTION_ENV, "").strip()
    if not raw:
        return DEFAULT_RETENTION_S
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if not value >= 0 or value == float("inf"):  # also catches NaN
        logger.warning("idempotency_retention_invalid", env=IDEMPOTENCY_RETENTION_ENV, default=DEFAULT_RETENTION_S)
        return DEFAULT_RETENTION_S
    return value


class IdempotencyRecord(BaseModel):
    """What a key started: kept in memory and, with a state dir, on disk. Not part of the wire contract."""

    key_sha256: str
    run_id: str
    fingerprint: str
    created_at: str  # ISO-8601 UTC


def _now() -> datetime:
    return datetime.now(UTC)


class IdempotencyStore:
    """Keys -> the run each started, with lazy expiry and a lock per key.

    Every change to a key's record happens while holding that key's lock
    (:meth:`hold`), so a check-and-create cannot race another request carrying
    the same key, and a prune cannot delete a record written after it decided.
    """

    def __init__(
        self,
        state_dir: Path | None,
        retention_s: float = DEFAULT_RETENTION_S,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        """Loads the saved records (blocking, so call it at startup, not per request)."""
        self.dir = state_dir / "idempotency" if state_dir is not None else None
        self.retention_s = retention_s
        self.clock = clock
        self._records: dict[str, IdempotencyRecord] = {}
        # key hash -> [lock, holders]; an entry is dropped when its last holder leaves.
        self._locks: dict[str, list] = {}
        self._prune_lock = asyncio.Lock()
        if self.dir is not None:
            self._records = self._load()

    # --- locking ---

    @asynccontextmanager
    async def hold(self, khash: str) -> AsyncIterator[None]:
        """Hold *khash*'s lock for a check-and-create (or a prune) of its record."""
        entry = self._locks.get(khash)
        if entry is None:
            entry = self._locks[khash] = [asyncio.Lock(), 0]
        entry[1] += 1
        try:
            async with entry[0]:
                yield
        finally:
            entry[1] -= 1
            if entry[1] == 0:
                self._locks.pop(khash, None)

    # --- records ---

    def get(self, khash: str) -> IdempotencyRecord | None:
        return self._records.get(khash)

    def put(self, record: IdempotencyRecord) -> None:
        """Remember *record* in memory (call :meth:`save` off the loop to persist it)."""
        self._records[record.key_sha256] = record

    def new_record(self, khash: str, run_id: str, fingerprint: str) -> IdempotencyRecord:
        return IdempotencyRecord(
            key_sha256=khash, run_id=run_id, fingerprint=fingerprint, created_at=self.clock().isoformat()
        )

    def expired(self, record: IdempotencyRecord) -> bool:
        """Older than the retention window (an unreadable timestamp counts as expired)."""
        try:
            created = datetime.fromisoformat(record.created_at)
        except ValueError:
            return True
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        return (self.clock() - created).total_seconds() > self.retention_s

    async def prune(self, active: Callable[[str], bool]) -> None:
        """Drop expired records whose run is not *active* (lazy expiry).

        Must not be called while holding a key's lock (it takes each expired
        key's lock in turn). Concurrent calls collapse into one.
        """
        if self._prune_lock.locked():
            return
        async with self._prune_lock:
            for khash in [h for h, rec in self._records.items() if self.expired(rec) and not active(rec.run_id)]:
                async with self.hold(khash):
                    rec = self._records.get(khash)
                    # Re-checked under the lock: a request may have replaced it meanwhile.
                    if rec is None or not self.expired(rec) or active(rec.run_id):
                        continue
                    del self._records[khash]
                    if self.dir is not None:
                        await asyncio.to_thread(self._delete, khash)
                    logger.info("idempotency_key_pruned", run_id=rec.run_id)

    # --- disk (blocking: call via asyncio.to_thread from a route) ---

    def _path(self, khash: str) -> Path:
        assert self.dir is not None
        return self.dir / f"{khash}.json"

    def save(self, record: IdempotencyRecord) -> None:
        """Write *record* atomically; a failure is logged, not raised (the run has already started)."""
        if self.dir is None:
            return
        path = self._path(record.key_sha256)
        tmp = path.with_name(f".{path.name}.tmp")
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp.write_text(record.model_dump_json(), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            logger.error("idempotency_record_save_failed", run_id=record.run_id, error=repr(exc))

    def _delete(self, khash: str) -> None:
        try:
            self._path(khash).unlink(missing_ok=True)
        except OSError as exc:
            logger.error("idempotency_record_delete_failed", error=repr(exc))

    def _load(self) -> dict[str, IdempotencyRecord]:
        assert self.dir is not None
        if not self.dir.is_dir():
            return {}
        records: dict[str, IdempotencyRecord] = {}
        for path in sorted(self.dir.glob("*.json")):
            if not _KEY_HASH.fullmatch(path.stem):
                continue
            try:
                record = IdempotencyRecord.model_validate_json(path.read_bytes())
            except (OSError, ValidationError) as exc:
                logger.warning("idempotency_record_unreadable", path=str(path), error=repr(exc))
                continue
            if record.key_sha256 != path.stem:
                logger.warning("idempotency_record_mismatched", path=str(path))
                continue
            records[record.key_sha256] = record
        return records

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class SeedAllocation:
    key: str
    seed: int
    previous_seed: Optional[int]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _read_seed_map_from_text(text: str) -> Dict[str, int]:
    if not text.strip():
        return {}
    seed_map: Dict[str, int] = {}

    buf = io.StringIO(text)
    reader = csv.reader(buf)
    rows = list(reader)
    if not rows:
        return {}

    header = [c.strip() for c in (rows[0] or [])]
    header_lower = [c.lower() for c in header]
    has_header = "key" in header_lower or "run_id" in header_lower or "run" in header_lower

    if has_header:
        buf.seek(0)
        dict_reader = csv.DictReader(buf)
        for row in dict_reader:
            if not row:
                continue
            key = (row.get("key") or row.get("run") or row.get("run_id") or "").strip()
            if not key:
                continue
            seed_val = _parse_int(row.get("last_seed") or row.get("seed"))
            if seed_val is None:
                continue
            seed_map[key] = seed_val
        return seed_map

    # Headerless fallback: assume first col is key, second col is last_seed.
    for row in rows:
        if not row:
            continue
        key = str(row[0]).strip() if len(row) >= 1 else ""
        if not key:
            continue
        seed_val = _parse_int(row[1]) if len(row) >= 2 else None
        if seed_val is None:
            continue
        seed_map[key] = seed_val
    return seed_map


def _write_seed_map_to_handle(handle, seed_map: Dict[str, int]) -> None:
    handle.seek(0)
    handle.truncate()
    writer = csv.DictWriter(handle, fieldnames=["key", "last_seed", "updated_at"])
    writer.writeheader()
    now = _utc_now_iso()
    for key in sorted(seed_map.keys()):
        writer.writerow({"key": key, "last_seed": seed_map[key], "updated_at": now})
    handle.flush()
    os.fsync(handle.fileno())


def allocate_incrementing_seed(
    *,
    key: str,
    csv_path: Path,
    first_seed: int = 1338,
) -> SeedAllocation:
    """Allocate a monotonically increasing seed for a given logical run key.

    - Uses an on-disk CSV so multiple `runs_autopilot` processes can coordinate.
    - Concurrency-safe on Linux via an exclusive `fcntl.flock` lock.
    - The CSV stores the *last seed used* for each key; each allocation increments by +1.

    The first time a key is seen, it returns `first_seed`.
    """

    if not key or not str(key).strip():
        raise ValueError("seed allocation key must be a non-empty string")

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Open in a+ so the file exists for locking even on first use.
    with open(csv_path, "a+", encoding="utf-8", newline="") as lock_f:
        try:
            import fcntl

            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        except Exception:
            # If flock is unavailable for some reason, fall back to best-effort.
            # (On Linux this should basically never happen.)
            pass

        lock_f.seek(0)
        seed_map = _read_seed_map_from_text(lock_f.read())
        prev = seed_map.get(key)
        if prev is None:
            next_seed = int(first_seed)
        else:
            next_seed = int(prev) + 1

        seed_map[key] = next_seed
        _write_seed_map_to_handle(lock_f, seed_map)

        try:
            import fcntl

            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass

    return SeedAllocation(key=key, seed=next_seed, previous_seed=prev)

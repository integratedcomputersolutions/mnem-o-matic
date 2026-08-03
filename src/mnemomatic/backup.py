"""Scheduled on-server backups of the export archive.

Opt-in via MNEMOMATIC_BACKUP_DIR: the server writes the full export zip
(every namespace) into that directory on a fixed interval and prunes
archives beyond the retention count. Backups reuse the export format, so
any backup can be restored like a manual export.

Backups are named ``mnemomatic-backup-YYYYMMDD-HHMMSS.zip`` (UTC) —
a prefix distinct from manual exports (``mnemomatic-export-...``) — and
pruning only ever touches that pattern, so exports or anything else stored
alongside are never deleted.

The schedule survives restarts: the next backup is due one interval after
the newest existing archive, not one interval after boot, so frequent
restarts neither skip backups nor churn the retention window.
"""

import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from mnemomatic.export import build_export_zip

logger = logging.getLogger("mnemomatic")

_PATTERN = "mnemomatic-backup-*.zip"


def _backups(backup_dir: Path) -> list[Path]:
    """Existing backup archives, oldest first (the name's timestamp sorts)."""
    return sorted(backup_dir.glob(_PATTERN))


def run_backup(db, backup_dir: Path, keep: int, server_version: str) -> Path:
    """Write one backup archive atomically, prune beyond *keep*, return its path."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    data, _ = build_export_zip(db, server_version=server_version)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = backup_dir / f"mnemomatic-backup-{stamp}.zip"
    part = target.with_name(target.name + ".part")
    part.write_bytes(data)
    part.replace(target)
    for stale in _backups(backup_dir)[:-keep]:
        stale.unlink()
        logger.info("Backup pruned: %s", stale.name)
    return target


def next_delay(backup_dir: Path, interval: float) -> float:
    """Seconds until the next backup is due, based on the newest existing archive."""
    existing = _backups(backup_dir) if backup_dir.is_dir() else []
    if not existing:
        return 0.0
    age = time.time() - existing[-1].stat().st_mtime
    return max(0.0, interval - age)


def backup_loop(db_getter, backup_dir: Path, interval: float, keep: int,
                server_version: str, stop: threading.Event) -> None:
    """Back up whenever the newest archive is *interval* seconds old, until *stop*."""
    while not stop.wait(next_delay(backup_dir, interval)):
        try:
            target = run_backup(db_getter(), backup_dir, keep, server_version)
            logger.info("Backup written: %s", target)
        except Exception as e:
            logger.error("Backup failed: %s: %s", type(e).__name__, e)
            # The newest archive is still old, so next_delay stays 0 while the
            # failure persists — wait a full interval instead of hot-looping.
            if stop.wait(interval):
                return


def start_backup_thread(db_getter, backup_dir: Path, *, interval_hours: float,
                        keep: int, server_version: str) -> threading.Thread:
    """Start the backup loop on a daemon thread and return it."""
    if interval_hours <= 0:
        raise ValueError(f"MNEMOMATIC_BACKUP_INTERVAL must be positive, got {interval_hours}")
    if keep < 1:
        raise ValueError(f"MNEMOMATIC_BACKUP_KEEP must be at least 1, got {keep}")
    thread = threading.Thread(
        target=backup_loop,
        args=(db_getter, backup_dir, interval_hours * 3600.0, keep,
              server_version, threading.Event()),
        name="mnemomatic-backup",
        daemon=True,
    )
    thread.start()
    return thread

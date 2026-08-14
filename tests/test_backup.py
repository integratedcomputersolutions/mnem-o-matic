"""Tests for scheduled backups (mnemomatic.backup).

Covers the archive write (atomic, correctly named, valid export zip),
retention pruning (only backup-pattern files touched), restart-aware
scheduling via next_delay, and the loop's failure handling.
"""

import io
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path

from mnemomatic.backup import backup_loop, next_delay, run_backup, start_backup_thread
from mnemomatic.db import Database
from mnemomatic.models import Note


class BackupDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        # File-backed db: the loop runs on its own thread and thread-local
        # connections each see a distinct, empty ":memory:" database.
        self.db = Database(str(Path(self._tmp.name) / "test.db"))
        self.db.store_note(Note(namespace="proj", title="n", content="x"), embedding=None)

    def tearDown(self):
        self.db.close()
        self._tmp.cleanup()


class TestRunBackup(BackupDirTestCase):
    def test_writes_valid_export_zip(self):
        target = run_backup(self.db, self.dir, keep=7, server_version="0.0.0-test")
        self.assertRegex(target.name, r"^mnemomatic-backup-\d{8}-\d{6}\.zip$")
        with zipfile.ZipFile(io.BytesIO(target.read_bytes())) as zf:
            self.assertIn("proj/notes/n.md", zf.namelist())
        # Atomic write: no .part left behind.
        self.assertEqual(list(self.dir.glob("*.part")), [])

    def test_creates_missing_directory(self):
        nested = self.dir / "a" / "b"
        target = run_backup(self.db, nested, keep=1, server_version="0.0.0-test")
        self.assertTrue(target.exists())

    def test_prunes_only_backup_files_beyond_keep(self):
        for stamp in ("20260101-000000", "20260102-000000", "20260103-000000"):
            (self.dir / f"mnemomatic-backup-{stamp}.zip").write_bytes(b"old")
        keepers = [self.dir / "mnemomatic-export-2026-01-01.zip", self.dir / "unrelated.txt"]
        for p in keepers:
            p.write_bytes(b"keep")

        run_backup(self.db, self.dir, keep=2, server_version="0.0.0-test")

        backups = sorted(p.name for p in self.dir.glob("mnemomatic-backup-*.zip"))
        self.assertEqual(len(backups), 2)
        # The two oldest fakes are gone; the newest fake + the real one remain.
        self.assertEqual(backups[0], "mnemomatic-backup-20260103-000000.zip")
        for p in keepers:
            self.assertTrue(p.exists())


class TestNextDelay(BackupDirTestCase):
    def test_no_archives_means_due_now(self):
        self.assertEqual(next_delay(self.dir, 3600.0), 0.0)

    def test_missing_directory_means_due_now(self):
        self.assertEqual(next_delay(self.dir / "nope", 3600.0), 0.0)

    def test_fresh_archive_waits_out_the_remainder(self):
        (self.dir / "mnemomatic-backup-20260101-000000.zip").write_bytes(b"x")
        delay = next_delay(self.dir, 3600.0)
        self.assertGreater(delay, 3500.0)
        self.assertLessEqual(delay, 3600.0)

    def test_old_archive_is_overdue(self):
        stale = self.dir / "mnemomatic-backup-20260101-000000.zip"
        stale.write_bytes(b"x")
        import os
        os.utime(stale, (time.time() - 7200, time.time() - 7200))
        self.assertEqual(next_delay(self.dir, 3600.0), 0.0)


class TestBackupLoop(BackupDirTestCase):
    def _run_loop(self, db_getter, interval, stop):
        thread = threading.Thread(
            target=backup_loop,
            args=(db_getter, self.dir, interval, 7, "0.0.0-test", stop),
            daemon=True,
        )
        thread.start()
        return thread

    def _wait_for(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def test_backs_up_once_then_waits(self):
        stop = threading.Event()
        thread = self._run_loop(lambda: self.db, interval=3600.0, stop=stop)
        self.assertTrue(self._wait_for(lambda: list(self.dir.glob("mnemomatic-backup-*.zip"))))
        stop.set()
        thread.join(timeout=5.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(list(self.dir.glob("mnemomatic-backup-*.zip"))), 1)

    def test_failure_is_caught_and_loop_stays_stoppable(self):
        def broken():
            raise RuntimeError("db unavailable")

        stop = threading.Event()
        thread = self._run_loop(broken, interval=3600.0, stop=stop)
        # Give the loop a moment to hit the failure path, then stop it.
        time.sleep(0.1)
        self.assertTrue(thread.is_alive())
        stop.set()
        thread.join(timeout=5.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(list(self.dir.glob("*.zip")), [])


class TestStartValidation(BackupDirTestCase):
    def test_rejects_bad_config(self):
        with self.assertRaises(ValueError):
            start_backup_thread(lambda: self.db, self.dir,
                                interval_hours=0, keep=7, server_version="t")
        with self.assertRaises(ValueError):
            start_backup_thread(lambda: self.db, self.dir,
                                interval_hours=24, keep=0, server_version="t")


if __name__ == "__main__":
    unittest.main()

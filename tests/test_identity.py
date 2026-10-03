"""Tests for the identity module: password hashing, users and roles,
sessions, API tokens, the login throttle, and first-run bootstrap."""

import io
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from mnemomatic import identity
from mnemomatic.db import Database
from mnemomatic.identity import (
    DEVICE_COOKIE_TTL,
    DUMMY_HASH,
    FirstRun,
    Identity,
    IdentityError,
    LoginThrottle,
    ensure_bootstrap,
    generate_setup_code,
    generate_token,
    hash_password,
    needs_rehash,
    verify_password,
)
from mnemomatic.throttle import FailureThrottle, client_key


def _fast_scrypt():
    """Patch the work factor down so the suite does not spend seconds hashing."""
    return patch.multiple(identity, SCRYPT_LOG_N=10, SCRYPT_P=1)


class TestPasswordHashing(unittest.TestCase):
    def test_round_trip_and_format(self):
        with _fast_scrypt():
            stored = hash_password("correct horse battery")
            self.assertTrue(stored.startswith("$scrypt$ln=10,r=8,p=1$"))
            self.assertEqual(stored.count("$"), 4)
            self.assertTrue(verify_password(stored, "correct horse battery"))
            self.assertFalse(verify_password(stored, "correct horse batter"))

    def test_salts_differ(self):
        with _fast_scrypt():
            self.assertNotEqual(hash_password("same"), hash_password("same"))

    def test_malformed_hash_verifies_false(self):
        self.assertFalse(verify_password("", "x"))
        self.assertFalse(verify_password("$scrypt$ln=zz$a$b", "x"))
        self.assertFalse(verify_password("$bcrypt$x$y$z", "x"))

    def test_needs_rehash_tracks_parameters(self):
        with _fast_scrypt():
            weak = hash_password("pw")
        self.assertTrue(needs_rehash(weak))
        self.assertTrue(needs_rehash("garbage"))
        with _fast_scrypt():
            # Under the patched constants the same hash counts as current.
            self.assertFalse(needs_rehash(weak))

    def test_dummy_hash_is_valid_format(self):
        self.assertTrue(DUMMY_HASH.startswith("$scrypt$"))
        self.assertFalse(verify_password(DUMMY_HASH, ""))


class TestGenerators(unittest.TestCase):
    def test_token_shape(self):
        tok = generate_token()
        self.assertTrue(tok.startswith("mnm_"))
        self.assertGreaterEqual(len(tok), 4 + 43)
        self.assertNotEqual(tok, generate_token())

    def test_setup_code_shape(self):
        code = generate_setup_code()
        self.assertRegex(code, r"^[A-Z2-9]{4}-[A-Z2-9]{4}-[A-Z2-9]{4}$")
        for bad in "0O1IL":
            self.assertNotIn(bad, code)


class IdentityCase(unittest.TestCase):
    """A temp-file database (per-thread connections need a real file)."""

    def setUp(self):
        self._patch = _fast_scrypt()
        self._patch.start()
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        self.path = Path(tmp.name)
        self.db = Database(str(self.path))
        self.ident = Identity(self.db)

    def tearDown(self):
        self.db.close()
        self._patch.stop()
        for p in (self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm")):
            p.unlink(missing_ok=True)

    def admin(self, name="root", password="rootpassword1"):
        return self.ident.create_user(name, role="admin", password=password)[0]


class TestUsers(IdentityCase):
    def test_create_with_password_and_authenticate(self):
        user = self.admin()
        self.assertEqual(user.role, "admin")
        self.assertFalse(user.must_change_password)
        self.assertEqual(self.ident.authenticate("ROOT", "rootpassword1").id, user.id)

    def test_username_normalized_and_validated(self):
        user, _ = self.ident.create_user("  Alice.Smith ", password="alicepassword")
        self.assertEqual(user.username, "alice.smith")
        for bad in ("a", "-dash", "has space", "x" * 33, "üml"):
            with self.assertRaises(IdentityError) as cm:
                self.ident.create_user(bad, password="validpassword1")
            self.assertEqual(cm.exception.code, "invalid_username")

    def test_duplicate_is_case_insensitive(self):
        self.admin("bob")
        with self.assertRaises(IdentityError) as cm:
            self.ident.create_user("BOB", password="anotherpassword")
        self.assertEqual(cm.exception.status, 409)

    def test_weak_password_refused(self):
        with self.assertRaises(IdentityError) as cm:
            self.ident.create_user("carol", password="short")
        self.assertEqual(cm.exception.code, "weak_password")
        with self.assertRaises(IdentityError):
            self.ident.create_user("carol", password="x" * 1025)

    def test_temporary_password_flow(self):
        user, temp = self.ident.create_user("dave")
        self.assertTrue(user.must_change_password)
        self.assertIsNotNone(user.temp_password_expires_at)
        self.assertEqual(len(temp), 16)
        authed = self.ident.authenticate("dave", temp)
        self.assertTrue(authed.must_change_password)
        self.ident.change_password(user.id, temp, "brand new password")
        self.assertFalse(self.ident.get_user(user.id).must_change_password)
        self.assertIsNone(self.ident.get_user(user.id).temp_password_expires_at)

    def test_expired_temporary_password(self):
        user, temp = self.ident.create_user("erin")
        conn = self.db.connection()
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        conn.execute("UPDATE users SET temp_password_expires_at = ? WHERE id = ?", (past, user.id))
        conn.commit()
        with self.assertRaises(IdentityError) as cm:
            self.ident.authenticate("erin", temp)
        self.assertEqual(cm.exception.code, "temp_password_expired")

    def test_authenticate_failures(self):
        self.admin()
        with self.assertRaises(IdentityError) as cm:
            self.ident.authenticate("root", "wrong")
        self.assertEqual(cm.exception.code, "invalid_credentials")
        with self.assertRaises(IdentityError) as cm:
            self.ident.authenticate("nobody", "rootpassword1")
        self.assertEqual(cm.exception.code, "invalid_credentials")

    def test_unknown_user_still_runs_one_verification(self):
        with patch.object(identity, "verify_password", wraps=identity.verify_password) as v:
            with self.assertRaises(IdentityError):
                self.ident.authenticate("ghost", "whatever")
            v.assert_called_once()
            self.assertEqual(v.call_args.args[0], DUMMY_HASH)

    def test_rehash_on_login(self):
        self.admin()
        conn = self.db.connection()
        weaker = identity.hash_password("rootpassword1", log_n=9, p=1)
        conn.execute("UPDATE users SET password_hash = ? WHERE username = 'root'", (weaker,))
        conn.commit()
        self.ident.authenticate("root", "rootpassword1")
        stored = conn.execute("SELECT password_hash FROM users WHERE username = 'root'").fetchone()
        self.assertIn("ln=10,", stored["password_hash"])
        self.assertTrue(verify_password(stored["password_hash"], "rootpassword1"))

    def test_change_password_checks_current_and_policy(self):
        user = self.admin()
        with self.assertRaises(IdentityError) as cm:
            self.ident.change_password(user.id, "nope", "something long enough")
        self.assertEqual(cm.exception.code, "wrong_password")
        with self.assertRaises(IdentityError):
            self.ident.change_password(user.id, "rootpassword1", "short")
        with self.assertRaises(IdentityError):
            self.ident.change_password(user.id, "rootpassword1", "rootpassword1")

    def test_change_password_keeps_only_the_current_session(self):
        user = self.admin()
        keep = self.ident.create_session(user.id)
        other = self.ident.create_session(user.id)
        self.ident.change_password(user.id, "rootpassword1", "a different password", keep_session=keep)
        self.assertIsNotNone(self.ident.resolve_session(keep))
        self.assertIsNone(self.ident.resolve_session(other))

    def test_end_all_sessions(self):
        user = self.admin()
        sessions = [self.ident.create_session(user.id) for _ in range(3)]
        self.assertEqual(self.ident.end_all_sessions(), 3)
        for raw in sessions:
            self.assertIsNone(self.ident.resolve_session(raw))
        self.assertEqual(self.ident.end_all_sessions(), 0)


class TestAdminGuards(IdentityCase):
    def test_self_actions_refused(self):
        root = self.admin()
        for call in (lambda: self.ident.set_active(root.id, False, acting_user_id=root.id),
                     lambda: self.ident.set_role(root.id, "user", acting_user_id=root.id),
                     lambda: self.ident.delete_user(root.id, acting_user_id=root.id),
                     lambda: self.ident.reset_password(root.id, acting_user_id=root.id)):
            with self.assertRaises(IdentityError) as cm:
                call()
            self.assertEqual(cm.exception.code, "self_action")

    def test_last_admin_guard(self):
        root = self.admin()
        other = self.admin("second", "secondpassword")
        # Two admins: demoting, deactivating, deleting one is fine.
        self.ident.set_role(other.id, "user", acting_user_id=root.id)
        # Now root is the only admin; nobody may remove it.
        for call in (lambda: self.ident.set_active(root.id, False, acting_user_id=other.id),
                     lambda: self.ident.set_role(root.id, "user", acting_user_id=other.id),
                     lambda: self.ident.delete_user(root.id, acting_user_id=other.id)):
            with self.assertRaises(IdentityError) as cm:
                call()
            self.assertEqual(cm.exception.code, "last_admin")
        # Promoting back makes root removable again.
        self.ident.set_role(other.id, "admin", acting_user_id=root.id)
        self.ident.delete_user(root.id, acting_user_id=other.id)
        self.assertIsNone(self.ident.get_user(root.id))

    def test_inactive_admin_does_not_count(self):
        root = self.admin()
        sleeper = self.admin("sleeper", "sleeperpassword")
        self.ident.set_active(sleeper.id, False, acting_user_id=root.id)
        with self.assertRaises(IdentityError) as cm:
            self.ident.set_role(root.id, "user", acting_user_id=sleeper.id)
        self.assertEqual(cm.exception.code, "last_admin")

    def test_deactivate_ends_sessions_and_revokes_tokens(self):
        root = self.admin()
        user, _ = self.ident.create_user("frank", password="frankpassword")
        sid = self.ident.create_session(user.id)
        _, raw = self.ident.create_token(user.id, "laptop")
        _, revoked = self.ident.set_active(user.id, False, acting_user_id=root.id)
        self.assertEqual(revoked, 1)
        self.assertIsNone(self.ident.resolve_session(sid))
        self.assertIsNone(self.ident.resolve_token(raw))
        with self.assertRaises(IdentityError) as cm:
            self.ident.authenticate("frank", "frankpassword")
        self.assertEqual(cm.exception.code, "account_disabled")
        # Reactivating must not revive the old tokens: an account is usually
        # disabled because a credential leaked.
        self.ident.set_active(user.id, True, acting_user_id=root.id)
        self.assertIsNone(self.ident.resolve_token(raw))
        self.assertIsNotNone(self.ident.authenticate("frank", "frankpassword"))

    def test_delete_cascades(self):
        root = self.admin()
        user, _ = self.ident.create_user("gina", password="ginapassword1")
        self.ident.create_session(user.id)
        self.ident.create_token(user.id, "t")
        self.ident.delete_user(user.id, acting_user_id=root.id)
        conn = self.db.connection()
        self.assertEqual(conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"], 0)
        self.assertEqual(conn.execute("SELECT COUNT(*) AS n FROM api_tokens").fetchone()["n"], 0)

    def test_reset_password_ends_sessions_keeps_tokens(self):
        root = self.admin()
        user, _ = self.ident.create_user("hank", password="hankpassword1")
        sid = self.ident.create_session(user.id)
        _, raw = self.ident.create_token(user.id, "agent")
        temp, expires = self.ident.reset_password(user.id, acting_user_id=root.id)
        self.assertIsNone(self.ident.resolve_session(sid))
        self.assertIsNotNone(self.ident.resolve_token(raw))
        self.assertTrue(self.ident.authenticate("hank", temp).must_change_password)

    def test_list_users_orders_admins_first_with_token_counts(self):
        self.ident.create_user("zed", password="zedpassword12")
        root = self.admin()
        self.ident.create_token(root.id, "a")
        self.ident.create_token(root.id, "b")
        users = self.ident.list_users()
        self.assertEqual([u["username"] for u in users], ["root", "zed"])
        self.assertEqual(users[0]["token_count"], 2)
        self.assertNotIn("password_hash", users[0])


class TestSessions(IdentityCase):
    def test_round_trip(self):
        user = self.admin()
        raw = self.ident.create_session(user.id)
        principal = self.ident.resolve_session(raw)
        self.assertEqual(principal.user.id, user.id)
        self.assertEqual(principal.via, "session")
        self.assertIsNone(self.ident.resolve_session(raw + "x"))
        self.assertIsNone(self.ident.resolve_session(""))

    def test_stored_hashed(self):
        user = self.admin()
        raw = self.ident.create_session(user.id)
        row = self.db.connection().execute("SELECT token_hash FROM sessions").fetchone()
        self.assertNotEqual(row["token_hash"], raw)
        self.assertEqual(len(row["token_hash"]), 64)

    def _age(self, column, delta):
        conn = self.db.connection()
        when = (datetime.now(timezone.utc) - delta).isoformat()
        conn.execute(f"UPDATE sessions SET {column} = ?", (when,))
        conn.commit()

    def test_expiry_and_idle(self):
        user = self.admin()
        raw = self.ident.create_session(user.id)
        self._age("last_seen_at", timedelta(hours=3))
        self.assertIsNone(self.ident.resolve_session(raw))
        raw = self.ident.create_session(user.id)
        self._age("expires_at", timedelta(seconds=1))
        self.assertIsNone(self.ident.resolve_session(raw))
        self.assertEqual(self.db.connection().execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"], 0)

    def test_touch_is_rate_limited(self):
        user = self.admin()
        raw = self.ident.create_session(user.id)
        self._age("last_seen_at", timedelta(seconds=30))
        before = self.db.connection().execute("SELECT last_seen_at FROM sessions").fetchone()["last_seen_at"]
        self.ident.resolve_session(raw)
        self.assertEqual(self.db.connection().execute("SELECT last_seen_at FROM sessions").fetchone()["last_seen_at"], before)
        self._age("last_seen_at", timedelta(seconds=90))
        self.ident.resolve_session(raw)
        self.assertNotEqual(self.db.connection().execute("SELECT last_seen_at FROM sessions").fetchone()["last_seen_at"], before)

    def test_delete_and_prune(self):
        user = self.admin()
        raw = self.ident.create_session(user.id)
        self.ident.delete_session(raw)
        self.assertIsNone(self.ident.resolve_session(raw))
        self.ident.create_session(user.id)
        self.ident.create_session(user.id)
        self._age("last_seen_at", timedelta(hours=5))
        self.assertEqual(self.ident.prune_sessions(), 2)

    def test_new_session_sweeps_dead_ones(self):
        # Browsers that never come back must not leave rows forever.
        user = self.admin()
        for _ in range(3):
            self.ident.create_session(user.id)
        self._age("last_seen_at", timedelta(hours=5))
        live = self.ident.create_session(user.id)
        rows = self.db.connection().execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"]
        self.assertEqual(rows, 1)
        self.assertIsNotNone(self.ident.resolve_session(live))


class TestTokens(IdentityCase):
    def test_create_resolve_revoke(self):
        user = self.admin()
        rec, raw = self.ident.create_token(user.id, "laptop", expires_in_days=0)
        self.assertTrue(raw.startswith("mnm_"))
        self.assertEqual(rec["hint"], raw[:10])
        self.assertIsNone(rec["expires_at"])
        principal = self.ident.resolve_token(raw)
        self.assertEqual(principal.user.username, "root")
        self.assertEqual(principal.via, "token")
        self.assertEqual(principal.token_id, rec["id"])
        self.assertEqual(principal.token_hint, rec["hint"])
        self.assertEqual(principal.token_name, "laptop")
        revoked = self.ident.revoke_token(user.id, rec["id"])
        self.assertIsNotNone(revoked["revoked_at"])
        self.assertIsNone(self.ident.resolve_token(raw))
        # Revoking twice is harmless; another user's token is not found.
        self.assertIsNotNone(self.ident.revoke_token(user.id, rec["id"]))
        other, _ = self.ident.create_user("ivy", password="ivypassword12")
        self.assertIsNone(self.ident.revoke_token(other.id, rec["id"]))

    def test_unknown_or_wrong_prefix(self):
        self.assertIsNone(self.ident.resolve_token("mnm_nope"))
        self.assertIsNone(self.ident.resolve_token("sk-nope"))
        self.assertIsNone(self.ident.resolve_token(""))

    def test_expiry(self):
        user = self.admin()
        rec, raw = self.ident.create_token(user.id, "short", expires_in_days=1)
        self.assertIsNotNone(rec["expires_at"])
        self.assertIsNotNone(self.ident.resolve_token(raw))
        conn = self.db.connection()
        conn.execute("UPDATE api_tokens SET expires_at = ? WHERE id = ?",
                     ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), rec["id"]))
        conn.commit()
        self.assertIsNone(self.ident.resolve_token(raw))

    def test_validation(self):
        user = self.admin()
        for name in ("", " ", "x" * 65):
            with self.assertRaises(IdentityError) as cm:
                self.ident.create_token(user.id, name)
            self.assertEqual(cm.exception.code, "invalid_name")
        for days in (-1, 3651, "abc"):
            with self.assertRaises(IdentityError) as cm:
                self.ident.create_token(user.id, "ok", expires_in_days=days)
            self.assertEqual(cm.exception.code, "invalid_expiry")

    def test_active_cap_counts_only_live_tokens(self):
        user = self.admin()
        ids = [self.ident.create_token(user.id, f"t{i}")[0]["id"] for i in range(identity.MAX_ACTIVE_TOKENS)]
        with self.assertRaises(IdentityError) as cm:
            self.ident.create_token(user.id, "one too many")
        self.assertEqual(cm.exception.code, "token_limit")
        self.ident.revoke_token(user.id, ids[0])
        self.ident.create_token(user.id, "fits again")

    def test_last_used_touch(self):
        user = self.admin()
        rec, raw = self.ident.create_token(user.id, "t")
        self.assertIsNone(rec["last_used_at"])
        self.ident.resolve_token(raw)
        first = self.ident.list_tokens(user.id)[0]["last_used_at"]
        self.assertIsNotNone(first)
        self.ident.resolve_token(raw)
        self.assertEqual(self.ident.list_tokens(user.id)[0]["last_used_at"], first)

    def test_list_is_per_user_newest_first(self):
        user = self.admin()
        other, _ = self.ident.create_user("jo", password="jopassword123")
        self.ident.create_token(user.id, "first")
        self.ident.create_token(user.id, "second")
        self.ident.create_token(other.id, "theirs")
        names = [t["name"] for t in self.ident.list_tokens(user.id)]
        self.assertEqual(names, ["second", "first"])


class TestLoginThrottle(unittest.TestCase):
    def test_per_account_and_ip(self):
        t = LoginThrottle()
        for _ in range(4):
            t.record_failure("alice", "10.0.0.1")
        self.assertEqual(t.retry_after("alice", "10.0.0.1"), 0)
        t.record_failure("alice", "10.0.0.1")
        self.assertGreater(t.retry_after("alice", "10.0.0.1"), 0)      # the guessing address is out
        self.assertEqual(t.retry_after("alice", "10.0.0.2"), 0)        # the owner elsewhere is not
        self.assertEqual(t.retry_after("bob", "10.0.0.1"), 0)          # ip still under its own limit

    def test_per_ip(self):
        t = LoginThrottle()
        for i in range(20):
            t.record_failure(f"user{i}", "10.0.0.9")
        self.assertGreater(t.retry_after("fresh", "10.0.0.9"), 0)      # ip locked for any name

    def test_account_wide_limit_spares_known_devices(self):
        t = LoginThrottle()
        for i in range(100):                                           # 25 addresses, 4 guesses each
            t.record_failure("alice", f"10.1.{i // 4}.1")
        self.assertGreater(t.retry_after("alice", "10.9.9.9"), 0)
        self.assertEqual(t.retry_after("alice", "10.9.9.9", known_device=True), 0)

    def test_impossible_names_share_one_bucket(self):
        t = LoginThrottle()
        for i in range(5):
            t.record_failure(f"No Such User {i}!", "10.0.0.1")
        self.assertGreater(t.retry_after("Another Junk Name", "10.0.0.1"), 0)
        self.assertEqual(t.retry_after("alice", "10.0.0.1"), 0)

    def test_success_clears(self):
        t = LoginThrottle()
        for _ in range(3):
            t.record_failure("alice", "ip")
        t.record_success("alice", "ip")
        for _ in range(4):
            t.record_failure("alice", "ip")
        self.assertEqual(t.retry_after("alice", "ip"), 0)


class TestClientKey(unittest.TestCase):
    def test_ipv4_is_per_address(self):
        self.assertEqual(client_key("10.0.0.1"), "10.0.0.1")

    def test_ipv6_groups_by_64(self):
        self.assertEqual(client_key("2001:db8:1:2:aaaa::1"), "2001:db8:1:2::/64")
        self.assertEqual(client_key("2001:DB8:1:2:ffff:ffff:ffff:ffff"), "2001:db8:1:2::/64")
        self.assertNotEqual(client_key("2001:db8:1:3::1"), client_key("2001:db8:1:2::1"))

    def test_ipv4_mapped_is_ipv4(self):
        self.assertEqual(client_key("::ffff:10.0.0.1"), "10.0.0.1")

    def test_unparsable_is_its_own_bucket(self):
        self.assertEqual(client_key("unknown"), "unknown")
        self.assertEqual(client_key("testclient"), "testclient")


class TestLoginThrottleIPv6(unittest.TestCase):
    def test_rotating_within_a_64_does_not_reset_the_limit(self):
        t = LoginThrottle()
        for i in range(5):
            t.record_failure("alice", f"2001:db8:0:1::{i + 1:x}")
        self.assertGreater(t.reserve("alice", "2001:db8:0:1::ffff"), 0)
        self.assertEqual(t.reserve("alice", "2001:db8:0:2::1"), 0)         # another /64 is not affected
        t.release("alice", "2001:db8:0:2::1")

    def test_per_ip_window_spans_the_64(self):
        t = LoginThrottle()
        for i in range(20):
            t.record_failure(f"user{i}", f"2001:db8::{i + 1:x}")
        self.assertGreater(t.retry_after("fresh", "2001:db8::abcd"), 0)


class TestThrottleReservation(unittest.TestCase):
    def test_in_flight_attempts_count_against_the_limit(self):
        t = FailureThrottle(max_failures=5, window=900.0, lockout=900.0)
        for _ in range(5):
            self.assertEqual(t.reserve("k"), 0)
        self.assertEqual(t.reserve("k"), 1)            # allowance is all in flight
        t.release("k")
        self.assertEqual(t.reserve("k"), 0)            # a finished attempt frees its slot

    def test_failures_and_in_flight_share_the_allowance(self):
        t = FailureThrottle(max_failures=5, window=900.0, lockout=900.0)
        for _ in range(4):
            t.record_failure("k")
        self.assertEqual(t.reserve("k"), 0)
        self.assertEqual(t.reserve("k"), 1)
        t.record_failure("k")
        t.release("k")
        self.assertGreater(t.reserve("k"), 1)          # now a real lockout

    def test_login_reserve_is_all_or_nothing(self):
        t = LoginThrottle()
        for i in range(19):
            t.record_failure(f"user{i}", "10.0.0.9")
        self.assertEqual(t.reserve("alice", "10.0.0.9"), 0)
        self.assertGreater(t.reserve("bob", "10.0.0.9"), 0)   # per-ip window full
        t.release("alice", "10.0.0.9")
        # bob's refused reservation left nothing behind in his other windows.
        for _ in range(5):
            self.assertEqual(t._by_account_ip.reserve("bob\x0010.0.0.9"), 0)

    def test_known_device_skips_the_account_window(self):
        t = LoginThrottle()
        for i in range(100):
            t.record_failure("alice", f"10.1.{i // 4}.1")
        self.assertGreater(t.reserve("alice", "10.9.9.9"), 0)
        self.assertEqual(t.reserve("alice", "10.9.9.9", known_device=True), 0)
        t.release("alice", "10.9.9.9", known_device=True)


class TestKnownDevice(IdentityCase):
    def setUp(self):
        super().setUp()
        self.root = self.admin()
        self.alice = self.ident.create_user("alice", password="alicepassword1")[0]
        self.ident.create_user("bob", password="bobpassword123")

    def test_proof_round_trip(self):
        proof = self.ident.issue_device_proof("alice")
        self.assertTrue(self.ident.is_known_device("alice", proof))
        self.assertFalse(self.ident.is_known_device("bob", proof))           # bound to the account
        flipped = proof[:-1] + ("1" if proof[-1] == "0" else "0")
        self.assertFalse(self.ident.is_known_device("alice", flipped))
        self.assertFalse(self.ident.is_known_device("alice", None))
        self.assertFalse(self.ident.is_known_device("alice", "garbage"))
        self.assertFalse(self.ident.is_known_device("nobody", proof))

    def test_proof_expires(self):
        issued = int(time.time()) - int(DEVICE_COOKIE_TTL.total_seconds()) - 60
        stale = f"{issued}.{self.ident._device_mac(self.ident._device_binding('alice'), issued)}"
        self.assertFalse(self.ident.is_known_device("alice", stale))

    def test_key_survives_a_new_identity_object(self):
        proof = self.ident.issue_device_proof("alice")
        self.assertTrue(Identity(self.db).is_known_device("alice", proof))

    def test_password_change_voids_proof(self):
        proof = self.ident.issue_device_proof("alice")
        self.ident.change_password(self.alice.id, "alicepassword1", "a brand new password")
        self.assertFalse(self.ident.is_known_device("alice", proof))
        self.assertTrue(self.ident.is_known_device("alice", self.ident.issue_device_proof("alice")))

    def test_reset_voids_proof(self):
        proof = self.ident.issue_device_proof("alice")
        self.ident.reset_password(self.alice.id, acting_user_id=self.root.id)
        self.assertFalse(self.ident.is_known_device("alice", proof))

    def test_deactivation_voids_proof_for_good(self):
        proof = self.ident.issue_device_proof("alice")
        self.ident.set_active(self.alice.id, False, acting_user_id=self.root.id)
        self.assertFalse(self.ident.is_known_device("alice", proof))
        self.ident.set_active(self.alice.id, True, acting_user_id=self.root.id)
        self.assertFalse(self.ident.is_known_device("alice", proof))

    def test_recreated_account_does_not_inherit_proof(self):
        proof = self.ident.issue_device_proof("alice")
        self.ident.delete_user(self.alice.id, acting_user_id=self.root.id)
        self.ident.create_user("alice", password="alicepassword1")
        self.assertFalse(self.ident.is_known_device("alice", proof))


class TestFirstRun(unittest.TestCase):
    def test_non_ascii_candidate_is_just_wrong(self):
        fr = FirstRun()
        fr.issue()
        self.assertFalse(fr.check("ÅÅÅÅ-ÅÅÅÅ-ÅÅÅÅ"))

    def test_issue_check_clear(self):
        fr = FirstRun()
        self.assertFalse(fr.check("ABCD-EFGH-JKMN"))
        code = fr.issue()
        self.assertTrue(fr.check(code))
        self.assertTrue(fr.check(code.lower()))
        self.assertTrue(fr.check(f" {code} "))
        self.assertFalse(fr.check(code[:-1] + ("Y" if code[-1] == "Z" else "Z")))   # always a different code
        fr.clear()
        self.assertFalse(fr.check(code))


class TestBootstrap(IdentityCase):
    def test_env_password_creates_admin_and_audits(self):
        fr = FirstRun()
        ensure_bootstrap(self.ident, fr, "env-admin-password")
        self.assertIsNone(fr.code)
        user = self.ident.authenticate("admin", "env-admin-password")
        self.assertTrue(user.is_admin)
        self.assertFalse(user.must_change_password)
        event = self.db.list_audit(op="admin.created")[0]
        self.assertEqual((event["actor"], event["item_type"], event["item_id"]), ("system", "user", "admin"))
        self.assertEqual(event["detail"], {"source": "env"})

    def test_no_password_prints_setup_code(self):
        fr = FirstRun()
        out = io.StringIO()
        with redirect_stdout(out):
            ensure_bootstrap(self.ident, fr, None)
        self.assertIsNotNone(fr.code)
        self.assertIn(fr.code, out.getvalue())
        self.assertEqual(self.ident.count_users(), 0)

    def test_existing_users_short_circuit(self):
        self.admin()
        fr = FirstRun()
        with self.assertLogs("mnemomatic", level="INFO") as logs:
            ensure_bootstrap(self.ident, fr, "ignored-password")
        self.assertIsNone(fr.code)
        self.assertEqual(self.ident.count_users(), 1)
        self.assertTrue(any("ignoring" in line for line in logs.output))

    def test_weak_env_password_fails_loudly(self):
        with self.assertRaises(IdentityError):
            ensure_bootstrap(self.ident, FirstRun(), "short")


if __name__ == "__main__":
    unittest.main()

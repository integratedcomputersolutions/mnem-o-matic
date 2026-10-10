"""Users, browser sessions, and per-user API tokens.

The server has two kinds of credential and this module owns both:

- A **session** is what a browser holds after logging in with a password:
  a random 256-bit value in an HttpOnly cookie, stored here only as its
  SHA-256 hash, expiring after a day or two idle hours.
- An **API token** is what an agent holds: ``mnm_`` plus 32 random bytes,
  shown once at creation, stored only as its SHA-256 hash, revocable one at a
  time. Every MCP request is attributed to the token's owner, which is what
  makes the audit log's ``actor`` an authenticated name rather than a header
  the client chose for itself.

A third kind of caller is a **trusted proxy** (see ``auth.py``): a gateway
that signs people in itself and, holding a secret only it knows, names the
user on each request. Such users are created here on first sight, keyed by
the proxy's identity for them (``external_id``), and have no password of
their own.

Passwords are hashed with scrypt from the standard library — no extra
dependency — in a self-describing PHC-style string, so the parameters can be
raised later and old hashes upgraded on the next successful login.

Nothing here writes to the audit log except the bootstrap path and a proxy
user's creation (both happen before any request identity exists); the HTTP
layer records login, token, and user-management events with the request's
identity attached. Keeping that out of here keeps these methods testable
without a request in flight.
"""

import base64
import hashlib
import hmac
import logging
import re
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from mnemomatic.audit import write_event
from mnemomatic.throttle import FailureThrottle, client_key

logger = logging.getLogger("mnemomatic")

# ── Policy constants ────────────────────────────────────────────────────────

USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,31}$")
USERNAME_MAX = 32
MIN_PASSWORD_LEN = 10
MAX_PASSWORD_LEN = 1024
ROLES = ("admin", "user")

# What sits in password_hash for a user a trusted proxy introduced: not a
# parseable hash, so it verifies as nothing, and recognisably "no password"
# rather than a hash of something. An admin's password reset replaces it.
NO_PASSWORD_HASH = "!"
EXTERNAL_ID_MAX = 254          # an email address at most, per RFC 5321

TOKEN_PREFIX = "mnm_"
TOKEN_HINT_LEN = len(TOKEN_PREFIX) + 6     # enough to tell tokens apart in a list, useless to guess from
MAX_ACTIVE_TOKENS = 25
MAX_TOKEN_DAYS = 3650

SESSION_TTL = timedelta(hours=24)
SESSION_IDLE = timedelta(hours=2)
TOUCH_INTERVAL = timedelta(seconds=60)      # how often last_seen / last_used are rewritten
TEMP_PASSWORD_TTL = timedelta(days=7)
DEVICE_COOKIE_TTL = timedelta(days=90)
DEVICE_KEY_SETTING = "device_cookie_key"

# No 0/O/1/I/L — these get read aloud and typed from a log line.
SETUP_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
TEMP_PASSWORD_ALPHABET = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKMNPQRSTUVWXYZ23456789"

# scrypt: 2^15 blocks of r=8 is 32 MiB of memory per hash, which is the
# OWASP-recommended tier and roughly 100 ms on current hardware. maxmem must
# be stated explicitly — 128·n·r lands exactly on OpenSSL's default ceiling.
SCRYPT_LOG_N = 15
SCRYPT_R = 8
SCRYPT_P = 3
SCRYPT_DKLEN = 32
SCRYPT_MAXMEM = 64 * 1024 * 1024
_SALT_BYTES = 16


# ── Password hashing ────────────────────────────────────────────────────────

def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str, *, log_n: int | None = None, r: int | None = None,
                  p: int | None = None) -> str:
    """scrypt hash in the form ``$scrypt$ln=15,r=8,p=3$<salt>$<key>``.

    The parameters travel with the hash, so verification never depends on the
    current constants and a later increase only affects new hashes (plus any
    old one re-hashed after a successful login — see needs_rehash). Defaults
    are read at call time so tests can lower the work factor by patching the
    module constants.
    """
    log_n = SCRYPT_LOG_N if log_n is None else log_n
    r = SCRYPT_R if r is None else r
    p = SCRYPT_P if p is None else p
    salt = secrets.token_bytes(_SALT_BYTES)
    key = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=1 << log_n, r=r, p=p,
                         dklen=SCRYPT_DKLEN, maxmem=SCRYPT_MAXMEM)
    return f"$scrypt$ln={log_n},r={r},p={p}${_b64(salt)}${_b64(key)}"


def _parse_hash(stored: str) -> tuple[dict[str, int], bytes, bytes] | None:
    try:
        _, scheme, params, salt, key = stored.split("$")
        if scheme != "scrypt":
            return None
        parsed = {k: int(v) for k, v in (kv.split("=") for kv in params.split(","))}
        return parsed, _unb64(salt), _unb64(key)
    except (ValueError, TypeError):
        return None


def verify_password(stored: str, candidate: str) -> bool:
    """Constant-time check of `candidate` against a stored hash. Malformed
    hashes verify as False rather than raising — a bad row must not become a
    way to crash the login handler."""
    parsed = _parse_hash(stored)
    if parsed is None:
        return False
    params, salt, key = parsed
    try:
        computed = hashlib.scrypt(candidate.encode("utf-8"), salt=salt, n=1 << params["ln"],
                                  r=params["r"], p=params["p"], dklen=len(key),
                                  maxmem=SCRYPT_MAXMEM)
    except (ValueError, KeyError):
        return False
    return hmac.compare_digest(computed, key)


def needs_rehash(stored: str) -> bool:
    """True when the stored hash uses weaker parameters than the current ones."""
    parsed = _parse_hash(stored)
    if parsed is None:
        return True
    params = parsed[0]
    return (params.get("ln"), params.get("r"), params.get("p")) != (SCRYPT_LOG_N, SCRYPT_R, SCRYPT_P)


# Compared against when the username is unknown or inactive, so a login
# attempt costs the same whether or not the account exists.
DUMMY_HASH = hash_password(secrets.token_hex(16))


# ── Small generators ────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse(ts: str | None) -> datetime | None:
    return datetime.fromisoformat(ts) if ts else None


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def generate_token() -> str:
    """A fresh API token: the prefix plus 32 random bytes, URL-safe."""
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


def generate_temp_password(length: int = 16) -> str:
    return "".join(secrets.choice(TEMP_PASSWORD_ALPHABET) for _ in range(length))


def generate_setup_code() -> str:
    """``XXXX-XXXX-XXXX`` from an alphabet without look-alike characters."""
    chars = [secrets.choice(SETUP_CODE_ALPHABET) for _ in range(12)]
    return "-".join("".join(chars[i:i + 4]) for i in (0, 4, 8))


def normalize_username(name: str) -> str:
    return name.strip().lower()


def validate_username(name: str) -> str:
    """The normalized username, or raise IdentityError with the rule."""
    name = normalize_username(name)
    if not USERNAME_RE.match(name):
        raise IdentityError(
            "invalid_username", 400,
            "Usernames are 2–32 characters: lowercase letters, digits, '.', '_' or '-', "
            "starting with a letter or digit.",
        )
    return name


def normalize_external_id(value: str) -> str:
    """A proxy's identity for a user as stored: stripped and lower-cased,
    since it is usually an email address and case is not identity there.
    Raises for an empty value, one with whitespace or control characters,
    or one longer than an address can be."""
    value = (value or "").strip().lower()
    if not value or len(value) > EXTERNAL_ID_MAX or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value):
        raise IdentityError("invalid_external_id", 400,
                            f"The user identity must be 1–{EXTERNAL_ID_MAX} characters with no whitespace.")
    return value


def username_from_external_id(external_id: str) -> str:
    """A username in the local format for a proxy identity: what precedes any
    '@', with each run of characters outside [a-z0-9._-] replaced by '-' and
    leading punctuation dropped, trimmed to fit. 'user' if nothing usable
    is left. Uniqueness is the caller's problem."""
    local = external_id.split("@", 1)[0].lower()
    base = re.sub(r"[^a-z0-9._-]+", "-", local).lstrip("._-")
    if len(base) < 2:
        base = "user"
    return base[:USERNAME_MAX]


def validate_password(password: str) -> None:
    if not isinstance(password, str) or len(password) < MIN_PASSWORD_LEN:
        raise IdentityError("weak_password", 400,
                            f"Passwords must be at least {MIN_PASSWORD_LEN} characters.")
    if len(password) > MAX_PASSWORD_LEN:
        raise IdentityError("weak_password", 400,
                            f"Passwords must be at most {MAX_PASSWORD_LEN} characters.")


# ── Errors and records ──────────────────────────────────────────────────────

class IdentityError(Exception):
    """A refused identity operation, carrying the API's error code and status."""

    def __init__(self, code: str, status: int, details: str):
        super().__init__(details)
        self.code = code
        self.status = status
        self.details = details


@dataclass(frozen=True)
class User:
    id: int
    username: str
    display_name: str
    role: str
    active: bool
    must_change_password: bool
    temp_password_expires_at: str | None
    created_at: str
    external_id: str | None = None      # a trusted proxy's name for this user

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def public(self) -> dict:
        """What the API shows about a user — never the hash."""
        return {
            "id": self.id,
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role,
            "active": self.active,
            "must_change_password": self.must_change_password,
            "temp_password_expires_at": self.temp_password_expires_at,
            "created_at": self.created_at,
            "external_id": self.external_id,
        }


def _row_to_user(row: dict) -> User:
    return User(
        id=row["id"], username=row["username"], display_name=row["display_name"],
        role=row["role"], active=bool(row["active"]),
        must_change_password=bool(row["must_change_password"]),
        temp_password_expires_at=row["temp_password_expires_at"], created_at=row["created_at"],
        external_id=row["external_id"],
    )


@dataclass(frozen=True)
class Principal:
    """Who a request is acting as, and through which credential."""
    user: User
    via: str                     # "session", "token" or "proxy"
    token_id: int | None = None
    token_hint: str | None = None
    token_name: str | None = None

    @property
    def actor(self) -> str:
        """The name the audit log records for this user: the proxy's identity
        for a user it introduced (so history lines up across the proxy's own
        logs), the username otherwise. Independent of `via`, so one person
        is one actor however they arrived."""
        return self.user.external_id or self.user.username


# ── Login throttle ──────────────────────────────────────────────────────────

class LoginThrottle:
    """Three sliding windows over password attempts, fifteen minutes each:

    - per account *and* address (5): the everyday guard. Locks out only the
      address doing the guessing, so a stranger cannot lock the real owner
      out of their own account.
    - per address (20): one address cannot spray many names.
    - per account (100): many addresses cannot pool their guesses against
      one name. Browsers holding this account's known-device cookie (set on
      a successful sign-in) skip this one, so a distributed attack cannot
      lock the owner out either.

    An "address" is an IPv4 address or an IPv6 /64 (see throttle.client_key),
    so rotating through one IPv6 allocation does not buy fresh allowances.

    Usernames that cannot exist share a single bucket, so junk names neither
    grow the tables nor get a fresh allowance each.
    """

    def __init__(self):
        self._by_account_ip = FailureThrottle(max_failures=5, window=900.0, lockout=900.0)
        self._by_ip = FailureThrottle(max_failures=20, window=900.0, lockout=900.0)
        self._by_account = FailureThrottle(max_failures=100, window=900.0, lockout=900.0)

    @staticmethod
    def _account(username: str) -> str:
        return username if USERNAME_RE.match(username) else "\0invalid"

    def _buckets(self, username: str, ip: str, known_device: bool):
        account, ip = self._account(username), client_key(ip)
        buckets = [(self._by_account_ip, f"{account}\0{ip}"), (self._by_ip, ip)]
        if not known_device:
            buckets.append((self._by_account, account))
        return buckets

    def retry_after(self, username: str, ip: str, *, known_device: bool = False) -> int:
        return max(t.retry_after(key) for t, key in self._buckets(username, ip, known_device))

    def reserve(self, username: str, ip: str, *, known_device: bool = False) -> int:
        """Claim a sign-in attempt in every window; see FailureThrottle.reserve.

        Returns 0 when admitted, and the caller must `release()` with the same
        arguments afterwards. Otherwise returns the seconds to wait.
        """
        taken = []
        for t, key in self._buckets(username, ip, known_device):
            wait = t.reserve(key)
            if wait:
                for held, held_key in taken:
                    held.release(held_key)
                return max(wait, self.retry_after(username, ip, known_device=known_device))
            taken.append((t, key))
        return 0

    def release(self, username: str, ip: str, *, known_device: bool = False) -> None:
        for t, key in self._buckets(username, ip, known_device):
            t.release(key)

    def record_failure(self, username: str, ip: str) -> None:
        account, ip = self._account(username), client_key(ip)
        self._by_account_ip.record_failure(f"{account}\0{ip}")
        self._by_ip.record_failure(ip)
        self._by_account.record_failure(account)

    def record_success(self, username: str, ip: str) -> None:
        account, ip = self._account(username), client_key(ip)
        self._by_account_ip.record_success(f"{account}\0{ip}")
        self._by_ip.record_success(ip)
        self._by_account.record_success(account)


# ── First run ───────────────────────────────────────────────────────────────

class FirstRun:
    """Holds the one-time setup code printed at startup when no user exists.

    Lives in memory only: a restart prints a new one, and the code stops
    working the moment the first admin is created, whichever way that happens.
    """

    def __init__(self):
        self.code: str | None = None

    def issue(self) -> str:
        self.code = generate_setup_code()
        return self.code

    def check(self, candidate: str) -> bool:
        if not self.code:
            return False
        normalized = candidate.strip().upper().replace(" ", "")
        # Bytes: compare_digest refuses non-ASCII str, which would be a 500
        # (and an attempt the first-run throttle never counts).
        return hmac.compare_digest(normalized.encode(), self.code.encode())

    def clear(self) -> None:
        self.code = None


# ── The store ───────────────────────────────────────────────────────────────

class Identity:
    """Users, sessions, and tokens on top of the Database's connection.

    Every method is synchronous and uses this thread's connection, exactly
    like the content methods on Database. Raises IdentityError for anything
    the API should refuse with a specific code; returns None for lookups that
    simply find nothing.
    """

    def __init__(self, db):
        self._db = db

    def _conn(self) -> sqlite3.Connection:
        return self._db.connection()

    # ── users ──

    def count_users(self) -> int:
        return self._conn().execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]

    def get_user(self, user_id: int) -> User | None:
        row = self._conn().execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _row_to_user(row) if row else None

    def get_user_by_name(self, username: str) -> User | None:
        row = self._conn().execute(
            "SELECT * FROM users WHERE username = ?", (normalize_username(username),)
        ).fetchone()
        return _row_to_user(row) if row else None

    def list_users(self) -> list[dict]:
        """Every user, admins first, each with its count of live tokens."""
        rows = self._conn().execute("""
            SELECT u.*, (
                SELECT COUNT(*) FROM api_tokens t
                WHERE t.user_id = u.id AND t.revoked_at IS NULL
                  AND (t.expires_at IS NULL OR t.expires_at > ?)
            ) AS token_count
            FROM users u
            ORDER BY (u.role = 'admin') DESC, u.username
        """, (_iso(_now()),)).fetchall()
        return [{**_row_to_user(r).public(), "token_count": r["token_count"]} for r in rows]

    def create_user(self, username: str, *, role: str = "user", display_name: str = "",
                    password: str | None = None, only_if_no_users: bool = False) -> tuple[User, str | None]:
        """Create a user. With a password given, it is set outright; without
        one, a temporary password is generated and returned, and the user must
        choose a new one within TEMP_PASSWORD_TTL on their first login.

        only_if_no_users makes this the first user or nothing: the table is
        checked empty inside the same write transaction as the insert, so of
        several concurrent first-run requests exactly one gets through and
        the rest get already_set_up (409).

        Returns (user, temporary_password_or_None).
        """
        username = validate_username(username)
        if role not in ROLES:
            raise IdentityError("invalid_role", 400, f"Role must be one of: {', '.join(ROLES)}.")
        temp = None
        if password is None:
            temp = generate_temp_password()
            secret, must_change, temp_expires = temp, 1, _iso(_now() + TEMP_PASSWORD_TTL)
        else:
            validate_password(password)
            secret, must_change, temp_expires = password, 0, None
        password_hash = hash_password(secret)          # ~100 ms: not while holding the write lock
        try:
            with self._db.write() as conn:
                if only_if_no_users and conn.execute("SELECT 1 FROM users LIMIT 1").fetchone():
                    raise IdentityError("already_set_up", 409, "A user already exists; sign in instead.")
                cur = conn.execute(
                    "INSERT INTO users (username, display_name, role, password_hash, "
                    "must_change_password, temp_password_expires_at, active, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
                    (username, display_name.strip()[:128], role, password_hash,
                     must_change, temp_expires, _iso(_now())),
                )
        except sqlite3.IntegrityError:
            raise IdentityError("user_exists", 409, f"A user named {username!r} already exists.")
        return self.get_user(cur.lastrowid), temp

    def _require_user(self, user_id: int) -> User:
        user = self.get_user(user_id)
        if user is None:
            raise IdentityError("not_found", 404, "No such user.")
        return user

    def _guard_last_admin(self, target: User) -> None:
        """Refuse a change that would leave no active admin able to log in."""
        if not (target.is_admin and target.active):
            return
        others = self._conn().execute(
            "SELECT COUNT(*) AS n FROM users WHERE role = 'admin' AND active = 1 AND id != ?",
            (target.id,),
        ).fetchone()["n"]
        if others == 0:
            raise IdentityError("last_admin", 409,
                                "This is the only active administrator; add another first.")

    def _target(self, user_id: int, acting_user_id: int, *, self_message: str,
                may_remove_admin: bool) -> User:
        """The user an admin operation acts on, once the two rules every such
        operation shares have been checked: never on yourself, and never in a
        way that leaves no active administrator."""
        target = self._require_user(user_id)
        if target.id == acting_user_id:
            raise IdentityError("self_action", 403, self_message)
        if not may_remove_admin:
            self._guard_last_admin(target)
        return target

    def set_active(self, user_id: int, active: bool, *, acting_user_id: int) -> tuple[User, int]:
        """Enable or disable an account. Returns (user, tokens revoked).

        Disabling ends every session and revokes every live token for good:
        an account is usually disabled because something went wrong, and
        tokens that came back on reactivation would hand an attacker their
        foothold back. The person mints new tokens after reactivation."""
        # The last-admin guard reads inside the transaction, so two admins
        # disabling each other at once cannot both pass it.
        with self._db.write() as conn:
            self._target(user_id, acting_user_id, self_message="You cannot change your own account's status.",
                         may_remove_admin=active)
            conn.execute("UPDATE users SET active = ? WHERE id = ?", (1 if active else 0, user_id))
            revoked = 0
            if not active:
                conn.execute("UPDATE users SET credential_version = credential_version + 1 WHERE id = ?",
                             (user_id,))
                conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
                revoked = conn.execute(
                    "UPDATE api_tokens SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL",
                    (_iso(_now()), user_id),
                ).rowcount
        return self.get_user(user_id), revoked

    def set_role(self, user_id: int, role: str, *, acting_user_id: int) -> User:
        if role not in ROLES:
            raise IdentityError("invalid_role", 400, f"Role must be one of: {', '.join(ROLES)}.")
        with self._db.write() as conn:
            self._target(user_id, acting_user_id, self_message="You cannot change your own role.",
                         may_remove_admin=(role == "admin"))
            conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
        return self.get_user(user_id)

    def delete_user(self, user_id: int, *, acting_user_id: int) -> User:
        with self._db.write() as conn:
            target = self._target(user_id, acting_user_id, self_message="You cannot delete your own account.",
                                  may_remove_admin=False)
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))   # cascades to sessions/tokens
        return target

    def reset_password(self, user_id: int, *, acting_user_id: int) -> tuple[str, str]:
        """Issue a temporary password for `user_id` and end their sessions.
        Returns (temporary_password, expires_at). Tokens keep working: resetting
        a forgotten password should not silently break the person's agents."""
        self._target(user_id, acting_user_id,
                     self_message="Change your own password from the account page instead.",
                     may_remove_admin=True)       # a reset leaves the account in place
        temp = generate_temp_password()
        expires = _iso(_now() + TEMP_PASSWORD_TTL)
        password_hash = hash_password(temp)
        with self._db.write() as conn:
            conn.execute(
                "UPDATE users SET password_hash = ?, must_change_password = 1, "
                "temp_password_expires_at = ?, credential_version = credential_version + 1 WHERE id = ?",
                (password_hash, expires, user_id),
            )
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        return temp, expires

    def change_password(self, user_id: int, current: str, new: str, *,
                        keep_session: str | None = None) -> None:
        """Set a new password after verifying the current one. Every other
        session of the user is ended; the one presenting `keep_session` (the
        raw cookie value) survives so the browser doing the change stays in."""
        user = self._require_user(user_id)
        row = self._conn().execute("SELECT password_hash FROM users WHERE id = ?", (user_id,)).fetchone()
        if not verify_password(row["password_hash"], current):
            raise IdentityError("wrong_password", 401, "The current password is wrong.")
        validate_password(new)
        if new == current:
            raise IdentityError("weak_password", 400, "The new password must differ from the current one.")
        password_hash = hash_password(new)
        with self._db.write() as conn:
            conn.execute(
                "UPDATE users SET password_hash = ?, must_change_password = 0, "
                "temp_password_expires_at = NULL, credential_version = credential_version + 1 WHERE id = ?",
                (password_hash, user_id),
            )
            if keep_session:
                conn.execute("DELETE FROM sessions WHERE user_id = ? AND token_hash != ?",
                             (user_id, _sha256(keep_session)))
            else:
                conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        logger.info("Password changed for %s", user.username)

    def authenticate(self, username: str, password: str) -> User:
        """Check a password. Exactly one scrypt verification runs whether or
        not the user exists, so timing does not reveal valid usernames.

        Raises IdentityError: invalid_credentials (401), account_disabled
        (403), temp_password_expired (403).
        """
        username = normalize_username(username)
        row = self._conn().execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        # A proxy-introduced user has no password: verify against the dummy
        # so refusing them costs the same as refusing an unknown name.
        stored = row["password_hash"] if row and row["password_hash"] != NO_PASSWORD_HASH else DUMMY_HASH
        ok = verify_password(stored, password)
        if row is None or not ok:
            raise IdentityError("invalid_credentials", 401, "Wrong username or password.")
        user = _row_to_user(row)
        if not user.active:
            raise IdentityError("account_disabled", 403, "This account is disabled.")
        if user.must_change_password and user.temp_password_expires_at:
            if _parse(user.temp_password_expires_at) < _now():
                raise IdentityError("temp_password_expired", 403,
                                    "The temporary password has expired; ask an admin for a new one.")
        if needs_rehash(stored):
            password_hash = hash_password(password)
            with self._db.write() as conn:
                conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user.id))
        return user

    # ── known devices ──
    # A browser that signed in to an account gets a cookie proving it, which
    # exempts it from the account-wide login throttle (see LoginThrottle). It
    # is an HMAC over (user id, username, credential version, issue time)
    # under a key kept in the settings table, so it survives restarts, cannot
    # be forged, and needs no table. The credential version moves on every
    # password change, reset and deactivation, so a browser that once held
    # the old password loses the exemption with it.

    def _device_key(self) -> bytes:
        stored = self._db.get_setting(DEVICE_KEY_SETTING)
        if stored is None:
            # First use. INSERT OR IGNORE so two racing first sign-ins agree
            # on one key; re-read to get whichever won.
            with self._db.write() as conn:
                conn.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                             (DEVICE_KEY_SETTING, secrets.token_hex(32)))
            stored = self._db.get_setting(DEVICE_KEY_SETTING)
        return bytes.fromhex(stored)

    def _device_binding(self, username: str) -> str | None:
        """What a proof for `username` is bound to, or None when no active
        account has that name."""
        row = self._conn().execute(
            "SELECT id, username, credential_version FROM users WHERE username = ? AND active = 1",
            (username,),
        ).fetchone()
        return f"{row['id']}|{row['username']}|{row['credential_version']}" if row else None

    def _device_mac(self, binding: str, issued: int) -> str:
        return hmac.new(self._device_key(), f"{binding}|{issued}".encode(), hashlib.sha256).hexdigest()

    def issue_device_proof(self, username: str) -> str:
        binding = self._device_binding(username)
        if binding is None:
            raise IdentityError("not_found", 404, "No such active user.")
        issued = int(_now().timestamp())
        return f"{issued}.{self._device_mac(binding, issued)}"

    def is_known_device(self, username: str, proof: str | None) -> bool:
        if not proof or not USERNAME_RE.match(username):
            return False
        issued_raw, _, mac = proof.partition(".")
        try:
            issued = int(issued_raw)
        except ValueError:
            return False
        if _now().timestamp() - issued > DEVICE_COOKIE_TTL.total_seconds():
            return False
        binding = self._device_binding(username)
        if binding is None:
            return False
        return hmac.compare_digest(mac.encode(), self._device_mac(binding, issued).encode())

    # ── sessions ──

    def create_session(self, user_id: int) -> str:
        """Start a session; returns the raw cookie value (stored only hashed).

        Dead sessions go first. resolve_session only deletes one when its
        cookie comes back, so browsers that never return would otherwise
        leave their rows behind forever; a sign-in is a natural moment to
        sweep, and the table then never holds more than the live sessions."""
        self.prune_sessions()
        raw = secrets.token_urlsafe(32)
        now = _now()
        with self._db.write() as conn:
            conn.execute(
                "INSERT INTO sessions (token_hash, user_id, created_at, last_seen_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (_sha256(raw), user_id, _iso(now), _iso(now), _iso(now + SESSION_TTL)),
            )
        return raw

    def resolve_session(self, raw: str) -> Principal | None:
        """The principal behind a cookie value, or None if it is unknown,
        expired, idle too long, or belongs to a disabled user. Dead sessions
        are deleted as they are discovered."""
        if not raw:
            return None
        conn = self._conn()
        row = conn.execute(
            "SELECT s.id AS sid, s.last_seen_at, s.expires_at, u.* FROM sessions s "
            "JOIN users u ON u.id = s.user_id WHERE s.token_hash = ?",
            (_sha256(raw),),
        ).fetchone()
        if row is None:
            return None
        now = _now()
        if (_parse(row["expires_at"]) < now
                or now - _parse(row["last_seen_at"]) > SESSION_IDLE
                or not row["active"]):
            with self._db.write() as w:
                w.execute("DELETE FROM sessions WHERE id = ?", (row["sid"],))
            return None
        if now - _parse(row["last_seen_at"]) > TOUCH_INTERVAL:
            with self._db.write() as w:
                w.execute("UPDATE sessions SET last_seen_at = ? WHERE id = ?", (_iso(now), row["sid"]))
        return Principal(user=_row_to_user(row), via="session")

    def delete_session(self, raw: str) -> None:
        if not raw:
            return
        with self._db.write() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_sha256(raw),))

    def end_all_sessions(self) -> int:
        """Sign every browser out. Returns how many sessions there were."""
        with self._db.write() as conn:
            return conn.execute("DELETE FROM sessions").rowcount

    def prune_sessions(self) -> int:
        """Drop sessions past their lifetime or idle limit. Returns how many."""
        now = _now()
        with self._db.write() as conn:
            return conn.execute(
                "DELETE FROM sessions WHERE expires_at < ? OR last_seen_at < ?",
                (_iso(now), _iso(now - SESSION_IDLE)),
            ).rowcount

    # ── API tokens ──

    @staticmethod
    def _token_public(row: dict) -> dict:
        return {k: row[k] for k in ("id", "name", "hint", "created_at", "expires_at",
                                    "last_used_at", "revoked_at")}

    def list_tokens(self, user_id: int) -> list[dict]:
        rows = self._conn().execute(
            "SELECT * FROM api_tokens WHERE user_id = ? ORDER BY id DESC", (user_id,)
        ).fetchall()
        return [self._token_public(r) for r in rows]

    def create_token(self, user_id: int, name: str, expires_in_days: int = 0) -> tuple[dict, str]:
        """Mint a token for `user_id`. Returns (public record, raw token); the
        raw value is never recoverable afterwards."""
        name = (name or "").strip()
        if not 1 <= len(name) <= 64:
            raise IdentityError("invalid_name", 400, "Token names are 1–64 characters.")
        try:
            days = int(expires_in_days or 0)
        except (TypeError, ValueError):
            raise IdentityError("invalid_expiry", 400, "expires_in_days must be a whole number.")
        if not 0 <= days <= MAX_TOKEN_DAYS:
            raise IdentityError("invalid_expiry", 400,
                                f"expires_in_days must be between 0 (never) and {MAX_TOKEN_DAYS}.")
        now = _now()
        # The limit check and the insert share one transaction, so a burst of
        # requests cannot all see room for one more token.
        with self._db.write() as conn:
            live = conn.execute(
                "SELECT COUNT(*) AS n FROM api_tokens WHERE user_id = ? AND revoked_at IS NULL "
                "AND (expires_at IS NULL OR expires_at > ?)",
                (user_id, _iso(now)),
            ).fetchone()["n"]
            if live >= MAX_ACTIVE_TOKENS:
                raise IdentityError("token_limit", 409,
                                    f"At most {MAX_ACTIVE_TOKENS} active tokens per user; revoke one first.")
            raw = generate_token()
            expires = _iso(now + timedelta(days=days)) if days else None
            cur = conn.execute(
                "INSERT INTO api_tokens (user_id, name, token_hash, hint, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, name, _sha256(raw), raw[:TOKEN_HINT_LEN], _iso(now), expires),
            )
            row = conn.execute("SELECT * FROM api_tokens WHERE id = ?", (cur.lastrowid,)).fetchone()
            return self._token_public(row), raw

    def revoke_token(self, user_id: int, token_id: int) -> dict | None:
        """Revoke one of `user_id`'s tokens; None when it is not theirs."""
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM api_tokens WHERE id = ? AND user_id = ?", (token_id, user_id)
        ).fetchone()
        if row is None:
            return None
        if row["revoked_at"] is None:
            with self._db.write() as w:
                w.execute("UPDATE api_tokens SET revoked_at = ? WHERE id = ?", (_iso(_now()), token_id))
            row = conn.execute("SELECT * FROM api_tokens WHERE id = ?", (token_id,)).fetchone()
        return self._token_public(row)

    # ── trusted proxy ──

    def resolve_proxy_user(self, external_id: str, *, display_name: str | None = None,
                           role: str | None = None) -> Principal:
        """The principal a trusted proxy vouches for, created on first sight.

        Called only after AuthMiddleware has checked the proxy's secret, so
        the identity is believed. A new user gets a username derived from the
        identity (suffixed until unique), the role the proxy named or `user`,
        the display name it sent or the identity itself, and no password:
        nobody can sign in to the account until an admin issues a temporary
        one. A returning user is matched on the identity alone; a role or
        display name the proxy sends updates the row, so the proxy stays the
        source of truth for whatever it chooses to say. Raises
        account_disabled (403) for a user an admin has deactivated here —
        the proxy does not get to override that.
        """
        external_id = normalize_external_id(external_id)
        if role is not None and role not in ROLES:
            raise IdentityError("invalid_role", 400, f"Role must be one of: {', '.join(ROLES)}.")
        display = (display_name or "").strip()[:128]
        row = self._conn().execute("SELECT * FROM users WHERE external_id = ?", (external_id,)).fetchone()
        if row is None:
            row = self._create_proxy_user(external_id, display or external_id, role or "user")
        user = _row_to_user(row)
        if not user.active:
            raise IdentityError("account_disabled", 403, "This account is disabled.")
        updates: dict[str, str] = {}
        if role is not None and role != user.role:
            try:
                self._guard_last_admin(user)
                updates["role"] = role
            except IdentityError:
                logger.warning("Proxy asked to demote %r, the only active administrator; keeping admin",
                               user.username)
        if display and display != user.display_name:
            updates["display_name"] = display
        if updates:
            with self._db.write() as conn:
                conn.execute(f"UPDATE users SET {', '.join(f'{k} = ?' for k in updates)} WHERE id = ?",
                             (*updates.values(), user.id))
            user = self.get_user(user.id)
        return Principal(user=user, via="proxy")

    def _create_proxy_user(self, external_id: str, display_name: str, role: str) -> dict:
        """Insert the row for a proxy identity seen for the first time and
        return it. Two first requests for one person can race; the unique
        index makes one of them lose, and the loser reads the winner's row."""
        base = username_from_external_id(external_id)
        try:
            with self._db.write() as conn:
                taken = {r["username"].lower() for r in conn.execute(
                    "SELECT username FROM users WHERE username = ? OR username LIKE ?", (base, base + "-%"))}
                username, n = base, 2
                while username in taken:
                    suffix = f"-{n}"
                    username, n = base[:USERNAME_MAX - len(suffix)] + suffix, n + 1
                cur = conn.execute(
                    "INSERT INTO users (username, display_name, role, password_hash, must_change_password, "
                    "temp_password_expires_at, active, created_at, external_id) VALUES (?, ?, ?, ?, 0, NULL, 1, ?, ?)",
                    (username, display_name, role, NO_PASSWORD_HASH, _iso(_now()), external_id),
                )
                row_id = cur.lastrowid
        except sqlite3.IntegrityError:
            row = self._conn().execute("SELECT * FROM users WHERE external_id = ?", (external_id,)).fetchone()
            if row is None:
                raise
            return row
        write_event(self._db, "user.created", actor=external_id, item_type="user", item_id=username,
                    role=role, source="proxy")
        logger.info("Created user %r for proxy identity %r", username, external_id)
        return self._conn().execute("SELECT * FROM users WHERE id = ?", (row_id,)).fetchone()

    def resolve_token(self, raw: str) -> Principal | None:
        """The principal behind a bearer value, or None when it is unknown,
        revoked, expired, or its owner is disabled."""
        if not raw or not raw.startswith(TOKEN_PREFIX):
            return None
        conn = self._conn()
        row = conn.execute(
            "SELECT t.id AS tid, t.hint, t.name AS tname, t.expires_at, t.last_used_at, u.* FROM api_tokens t "
            "JOIN users u ON u.id = t.user_id "
            "WHERE t.token_hash = ? AND t.revoked_at IS NULL AND u.active = 1",
            (_sha256(raw),),
        ).fetchone()
        if row is None:
            return None
        now = _now()
        if row["expires_at"] and _parse(row["expires_at"]) < now:
            return None
        last = _parse(row["last_used_at"])
        if last is None or now - last > TOUCH_INTERVAL:
            with self._db.write() as w:
                w.execute("UPDATE api_tokens SET last_used_at = ? WHERE id = ?", (_iso(now), row["tid"]))
        return Principal(user=_row_to_user(row), via="token", token_id=row["tid"], token_hint=row["hint"],
                         token_name=row["tname"])


# ── Bootstrap ───────────────────────────────────────────────────────────────

def ensure_bootstrap(identity: Identity, first_run: FirstRun, admin_password: str | None) -> None:
    """Make sure someone can log in.

    With users present this does nothing (and says so if an admin password
    was supplied anyway, since it would be silently unused). With none:
    an `admin_password` creates `admin` directly — the headless path for
    compose files and CI — otherwise a setup code is printed for the browser's
    first-run screen to consume.
    """
    if identity.count_users() > 0:
        if admin_password:
            logger.info("MNEMOMATIC_ADMIN_PASSWORD is set but users already exist — ignoring it")
        return
    if admin_password:
        identity.create_user("admin", role="admin", display_name="Administrator",
                             password=admin_password)
        write_event(identity._db, "admin.created", actor="system", item_type="user", item_id="admin", source="env")
        logger.info("Created the initial 'admin' user from MNEMOMATIC_ADMIN_PASSWORD")
        return
    code = first_run.issue()
    banner = (
        "\n"
        "============================================================\n"
        "  No users yet. Open the web UI and enter this setup code\n"
        "  to create the first administrator:\n"
        "\n"
        f"      {code}\n"
        "\n"
        "  The code is valid until a user exists or the server restarts.\n"
        "============================================================\n"
    )
    print(banner, flush=True)
    logger.warning("First-run setup code issued (see the banner above)")

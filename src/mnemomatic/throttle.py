"""In-memory brute-force throttle for credential checks.

Tracks failed attempts per client (usually an IP address) in a sliding
window; once a client accumulates too many failures it is locked out for a
fixed period regardless of what it sends. State is process-local — restarts
clear it — which is the right trade-off for a single-instance server: it
raises the cost of online guessing without adding storage or configuration.

Note: the client key is the connection's peer address, so behind a reverse
proxy every request shares the proxy's IP and one attacker's lockout blocks
everyone behind it. Set MNEMOMATIC_TRUSTED_PROXIES so uvicorn resolves the
real client from X-Forwarded-For and each client gets its own bucket.

A check that awaits slow work (a password hash) before it can record the
outcome must `reserve()` the attempt first and `release()` it afterwards:
otherwise every request in a concurrent burst passes the check before the
first failure is counted.
"""

import ipaddress
import threading
import time

# Prune bookkeeping for idle clients once the table grows past this.
_MAX_TRACKED_CLIENTS = 1024


def client_key(ip: str) -> str:
    """The throttle bucket for a client address.

    An IPv6 host is usually handed a whole /64 and can pick a fresh address
    from it for every request, so per-address limits would reset on each
    one: IPv6 addresses share their /64's bucket. IPv4 (and IPv4-mapped IPv6)
    stays per address. Anything unparsable ("unknown", a test client name)
    is its own bucket.
    """
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return ip
    if addr.version == 6:
        if addr.ipv4_mapped:
            return str(addr.ipv4_mapped)
        return str(ipaddress.IPv6Network((addr, 64), strict=False))
    return str(addr)


class FailureThrottle:
    """Per-client failure counter with a sliding window and lockout."""

    def __init__(self, max_failures: int = 5, window: float = 60.0, lockout: float = 300.0):
        """Args:
            max_failures: Failures within `window` seconds that trigger a lockout.
            window: Sliding window (seconds) over which failures are counted.
            lockout: How long (seconds) a locked-out client stays blocked.
        """
        self.max_failures = max_failures
        self.window = window
        self.lockout = lockout
        self._lock = threading.Lock()
        self._failures: dict[str, list[float]] = {}
        self._locked_until: dict[str, float] = {}
        self._in_flight: dict[str, int] = {}

    def retry_after(self, client: str) -> int:
        """Seconds until `client` may try again; 0 when not locked out."""
        now = time.monotonic()
        with self._lock:
            until = self._locked_until.get(client, 0.0)
            if until <= now:
                return 0
            # Round up so a client that waits exactly this long is admitted.
            return int(until - now) + 1

    def reserve(self, client: str) -> int:
        """Claim an attempt for `client` before checking its credential.

        Returns 0 when admitted; the caller must then `release()` once the
        attempt is over, after recording its outcome. Otherwise returns the
        seconds to wait: the lockout, or 1 when attempts already in flight
        would use up what is left of the allowance.
        """
        now = time.monotonic()
        with self._lock:
            until = self._locked_until.get(client, 0.0)
            if until > now:
                return int(until - now) + 1
            recent = sum(1 for t in self._failures.get(client, []) if now - t < self.window)
            pending = self._in_flight.get(client, 0)
            if recent + pending >= self.max_failures:
                return 1
            self._in_flight[client] = pending + 1
            return 0

    def release(self, client: str) -> None:
        """End an attempt admitted by `reserve()`."""
        with self._lock:
            pending = self._in_flight.get(client, 0) - 1
            if pending > 0:
                self._in_flight[client] = pending
            else:
                self._in_flight.pop(client, None)

    def record_failure(self, client: str) -> None:
        now = time.monotonic()
        with self._lock:
            recent = [t for t in self._failures.get(client, []) if now - t < self.window]
            recent.append(now)
            if len(recent) >= self.max_failures:
                self._locked_until[client] = now + self.lockout
                self._failures.pop(client, None)
            else:
                self._failures[client] = recent
            if len(self._failures) + len(self._locked_until) > _MAX_TRACKED_CLIENTS:
                self._prune(now)

    def record_success(self, client: str) -> None:
        with self._lock:
            self._failures.pop(client, None)

    def _prune(self, now: float) -> None:
        """Drop expired lockouts and stale failure lists. Caller holds the lock."""
        self._locked_until = {c: t for c, t in self._locked_until.items() if t > now}
        self._failures = {
            c: times for c, times in self._failures.items()
            if times and now - times[-1] < self.window
        }

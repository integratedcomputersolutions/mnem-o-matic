"""In-memory brute-force throttle for credential checks.

Tracks failed attempts per client (usually an IP address) in a sliding
window; once a client accumulates too many failures it is locked out for a
fixed period regardless of what it sends. State is process-local — restarts
clear it — which is the right trade-off for a single-instance server: it
raises the cost of online guessing without adding storage or configuration.

Note: when the server sits behind a reverse proxy all requests share the
proxy's IP, so a lockout triggered by one attacker also blocks other clients
behind that proxy until it expires. That is the safe failure mode.
"""

import threading
import time

# Prune bookkeeping for idle clients once the table grows past this.
_MAX_TRACKED_CLIENTS = 1024


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

    def retry_after(self, client: str) -> int:
        """Seconds until `client` may try again; 0 when not locked out."""
        now = time.monotonic()
        with self._lock:
            until = self._locked_until.get(client, 0.0)
            if until <= now:
                return 0
            # Round up so a client that waits exactly this long is admitted.
            return int(until - now) + 1

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

"""Bounded, thread-safe process-local storage and request windows.

Run the HTTP application with one worker and one instance, or use shared
storage before distributing experiment requests across processes.
"""

import math
import secrets
import sys
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, fields, is_dataclass

import numpy as np


class ResultUnavailable(Exception):
    """The requested temporary result is unavailable or expired."""


class ResultTooLarge(Exception):
    """A single result exceeds the store's memory budget."""


@dataclass(frozen=True)
class _Entry:
    result: object
    expires_at: float
    size: int


def result_size(result):
    """Conservatively count owned arrays, containers, and experiment metadata."""
    seen = set()

    def size(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        count = sys.getsizeof(value)
        if isinstance(value, np.ndarray):
            return count if value.flags.owndata else count + value.nbytes
        if is_dataclass(value) and not isinstance(value, type):
            if hasattr(value, "__dict__"):
                return count + size(vars(value))
            return count + sum(size(getattr(value, field.name)) for field in fields(value))
        if isinstance(value, (tuple, list)):
            return count + sum(size(item) for item in value)
        if isinstance(value, dict):
            return count + sum(size(key) + size(item) for key, item in value.items())
        return count

    return size(result)


class ResultStore:
    def __init__(
        self, max_bytes=96 * 1024**2, max_results=4, ttl_seconds=20 * 60, clock=time.monotonic
    ):
        if max_bytes < 1 or max_results < 1 or ttl_seconds <= 0:
            raise ValueError("Result-store limits must be positive.")
        self.max_bytes = max_bytes
        self.max_results = max_results
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def _prune(self, now):
        for identifier, entry in tuple(self._entries.items()):
            if entry.expires_at <= now:
                self._bytes -= entry.size
                del self._entries[identifier]

    def add(self, result):
        size = result_size(result)
        if size > self.max_bytes:
            raise ResultTooLarge
        identifier = secrets.token_urlsafe(32)
        with self._lock:
            now = self._clock()
            self._prune(now)
            while len(self._entries) >= self.max_results or self._bytes + size > self.max_bytes:
                _, removed = self._entries.popitem(last=False)
                self._bytes -= removed.size
            self._entries[identifier] = _Entry(result, now + self.ttl_seconds, size)
            self._bytes += size
        return identifier

    def get(self, identifier):
        with self._lock:
            self._prune(self._clock())
            if identifier not in self._entries:
                raise ResultUnavailable
            self._entries.move_to_end(identifier)
            return self._entries[identifier].result

    @property
    def memory_bytes(self):
        with self._lock:
            self._prune(self._clock())
            return self._bytes

    @property
    def count(self):
        with self._lock:
            self._prune(self._clock())
            return len(self._entries)


class RateLimitExceeded(Exception):
    def __init__(self, retry_after):
        super().__init__("The request limit has been reached.")
        self.retry_after = retry_after


class RequestLimiter:
    def __init__(
        self, read_limit=120, run_limit=6, max_clients=1024, window_seconds=60, clock=time.monotonic
    ):
        if min(read_limit, run_limit, max_clients, window_seconds) <= 0:
            raise ValueError("Request-limit settings must be positive.")
        self._limits = {"read": read_limit, "run": run_limit}
        self._max_clients = max_clients
        self._window = window_seconds
        self._clock = clock
        self._clients = OrderedDict()
        self._lock = threading.Lock()

    def check(self, client, kind):
        with self._lock:
            now = self._clock()
            if client not in self._clients:
                if len(self._clients) >= self._max_clients:
                    self._clients.popitem(last=False)
                self._clients[client] = {"read": deque(), "run": deque()}
            self._clients.move_to_end(client)
            requests = self._clients[client][kind]
            while requests and requests[0] <= now - self._window:
                requests.popleft()
            if len(requests) >= self._limits[kind]:
                raise RateLimitExceeded(max(1, math.ceil(requests[0] + self._window - now)))
            requests.append(now)

    @property
    def client_count(self):
        with self._lock:
            return len(self._clients)

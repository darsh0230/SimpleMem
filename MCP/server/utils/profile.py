import time
import json
import logging
import threading
import os
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

logger = logging.getLogger("simplemem.profile")


class Profiler:
    """
    Simple profiler that logs duration and generates Chrome Trace Events.
    Thread-safe using thread-local storage.
    """

    def __init__(self, enabled: bool = True):
        self._local = threading.local()
        self._lock = threading.Lock()
        self._events: List[Dict[str, Any]] = []
        self._enabled = enabled

    def set_enabled(self, enabled: bool):
        """Enable or disable profiling"""
        self._enabled = enabled

    def _get_thread_events(self) -> List[Dict[str, Any]]:
        if not hasattr(self._local, "events"):
            self._local.events = []
        return self._local.events

    @contextmanager
    def profile(
        self, name: str, category: str = "PERF", args: Optional[Dict[str, Any]] = None
    ):
        """Context manager to profile a block of code."""
        if not self._enabled:
            yield
            return

        start_time = time.time()
        start_ts_us = int(start_time * 1_000_000)
        thread_id = threading.get_ident()
        process_id = os.getpid()

        try:
            yield
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Log to debug
            logger.debug(f"[PROFILE] {name} took {duration_ms:.2f}ms")

            # Create trace event (Complete Event 'X')
            event = {
                "name": name,
                "cat": category,
                "ph": "X",
                "ts": start_ts_us,
                "dur": int(duration_ms * 1000),  # microseconds
                "pid": process_id,
                "tid": thread_id,
                "args": args or {},
            }

            # Add to thread-local storage temporarily
            # In a real async environment, we might want to consolidate periodically
            # For now, we'll append to the global list with a lock to be safe
            with self._lock:
                self._events.append(event)

    def dump_trace(self, path: str):
        """Dump collected events to a JSON file in Chrome Trace format."""
        with self._lock:
            # Copy events to avoid issues
            events_to_dump = list(self._events)

        try:
            with open(path, "w") as f:
                json.dump(events_to_dump, f, indent=2)
            logger.info(f"Trace dumped to {path}")
        except Exception as e:
            logger.error(f"Failed to dump trace: {e}")

    def clear(self):
        with self._lock:
            self._events = []


# Global profiler instance
profiler = Profiler()

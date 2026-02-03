from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Event:
    type: str
    payload: dict[str, Any]


class EventBus:
    """Simple in-process event bus.

    NOTE about real-time audio:
      audio.capture can publish events at 50+ Hz. If downstream processing is slower,
      an unbounded queue will create latency ("lag") that grows over time.

    This bus provides `drain()` so runtimes can coalesce high-rate events (audio.chunk,
    vad.score, wake_word.scores, partial ASR, etc.) and stay real-time by keeping the
    newest events and dropping stale ones.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[Event] = queue.Queue()
        self._subscribers: dict[str, list[Callable[[Event], None]]] = {}
        self._lock = threading.Lock()

    def publish(self, event: Event) -> None:
        self._queue.put(event)

    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(handler)

    def poll(self, timeout: float = 0.1) -> Event | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self, max_items: int = 200) -> list[Event]:
        """Non-blocking drain up to max_items events."""
        out: list[Event] = []
        for _ in range(max_items):
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return out

    def qsize(self) -> int:
        try:
            return int(self._queue.qsize())
        except Exception:
            return 0

    def dispatch(self, event: Event) -> None:
        handlers: list[Callable[[Event], None]] = []
        with self._lock:
            handlers = list(self._subscribers.get(event.type, []))
            handlers += self._subscribers.get("*", [])
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                logging.getLogger("voice_agent.bus").exception(
                    "Unhandled exception in handler for event=%s handler=%s",
                    event.type,
                    getattr(handler, "__name__", str(handler)),
                )

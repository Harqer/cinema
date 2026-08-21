"""
api/services/event_bus.py
─────────────────────────────────────────────────────────────────────────────
Lightweight in-process SSE event bus.

Callers write events via `publish(channel, payload)`.
SSE endpoints subscribe with an async `subscribe(channel)` generator that
yields JSON-serialised messages until the client disconnects.

Channels:
  ingest:{project_id}     — book digestion progress
  generate:{project_id}:{episode_number}  — per-episode generation progress
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import AsyncGenerator


_queues: dict[str, list[asyncio.Queue]] = defaultdict(list)


def publish(channel: str, payload: dict) -> None:
    """Publish an event to all subscribers of *channel* (sync-safe)."""
    message = json.dumps(payload)
    dead: list[asyncio.Queue] = []
    for q in _queues[channel]:
        try:
            q.put_nowait(message)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        try:
            _queues[channel].remove(q)
        except ValueError:
            pass


async def subscribe(channel: str) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE-formatted messages from *channel*.

    Yields strings in the ``data: {json}\\n\\n`` format expected by
    EventSource on the browser.  Stops when the client drops or a
    ``{"type": "done"}`` sentinel is received.
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=512)
    _queues[channel].append(q)
    try:
        while True:
            message = await q.get()
            yield f"data: {message}\n\n"
            data = json.loads(message)
            if data.get("type") == "done":
                break
    finally:
        try:
            _queues[channel].remove(q)
        except ValueError:
            pass


def ingest_channel(project_id: int) -> str:
    return f"ingest:{project_id}"


def generate_channel(project_id: int, episode_number: int) -> str:
    return f"generate:{project_id}:{episode_number}"

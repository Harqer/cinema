"""
cinema_studio/memory.py
─────────────────────────────────────────────────────────────────────────────
Memory layer for the movie production LangGraph.

Two types of memory wired in:
  SHORT-TERM  — LangGraph checkpointer (PostgreSQL in prod, InMemorySaver in dev)
                Survives crashes; a 200-scene feature-film run can resume from
                the exact scene where it interrupted.

  LONG-TERM   — LangGraph store (PostgreSQL in prod, InMemoryStore in dev)
                Persists character appearance decisions, location visual notes,
                quality feedback, and style choices ACROSS production sessions
                (different books, re-runs of the same book).

Namespaces in the store
───────────────────────
  ("cinema", book_title, "characters")     — per-book character decisions
  ("cinema", book_title, "locations")      — per-book location decisions + Maps data
  ("cinema", book_title, "scene_quality")  — per-scene quality feedback history
  ("cinema", book_title, "style_guide")    — evolving cinematography style guide

Usage
─────
    # Dev (no Postgres needed)
    cp, store = get_memory()

    # Production
    cp, store = get_memory(
        postgres_uri="postgresql://user:pass@host:5432/cinema"
    )

    graph = build_movie_graph(checkpointer=cp, store=store)
    graph.invoke(state, {"configurable": {"thread_id": book_title}})
"""

from __future__ import annotations

import os
from typing import Any

# ── Type aliases ──────────────────────────────────────────────────────────────
# Keep imports lazy so the package loads without psycopg installed in dev
CheckpointerType = Any
StoreType = Any


def get_memory(
    postgres_uri: str | None = None,
    *,
    setup: bool = True,
) -> tuple[CheckpointerType, StoreType]:
    """
    Return (checkpointer, store) for the movie graph.

    If ``postgres_uri`` is provided (or the env var ``CINEMA_POSTGRES_URI`` is
    set), returns PostgreSQL-backed instances.  Otherwise returns in-memory
    instances suitable for development and testing.

    Args:
        postgres_uri: PostgreSQL connection string.
                      Falls back to env var CINEMA_POSTGRES_URI.
        setup:        If True and using Postgres, run schema migrations on first
                      use (idempotent — safe to leave True always).

    Returns:
        (checkpointer, store) — both ready to pass to build_movie_graph().
    """
    uri = postgres_uri or os.environ.get("CINEMA_POSTGRES_URI", "")

    if uri:
        return _postgres_memory(uri, setup=setup)
    return _inmemory_memory()


def _inmemory_memory() -> tuple:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.store.memory import InMemoryStore

    checkpointer = MemorySaver()
    store = InMemoryStore()
    return checkpointer, store


def _postgres_memory(uri: str, *, setup: bool) -> tuple:
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.store.postgres import PostgresStore

    checkpointer = PostgresSaver.from_conn_string(uri)
    store = PostgresStore.from_conn_string(uri)

    if setup:
        checkpointer.setup()
        store.setup()

    return checkpointer, store


# ── Store namespace helpers ───────────────────────────────────────────────────

def ns_characters(book_title: str) -> tuple:
    return ("cinema", _slug(book_title), "characters")

def ns_locations(book_title: str) -> tuple:
    return ("cinema", _slug(book_title), "locations")

def ns_scene_quality(book_title: str) -> tuple:
    return ("cinema", _slug(book_title), "scene_quality")

def ns_style_guide(book_title: str) -> tuple:
    return ("cinema", _slug(book_title), "style_guide")

def ns_production_log(book_title: str) -> tuple:
    return ("cinema", _slug(book_title), "production_log")

def _slug(title: str) -> str:
    return title.lower().replace(" ", "_")[:60]


# ── Store read/write helpers used by memory_node ─────────────────────────────

def store_character_decision(
    store: StoreType,
    book_title: str,
    character_name: str,
    visual_description: str,
    tts_voice: str,
) -> None:
    """Write a confirmed character appearance decision to long-term memory."""
    store.put(
        ns_characters(book_title),
        _slug(character_name),
        {
            "name": character_name,
            "visual_description": visual_description,
            "tts_voice": tts_voice,
            "confirmed": True,
        },
    )


def load_character_decisions(
    store: StoreType, book_title: str
) -> dict[str, dict]:
    """
    Load all confirmed character appearance decisions for a book.
    Returns {character_slug: {name, visual_description, tts_voice}}.
    """
    items = store.search(ns_characters(book_title), limit=200)
    return {item.key: item.value for item in items}


def store_location_decision(
    store: StoreType,
    book_title: str,
    location_name: str,
    maps_address: str,
    maps_description: str,
    visual_description: str,
) -> None:
    """Persist enriched location data so Maps grounding runs only once per book."""
    store.put(
        ns_locations(book_title),
        _slug(location_name),
        {
            "name": location_name,
            "maps_address": maps_address,
            "maps_description": maps_description,
            "visual_description": visual_description,
            "grounded": bool(maps_description),
        },
    )


def load_location_decisions(
    store: StoreType, book_title: str
) -> dict[str, dict]:
    items = store.search(ns_locations(book_title), limit=200)
    return {item.key: item.value for item in items}


def store_scene_quality(
    store: StoreType,
    book_title: str,
    chapter: int,
    scene: int,
    quality_score: float,
    issues: list[str],
    video_path: str,
) -> None:
    """Record quality outcome for a scene — used on re-run to skip already-good scenes."""
    key = f"ch{chapter:03d}_sc{scene:03d}"
    store.put(
        ns_scene_quality(book_title),
        key,
        {
            "chapter": chapter,
            "scene": scene,
            "quality_score": quality_score,
            "issues": issues,
            "video_path": video_path,
            "passed": quality_score >= 0.68,
        },
    )


def load_passed_scenes(store: StoreType, book_title: str) -> set[str]:
    """Return set of 'ch001_sc003'-style keys for scenes that already passed QA."""
    items = store.search(ns_scene_quality(book_title), limit=10000)
    return {item.key for item in items if item.value.get("passed")}


def store_style_note(
    store: StoreType,
    book_title: str,
    note: str,
    source: str = "quality_gate",
) -> None:
    """Accumulate style lessons learned during production."""
    import uuid
    store.put(
        ns_style_guide(book_title),
        str(uuid.uuid4())[:8],
        {"note": note, "source": source},
    )


def load_style_notes(store: StoreType, book_title: str) -> list[str]:
    items = store.search(ns_style_guide(book_title), limit=50)
    return [item.value.get("note", "") for item in items]

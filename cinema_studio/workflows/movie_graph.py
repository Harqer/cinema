"""
cinema_studio/workflows/movie_graph.py
─────────────────────────────────────────────────────────────────────────────
Full-length movie production LangGraph — with short-term + long-term memory.

Graph topology:

  START
    │
    ▼
  load_book              ← ingest PDF → BookAnalysis (chapter+scene structure)
    │
    ▼
  build_registries       ← CharacterRegistry + LocationRegistry (Maps grounding)
    │
    ▼
  restore_from_memory    ← merge previously confirmed char/location decisions
    │                      from long-term store; mark already-passed scenes
    ▼
  ┌─► advance_scene ─────────────────────── done? ──► assemble_movie ──► END
  │     │                                   ▲
  │     ▼                                   │
  │   write_script  (write_scene_script tool call)
  │     │
  │     ▼
  │   generate_storyboard  (generate_storyboard_image tool call)
  │     │
  │     ▼
  │   generate_clip  (generate_scene_video tool call)
  │     │
  │     ▼
  │   validate_clip ─── pass ──► commit_scene_to_memory
  │                │                        │
  │                │                        ▼
  │                │             summarize_log ──► advance_scene (loop)
  │                └── retry ──► generate_storyboard (redo storyboard+clip)
  └──────────────────── skip (max retries) ──► advance_scene

Short-term memory (checkpointer):
  Every node's output is checkpointed.  If the process crashes mid-run
  (e.g. after scene 47 of 200), restart with the same thread_id and the
  graph resumes from the last checkpoint automatically.

Long-term memory (store):
  Character appearances, location Maps data, and scene quality outcomes
  are written to the store after each passing scene.  On re-run of the
  same book title, restore_from_memory reads this data back so:
    - Characters look identical to the previous run
    - Maps grounding doesn't repeat for locations already resolved
    - Already-passed scenes are skipped (reused from prior run)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from cinema_studio.workflows.movie_state import MovieWorkflowState
from cinema_studio.workflows.nodes.book_nodes import (
    load_book_node,
    build_registries_node,
    advance_scene_node,
    assemble_movie_node,
)
from cinema_studio.workflows.nodes.scene_nodes import (
    write_script_node,
    generate_storyboard_node,
    generate_clip_node,
    validate_clip_node,
)
from cinema_studio.workflows.nodes.memory_nodes import (
    restore_from_memory_node,
    commit_scene_to_memory_node,
    summarize_log_node,
)


# ── Routing functions ─────────────────────────────────────────────────────────

def after_advance(state: MovieWorkflowState) -> Literal["write_script", "assemble_movie"]:
    """Route after advance_scene: if all chapters done → assemble, else write script."""
    if state.get("status") == "complete":
        return "assemble_movie"
    return "write_script"


def after_validate(
    state: MovieWorkflowState,
) -> Literal["commit_scene_to_memory", "generate_storyboard", "advance_scene"]:
    """
    Route after validate_clip:
      pass           → commit_scene_to_memory (write to store, then advance)
      retry          → generate_storyboard (redo from storyboard for visual fix)
      max retries    → advance_scene (skip, log it, don't write bad scene to store)
    """
    score = state.get("current_quality_score", 0.0)
    retry = state.get("current_retry_count", 0)
    max_retries = state.get("max_clip_retries", 2)
    status = state.get("status", "running")

    if status == "complete":
        return "commit_scene_to_memory"

    if score >= 0.68:
        return "commit_scene_to_memory"

    if retry >= max_retries:
        return "advance_scene"   # skip — don't commit bad scene

    return "generate_storyboard"


def after_commit(
    state: MovieWorkflowState,
) -> Literal["summarize_log", "advance_scene"]:
    """After committing to memory, decide whether to summarize the log."""
    clips_done = len(state.get("completed_clips", []))
    if clips_done > 0 and clips_done % 10 == 0:
        return "summarize_log"
    return "advance_scene"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_movie_graph(
    checkpointer: Any = None,
    store: Any = None,
) -> StateGraph:
    """
    Compile the full book-to-movie LangGraph with memory.

    Args:
        checkpointer: Short-term memory — checkpoints every node output.
                      Pass ``MemorySaver()`` for dev or a ``PostgresSaver``
                      for production.  Defaults to ``MemorySaver()``.
        store:        Long-term memory — persists cross-session character/
                      location data.  Pass ``InMemoryStore()`` for dev or a
                      ``PostgresStore`` for production.  None = disabled.

    Returns:
        Compiled LangGraph app ready for .invoke() / .stream().
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    g = StateGraph(MovieWorkflowState)

    # ── Register all nodes ────────────────────────────────────────────────────
    g.add_node("load_book", load_book_node)
    g.add_node("build_registries", build_registries_node)
    g.add_node("restore_from_memory", restore_from_memory_node)
    g.add_node("advance_scene", advance_scene_node)
    g.add_node("write_script", write_script_node)
    g.add_node("generate_storyboard", generate_storyboard_node)
    g.add_node("generate_clip", generate_clip_node)
    g.add_node("validate_clip", validate_clip_node)
    g.add_node("commit_scene_to_memory", commit_scene_to_memory_node)
    g.add_node("summarize_log", summarize_log_node)
    g.add_node("assemble_movie", assemble_movie_node)

    # ── Fixed edges ───────────────────────────────────────────────────────────
    # Startup chain
    g.add_edge(START, "load_book")
    g.add_edge("load_book", "build_registries")
    g.add_edge("build_registries", "restore_from_memory")
    g.add_edge("restore_from_memory", "advance_scene")

    # Inner scene pipeline — fixed
    g.add_edge("write_script", "generate_storyboard")
    g.add_edge("generate_storyboard", "generate_clip")
    g.add_edge("generate_clip", "validate_clip")

    # summarize_log always flows back to advance_scene
    g.add_edge("summarize_log", "advance_scene")

    # Assembly → done
    g.add_edge("assemble_movie", END)

    # ── Conditional edges ─────────────────────────────────────────────────────
    # After advance_scene: next scene or done
    g.add_conditional_edges(
        "advance_scene",
        after_advance,
        {
            "write_script": "write_script",
            "assemble_movie": "assemble_movie",
        },
    )

    # After validate_clip: pass → commit | retry → redo storyboard | skip
    g.add_conditional_edges(
        "validate_clip",
        after_validate,
        {
            "commit_scene_to_memory": "commit_scene_to_memory",
            "generate_storyboard": "generate_storyboard",
            "advance_scene": "advance_scene",
        },
    )

    # After commit: maybe summarize log, then advance
    g.add_conditional_edges(
        "commit_scene_to_memory",
        after_commit,
        {
            "summarize_log": "summarize_log",
            "advance_scene": "advance_scene",
        },
    )

    return g.compile(checkpointer=checkpointer, store=store)


# ── High-level runner ─────────────────────────────────────────────────────────

def produce_movie(
    book_source: str | Path,
    book_title: str,
    book_author: str = "",
    genre: str = "drama",
    style: str = "cinematic",
    clip_duration_seconds: int = 8,
    output_dir: str | Path = "output/movie",
    max_clip_retries: int = 2,
    thread_id: str | None = None,
    stream: bool = True,
    checkpointer: Any = None,
    store: Any = None,
    postgres_uri: str | None = None,
) -> MovieWorkflowState:
    """
    Produce a full-length movie from a book.

    Memory options (in priority order):
      1. Pass explicit ``checkpointer`` + ``store`` objects.
      2. Pass ``postgres_uri`` — memory module creates Postgres instances.
      3. Neither — uses in-memory (dev mode, not resumable across process restarts).

    Args:
        book_source:           Path to PDF/text or raw text string.
        book_title:            Title (used for thread_id and output filenames).
        book_author:           Author name.
        genre:                 Genre string for market research.
        style:                 Visual style — cinematic | documentary | noir | etc.
        clip_duration_seconds: Length of each scene clip in seconds.
        output_dir:            Root directory for all output files.
        max_clip_retries:      Max attempts per scene before skipping.
        thread_id:             LangGraph checkpoint thread ID.
                               Defaults to the slugified book title so the same
                               book always resumes the same run.
        stream:                Print progress as nodes execute.
        checkpointer:          Explicit LangGraph checkpointer instance.
        store:                 Explicit LangGraph store instance.
        postgres_uri:          PostgreSQL URI — convenience shortcut to use
                               Postgres for both checkpointer and store.

    Returns:
        Final :class:`MovieWorkflowState` with movie_path set.
    """
    # ── Resolve memory ────────────────────────────────────────────────────────
    if checkpointer is None or store is None:
        from cinema_studio.memory import get_memory
        _cp, _st = get_memory(postgres_uri=postgres_uri)
        if checkpointer is None:
            checkpointer = _cp
        if store is None:
            store = _st

    app = build_movie_graph(checkpointer=checkpointer, store=store)

    # Default thread_id = slugified book title so same book always resumes
    if thread_id is None:
        thread_id = book_title.lower().replace(" ", "_")[:60]

    initial: MovieWorkflowState = {
        "book_source": str(book_source),
        "book_title": book_title,
        "book_author": book_author,
        "genre": genre,
        "style": style,
        "clip_duration_seconds": clip_duration_seconds,
        "output_dir": str(output_dir),
        "max_clip_retries": max_clip_retries,
        "completed_clips": [],
        "passed_scene_keys": [],
        "style_notes": [],
        "log": [],
        "status": "running",
        "current_chapter_index": 0,
        "current_scene_index": 0,
        "current_retry_count": 0,
        "current_quality_issues": [],
    }

    config = {"configurable": {"thread_id": thread_id}}

    if stream:
        final: MovieWorkflowState = {}
        for event in app.stream(initial, config=config, stream_mode="updates"):
            for node_name, node_state in event.items():
                clips_done = len(node_state.get("completed_clips", []))
                total = node_state.get("total_scenes", "?")
                for line in node_state.get("log", []):
                    print(f"  {line}")
                if clips_done:
                    print(f"    clips: {clips_done}/{total}")
                final = {**final, **node_state}
        return final
    else:
        return app.invoke(initial, config=config)

"""
cinema_studio/workflows/nodes/memory_nodes.py
─────────────────────────────────────────────────────────────────────────────
Memory nodes — long-term store reads/writes that make the movie graph
resumable and context-aware across sessions.

  restore_from_memory_node
      Runs once at startup (after build_registries).
      Reads previously confirmed character/location decisions from the store
      and merges them back into the live registries — so a resumed run doesn't
      regenerate characters that were already locked in.
      Also skips scenes that already passed QA in a previous run.

  commit_scene_to_memory_node
      Runs after validate_clip passes.
      Writes the scene's quality outcome, character appearance, and location
      data to the long-term store so they survive across sessions.

  summarize_log_node
      Runs every N scenes to trim the in-state log list.
      Writes a rolling summary of production decisions to the store and
      replaces the full log list with just the summary line.
      Prevents the log list from growing unbounded across a 200-scene film.
"""

from __future__ import annotations

from cinema_studio.memory import (
    load_character_decisions,
    load_location_decisions,
    load_passed_scenes,
    load_style_notes,
    store_character_decision,
    store_location_decision,
    store_scene_quality,
    store_style_note,
    ns_production_log,
    _slug,
)
from cinema_studio.workflows.movie_state import MovieWorkflowState

# How often (in completed clips) we run log summarization
_SUMMARIZE_EVERY = 10


# ── NODE: RestoreFromMemory ───────────────────────────────────────────────────

def restore_from_memory_node(state: MovieWorkflowState, *, store=None) -> MovieWorkflowState:
    """
    At graph startup, read long-term memory and:
      1. Overlay confirmed character appearances onto the registry dict.
      2. Overlay enriched location data onto the registry dict.
      3. Build a set of already-passed scene keys so validate_clip can skip them.

    If the store has no data for this book yet (first run), this is a no-op.

    LangGraph injects the store automatically when the graph is compiled with
    ``store=store`` — the ``store`` keyword argument is the injection point.
    """
    if store is None:
        # No store configured — skip (dev mode)
        return {**state, "log": ["[memory] No store configured — skipping restore."]}

    book_title = state.get("book_title", "unknown")

    # ── Restore character registry ─────────────────────────────────────────────
    stored_chars = load_character_decisions(store, book_title)
    char_reg = dict(state.get("character_registry", {}))
    for slug, data in stored_chars.items():
        if slug in char_reg:
            # Update visual description with the stored (confirmed) version
            char_reg[slug]["visual_description"] = data.get(
                "visual_description", char_reg[slug].get("visual_description", "")
            )
            char_reg[slug]["tts_voice_name"] = data.get(
                "tts_voice", char_reg[slug].get("tts_voice_name", "Fenrir")
            )
        else:
            char_reg[slug] = data

    # ── Restore location registry ──────────────────────────────────────────────
    stored_locs = load_location_decisions(store, book_title)
    loc_reg = dict(state.get("location_registry", {}))
    for slug, data in stored_locs.items():
        if data.get("grounded"):
            if slug not in loc_reg:
                loc_reg[slug] = {}
            loc_reg[slug]["maps_description"] = data.get("maps_description", "")
            loc_reg[slug]["maps_formatted_address"] = data.get("maps_address", "")
            loc_reg[slug]["maps_grounded"] = True

    # ── Load style notes ───────────────────────────────────────────────────────
    style_notes = load_style_notes(store, book_title)

    # ── Load already-passed scene keys (for skip-on-resume) ───────────────────
    passed_scenes = load_passed_scenes(store, book_title)

    msgs = [
        f"[memory] Restored {len(stored_chars)} character decisions, "
        f"{len(stored_locs)} location decisions, "
        f"{len(passed_scenes)} previously passed scenes."
    ]
    if style_notes:
        msgs.append(f"[memory] {len(style_notes)} style notes loaded.")

    return {
        **state,
        "character_registry": char_reg,
        "location_registry": loc_reg,
        # Store passed-scene keys as a JSON-serialisable list on state
        # (validate_clip_node reads this to skip already-good scenes)
        "passed_scene_keys": list(passed_scenes),
        "style_notes": style_notes,
        "log": msgs,
    }


# ── NODE: CommitSceneToMemory ─────────────────────────────────────────────────

def commit_scene_to_memory_node(state: MovieWorkflowState, *, store=None) -> MovieWorkflowState:
    """
    After a scene passes QA, persist its data to long-term memory.

    Writes:
      - Scene quality outcome (so re-runs skip it)
      - Character appearances seen in this scene (locks them in)
      - Location data (so Maps grounding isn't repeated)
      - Any style notes from quality issues (for future scene generation)
    """
    if store is None:
        return state

    book_title = state.get("book_title", "unknown")
    ch_idx = state.get("current_chapter_index", 0)
    sc_idx = state.get("current_scene_index", 0)
    scene = state.get("current_scene", {})
    char_reg = state.get("character_registry", {})
    loc_reg = state.get("location_registry", {})

    # ── Commit quality outcome ─────────────────────────────────────────────────
    store_scene_quality(
        store=store,
        book_title=book_title,
        chapter=ch_idx + 1,
        scene=sc_idx + 1,
        quality_score=state.get("current_quality_score", 0.0),
        issues=state.get("current_quality_issues", []),
        video_path=state.get("current_video_path", ""),
    )

    # ── Commit character appearances ───────────────────────────────────────────
    for cp in scene.get("characters_present", []):
        name = cp.get("character_name", cp.get("name", ""))
        slug = _slug(name)
        rec = char_reg.get(slug, {})
        store_character_decision(
            store=store,
            book_title=book_title,
            character_name=name,
            visual_description=rec.get("visual_description", ""),
            tts_voice=rec.get("tts_voice_name", "Fenrir"),
        )

    # ── Commit location data ───────────────────────────────────────────────────
    loc_name = scene.get("location_name", "")
    if loc_name:
        slug = _slug(loc_name)
        rec = loc_reg.get(slug, {})
        if rec.get("maps_grounded"):
            store_location_decision(
                store=store,
                book_title=book_title,
                location_name=loc_name,
                maps_address=rec.get("maps_formatted_address", ""),
                maps_description=rec.get("maps_description", ""),
                visual_description=rec.get("visual_description", ""),
            )

    # ── Commit any style lessons from quality issues ───────────────────────────
    for issue in state.get("current_quality_issues", []):
        if issue.startswith("Improvement hint:"):
            store_style_note(store=store, book_title=book_title, note=issue)

    return {
        **state,
        "log": [f"[memory] Scene Ch{ch_idx+1}/Sc{sc_idx+1} committed to store."],
    }


# ── NODE: SummarizeLog ────────────────────────────────────────────────────────

def summarize_log_node(state: MovieWorkflowState, *, store=None) -> MovieWorkflowState:
    """
    Rolling log summarization — runs every _SUMMARIZE_EVERY completed scenes.

    Prevents the log list from growing into hundreds of entries across a
    200-scene production run (which would bloat every checkpoint).

    Writes the full log to the store as a timestamped entry, then replaces
    the in-state log with a single summary line.
    """
    clips_done = len(state.get("completed_clips", []))
    if clips_done % _SUMMARIZE_EVERY != 0 or clips_done == 0:
        return state  # Not time yet

    log = state.get("log", [])
    total = state.get("total_scenes", "?")
    summary_line = (
        f"[log summary] {clips_done}/{total} scenes complete. "
        f"Last {len(log)} log entries archived to store."
    )

    if store is not None:
        import uuid, datetime
        book_title = state.get("book_title", "unknown")
        store.put(
            ns_production_log(book_title),
            str(uuid.uuid4())[:8],
            {
                "clips_done": clips_done,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "entries": log,
            },
        )

    return {
        **state,
        # Replace full log with just the summary — checkpoints stay small
        "log": [summary_line],
    }

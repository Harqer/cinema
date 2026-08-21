"""
cinema_studio/workflows/nodes/book_nodes.py
─────────────────────────────────────────────────────────────────────────────
Outer-loop nodes — run once per movie production job.

  load_book_node        — ingest PDF/text → BookAnalysis
  build_registries_node — CharacterRegistry + LocationRegistry (Maps grounding)
  advance_scene_node    — load the next scene's data onto state for inner nodes
  assemble_movie_node   — concatenate all clips into a single MP4
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from cinema_studio.agents.document_processor import process_book
from cinema_studio.production.character_registry import CharacterRegistry
from cinema_studio.production.location_registry import LocationRegistry
from cinema_studio.workflows.movie_state import MovieWorkflowState


# ── NODE: LoadBook ────────────────────────────────────────────────────────────

def load_book_node(state: MovieWorkflowState) -> MovieWorkflowState:
    """
    Ingest the book and build a chapter-by-chapter BookAnalysis.
    Stores the result serialised as a dict on state.
    """
    source = state.get("book_source", "")
    title = state.get("book_title", "")

    print(f"  📖  Loading book: {title or source}")
    analysis = process_book(source=source, title=title or None)

    total_scenes = sum(len(ch.scenes) for ch in analysis.chapters)

    return {
        **state,
        "book_analysis": analysis.model_dump(),
        "total_chapters": len(analysis.chapters),
        "total_scenes": total_scenes,
        "current_chapter_index": 0,
        "current_scene_index": 0,
        "current_retry_count": 0,
        "completed_clips": [],
        "log": [
            f"Book loaded: {analysis.title} by {analysis.author}. "
            f"{len(analysis.chapters)} chapters, {total_scenes} scenes."
        ],
    }


# ── NODE: BuildRegistries ─────────────────────────────────────────────────────

def build_registries_node(state: MovieWorkflowState) -> MovieWorkflowState:
    """
    Build the CharacterRegistry and LocationRegistry from the BookAnalysis.
    Runs Google Maps grounding for all locations with real-world hints.
    """
    from cinema_studio.agents.document_processor import BookAnalysis

    analysis = BookAnalysis.model_validate(state["book_analysis"])

    print("  🧑‍🎭  Building character registry...")
    char_registry = CharacterRegistry.from_analysis(analysis)

    print("  🗺   Building location registry + Maps grounding...")
    loc_registry = LocationRegistry.from_analysis(analysis)
    loc_registry.ground_all()   # enriches each location with Maps data

    return {
        **state,
        "character_registry": char_registry.to_dict(),
        "location_registry": loc_registry.to_dict(),
        "log": [
            f"Registries built: {len(char_registry.all_characters())} characters, "
            f"{len(loc_registry.all_locations())} locations."
        ],
    }


# ── NODE: AdvanceScene ────────────────────────────────────────────────────────

def advance_scene_node(state: MovieWorkflowState) -> MovieWorkflowState:
    """
    Load the current scene (identified by chapter_index + scene_index) onto
    state.current_scene so the inner scene nodes can read it.

    Also resets per-scene artefacts.
    """
    analysis_dict = state.get("book_analysis", {})
    chapters = analysis_dict.get("chapters", [])
    ch_idx = state.get("current_chapter_index", 0)
    sc_idx = state.get("current_scene_index", 0)

    if ch_idx >= len(chapters):
        # All chapters done — signal completion
        return {**state, "status": "complete"}

    chapter = chapters[ch_idx]
    scenes = chapter.get("scenes", [])

    if sc_idx >= len(scenes):
        # Chapter exhausted — advance to next
        new_ch = ch_idx + 1
        if new_ch >= len(chapters):
            return {**state, "status": "complete"}
        return {
            **state,
            "current_chapter_index": new_ch,
            "current_scene_index": 0,
        }

    scene = scenes[sc_idx]
    total = state.get("total_scenes", 1)
    done = sum(len(c.get("scenes", [])) for c in chapters[:ch_idx]) + sc_idx + 1

    print(
        f"  🎬  Scene {done}/{total} — "
        f"Ch{ch_idx+1} '{chapter.get('title','')}' | {scene.get('slug','')}"
    )

    return {
        **state,
        "current_scene": scene,
        "current_script": {},
        "current_storyboard_path": "",
        "current_storyboard_mime": "",
        "current_video_path": "",
        "current_video_mime": "",
        "current_quality_score": 0.0,
        "current_retry_count": state.get("current_retry_count", 0),
        "current_quality_issues": [],
        "status": "running",
    }


# ── NODE: AssembleMovie ───────────────────────────────────────────────────────

def assemble_movie_node(state: MovieWorkflowState) -> MovieWorkflowState:
    """
    Concatenate all completed clips into a single full-length MP4.

    Uses ffmpeg concat demuxer — clips are joined in chapter/scene order
    with no re-encoding (stream copy) for speed.

    Falls back gracefully if ffmpeg is not installed: writes a JSON manifest
    listing all clip paths in order.
    """
    clips: list[dict] = state.get("completed_clips", [])
    output_dir = Path(state.get("output_dir", "output"))
    title = state.get("book_title", "movie").replace(" ", "_")

    if not clips:
        return {
            **state,
            "status": "failed",
            "error": "No clips to assemble.",
        }

    # Sort by chapter then scene
    ordered = sorted(clips, key=lambda c: (c.get("chapter_number", 0), c.get("scene_number", 0)))
    valid = [c for c in ordered if c.get("video_path") and Path(c["video_path"]).exists()]

    output_dir.mkdir(parents=True, exist_ok=True)
    movie_path = str(output_dir / f"{title}_full_movie.mp4")
    manifest_path = str(output_dir / f"{title}_clip_manifest.json")

    # Always write manifest
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "title": state.get("book_title", ""),
                "author": state.get("book_author", ""),
                "total_clips": len(valid),
                "clips": [
                    {
                        "order": i + 1,
                        "chapter": c.get("chapter_number"),
                        "scene": c.get("scene_number"),
                        "slug": c.get("slug"),
                        "path": c.get("video_path"),
                        "duration_seconds": c.get("duration_seconds", 8),
                    }
                    for i, c in enumerate(valid)
                ],
            },
            f,
            indent=2,
        )

    # Try ffmpeg concat
    try:
        concat_list = output_dir / "concat_list.txt"
        with open(concat_list, "w") as f:
            for clip in valid:
                f.write(f"file '{Path(clip['video_path']).resolve()}'\n")

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-c", "copy",
                movie_path,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr[-500:])

        total_duration = sum(c.get("duration_seconds", 8) for c in valid)

        return {
            **state,
            "movie_path": movie_path,
            "movie_duration_seconds": float(total_duration),
            "status": "complete",
            "log": [
                f"🎞  Movie assembled: {movie_path} "
                f"({len(valid)} clips, ~{total_duration//60}m{total_duration%60}s)"
            ],
        }

    except (FileNotFoundError, RuntimeError) as exc:
        # ffmpeg not available or failed — manifest-only mode
        return {
            **state,
            "movie_path": manifest_path,
            "movie_duration_seconds": float(sum(c.get("duration_seconds", 8) for c in valid)),
            "status": "complete",
            "log": [
                f"⚠  ffmpeg unavailable ({exc}). "
                f"Clip manifest written to {manifest_path}. "
                f"Run: ffmpeg -f concat -safe 0 -i concat_list.txt -c copy movie.mp4"
            ],
        }

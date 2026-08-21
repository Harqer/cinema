"""
cinema_studio/workflows/movie_state.py
─────────────────────────────────────────────────────────────────────────────
Full-length movie production state — the wire that carries data through the
entire book → movie LangGraph.

Organised in layers:
  1. Book inputs (paths, raw text)
  2. Semantic analysis (BookAnalysis, registries)
  3. Chapter iteration cursor (which chapter/scene we are generating now)
  4. Per-scene artefacts (script, storyboard, video clip)
  5. Clip manifest (growing list of completed clips)
  6. Assembly (final movie file path)
  7. Control (status, errors, retries)
"""

from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict


def _append(existing: list, new: list) -> list:
    return (existing or []) + (new or [])


class ClipRecord(TypedDict, total=False):
    """Metadata for one generated video clip."""
    chapter_number: int
    scene_number: int
    slug: str
    video_path: str
    video_mime: str
    storyboard_path: str
    script: dict             # output of write_scene_script
    duration_seconds: int
    quality_score: float
    characters: list[str]
    location_name: str


class MovieWorkflowState(TypedDict, total=False):
    # ── Inputs ────────────────────────────────────────────────────────────────
    book_source: str          # Path to PDF or raw text
    book_title: str
    book_author: str
    genre: str
    style: str                # "cinematic" | "documentary" | ...
    clip_duration_seconds: int        # seconds per scene clip (default 8)
    output_dir: str
    max_clip_retries: int

    # ── Semantic analysis ─────────────────────────────────────────────────────
    book_analysis: dict       # BookAnalysis serialised to dict
    character_registry: dict  # CharacterRegistry.to_dict()
    location_registry: dict   # LocationRegistry.to_dict()

    # ── Chapter iteration cursor ──────────────────────────────────────────────
    total_chapters: int
    total_scenes: int
    current_chapter_index: int   # 0-based index into book_analysis.chapters
    current_scene_index: int     # 0-based index into current chapter's scenes
    previous_scene_summary: str  # last generated scene description (for continuity)

    # ── Per-scene artefacts (reset each scene) ────────────────────────────────
    current_scene: dict          # BookScene serialised to dict
    current_script: dict         # write_scene_script output
    current_storyboard_path: str
    current_storyboard_mime: str
    current_video_path: str
    current_video_mime: str
    current_quality_score: float
    current_retry_count: int
    current_quality_issues: Annotated[list[str], _append]

    # ── Clip manifest (append-only) ───────────────────────────────────────────
    completed_clips: Annotated[list[ClipRecord], _append]

    # ── Long-term memory overlays (populated by restore_from_memory_node) ─────
    passed_scene_keys: list[str]   # "ch001_sc003" keys already passed in prior runs
    style_notes: list[str]         # Accumulated cinematography style lessons

    # ── Assembly ──────────────────────────────────────────────────────────────
    movie_path: str           # Path to the assembled full-length MP4
    movie_duration_seconds: float

    # ── Control ───────────────────────────────────────────────────────────────
    status: str               # "running" | "complete" | "failed"
    error: str
    log: Annotated[list[str], _append]

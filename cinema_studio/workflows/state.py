"""
cinema_studio/workflows/state.py
─────────────────────────────────────────────────────────────────────────────
Shared state schema that flows through every node in the image-to-video
LangGraph workflow.

Think of this like the "wire" in ComfyUI — all nodes read from and write into
a single typed dict that LangGraph checkpoints at every step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from typing_extensions import TypedDict


# ── Operator helpers ──────────────────────────────────────────────────────────
# LangGraph uses "reducers" to merge parallel writes to the same key.
# For simple scalar fields we just overwrite (default); for lists we append.

def _append(existing: list, new: list) -> list:
    return (existing or []) + (new or [])


# ── State ─────────────────────────────────────────────────────────────────────

class VideoWorkflowState(TypedDict, total=False):
    # ── Inputs ────────────────────────────────────────────────────────────────
    image_path: str              # Path to the input image file
    image_bytes: bytes           # Raw image bytes (loaded by LoadImage node)
    image_mime: str              # MIME type, e.g. "image/jpeg"
    user_prompt: str             # Optional user hint for the video content
    style: str                   # "cinematic" | "documentary" | "surreal" | ...
    duration_seconds: int        # Target video length (e.g. 5, 10, 15)
    output_dir: str              # Where to write final assets

    # ── Analysis ─────────────────────────────────────────────────────────────
    image_analysis: dict         # Structured scene metadata from AnalyzeImage
    # {subject, setting, mood, time_of_day, color_palette, camera_angle, ...}

    # ── Prompt engineering ────────────────────────────────────────────────────
    base_prompt: str             # Draft prompt from analysis
    enhanced_prompt: str         # Upscaled prompt from EnhancePrompt node
    parallel_research: str       # Cinematography research from Parallel web search
    negative_prompt: str         # What to avoid in the video

    # ── Video generation ─────────────────────────────────────────────────────
    video_bytes: bytes           # Raw generated video bytes
    video_path: str              # Path to saved video file
    video_mime: str              # e.g. "video/mp4"

    # ── Quality gate ─────────────────────────────────────────────────────────
    quality_score: float         # 0.0–1.0 quality estimate
    quality_issues: Annotated[list[str], _append]  # Logged issues
    retry_count: int             # How many regeneration attempts so far
    max_retries: int             # Hard limit on retries (default 2)

    # ── Routing / control ─────────────────────────────────────────────────────
    error: str                   # Set on any unrecoverable failure
    status: str                  # "running" | "complete" | "failed"

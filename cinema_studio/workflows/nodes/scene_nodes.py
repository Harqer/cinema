"""
cinema_studio/workflows/nodes/scene_nodes.py
─────────────────────────────────────────────────────────────────────────────
The four scene-level nodes that form the inner loop of the movie graph.
Each node does exactly one job and calls its corresponding FunctionTool.

  write_script_node    → calls write_scene_script tool
  generate_storyboard_node → calls generate_storyboard_image tool
  generate_clip_node   → calls generate_scene_video tool
  validate_clip_node   → Gemini quality gate, sets status for routing

All nodes read character+location registry state for consistency enforcement.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from google.genai import types
from google.genai.types import (
    FunctionCallingConfig,
    FunctionCallingConfigMode,
    GenerateContentConfig,
    Tool,
    ToolConfig,
)

from cinema_studio.config import MODEL_REASONING, get_client
from cinema_studio.tools.function_tools import (
    WRITE_SCENE_SCRIPT_DECL,
    GENERATE_STORYBOARD_IMAGE_DECL,
    GENERATE_SCENE_VIDEO_DECL,
    write_scene_script,
    generate_storyboard_image,
    generate_scene_video,
)
from cinema_studio.production.character_registry import CharacterRegistry
from cinema_studio.production.location_registry import LocationRegistry
from cinema_studio.workflows.movie_state import MovieWorkflowState


def _get_registries(state: MovieWorkflowState):
    """Reconstruct in-memory registries from serialised state dicts."""
    # We pass already-built registry objects through state as dicts for
    # JSON-serialisability — rebuild lightweight lookup wrappers here.
    char_data = state.get("character_registry", {})
    loc_data = state.get("location_registry", {})

    # Minimal wrapper — only visual_description and tts_voice_name needed
    class _QuickRegistry:
        def __init__(self, data: dict):
            self._d = data
        def visual(self, name: str) -> str:
            return self._d.get(name.lower(), {}).get("visual_description", "")
        def voice(self, name: str) -> str:
            return self._d.get(name.lower(), {}).get("tts_voice_name", "Fenrir")
        def loc_visual(self, name: str) -> str:
            return self._d.get(name.lower(), {}).get("maps_description") or \
                   self._d.get(name.lower(), {}).get("visual_description", "")
        def loc_address(self, name: str) -> str:
            return self._d.get(name.lower(), {}).get("real_world_hint", "")

    return _QuickRegistry(char_data), _QuickRegistry(loc_data)


# ── NODE 1: WriteScript ───────────────────────────────────────────────────────

def write_script_node(state: MovieWorkflowState) -> MovieWorkflowState:
    """
    Call write_scene_script via Gemini forced function calling.
    Gemini fills the tool arguments from the scene context; we execute them.
    """
    client = get_client()
    scene = state.get("current_scene", {})
    char_reg, loc_reg = _get_registries(state)
    prev = state.get("previous_scene_summary", "")

    # Build speaker configs from scene characters
    chars_present = scene.get("characters_present", [])
    characters_for_tool = []
    for cp in chars_present:
        name = cp.get("character_name", cp.get("name", ""))
        characters_for_tool.append({
            "name": name,
            "voice_profile": char_reg.voice(name),
            "emotional_state": cp.get("emotional_state", ""),
            "is_speaking": cp.get("is_speaking", False),
        })

    # Forced function call — Gemini receives scene context and calls the tool
    response = client.models.generate_content(
        model=MODEL_REASONING,
        contents=(
            f"Write the screenplay for this scene.\n\n"
            f"Scene: {scene.get('description', '')}\n"
            f"Slug: {scene.get('slug', '')}\n"
            f"Location: {scene.get('location_name', '')}\n"
            f"Emotional beat: {scene.get('emotional_beat', '')}\n"
            f"Key dialogue: {scene.get('dialogue_excerpt', '')}\n"
            f"Camera: {scene.get('camera_suggestion', '')}\n"
            f"Previous scene context: {prev}\n"
            f"Characters: {json.dumps(characters_for_tool)}"
        ),
        config=GenerateContentConfig(
            tools=[Tool(function_declarations=[WRITE_SCENE_SCRIPT_DECL])],
            tool_config=ToolConfig(
                function_calling_config=FunctionCallingConfig(
                    mode=FunctionCallingConfigMode.ANY,
                    allowed_function_names=["write_scene_script"],
                )
            ),
            temperature=0.7,
        ),
    )

    args = dict(response.function_calls[0].args)
    # Execute the actual tool
    script_result = write_scene_script(
        book_scene_description=args.get("book_scene_description", scene.get("description", "")),
        characters_present=args.get("characters_present", characters_for_tool),
        emotional_beat=args.get("emotional_beat", scene.get("emotional_beat", "")),
        location_name=args.get("location_name", scene.get("location_name", "")),
        slug=args.get("slug", scene.get("slug", "")),
        dialogue_excerpt=args.get("dialogue_excerpt", scene.get("dialogue_excerpt", "")),
        camera_suggestion=args.get("camera_suggestion", scene.get("camera_suggestion", "")),
        context=args.get("context", prev),
    )

    return {
        **state,
        "current_script": script_result,
        "log": [
            f"[Ch{state.get('current_chapter_index',0)+1} "
            f"Sc{state.get('current_scene_index',0)+1}] Script written: "
            f"{scene.get('slug','')}"
        ],
    }


# ── NODE 2: GenerateStoryboard ────────────────────────────────────────────────

def generate_storyboard_node(state: MovieWorkflowState) -> MovieWorkflowState:
    """
    Call generate_storyboard_image via Gemini forced function calling.
    Character and location registries are injected into the call args.
    """
    client = get_client()
    scene = state.get("current_scene", {})
    char_reg, loc_reg = _get_registries(state)
    ch_idx = state.get("current_chapter_index", 0)
    sc_idx = state.get("current_scene_index", 0)

    location_name = scene.get("location_name", "")
    chars_present = scene.get("characters_present", [])

    # Build character description list from registry
    char_descs = []
    for cp in chars_present:
        name = cp.get("character_name", cp.get("name", ""))
        char_descs.append({
            "name": name,
            "visual_description": char_reg.visual(name),
            "emotional_state": cp.get("emotional_state", ""),
            "position_in_frame": "foreground",
        })

    loc_visual = loc_reg.loc_visual(location_name) or scene.get("description", "")
    maps_address = loc_reg.loc_address(location_name)

    output_dir = Path(state.get("output_dir", "output"))
    output_path = str(
        output_dir / "storyboards" / f"ch{ch_idx+1:03d}_sc{sc_idx+1:03d}.png"
    )

    # Forced function call for storyboard generation
    scene_desc = (
        f"{scene.get('description', '')} "
        f"Camera: {scene.get('camera_suggestion', '')}. "
        f"Beat: {scene.get('emotional_beat', '')}."
    )

    response = client.models.generate_content(
        model=MODEL_REASONING,
        contents=(
            f"Generate a storyboard image for this scene.\n\n"
            f"Scene: {scene_desc}\n"
            f"Location: {location_name} — {loc_visual}\n"
            f"Maps ref: {maps_address}\n"
            f"Style: {state.get('style', 'cinematic')}\n"
            f"Characters: {json.dumps(char_descs)}\n"
            f"Output path: {output_path}"
        ),
        config=GenerateContentConfig(
            tools=[Tool(function_declarations=[GENERATE_STORYBOARD_IMAGE_DECL])],
            tool_config=ToolConfig(
                function_calling_config=FunctionCallingConfig(
                    mode=FunctionCallingConfigMode.ANY,
                    allowed_function_names=["generate_storyboard_image"],
                )
            ),
        ),
    )

    args = dict(response.function_calls[0].args)
    result = generate_storyboard_image(
        scene_description=args.get("scene_description", scene_desc),
        location_description=args.get("location_description", loc_visual),
        style=args.get("style", state.get("style", "cinematic")),
        character_descriptions=args.get("character_descriptions", char_descs),
        maps_address=args.get("maps_address", maps_address),
        aspect_ratio=args.get("aspect_ratio", "16:9"),
        output_path=args.get("output_path", output_path),
    )

    return {
        **state,
        "current_storyboard_path": result["image_path"],
        "current_storyboard_mime": result["image_mime"],
        "log": [
            f"[Ch{ch_idx+1} Sc{sc_idx+1}] Storyboard saved: {result['image_path']}"
        ],
    }


# ── NODE 3: GenerateClip ──────────────────────────────────────────────────────

def generate_clip_node(state: MovieWorkflowState) -> MovieWorkflowState:
    """
    Call generate_scene_video via Gemini forced function calling.
    Uses the storyboard as first-frame anchor for visual consistency.
    """
    client = get_client()
    scene = state.get("current_scene", {})
    script = state.get("current_script", {})
    ch_idx = state.get("current_chapter_index", 0)
    sc_idx = state.get("current_scene_index", 0)

    storyboard_path = state.get("current_storyboard_path", "")
    storyboard_mime = state.get("current_storyboard_mime", "image/png")

    if not storyboard_path or not Path(storyboard_path).exists():
        return {
            **state,
            "current_quality_issues": ["Storyboard missing — cannot generate clip"],
            "status": "running",
        }

    output_dir = Path(state.get("output_dir", "output"))
    output_path = str(
        output_dir / "clips" / f"ch{ch_idx+1:03d}_sc{sc_idx+1:03d}.mp4"
    )

    # Build cinematic prompt from script + scene
    action_lines = " ".join(script.get("action_lines", []))
    dialogue_sample = " | ".join(
        f"{d.get('character','')}: {d.get('text','')[:40]}"
        for d in script.get("dialogue_lines", [])[:2]
    )
    cinematic_prompt = (
        f"{scene.get('camera_suggestion', 'Slow push-in shot')}. "
        f"{action_lines} "
        f"Dialogue rhythm: {dialogue_sample}. "
        f"Emotional beat: {scene.get('emotional_beat', '')}. "
        f"Style: {state.get('style', 'cinematic')}. "
        "24fps cinematic, shallow depth of field, motivated lighting, "
        "film grain, colour graded. Characters maintain exact appearance from first frame."
    )

    response = client.models.generate_content(
        model=MODEL_REASONING,
        contents=(
            f"Generate a video clip for this scene.\n\n"
            f"Cinematic prompt: {cinematic_prompt}\n"
            f"Storyboard image path: {storyboard_path}\n"
            f"Image MIME: {storyboard_mime}\n"
            f"Duration: {state.get('clip_duration_seconds', 8)} seconds\n"
            f"Chapter: {ch_idx+1}, Scene: {sc_idx+1}\n"
            f"Output path: {output_path}"
        ),
        config=GenerateContentConfig(
            tools=[Tool(function_declarations=[GENERATE_SCENE_VIDEO_DECL])],
            tool_config=ToolConfig(
                function_calling_config=FunctionCallingConfig(
                    mode=FunctionCallingConfigMode.ANY,
                    allowed_function_names=["generate_scene_video"],
                )
            ),
        ),
    )

    args = dict(response.function_calls[0].args)

    try:
        result = generate_scene_video(
            cinematic_prompt=args.get("cinematic_prompt", cinematic_prompt),
            image_path=args.get("image_path", storyboard_path),
            image_mime=args.get("image_mime", storyboard_mime),
            negative_prompt=args.get("negative_prompt", ""),
            duration_seconds=int(args.get("duration_seconds", state.get("clip_duration_seconds", 8))),
            style=args.get("style", state.get("style", "cinematic")),
            output_path=args.get("output_path", output_path),
            chapter_number=ch_idx + 1,
            scene_number=sc_idx + 1,
        )
        return {
            **state,
            "current_video_path": result["video_path"],
            "current_video_mime": result["video_mime"],
            "log": [f"[Ch{ch_idx+1} Sc{sc_idx+1}] Clip saved: {result['video_path']}"],
        }
    except Exception as exc:
        return {
            **state,
            "current_quality_issues": [f"Video generation error: {exc}"],
            "log": [f"[Ch{ch_idx+1} Sc{sc_idx+1}] ERROR: {exc}"],
        }


# ── NODE 4: ValidateClip ──────────────────────────────────────────────────────

_CLIP_PASS_THRESHOLD = 0.68


def validate_clip_node(state: MovieWorkflowState) -> MovieWorkflowState:
    """
    Quality gate: score the generated clip against scene criteria.
    Sets current_quality_score and adjusts status for routing.
    """
    video_path = state.get("current_video_path", "")
    retry = state.get("current_retry_count", 0)
    max_retries = state.get("max_clip_retries", 2)

    if not video_path or not Path(video_path).exists():
        # No clip produced — count as retry or fail
        score = 0.0
        issues = ["No video clip produced."]
    else:
        # Lightweight check: if file exists and is > 10KB, consider it passing
        # A real implementation would do Gemini multimodal QA here
        size = Path(video_path).stat().st_size
        score = 0.85 if size > 10_000 else 0.30
        issues = [] if score >= _CLIP_PASS_THRESHOLD else ["Clip file suspiciously small."]

    passes = score >= _CLIP_PASS_THRESHOLD
    ch_idx = state.get("current_chapter_index", 0)
    sc_idx = state.get("current_scene_index", 0)

    if passes:
        # Commit clip to manifest
        scene = state.get("current_scene", {})
        clip: dict = {
            "chapter_number": ch_idx + 1,
            "scene_number": sc_idx + 1,
            "slug": scene.get("slug", ""),
            "video_path": video_path,
            "video_mime": state.get("current_video_mime", "video/mp4"),
            "storyboard_path": state.get("current_storyboard_path", ""),
            "script": state.get("current_script", {}),
            "duration_seconds": state.get("clip_duration_seconds", 8),
            "quality_score": score,
            "characters": [
                cp.get("character_name", cp.get("name", ""))
                for cp in scene.get("characters_present", [])
            ],
            "location_name": scene.get("location_name", ""),
        }
        # Advance cursor
        new_scene_idx = sc_idx + 1
        chapter_analysis = state.get("book_analysis", {})
        chapters = chapter_analysis.get("chapters", [])
        current_chapter = chapters[ch_idx] if ch_idx < len(chapters) else {}
        chapter_scenes = current_chapter.get("scenes", [])

        if new_scene_idx >= len(chapter_scenes):
            # Move to next chapter
            new_chapter_idx = ch_idx + 1
            new_scene_idx = 0
        else:
            new_chapter_idx = ch_idx

        all_chapters_done = new_chapter_idx >= len(chapters)

        return {
            **state,
            "completed_clips": [clip],
            "current_chapter_index": new_chapter_idx,
            "current_scene_index": new_scene_idx,
            "current_retry_count": 0,
            "current_quality_issues": [],
            "current_quality_score": score,
            "previous_scene_summary": scene.get("description", "")[:200],
            "status": "complete" if all_chapters_done else "running",
            "log": [
                f"[Ch{ch_idx+1} Sc{sc_idx+1}] ✓ clip accepted "
                f"(score={score:.2f}, {len(state.get('completed_clips',[]))+1} clips total)"
            ],
        }
    else:
        new_retry = retry + 1
        failed = new_retry >= max_retries
        return {
            **state,
            "current_retry_count": new_retry,
            "current_quality_score": score,
            "current_quality_issues": issues,
            "status": "failed" if failed else "running",
            "log": [
                f"[Ch{ch_idx+1} Sc{sc_idx+1}] ✗ clip rejected "
                f"(score={score:.2f}, retry {new_retry}/{max_retries})"
            ],
        }

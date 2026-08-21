"""
cinema_studio/tools/function_tools.py
─────────────────────────────────────────────────────────────────────────────
All three core generative operations exposed as Gemini FunctionDeclarations.

  generate_storyboard_image   — Imagen 3 / Gemini image gen
  generate_scene_video        — Gemini Omni Flash video gen
  write_scene_script          — Gemini screenplay writer (forced structured output)

Each is defined twice:
  1. As a FunctionDeclaration for the Gemini tool-calling schema
  2. As a real Python implementation that executes when Gemini calls it

The LangGraph nodes import the implementations.  The orchestrator agent
imports the FunctionDeclarations to give Gemini the tool palette.
"""

from __future__ import annotations

import time
from pathlib import Path

from google.genai import types
from google.genai.types import FunctionDeclaration

from cinema_studio.config import MODEL_IMAGE_GEN, MODEL_REASONING, get_client


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 — generate_storyboard_image
# ─────────────────────────────────────────────────────────────────────────────

GENERATE_STORYBOARD_IMAGE_DECL = FunctionDeclaration(
    name="generate_storyboard_image",
    description=(
        "Generate a high-fidelity storyboard image for a single film scene. "
        "The image serves as the visual anchor for video generation. "
        "Characters are rendered with strict visual consistency — always pass "
        "character_descriptions so the same person looks the same across all scenes."
    ),
    parameters={
        "type": "object",
        "required": ["scene_description", "location_description", "style"],
        "properties": {
            "scene_description": {
                "type": "string",
                "description": (
                    "What is happening in the scene: action, atmosphere, camera angle. "
                    "100–200 words."
                ),
            },
            "character_descriptions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "visual_description": {
                            "type": "string",
                            "description": "Full appearance string from character registry",
                        },
                        "emotional_state": {"type": "string"},
                        "position_in_frame": {
                            "type": "string",
                            "description": "foreground left | centre | background right | etc.",
                        },
                    },
                },
                "description": "All characters visible in this scene with their exact appearance.",
            },
            "location_description": {
                "type": "string",
                "description": "Visual description of the setting — architecture, lighting, time of day.",
            },
            "maps_address": {
                "type": "string",
                "description": (
                    "Real-world address from Google Maps grounding "
                    "(e.g. 'Eiffel Tower, Paris'). Used to pull architectural reference."
                ),
            },
            "style": {
                "type": "string",
                "enum": ["cinematic", "documentary", "noir", "surreal", "horror", "romantic"],
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["16:9", "2.39:1", "4:3", "1:1"],
                "description": "Film aspect ratio for the frame.",
            },
            "output_path": {
                "type": "string",
                "description": "Absolute path where the generated image should be saved.",
            },
        },
    },
)


def generate_storyboard_image(
    scene_description: str,
    location_description: str,
    style: str = "cinematic",
    character_descriptions: list[dict] | None = None,
    maps_address: str = "",
    aspect_ratio: str = "16:9",
    output_path: str = "",
) -> dict:
    """
    Implementation of the generate_storyboard_image tool.

    Returns a dict with:
      image_path   — path to saved PNG
      image_bytes  — raw bytes (kept in memory for video gen anchor)
      image_mime   — "image/png"
      prompt_used  — the full prompt sent to the model
    """
    client = get_client()

    # Build character block
    char_block = ""
    if character_descriptions:
        char_lines = []
        for ch in character_descriptions:
            name = ch.get("name", "Character")
            visual = ch.get("visual_description", "")
            emotion = ch.get("emotional_state", "")
            pos = ch.get("position_in_frame", "")
            char_lines.append(
                f"{name} ({pos}): {visual}. Current emotion: {emotion}."
            )
        char_block = "Characters in frame:\n" + "\n".join(char_lines) + "\n\n"

    maps_ref = f"Real-world location reference: {maps_address}. " if maps_address else ""

    prompt = (
        f"Cinematic storyboard image. {style.capitalize()} style. "
        f"Aspect ratio {aspect_ratio}. "
        f"Scene: {scene_description}\n\n"
        f"Setting: {location_description}\n\n"
        f"{maps_ref}"
        f"{char_block}"
        "Render as a photorealistic film still. "
        "Professional cinematography, motivated lighting, shallow depth of field. "
        "No text overlays. No watermarks. No cartoon style."
    )

    response = client.models.generate_content(
        model=MODEL_IMAGE_GEN,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        ),
    )

    image_part = response.candidates[0].content.parts[0]
    image_bytes: bytes = image_part.inline_data.data
    image_mime: str = image_part.inline_data.mime_type or "image/png"

    if not output_path:
        output_path = f"output/storyboards/storyboard_{int(time.time())}.png"

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(image_bytes)

    return {
        "image_path": str(out),
        "image_bytes": image_bytes,
        "image_mime": image_mime,
        "prompt_used": prompt,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 — generate_scene_video
# ─────────────────────────────────────────────────────────────────────────────

GENERATE_SCENE_VIDEO_DECL = FunctionDeclaration(
    name="generate_scene_video",
    description=(
        "Generate a 5–15 second high-fidelity video clip for one film scene. "
        "Requires a storyboard image as the first-frame anchor for visual consistency. "
        "The clip is saved to disk and its path returned for later assembly."
    ),
    parameters={
        "type": "object",
        "required": ["cinematic_prompt", "image_path", "image_mime"],
        "properties": {
            "cinematic_prompt": {
                "type": "string",
                "description": (
                    "Full cinematic video generation prompt — 150–250 words including "
                    "camera movement, character motion, lighting, colour grade, "
                    "atmosphere, and sound design hints."
                ),
            },
            "image_path": {
                "type": "string",
                "description": "Path to the storyboard image used as first frame.",
            },
            "image_mime": {"type": "string", "description": "MIME of the storyboard image."},
            "negative_prompt": {
                "type": "string",
                "description": "What to explicitly avoid: artifacts, distortions, wrong characters, etc.",
            },
            "duration_seconds": {
                "type": "integer",
                "description": "Target clip length 5–15 seconds.",
            },
            "style": {"type": "string"},
            "output_path": {
                "type": "string",
                "description": "Absolute path where the video clip should be saved.",
            },
            "chapter_number": {"type": "integer"},
            "scene_number": {"type": "integer"},
        },
    },
)


def generate_scene_video(
    cinematic_prompt: str,
    image_path: str,
    image_mime: str,
    negative_prompt: str = "",
    duration_seconds: int = 8,
    style: str = "cinematic",
    output_path: str = "",
    chapter_number: int = 0,
    scene_number: int = 0,
) -> dict:
    """
    Implementation of generate_scene_video.

    Returns:
      video_path   — path to saved MP4
      video_bytes  — raw bytes
      video_mime   — "video/mp4"
    """
    client = get_client()

    image_bytes = Path(image_path).read_bytes()

    full_prompt = (
        f"{cinematic_prompt}\n\n"
        f"Style: {style}. Duration: {duration_seconds} seconds. "
        "Start from the provided storyboard image as the first frame. "
        "Maintain exact character appearances, location, and lighting from the image. "
        "Smooth cinematic motion. No jump cuts. Professional grade."
    )
    if negative_prompt:
        full_prompt += f"\n\nAvoid: {negative_prompt}."

    if not output_path:
        output_path = (
            f"output/clips/ch{chapter_number:03d}_sc{scene_number:03d}_{int(time.time())}.mp4"
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    response = client.models.generate_content(
        model="gemini-2.0-flash-preview-image-generation",
        contents=types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=image_bytes, mime_type=image_mime),
                types.Part.from_text(text=full_prompt),
            ],
        ),
        config=types.GenerateContentConfig(response_modalities=["VIDEO"]),
    )

    video_part = response.candidates[0].content.parts[0]
    video_bytes: bytes = video_part.inline_data.data
    video_mime: str = video_part.inline_data.mime_type or "video/mp4"

    out.write_bytes(video_bytes)

    return {
        "video_path": str(out),
        "video_bytes": video_bytes,
        "video_mime": video_mime,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3 — write_scene_script
# ─────────────────────────────────────────────────────────────────────────────

WRITE_SCENE_SCRIPT_DECL = FunctionDeclaration(
    name="write_scene_script",
    description=(
        "Write the full screenplay for a single scene — dialogue, action lines, "
        "camera directions — grounded by the book's scene description and character voices. "
        "Returns structured script data for TTS and captioning."
    ),
    parameters={
        "type": "object",
        "required": ["book_scene_description", "characters_present", "emotional_beat"],
        "properties": {
            "book_scene_description": {
                "type": "string",
                "description": "The scene description extracted from the book.",
            },
            "characters_present": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "voice_profile": {"type": "string"},
                        "emotional_state": {"type": "string"},
                        "is_speaking": {"type": "boolean"},
                    },
                },
                "description": "Characters with voice profiles for dialogue assignment.",
            },
            "location_name": {"type": "string"},
            "slug": {
                "type": "string",
                "description": "INT./EXT. LOCATION - DAY/NIGHT",
            },
            "emotional_beat": {"type": "string"},
            "dialogue_excerpt": {
                "type": "string",
                "description": "Key dialogue from the book to preserve verbatim.",
            },
            "camera_suggestion": {"type": "string"},
            "context": {
                "type": "string",
                "description": "What happened in the previous scene (for continuity).",
            },
        },
    },
)


def write_scene_script(
    book_scene_description: str,
    characters_present: list[dict],
    emotional_beat: str,
    location_name: str = "",
    slug: str = "",
    dialogue_excerpt: str = "",
    camera_suggestion: str = "",
    context: str = "",
) -> dict:
    """
    Implementation of write_scene_script.

    Returns:
      slug             — formatted slug line
      action_lines     — list of action description strings
      dialogue_lines   — list of {character, text, parenthetical} dicts
      full_fountain    — Fountain-format string
    """
    client = get_client()

    char_block = "\n".join(
        f"- {c['name']}: voice={c.get('voice_profile','')}, "
        f"emotion={c.get('emotional_state','')}, speaking={c.get('is_speaking', False)}"
        for c in characters_present
    )

    prompt = f"""
Write a production-ready screenplay scene.

SLUG: {slug or f'INT./EXT. {location_name.upper()} - DAY'}
EMOTIONAL BEAT: {emotional_beat}
CAMERA: {camera_suggestion}
CONTEXT (previous scene): {context}

BOOK SOURCE:
{book_scene_description}

KEY DIALOGUE TO PRESERVE:
{dialogue_excerpt}

CHARACTERS:
{char_block}

Output as JSON with these exact keys:
- slug: string
- action_lines: array of strings (each is one action paragraph)
- dialogue_lines: array of objects with keys: character (string), parenthetical (string), text (string)
- full_fountain: the complete Fountain-format scene as a single string

Rules:
- Preserve the emotional beat precisely
- Every dialogue line must match the character's voice profile
- Action lines must be visual — describe what the camera sees, not what characters think
- Keep under 3 pages (3 minutes of screen time)
"""

    response = client.models.generate_content(
        model=MODEL_REASONING,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    import json
    data = json.loads(response.text)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry — all declarations in one list for agent config
# ─────────────────────────────────────────────────────────────────────────────

ALL_TOOL_DECLARATIONS = [
    GENERATE_STORYBOARD_IMAGE_DECL,
    GENERATE_SCENE_VIDEO_DECL,
    WRITE_SCENE_SCRIPT_DECL,
]

TOOL_IMPLEMENTATIONS = {
    "generate_storyboard_image": generate_storyboard_image,
    "generate_scene_video": generate_scene_video,
    "write_scene_script": write_scene_script,
}

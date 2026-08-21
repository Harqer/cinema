"""
cinema_studio/workflows/nodes/generate_video.py
─────────────────────────────────────────────────────────────────────────────
NODE: GenerateVideo
─────────────────────────────────────────────────────────────────────────────
ComfyUI analogy: "KSampler" — this is the core generation node.

Uses Gemini Omni Flash (gemini-2.0-flash-preview-image-generation) with the
multimodal Live API / video generation capability to turn the enhanced prompt
+ source image into a high-fidelity video clip.

The model receives:
  - The source image (for visual grounding / image-to-video)
  - The enhanced cinematic prompt
  - Style and duration hints

Inputs:  state.image_bytes, state.image_mime, state.enhanced_prompt,
         state.negative_prompt, state.style, state.duration_seconds, state.output_dir
Outputs: state.video_bytes, state.video_path, state.video_mime
"""

from __future__ import annotations

import time
from pathlib import Path

from google.genai import types

from cinema_studio.config import get_client
from cinema_studio.workflows.state import VideoWorkflowState


# Gemini Omni Flash video generation model
_VIDEO_MODEL = "gemini-2.0-flash-preview-image-generation"

# Fallback: if the preview model is unavailable, Lyria clip gives short video
_LYRIA_FALLBACK = "lyria-3-clip-preview"


def generate_video(state: VideoWorkflowState) -> VideoWorkflowState:
    """
    Generate a high-fidelity video from the source image + enhanced prompt
    using Gemini Omni Flash.

    The model is prompted with:
      - The source image (visual anchor)
      - The full enhanced cinematic prompt
      - Duration and style constraints
      - Negative prompt (what to avoid)

    On success writes the video bytes to disk and sets state.video_path.
    On failure sets state.error (the ValidateOutput node handles retries).
    """
    client = get_client()

    duration = state.get("duration_seconds", 5)
    style = state.get("style", "cinematic")
    enhanced_prompt = state.get("enhanced_prompt", state.get("base_prompt", ""))
    negative = state.get("negative_prompt", "")
    output_dir = Path(state.get("output_dir", "output/videos"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the generation prompt — the image is the "first frame" anchor
    full_prompt = (
        f"{enhanced_prompt}\n\n"
        f"Style: {style}. "
        f"Duration: approximately {duration} seconds. "
        f"Start from this exact image as the first frame. "
        f"Maintain visual consistency with the source image. "
        "Camera is mounted, smooth motion, professional cinematography. "
        "High dynamic range, sharp focus on subject. "
    )
    if negative:
        full_prompt += f"\n\nAvoid: {negative}."

    parts = [
        # Source image as visual anchor (image-to-video grounding)
        types.Part.from_bytes(
            data=state["image_bytes"],
            mime_type=state["image_mime"],
        ),
        types.Part.from_text(text=full_prompt),
    ]

    try:
        response = client.models.generate_content(
            model=_VIDEO_MODEL,
            contents=types.Content(role="user", parts=parts),
            config=types.GenerateContentConfig(
                response_modalities=["VIDEO"],
            ),
        )

        # Extract video bytes from inline data
        video_part = response.candidates[0].content.parts[0]
        video_bytes: bytes = video_part.inline_data.data
        video_mime: str = video_part.inline_data.mime_type or "video/mp4"

        ext = "mp4" if "mp4" in video_mime else video_mime.split("/")[-1]
        timestamp = int(time.time())
        out_path = output_dir / f"generated_{timestamp}.{ext}"
        out_path.write_bytes(video_bytes)

        return {
            **state,
            "video_bytes": video_bytes,
            "video_path": str(out_path),
            "video_mime": video_mime,
        }

    except Exception as exc:
        return {
            **state,
            "error": f"Video generation failed: {exc}",
            "quality_issues": [f"Generation error: {exc}"],
        }

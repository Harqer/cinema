"""
cinema_studio/workflows/nodes/analyze_image.py
─────────────────────────────────────────────────────────────────────────────
NODE: AnalyzeImage
─────────────────────────────────────────────────────────────────────────────
ComfyUI analogy: "CLIP Text Encode" + "Image Interrogate" — sends the image
to Gemini with a FunctionDeclaration schema and extracts rich scene metadata
as a structured dict.  Uses multimodal function calling (forced ANY mode) so
the output always conforms to the schema — no free-text parsing needed.

Inputs:  state.image_bytes, state.image_mime
Outputs: state.image_analysis, state.base_prompt
"""

from __future__ import annotations

from google.genai import types
from google.genai.types import (
    FunctionCallingConfig,
    FunctionCallingConfigMode,
    FunctionDeclaration,
    GenerateContentConfig,
    Tool,
    ToolConfig,
)

from cinema_studio.config import MODEL_REASONING, get_client
from cinema_studio.workflows.state import VideoWorkflowState


# ── Function schema (the "form" Gemini must fill in) ─────────────────────────

_ANALYZE_FN = FunctionDeclaration(
    name="describe_scene",
    description=(
        "Extract rich cinematographic metadata from an image to guide "
        "high-fidelity video generation."
    ),
    parameters={
        "type": "object",
        "required": [
            "subject",
            "setting",
            "mood",
            "time_of_day",
            "weather",
            "color_palette",
            "camera_angle",
            "camera_movement_suggestion",
            "lighting",
            "motion_elements",
            "suggested_video_prompt",
            "negative_elements",
        ],
        "properties": {
            "subject": {
                "type": "string",
                "description": "Primary subject(s) in the image — person, object, creature, etc.",
            },
            "setting": {
                "type": "string",
                "description": "Location and environment — urban street, forest, interior, etc.",
            },
            "mood": {
                "type": "string",
                "description": "Emotional tone — tense, peaceful, melancholic, euphoric, etc.",
            },
            "time_of_day": {
                "type": "string",
                "description": "Dawn, morning, midday, golden hour, dusk, night.",
            },
            "weather": {
                "type": "string",
                "description": "Clear, overcast, rainy, foggy, snowing, etc.",
            },
            "color_palette": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Dominant hex color codes (max 5), e.g. ['#2C3E50', '#E74C3C'].",
            },
            "camera_angle": {
                "type": "string",
                "description": "Eye-level, low-angle, high-angle, birds-eye, Dutch tilt, etc.",
            },
            "camera_movement_suggestion": {
                "type": "string",
                "description": (
                    "Best camera movement to bring this image to life — "
                    "slow push-in, dolly out, crane up, pan left, static, orbit, etc."
                ),
            },
            "lighting": {
                "type": "string",
                "description": "Hard/soft, direction, quality — rim light, backlit, flat, dramatic, etc.",
            },
            "motion_elements": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Elements that should naturally move in a video — "
                    "leaves, water, hair, smoke, crowd, etc."
                ),
            },
            "suggested_video_prompt": {
                "type": "string",
                "description": (
                    "A detailed, cinematic text prompt (100–200 words) for video generation "
                    "based on this image. Written for a diffusion video model."
                ),
            },
            "negative_elements": {
                "type": "string",
                "description": "Elements to avoid in the generated video — distortions, artifacts, etc.",
            },
        },
    },
)

_ANALYZE_TOOL = Tool(function_declarations=[_ANALYZE_FN])


# ── Node function ─────────────────────────────────────────────────────────────

def analyze_image(state: VideoWorkflowState) -> VideoWorkflowState:
    """
    Send the loaded image to Gemini and extract structured scene metadata
    using forced function calling (ANY mode — guarantees structured output).
    """
    client = get_client()

    response = client.models.generate_content(
        model=MODEL_REASONING,
        contents=types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    data=state["image_bytes"],
                    mime_type=state["image_mime"],
                ),
                types.Part.from_text(
                    text=(
                        "Analyse this image in depth as a cinematographer. "
                        "Call the describe_scene function with your analysis."
                    )
                ),
            ],
        ),
        config=GenerateContentConfig(
            tools=[_ANALYZE_TOOL],
            tool_config=ToolConfig(
                function_calling_config=FunctionCallingConfig(
                    mode=FunctionCallingConfigMode.ANY,
                    allowed_function_names=["describe_scene"],
                )
            ),
            temperature=0,
        ),
    )

    # Extract the structured args from the forced function call
    fn_call = response.function_calls[0]
    analysis: dict = dict(fn_call.args)

    # Build a draft base prompt from the analysis
    movement = analysis.get("camera_movement_suggestion", "slow push-in")
    subject = analysis.get("subject", "subject")
    setting = analysis.get("setting", "scene")
    mood = analysis.get("mood", "cinematic")
    motion = ", ".join(analysis.get("motion_elements", []))
    base_prompt = (
        f"{movement.capitalize()} shot. {analysis.get('suggested_video_prompt', '')} "
        f"Subject: {subject}. Setting: {setting}. Mood: {mood}. "
        f"Natural motion: {motion}. "
        f"Lighting: {analysis.get('lighting', '')}. "
        f"Ultra high definition, photorealistic, cinematic."
    )

    return {
        **state,
        "image_analysis": analysis,
        "base_prompt": base_prompt,
        "negative_prompt": analysis.get("negative_elements", ""),
    }

"""
cinema_studio/workflows/nodes/validate_output.py
─────────────────────────────────────────────────────────────────────────────
NODE: ValidateOutput
─────────────────────────────────────────────────────────────────────────────
ComfyUI analogy: "Preview Image" + quality gate — inspects the generated
video and decides whether it passes or needs to be regenerated.

Uses Gemini multimodal vision to sample the output and score it against the
original image + prompt criteria.  Returns a quality_score (0.0–1.0) and
a list of issues if score < threshold.

This node drives the conditional edge back to GenerateVideo (retry loop).

Inputs:  state.video_bytes, state.video_mime, state.image_bytes,
         state.image_analysis, state.enhanced_prompt
Outputs: state.quality_score, state.quality_issues, state.status
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
from pydantic import BaseModel

from cinema_studio.config import MODEL_REASONING, get_client
from cinema_studio.workflows.state import VideoWorkflowState


_QUALITY_THRESHOLD = 0.70   # score below this triggers a retry
_MAX_VIDEO_BYTES_FOR_REVIEW = 4 * 1024 * 1024  # send first 4 MB to Gemini


# ── Function schema for structured quality assessment ─────────────────────────

_QUALITY_FN = FunctionDeclaration(
    name="assess_video_quality",
    description="Assess the quality of a generated video against criteria.",
    parameters={
        "type": "object",
        "required": ["score", "issues", "passes"],
        "properties": {
            "score": {
                "type": "number",
                "description": "Overall quality score from 0.0 (worst) to 1.0 (best).",
            },
            "issues": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specific quality issues found (empty if none).",
            },
            "passes": {
                "type": "boolean",
                "description": "True if the video is acceptable for delivery.",
            },
            "improvement_hint": {
                "type": "string",
                "description": (
                    "One-sentence suggestion for the next generation attempt "
                    "if passes is False."
                ),
            },
        },
    },
)

_QUALITY_TOOL = Tool(function_declarations=[_QUALITY_FN])


def validate_output(state: VideoWorkflowState) -> VideoWorkflowState:
    """
    Score the generated video against the original image and prompt criteria.

    If the video bytes are unavailable (generation failed), score is 0.0 and
    the node immediately routes back to retry.
    """
    video_bytes = state.get("video_bytes")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    # ── Fast-fail: no video bytes ─────────────────────────────────────────────
    if not video_bytes:
        return {
            **state,
            "quality_score": 0.0,
            "quality_issues": ["No video bytes produced by generation step."],
            "status": "failed" if retry_count >= max_retries else "running",
        }

    client = get_client()

    # Sample the video (send first N bytes so we don't blow the context)
    sample = video_bytes[: _MAX_VIDEO_BYTES_FOR_REVIEW]
    video_mime = state.get("video_mime", "video/mp4")
    enhanced_prompt = state.get("enhanced_prompt", "")
    analysis = state.get("image_analysis", {})

    criteria = (
        f"Original image subject: {analysis.get('subject', 'unknown')}.\n"
        f"Expected setting: {analysis.get('setting', 'unknown')}.\n"
        f"Expected mood: {analysis.get('mood', 'unknown')}.\n"
        f"Expected camera movement: {analysis.get('camera_movement_suggestion', 'any')}.\n"
        f"Generation prompt used: {enhanced_prompt[:400]}"
    )

    response = client.models.generate_content(
        model=MODEL_REASONING,
        contents=types.Content(
            role="user",
            parts=[
                # Reference: original image
                types.Part.from_bytes(
                    data=state["image_bytes"],
                    mime_type=state["image_mime"],
                ),
                # Generated video sample
                types.Part.from_bytes(
                    data=sample,
                    mime_type=video_mime,
                ),
                types.Part.from_text(
                    text=(
                        "You are a quality control supervisor for AI-generated videos.\n"
                        "Compare the generated video against the original image and criteria below.\n\n"
                        f"Criteria:\n{criteria}\n\n"
                        "Assess: visual consistency with source image, prompt adherence, "
                        "absence of artifacts/distortions, motion smoothness, "
                        "cinematic quality. Call assess_video_quality with your findings."
                    )
                ),
            ],
        ),
        config=GenerateContentConfig(
            tools=[_QUALITY_TOOL],
            tool_config=ToolConfig(
                function_calling_config=FunctionCallingConfig(
                    mode=FunctionCallingConfigMode.ANY,
                    allowed_function_names=["assess_video_quality"],
                )
            ),
            temperature=0,
        ),
    )

    fn_args = dict(response.function_calls[0].args)
    score: float = float(fn_args.get("score", 0.5))
    issues: list[str] = list(fn_args.get("issues", []))
    passes: bool = bool(fn_args.get("passes", score >= _QUALITY_THRESHOLD))
    hint: str = fn_args.get("improvement_hint", "")

    # Append improvement hint to issues so EnhancePrompt can read it on retry
    if hint and not passes:
        issues.append(f"Improvement hint: {hint}")

    # Determine next status
    if passes:
        status = "complete"
    elif retry_count >= max_retries:
        status = "failed"
        issues.append(f"Max retries ({max_retries}) reached.")
    else:
        status = "running"  # triggers retry edge

    return {
        **state,
        "quality_score": score,
        "quality_issues": issues,
        "status": status,
        "retry_count": retry_count + (0 if passes else 1),
    }

"""
cinema_studio/workflows/nodes/enhance_prompt.py
─────────────────────────────────────────────────────────────────────────────
NODE: EnhancePrompt
─────────────────────────────────────────────────────────────────────────────
ComfyUI analogy: "Prompt Upscaler" / "Style Loader" — takes the base prompt
from AnalyzeImage and enriches it with:

  1. Parallel Web Search: looks up cinematography references and visual
     style guides for the detected mood/genre.
  2. Gemini reasoning: rewrites the prompt using those references into a
     high-fidelity video generation prompt.

Inputs:  state.base_prompt, state.image_analysis, state.style, state.user_prompt
Outputs: state.enhanced_prompt, state.parallel_research
"""

from __future__ import annotations

from google.genai import types

from cinema_studio.config import MODEL_REASONING, get_client
from cinema_studio.agents.research import _run_search, PARALLEL_MODE_RESEARCH
from cinema_studio.workflows.state import VideoWorkflowState


# ── Cinematography domain sources for Parallel ────────────────────────────────
_CINEMATIC_DOMAINS = [
    "nofilmschool.com",
    "studiobinder.com",
    "premiumbeat.com",
    "bhphotovideo.com",
    "theafilmmaker.com",
    "afi.com",
    "masterclass.com",
    "vimeo.com",
]


def enhance_prompt(state: VideoWorkflowState) -> VideoWorkflowState:
    """
    Enrich the base prompt with real-world cinematography references via
    Parallel Web Search, then use Gemini to write a polished video gen prompt.
    """
    analysis = state.get("image_analysis", {})
    mood = analysis.get("mood", "cinematic")
    style = state.get("style", "cinematic")
    user_hint = state.get("user_prompt", "")
    base = state.get("base_prompt", "")

    # ── Step 1: Parallel Web Search — cinematography references ───────────────
    search_query = (
        f"cinematography visual style guide for {style} {mood} films. "
        "Camera movements, color grading, lighting techniques, lens choices. "
        "Specific film references with distinctive visual style."
    )
    research = _run_search(
        query=search_query,
        mode=PARALLEL_MODE_RESEARCH,
        include_domains=_CINEMATIC_DOMAINS,
        max_results=8,
    )
    research_text = research.answer or ""

    # ── Step 2: Gemini prompt rewrite ─────────────────────────────────────────
    client = get_client()

    system_prompt = (
        "You are a world-class prompt engineer specialising in AI video generation. "
        "You write prompts for diffusion-based video models (like Veo / Sora) that produce "
        "cinematic, photorealistic, high-fidelity results. "
        "Your prompts are detailed (150–250 words), technically precise, and reference "
        "real filmmaking language: focal length, colour grade, film stock, motion blur, etc."
    )

    user_message = f"""
## Base prompt (from image analysis)
{base}

## User's additional direction
{user_hint if user_hint else "(none provided)"}

## Cinematography research (from live web via Parallel)
{research_text}

## Target style
{style}

## Task
Rewrite the base prompt into a premium video generation prompt.
Requirements:
- Start with the camera movement + shot type
- Describe subject motion in detail
- Include colour grade / LUT reference (e.g. "orange-teal grade", "Kodak 500T film stock")
- Specify frame rate feel (e.g. "24fps cinematic", "120fps slow-motion")
- Mention depth of field and bokeh
- End with quality tags: "8K, RAW, photorealistic, no artifacts, no distortion"
- DO NOT include actor names or copyrighted film titles

Output ONLY the final prompt text — no preamble, no explanation.
"""

    response = client.models.generate_content(
        model=MODEL_REASONING,
        contents=[
            types.Content(role="user", parts=[types.Part.from_text(text=system_prompt)]),
            types.Content(role="user", parts=[types.Part.from_text(text=user_message)]),
        ],
        config=types.GenerateContentConfig(temperature=0.7),
    )

    enhanced = (response.text or base).strip()

    return {
        **state,
        "enhanced_prompt": enhanced,
        "parallel_research": research_text,
    }

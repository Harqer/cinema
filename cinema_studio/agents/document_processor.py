"""
cinema_studio/agents/document_processor.py
─────────────────────────────────────────────────────────────────────────────
Document Processing Agent — ingests a full book PDF/text and extracts a
chapter-level, scene-level structured model that drives every downstream node.

Key additions over the original:
  - Chapter model with scene breakdown per chapter
  - CharacterAppearance — which characters appear in each scene, who is speaking
  - Visual appearance description per character (used for image-gen consistency)
  - Location enriched with real-world address hint (for Google Maps grounding)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from google.genai import types
from pydantic import BaseModel, Field

from cinema_studio.config import MODEL_LONG_CONTEXT, get_client


# ── Character models ──────────────────────────────────────────────────────────

class Character(BaseModel):
    name: str
    role: str = Field(description="protagonist | antagonist | supporting | minor")
    description: str
    arc: str = Field(description="How does this character change over the story?")
    # Visual consistency fields — used as seed prompts for image generation
    visual_description: str = Field(
        description=(
            "Exact visual appearance for image-gen consistency: age, build, "
            "hair colour/style, eye colour, skin tone, typical clothing, "
            "distinguishing features. Be specific enough to reproduce the "
            "same person across 100 different AI-generated images."
        )
    )
    voice_profile: str = Field(
        default="",
        description=(
            "Speaking style: pace, pitch, accent, signature phrases, "
            "emotional register. Used by TTS speaker mapping."
        ),
    )


class CharacterAppearance(BaseModel):
    character_name: str
    is_speaking: bool = Field(description="True if this character has dialogue in the scene")
    lines_preview: list[str] = Field(
        default_factory=list,
        description="First few words of each dialogue line (for speaker identification)",
    )
    emotional_state: str = Field(
        default="", description="Character's emotional state in this specific scene"
    )


# ── Location models ───────────────────────────────────────────────────────────

class Location(BaseModel):
    name: str
    description: str
    scenes: list[str] = Field(description="Brief list of scenes set here")
    period: str = Field(description="Historical or fictional time period, e.g. '1940s New York'")
    real_world_address_hint: str = Field(
        default="",
        description=(
            "Best-guess real-world address or landmark for Google Maps grounding, "
            "e.g. 'Grand Central Terminal, New York, NY' or 'Montmartre, Paris, France'. "
            "Leave empty for purely fictional locations."
        ),
    )
    visual_description: str = Field(
        default="",
        description=(
            "Detailed visual description of the location for image-gen: architecture style, "
            "materials, lighting conditions, time of day, weather typical."
        ),
    )


# ── Scene models (chapter-level breakdown) ────────────────────────────────────

class BookScene(BaseModel):
    """A single scene within a chapter — the atomic unit for video generation."""
    scene_number: int = Field(description="Sequential scene number within the chapter")
    slug: str = Field(description="INT./EXT. LOCATION - DAY/NIGHT (screenplay format)")
    location_name: str = Field(description="Matches a Location.name in the book analysis")
    description: str = Field(
        description=(
            "What physically happens in this scene — action, not feelings. "
            "50–150 words. Used directly as a video generation prompt."
        )
    )
    characters_present: list[CharacterAppearance]
    dialogue_excerpt: str = Field(
        default="",
        description="The most important 1–3 lines of dialogue in this scene",
    )
    emotional_beat: str = Field(
        description="The emotional shift that happens: e.g. 'hope → dread'"
    )
    camera_suggestion: str = Field(
        default="",
        description="Cinematographic approach: shot type, movement, lens feel",
    )
    vfx_notes: str = Field(default="")


class Chapter(BaseModel):
    """One chapter of the book, broken into filmable scenes."""
    chapter_number: int
    title: str
    summary: str = Field(description="2–3 sentence summary of what happens")
    scenes: list[BookScene]
    dominant_locations: list[str] = Field(
        description="Location names (from Location.name) used in this chapter"
    )
    dominant_characters: list[str] = Field(
        description="Character names present in this chapter"
    )
    act: int = Field(description="Which film act this chapter maps to: 1, 2, or 3")
    emotional_arc: str = Field(
        description="The emotional journey of this chapter as a whole"
    )


# ── Story-level models ────────────────────────────────────────────────────────

class StoryAct(BaseModel):
    act: int
    summary: str
    key_scenes: list[str]
    emotional_tone: str


class BookAnalysis(BaseModel):
    title: str
    author: str
    genre: list[str]
    logline: str = Field(description="One-sentence pitch for the adaptation")
    themes: list[str]
    time_period: str
    characters: list[Character]
    locations: list[Location]
    chapters: list[Chapter]
    story_structure: list[StoryAct]
    adaptation_notes: str = Field(
        description="Specific challenges or opportunities for a film adaptation"
    )
    total_scenes: int = Field(
        default=0,
        description="Total number of filmable scenes across all chapters"
    )


# ── Agent function ────────────────────────────────────────────────────────────

def process_book(source: str | Path | bytes, title: str | None = None) -> BookAnalysis:
    """
    Ingest a full book (PDF path, bytes, or plain text) and return a
    chapter-by-chapter, scene-by-scene :class:`BookAnalysis`.

    The output is the semantic backbone for the entire movie production
    pipeline — every downstream node (script writer, storyboard generator,
    video generator) reads from this structure.
    """
    client = get_client()

    parts: list[Any] = []

    if isinstance(source, Path) or (isinstance(source, str) and Path(source).exists()):
        pdf_bytes = Path(source).read_bytes()
        parts.append(types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"))
    elif isinstance(source, bytes):
        parts.append(types.Part.from_bytes(data=source, mime_type="application/pdf"))
    else:
        parts.append(types.Part.from_text(text=str(source)))

    title_hint = f'The book title is "{title}". ' if title else ""
    parts.append(
        types.Part.from_text(
            text=(
                f"{title_hint}"
                "Analyse this entire book and produce a complete chapter-by-chapter, "
                "scene-by-scene breakdown suitable for driving an AI film production pipeline.\n\n"
                "Requirements:\n"
                "1. Extract EVERY chapter. For each chapter, extract EVERY discrete scene "
                "   (a scene = a single location + continuous time block).\n"
                "2. For each character give a precise visual description that an image-gen model "
                "   can use to reproduce the same person consistently across 100+ generated images.\n"
                "3. For each location give a real-world address hint where possible "
                "   (e.g. 'Grand Central Terminal, New York' not just 'a train station').\n"
                "4. For each scene, list which characters are present, which are speaking, "
                "   and their exact emotional state in that moment.\n"
                "5. Map each chapter to a film act (1, 2, or 3).\n"
                "6. For adaptation_notes, specifically flag: unreliable narrators, "
                "   internal monologue, non-linear timeline, and propose cinematic solutions.\n"
                "7. total_scenes must equal the sum of scenes across all chapters."
            )
        )
    )

    response = client.models.generate_content(
        model=MODEL_LONG_CONTEXT,
        contents=types.Content(role="user", parts=parts),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BookAnalysis,
            thinking_config=types.ThinkingConfig(thinking_budget=16000),
        ),
    )

    analysis = BookAnalysis.model_validate_json(response.text)
    # Compute total_scenes if model didn't fill it
    if analysis.total_scenes == 0:
        analysis = analysis.model_copy(
            update={"total_scenes": sum(len(ch.scenes) for ch in analysis.chapters)}
        )
    return analysis


def analysis_to_markdown(analysis: BookAnalysis) -> str:
    """Render a BookAnalysis as a human-readable production brief."""
    lines = [
        f"# Production Brief: {analysis.title}",
        f"**Author:** {analysis.author}",
        f"**Genre:** {', '.join(analysis.genre)}",
        f"**Period:** {analysis.time_period}",
        f"**Total filmable scenes:** {analysis.total_scenes}",
        "",
        "## Logline",
        analysis.logline,
        "",
        "## Themes",
        *[f"- {t}" for t in analysis.themes],
        "",
        "## Characters",
    ]
    for c in analysis.characters:
        lines += [
            f"### {c.name} ({c.role})",
            c.description,
            f"*Visual:* {c.visual_description}",
            f"*Arc:* {c.arc}",
            "",
        ]

    lines += ["## Key Locations"]
    for loc in analysis.locations:
        lines += [
            f"### {loc.name}",
            f"*Period/Style:* {loc.period}",
            f"*Maps hint:* {loc.real_world_address_hint or '(fictional)'}",
            loc.description,
            "",
        ]

    lines += ["## Chapter Breakdown"]
    for ch in sorted(analysis.chapters, key=lambda c: c.chapter_number):
        lines += [
            f"### Chapter {ch.chapter_number}: {ch.title} (Act {ch.act})",
            ch.summary,
            f"*Scenes:* {len(ch.scenes)} | *Arc:* {ch.emotional_arc}",
            "",
        ]

    lines += ["## Adaptation Notes", analysis.adaptation_notes]
    return "\n".join(lines)

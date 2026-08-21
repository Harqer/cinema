"""Script Development Agent — generates a full screenplay from BookAnalysis.

Grounded by Parallel Web Search for film writing conventions and comparable films.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from google.genai import types

from cinema_studio.config import MODEL_REASONING, get_client
from cinema_studio.agents.document_processor import BookAnalysis
from cinema_studio.agents.research import research_comparable_films


# ── Output schema ─────────────────────────────────────────────────────────────

class SceneLine(BaseModel):
    type: str = Field(description="action | dialogue | parenthetical | transition")
    character: str | None = None  # set for dialogue lines
    text: str


class Scene(BaseModel):
    scene_number: int
    slug: str = Field(description="INT./EXT. LOCATION - DAY/NIGHT")
    act: int
    description: str = Field(description="What happens in this scene")
    lines: list[SceneLine]
    emotional_beat: str
    vfx_notes: str = Field(default="", description="Visual effects or special requirements")
    location_key: str = Field(description="Matches a Location.name from BookAnalysis")


class Screenplay(BaseModel):
    title: str
    written_by: str = "Agentic Cinema Studio"
    logline: str
    scenes: list[Scene]
    total_pages_estimate: int
    production_notes: str


# ── Agent function ─────────────────────────────────────────────────────────────

def develop_script(
    analysis: BookAnalysis,
    target_runtime_minutes: int = 110,
    rating_target: str = "PG-13",
) -> Screenplay:
    """
    Generate a complete screenplay from a :class:`BookAnalysis`.

    Calls Parallel Web Search first to ground the script in comparable films
    and current market expectations before any writing begins.

    Args:
        analysis:               Structured book analysis from document_processor.
        target_runtime_minutes: Target film length in minutes (default 110).
        rating_target:          MPAA rating target (G/PG/PG-13/R/NC-17).

    Returns:
        Full :class:`Screenplay` with all scenes and dialogue.
    """
    client = get_client()

    # ── Step 1: Ground in comparable films (Parallel Web Search) ─────────────
    genre_str = ", ".join(analysis.genre)
    market_research = research_comparable_films(analysis.title, genre_str)

    # ── Step 2: Build the screenplay generation prompt ────────────────────────
    characters_brief = "\n".join(
        f"- {c.name} ({c.role}): {c.description}. Arc: {c.arc}"
        for c in analysis.characters
    )
    locations_brief = "\n".join(
        f"- {loc.name} ({loc.period}): {loc.description}"
        for loc in analysis.locations
    )
    story_structure = "\n".join(
        f"Act {act.act} ({act.emotional_tone}): {act.summary}"
        for act in sorted(analysis.story_structure, key=lambda a: a.act)
    )

    prompt = f"""
You are an award-winning screenwriter adapting "{analysis.title}" by {analysis.author} into a feature film.

## Source Material
**Genre:** {genre_str}
**Period:** {analysis.time_period}
**Logline:** {analysis.logline}
**Themes:** {', '.join(analysis.themes)}

## Characters
{characters_brief}

## Locations
{locations_brief}

## Story Structure
{story_structure}

## Adaptation Notes from Story Analysis
{analysis.adaptation_notes}

## Market Research (from live web sources via Parallel)
{market_research.answer}

## Your Task
Write a complete, production-ready screenplay.

Constraints:
- Target runtime: {target_runtime_minutes} minutes (~{target_runtime_minutes} pages at 1 min/page)
- Rating target: {rating_target}
- Include proper slug lines (INT./EXT. LOCATION - DAY/NIGHT)
- All dialogue must be character-voice-consistent
- Mark VFX requirements clearly
- Each scene needs an emotional beat label

Output as structured JSON matching the Screenplay schema exactly.
Use proper screenplay formatting conventions informed by the comparable films above.
"""

    response = client.models.generate_content(
        model=MODEL_REASONING,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Screenplay,
            thinking_config=types.ThinkingConfig(thinking_budget=10000),
        ),
    )

    screenplay = Screenplay.model_validate_json(response.text)
    return screenplay


def screenplay_to_fountain(screenplay: Screenplay) -> str:
    """
    Export a :class:`Screenplay` to Fountain plain-text format.
    Fountain is the open standard for screenplays (readable by Final Draft, etc.).
    """
    lines = [
        f"Title: {screenplay.title}",
        f"Written by: {screenplay.written_by}",
        f"Logline: {screenplay.logline}",
        "",
        "===",
        "",
    ]

    for scene in screenplay.scenes:
        lines.append(scene.slug.upper())
        lines.append("")
        lines.append(scene.description)
        lines.append("")

        for line in scene.lines:
            if line.type == "action":
                lines.append(line.text)
                lines.append("")
            elif line.type == "dialogue" and line.character:
                lines.append(f"\t\t\t{line.character.upper()}")
                lines.append(f"\t\t{line.text}")
                lines.append("")
            elif line.type == "parenthetical":
                lines.append(f"\t\t\t({line.text})")
            elif line.type == "transition":
                lines.append(f"\t\t\t\t\t\t{line.text.upper()}:")
                lines.append("")

        if scene.vfx_notes:
            lines.append(f"/* VFX: {scene.vfx_notes} */")
            lines.append("")

    lines.append("")
    lines.append(f"/* Production Notes: {screenplay.production_notes} */")
    return "\n".join(lines)

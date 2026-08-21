"""Studio Orchestrator — ADK LlmAgent wiring all agents into one pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from cinema_studio.config import MODEL_REASONING, PROJECT_ID, LOCATION
from cinema_studio.agents.document_processor import (
    BookAnalysis,
    process_book,
    analysis_to_markdown,
)
from cinema_studio.agents.research import (
    research_comparable_films,
    verify_scene_accuracy,
    analyze_release_strategy,
    research_adaptation_rights,
    get_industry_news,
)
from cinema_studio.agents.script_writer import develop_script, screenplay_to_fountain
from cinema_studio.agents.location_scout import scout_locations, location_brief_markdown
from cinema_studio.agents.audio_production import produce_audio_package


# ── ADK Tool wrappers ─────────────────────────────────────────────────────────
# Each agent function is wrapped as an ADK FunctionTool so the orchestrator
# can call them as tool invocations and get structured state back.

def _tool_research_comparable_films(book_title: str, genre: str) -> str:
    """Research comparable book-to-film adaptations and market appetite via Parallel."""
    result = research_comparable_films(book_title, genre)
    return json.dumps({"research": result.answer, "sources": result.sources})


def _tool_verify_scene(scene_description: str, time_period: str) -> str:
    """Verify historical and technical accuracy of a scene via Parallel."""
    result = verify_scene_accuracy(scene_description, time_period)
    return json.dumps({"verdict": result.answer, "sources": result.sources})


def _tool_analyze_release(genre: str, comparable_films_csv: str) -> str:
    """Analyze distribution strategy for a genre via Parallel."""
    comps = [c.strip() for c in comparable_films_csv.split(",")]
    result = analyze_release_strategy(genre, comps)
    return json.dumps({"strategy": result.answer, "sources": result.sources})


def _tool_industry_news(topic: str) -> str:
    """Get current film industry news on a topic via Parallel."""
    result = get_industry_news(topic)
    return json.dumps({"news": result.answer, "sources": result.sources})


def _tool_adaptation_rights(book_title: str, author: str) -> str:
    """Check adaptation rights status for a book via Parallel."""
    result = research_adaptation_rights(book_title, author)
    return json.dumps({"rights_info": result.answer, "sources": result.sources})


# ── Orchestrator agent ─────────────────────────────────────────────────────────

ORCHESTRATOR_INSTRUCTION = """
You are the Studio Head of Agentic Cinema — an AI film production studio.

Your mission: given a book title and optionally a book PDF, coordinate all departments
to produce a complete, production-ready movie package.

## Your Tools (use them in this order for a full production run)
1. `research_adaptation_rights` — check if the IP is available before starting.
2. `research_comparable_films` — ground the project in current market data.
3. [Internal] `process_book` — extract characters, locations, story structure from the book.
4. [Internal] `develop_script` — generate the full screenplay (uses Parallel internally).
5. [Internal] `scout_locations` — find real-world filming locations (uses Google Maps).
6. `verify_scene_accuracy` — validate key scenes for historical/technical accuracy via Parallel.
7. [Internal] `produce_audio_package` — generate score and dialogue.
8. `analyze_release_strategy` — research distribution strategy via Parallel.

## Grounding philosophy
- Never make a creative or business decision without grounding it in live data first.
- Parallel Web Search is your primary source of truth for anything industry-facing.
- Google Maps is your primary source of truth for anything location-facing.

## Output format
For each production run, produce a JSON production bible with:
- book_analysis (characters, locations, story structure)
- screenplay (fountain format)
- location_package (scouted locations, shoot schedule)
- audio_manifest (score tracks, dialogue lines)
- market_research (comparable films, release strategy)
- rights_status
"""


def build_orchestrator() -> LlmAgent:
    """Build and return the Studio Orchestrator ADK agent."""
    tools = [
        FunctionTool(func=_tool_research_comparable_films),
        FunctionTool(func=_tool_verify_scene),
        FunctionTool(func=_tool_analyze_release),
        FunctionTool(func=_tool_industry_news),
        FunctionTool(func=_tool_adaptation_rights),
    ]

    agent = LlmAgent(
        name="studio_orchestrator",
        model=MODEL_REASONING,
        description="Master film production orchestrator — book PDF → full production package",
        instruction=ORCHESTRATOR_INSTRUCTION,
        tools=tools,
    )
    return agent


# ── Full pipeline (non-ADK, direct function call) ─────────────────────────────

def run_full_pipeline(
    book_source: str | Path | bytes,
    book_title: str,
    book_author: str,
    genre: str,
    output_dir: Path = Path("output"),
) -> dict[str, Any]:
    """
    Run the complete Agentic Cinema pipeline end-to-end.

    This is the direct (non-ADK-session) entry point for demos and CI runs.
    Returns a dictionary containing all production artefacts.

    Args:
        book_source:  PDF path/bytes or plain text of the book.
        book_title:   Title of the book.
        book_author:  Author name.
        genre:        Genre string (e.g. "thriller, drama").
        output_dir:   Root directory to write output files.

    Returns:
        Production bible as a dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] Checking adaptation rights for '{book_title}'...")
    rights = research_adaptation_rights(book_title, book_author)

    print(f"[2/7] Researching comparable films...")
    market = research_comparable_films(book_title, genre)

    print(f"[3/7] Processing book...")
    analysis = process_book(book_source, title=book_title)

    # Save analysis brief
    brief_path = output_dir / "production_brief.md"
    brief_path.write_text(analysis_to_markdown(analysis))

    print(f"[4/7] Writing screenplay...")
    screenplay = develop_script(analysis)

    # Save screenplay
    fountain_path = output_dir / "screenplay.fountain"
    fountain_path.write_text(screenplay_to_fountain(screenplay))

    print(f"[5/7] Scouting locations...")
    location_pkg = scout_locations(analysis, screenplay)

    # Save location brief
    location_path = output_dir / "location_brief.md"
    location_path.write_text(location_brief_markdown(location_pkg))

    print(f"[6/7] Verifying key scene accuracy...")
    # Verify the first 3 scenes with the most action description
    verification_results = []
    action_scenes = sorted(
        [s for s in screenplay.scenes if len(s.description) > 100],
        key=lambda s: -len(s.description),
    )[:3]
    for scene in action_scenes:
        result = verify_scene_accuracy(scene.description, analysis.time_period)
        verification_results.append(
            {"scene_number": scene.scene_number, "verdict": result.answer}
        )

    print(f"[7/7] Producing audio assets...")
    audio_pkg = produce_audio_package(
        screenplay=screenplay,
        genre=genre,
        output_dir=output_dir / "audio",
    )

    # ── Assemble production bible ─────────────────────────────────────────────
    bible = {
        "title": book_title,
        "author": book_author,
        "genre": genre,
        "rights_status": {
            "summary": rights.answer,
            "sources": rights.sources,
        },
        "market_research": {
            "comparable_films": market.answer,
            "sources": market.sources,
        },
        "book_analysis": analysis.model_dump(),
        "screenplay": {
            "title": screenplay.title,
            "total_pages_estimate": screenplay.total_pages_estimate,
            "scene_count": len(screenplay.scenes),
            "fountain_path": str(fountain_path),
        },
        "location_package": location_pkg.model_dump(),
        "continuity_checks": verification_results,
        "audio_package": {
            "score_tracks": len(audio_pkg.score_tracks),
            "dialogue_lines": len(audio_pkg.dialogue_lines),
            "total_assets": audio_pkg.total_assets,
        },
    }

    bible_path = output_dir / "production_bible.json"
    bible_path.write_text(json.dumps(bible, indent=2, default=str))
    print(f"\n✅  Production bible written to {bible_path}")

    return bible


# ── ADK session-based entry point ─────────────────────────────────────────────

async def run_orchestrator_session(user_message: str) -> str:
    """
    Run the Studio Orchestrator in an ADK session.

    This is the Agent Engine entry point — session state is managed by ADK.
    Use for interactive or streaming production runs.
    """
    agent = build_orchestrator()
    session_service = InMemorySessionService()

    runner = Runner(
        agent=agent,
        app_name="agentic_cinema",
        session_service=session_service,
    )

    session = await session_service.create_session(
        app_name="agentic_cinema",
        user_id="studio",
    )

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    final_response = ""
    async for event in runner.run_async(
        user_id="studio",
        session_id=session.id,
        new_message=content,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text or ""

    return final_response

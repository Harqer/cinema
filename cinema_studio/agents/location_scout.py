"""Location Scout Agent — grounds filming locations in real-world data via Google Maps."""

from __future__ import annotations

from pydantic import BaseModel, Field

from google.genai import types

from cinema_studio.config import MODEL_REASONING, get_client
from cinema_studio.agents.document_processor import BookAnalysis
from cinema_studio.agents.script_writer import Screenplay


# ── Output schema ─────────────────────────────────────────────────────────────

class ScoutedLocation(BaseModel):
    fictional_name: str = Field(description="Location name as it appears in the script")
    real_world_suggestion: str = Field(description="Real filming location suggestion")
    city: str
    country: str
    address_hint: str = Field(default="", description="Street or area for Maps lookup")
    why_it_works: str = Field(description="Why this location matches the scene requirements")
    permit_notes: str = Field(default="", description="Known permit complexity or seasonal constraints")
    scene_numbers: list[int] = Field(description="Scenes that would be shot here")


class ShootDay(BaseModel):
    day: int
    locations: list[str] = Field(description="ScoutedLocation.real_world_suggestion values")
    estimated_drive_minutes: int = Field(default=0)
    notes: str = Field(default="")


class LocationPackage(BaseModel):
    locations: list[ScoutedLocation]
    shoot_schedule: list[ShootDay]
    total_unique_locations: int
    geography_notes: str = Field(
        description="Overall geography and travel logistics notes for the production"
    )


# ── Agent function ─────────────────────────────────────────────────────────────

def scout_locations(analysis: BookAnalysis, screenplay: Screenplay) -> LocationPackage:
    """
    Suggest real-world filming locations for every location in the screenplay,
    grounded by the Google Maps Tool via ADK.

    The Google Maps grounding lets the model pull real place data — hours,
    accessibility, directions between locations — directly into the response.

    Args:
        analysis:   Book analysis with location descriptions.
        screenplay: Generated screenplay with scene slugs.

    Returns:
        :class:`LocationPackage` with scouted locations and a shoot schedule.
    """
    client = get_client()

    # Build slug list grouped by location key
    location_scenes: dict[str, list[int]] = {}
    for scene in screenplay.scenes:
        key = scene.location_key
        location_scenes.setdefault(key, []).append(scene.scene_number)

    locations_context = "\n".join(
        f"- **{loc.name}** ({loc.period}): {loc.description}. "
        f"Scenes: {', '.join(str(n) for n in location_scenes.get(loc.name, []))}"
        for loc in analysis.locations
    )

    prompt = f"""
You are a professional film location scout with deep knowledge of international filming locations.

## Film: "{analysis.title}"
**Period:** {analysis.time_period}
**Genre:** {', '.join(analysis.genre)}

## Required Fictional Locations (from screenplay)
{locations_context}

## Your Task
For each fictional location above:
1. Suggest a real-world filming location that visually and practically matches it.
2. Consider: period accuracy, accessibility, permit complexity, visual distinctiveness.
3. Look up real place information (accessibility, nearby facilities) using Maps.
4. Group locations geographically to minimise company moves.
5. Build an efficient shoot schedule (group nearby locations on the same day).

Practical constraints:
- Minimise international travel — prefer clusters in same city/region.
- Flag any locations with difficult permit situations (government buildings, private land, etc.).
- Note seasonal restrictions (weather, tourist crowds, daylight hours).

Output the full LocationPackage JSON.
"""

    response = client.models.generate_content(
        model=MODEL_REASONING,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_maps=types.ToolGoogleMaps())],
            response_mime_type="application/json",
            response_schema=LocationPackage,
        ),
    )

    return LocationPackage.model_validate_json(response.text)


def location_brief_markdown(pkg: LocationPackage) -> str:
    """Render a :class:`LocationPackage` as a production location brief."""
    lines = [
        "# Location Scout Report",
        f"**Total unique locations:** {pkg.total_unique_locations}",
        "",
        "## Geography & Logistics",
        pkg.geography_notes,
        "",
        "## Locations",
    ]
    for loc in pkg.locations:
        lines += [
            f"### {loc.fictional_name}",
            f"**Real suggestion:** {loc.real_world_suggestion}",
            f"**City/Country:** {loc.city}, {loc.country}",
            f"**Why it works:** {loc.why_it_works}",
        ]
        if loc.permit_notes:
            lines.append(f"**Permit notes:** {loc.permit_notes}")
        lines += [
            f"**Scenes:** {', '.join(str(n) for n in loc.scene_numbers)}",
            "",
        ]

    lines += ["## Proposed Shoot Schedule"]
    for day in pkg.shoot_schedule:
        lines += [
            f"### Day {day.day}",
            *[f"- {loc}" for loc in day.locations],
        ]
        if day.estimated_drive_minutes:
            lines.append(f"*Estimated travel: {day.estimated_drive_minutes} min*")
        if day.notes:
            lines.append(f"*Notes: {day.notes}*")
        lines.append("")

    return "\n".join(lines)

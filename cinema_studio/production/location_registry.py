"""
cinema_studio/production/location_registry.py
─────────────────────────────────────────────────────────────────────────────
Location Registry — Google Maps grounded location data.

For each location extracted from the book, fetches real-world place data
(place ID, coordinates, photo references, opening hours) via the Google Maps
Places API.  This data is injected into every storyboard and video prompt
so generated visuals match the real geography.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from google.genai import types

from cinema_studio.config import MODEL_REASONING, get_client


@dataclass
class LocationRecord:
    fictional_name: str
    real_world_hint: str          # from book analysis (address hint)
    visual_description: str       # from book analysis
    period: str                   # time period

    # Populated by Google Maps grounding
    maps_place_id: str = ""
    maps_display_name: str = ""
    maps_formatted_address: str = ""
    maps_coordinates: dict = field(default_factory=dict)   # {lat, lng}
    maps_photo_reference: str = ""
    maps_description: str = ""    # AI-synthesised description from Maps data
    maps_grounded: bool = False


class LocationRegistry:
    """
    Registry of all film locations, grounded in Google Maps data.

    Usage::

        registry = LocationRegistry.from_analysis(book_analysis)
        await registry.ground_all()   # fetches Maps data for every location

        desc = registry.visual_prompt_fragment("Baker Street Flat")
        # → "Baker Street, London NW1 6XE. Victorian terraced building..."
    """

    def __init__(self) -> None:
        self._by_name: dict[str, LocationRecord] = {}

    @classmethod
    def from_analysis(cls, analysis) -> "LocationRegistry":
        registry = cls()
        for loc in analysis.locations:
            record = LocationRecord(
                fictional_name=loc.name,
                real_world_hint=getattr(loc, "real_world_address_hint", ""),
                visual_description=getattr(loc, "visual_description", loc.description),
                period=loc.period,
            )
            registry._by_name[loc.name.lower()] = record
        return registry

    def get(self, name: str) -> Optional[LocationRecord]:
        return self._by_name.get(name.lower())

    def all_locations(self) -> list[LocationRecord]:
        return list(self._by_name.values())

    def ground_all(self) -> None:
        """
        Use Gemini with the Google Maps Tool to enrich every location that has a
        real-world address hint.  Populates maps_description and
        maps_formatted_address on each LocationRecord.
        """
        client = get_client()

        for record in self._by_name.values():
            if not record.real_world_hint or record.maps_grounded:
                continue
            self._ground_location(client, record)

    def _ground_location(self, client, record: LocationRecord) -> None:
        """
        Call Gemini with the Google Maps tool to pull real place data for one
        location and write it back onto the record.
        """
        prompt = (
            f"I need precise visual and architectural details about this real-world location "
            f"for use as a film shooting reference.\n\n"
            f"Location: {record.real_world_hint}\n"
            f"Time period context: {record.period}\n\n"
            f"Please look up this location using Maps and provide:\n"
            f"1. The exact formatted address\n"
            f"2. Architectural style and key visual features\n"
            f"3. Typical lighting conditions at different times of day\n"
            f"4. Any notable visual landmarks or distinctive features visible from the street\n"
            f"5. Seasonal considerations (foliage, weather, crowds)\n\n"
            f"Return a detailed description paragraph suitable for inclusion in an AI "
            f"image generation prompt."
        )

        try:
            response = client.models.generate_content(
                model=MODEL_REASONING,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_maps=types.ToolGoogleMaps())],
                ),
            )
            record.maps_description = response.text or ""
            record.maps_formatted_address = record.real_world_hint
            record.maps_grounded = True
        except Exception:
            # Maps grounding is best-effort — never block production
            record.maps_grounded = False

    def visual_prompt_fragment(self, name: str) -> str:
        """
        Return the full location description for injection into image-gen prompts.
        Prefers Maps-grounded description; falls back to book description.
        """
        record = self.get(name)
        if record is None:
            return f"Location: {name}"

        if record.maps_grounded and record.maps_description:
            return (
                f"Location: {record.fictional_name}. "
                f"Real-world reference: {record.maps_formatted_address}. "
                f"{record.maps_description}"
            )
        return (
            f"Location: {record.fictional_name}. "
            f"Period: {record.period}. "
            f"{record.visual_description}"
        )

    def to_dict(self) -> dict:
        return {
            name: {
                "fictional_name": r.fictional_name,
                "real_world_hint": r.real_world_hint,
                "maps_formatted_address": r.maps_formatted_address,
                "maps_grounded": r.maps_grounded,
                "period": r.period,
            }
            for name, r in self._by_name.items()
        }

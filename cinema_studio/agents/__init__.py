"""cinema_studio.agents sub-package."""

from cinema_studio.agents.document_processor import process_book, BookAnalysis
from cinema_studio.agents.research import (
    research_comparable_films,
    verify_scene_accuracy,
    analyze_release_strategy,
    get_industry_news,
    research_adaptation_rights,
)
from cinema_studio.agents.script_writer import develop_script, screenplay_to_fountain
from cinema_studio.agents.location_scout import scout_locations
from cinema_studio.agents.audio_production import produce_audio_package

__all__ = [
    "process_book",
    "BookAnalysis",
    "research_comparable_films",
    "verify_scene_accuracy",
    "analyze_release_strategy",
    "get_industry_news",
    "research_adaptation_rights",
    "develop_script",
    "screenplay_to_fountain",
    "scout_locations",
    "produce_audio_package",
]

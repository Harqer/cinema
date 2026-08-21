"""cinema_studio/production/__init__.py"""
from cinema_studio.production.character_registry import CharacterRegistry, CharacterRecord
from cinema_studio.production.location_registry import LocationRegistry, LocationRecord

__all__ = [
    "CharacterRegistry", "CharacterRecord",
    "LocationRegistry", "LocationRecord",
]

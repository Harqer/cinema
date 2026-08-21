"""
cinema_studio/production/character_registry.py
─────────────────────────────────────────────────────────────────────────────
Character Consistency Registry

Maintains a canonical visual description for every character across the entire
movie.  Every storyboard image prompt and video generation prompt queries this
registry so the same person looks identical in scene 1 and scene 87.

Also handles speaker identification — given a dialogue line, returns the
matched character and their TTS voice config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CharacterRecord:
    name: str
    role: str                      # protagonist | antagonist | supporting | minor
    visual_description: str        # The canonical image-gen appearance string
    voice_profile: str             # TTS voice style description
    tts_voice_name: str = ""       # Mapped Gemini TTS voice name (e.g. "Charon")
    appearances: list[int] = field(default_factory=list)   # scene numbers
    speaking_scenes: list[int] = field(default_factory=list)


# Default TTS voice assignments by role — override per character as needed
_ROLE_VOICE_DEFAULTS: dict[str, str] = {
    "protagonist": "Aoede",       # warm, expressive
    "antagonist": "Charon",       # deep, commanding
    "supporting": "Fenrir",       # neutral, clear
    "minor": "Kore",              # light, minimal
}


class CharacterRegistry:
    """
    Central store for character visual and voice consistency.

    Usage::

        registry = CharacterRegistry.from_analysis(book_analysis)

        # Get the prompt fragment to inject into every image generation call
        frag = registry.visual_prompt_fragment("Elizabeth Bennet")

        # Resolve who is speaking a given line
        speaker = registry.identify_speaker("I cannot bear another word of this!")
    """

    def __init__(self) -> None:
        self._by_name: dict[str, CharacterRecord] = {}

    @classmethod
    def from_analysis(cls, analysis) -> "CharacterRegistry":
        """
        Build a registry from a :class:`BookAnalysis`.

        Automatically assigns TTS voices based on role.
        """
        registry = cls()
        for char in analysis.characters:
            tts_voice = _ROLE_VOICE_DEFAULTS.get(char.role, "Fenrir")
            record = CharacterRecord(
                name=char.name,
                role=char.role,
                visual_description=char.visual_description,
                voice_profile=getattr(char, "voice_profile", ""),
                tts_voice_name=tts_voice,
            )
            registry._by_name[char.name.lower()] = record
        return registry

    def get(self, name: str) -> Optional[CharacterRecord]:
        return self._by_name.get(name.lower())

    def all_characters(self) -> list[CharacterRecord]:
        return list(self._by_name.values())

    def visual_prompt_fragment(self, name: str) -> str:
        """
        Return the visual description string to inject into image-gen prompts.
        Always use this — never write free-form character descriptions in prompts.
        """
        record = self.get(name)
        if record is None:
            return f"{name} (appearance unknown)"
        return f"{name}: {record.visual_description}"

    def scene_character_prompts(self, character_appearances: list) -> list[dict]:
        """
        Given a list of CharacterAppearance objects from BookScene, return the
        full character description dicts for the storyboard image tool call.
        """
        result = []
        for appearance in character_appearances:
            name = appearance.character_name
            record = self.get(name)
            visual = record.visual_description if record else "appearance unknown"
            result.append(
                {
                    "name": name,
                    "visual_description": visual,
                    "emotional_state": appearance.emotional_state,
                    "position_in_frame": "foreground centre",  # default; node can override
                }
            )
        return result

    def speaker_configs_for_scene(self, character_appearances: list) -> list[dict]:
        """
        Return TTS SpeakerVoiceConfig dicts for all speaking characters in a scene.
        """
        configs = []
        for appearance in character_appearances:
            if not appearance.is_speaking:
                continue
            name = appearance.character_name
            record = self.get(name)
            voice_name = record.tts_voice_name if record else "Fenrir"
            configs.append({"speaker": name, "voice_name": voice_name})
        return configs

    def identify_speaker(self, dialogue_line: str, candidates: list[str]) -> str:
        """
        Best-effort speaker identification from a dialogue line and candidate names.

        Used when the source text has unattributed dialogue — compares the line
        against each character's voice_profile description to pick the most likely
        speaker.  Falls back to the first candidate.
        """
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            return "NARRATOR"

        # Build a simple scoring heuristic based on voice_profile keywords
        line_lower = dialogue_line.lower()
        best = candidates[0]
        best_score = -1

        for name in candidates:
            record = self.get(name)
            if not record or not record.voice_profile:
                continue
            # Count keyword overlaps between the line and the voice profile
            profile_words = set(record.voice_profile.lower().split())
            line_words = set(line_lower.split())
            score = len(profile_words & line_words)
            if score > best_score:
                best_score = score
                best = name

        return best

    def record_appearance(self, name: str, scene_number: int, is_speaking: bool) -> None:
        record = self.get(name)
        if record:
            if scene_number not in record.appearances:
                record.appearances.append(scene_number)
            if is_speaking and scene_number not in record.speaking_scenes:
                record.speaking_scenes.append(scene_number)

    def to_dict(self) -> dict:
        return {
            name: {
                "role": r.role,
                "visual_description": r.visual_description,
                "voice_profile": r.voice_profile,
                "tts_voice_name": r.tts_voice_name,
                "appearances": r.appearances,
                "speaking_scenes": r.speaking_scenes,
            }
            for name, r in self._by_name.items()
        }

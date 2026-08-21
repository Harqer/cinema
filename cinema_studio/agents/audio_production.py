"""Audio Production Agent — score generation (Lyria 3) + dialogue synthesis (Gemini TTS)."""

from __future__ import annotations

import struct
import wave
from io import BytesIO
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from google.genai import types

from cinema_studio.config import (
    MODEL_LYRIA_CLIP,
    MODEL_LYRIA_TRACK,
    MODEL_TTS,
    get_client,
)
from cinema_studio.agents.script_writer import Scene, Screenplay


# ── Output schemas ─────────────────────────────────────────────────────────────

class AudioAsset(BaseModel):
    name: str
    description: str
    file_path: str = Field(default="")  # set after generation + save
    duration_seconds: float = Field(default=0.0)


class ScoreTrack(AudioAsset):
    scene_numbers: list[int]
    mood: str
    prompt_used: str


class DialogueLine(AudioAsset):
    scene_number: int
    character: str
    line_text: str
    voice_config: dict = Field(default_factory=dict)


class AudioPackage(BaseModel):
    score_tracks: list[ScoreTrack]
    dialogue_lines: list[DialogueLine]
    total_assets: int


# ── PCM helpers ───────────────────────────────────────────────────────────────

def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    """Convert raw 24kHz 16-bit mono PCM to a WAV container."""
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _save_audio(pcm_bytes: bytes, output_path: Path, sample_rate: int = 24000) -> None:
    """Save raw PCM bytes as a WAV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_pcm_to_wav(pcm_bytes, sample_rate))


# ── Score generation (Lyria 3) ────────────────────────────────────────────────

def generate_score_track(
    scene_description: str,
    mood: str,
    genre: str,
    duration_hint: str = "2-3 minutes",
    reference_image_bytes: bytes | None = None,
    output_dir: Path = Path("output/audio/score"),
    track_name: str = "score_track",
) -> ScoreTrack:
    """
    Generate a film score track using Lyria 3.

    Uses ``lyria-3-clip-preview`` when a reference image is provided (supports
    image conditioning), otherwise uses ``lyria-3-pro-preview`` for full tracks.

    Args:
        scene_description:    What the scene is about (feeds the music prompt).
        mood:                 Emotional tone (e.g. "tense", "romantic", "epic").
        genre:                Film genre for style guidance.
        duration_hint:        Natural language duration hint embedded in the prompt.
        reference_image_bytes: Optional image bytes for image-conditioned generation.
        output_dir:           Directory to save the generated WAV.
        track_name:           Base filename (without extension).

    Returns:
        :class:`ScoreTrack` with file_path set after saving.
    """
    client = get_client()

    music_prompt = (
        f"Film score for a {genre} film. "
        f"Scene: {scene_description}. "
        f"Mood: {mood}. "
        f"Duration: {duration_hint}. "
        "Orchestral instrumentation with cinematic dynamics. "
        "No vocals. No lyrics."
    )

    parts: list = [types.Part.from_text(text=music_prompt)]
    if reference_image_bytes:
        parts.insert(
            0,
            types.Part.from_bytes(data=reference_image_bytes, mime_type="image/jpeg"),
        )
        model = MODEL_LYRIA_CLIP
    else:
        model = MODEL_LYRIA_TRACK

    response = client.models.generate_content(
        model=model,
        contents=types.Content(role="user", parts=parts),
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
        ),
    )

    audio_data = response.candidates[0].content.parts[0].inline_data.data
    output_path = output_dir / f"{track_name}.wav"
    _save_audio(audio_data, output_path)

    return ScoreTrack(
        name=track_name,
        description=f"{mood} score for: {scene_description[:80]}",
        file_path=str(output_path),
        prompt_used=music_prompt,
        scene_numbers=[],  # caller fills this in
        mood=mood,
    )


# ── Dialogue synthesis (Gemini TTS) ──────────────────────────────────────────

# Character voice mappings — chosen from Gemini TTS voice catalogue.
# These are illustrative; the real system derives them from character analysis.
_DEFAULT_VOICE_MAP: dict[str, dict] = {
    "NARRATOR": {"voice_name": "Charon"},  # deep, authoritative
    "DEFAULT_MALE": {"voice_name": "Fenrir"},  # neutral male
    "DEFAULT_FEMALE": {"voice_name": "Aoede"},  # neutral female
}


def _voice_for_character(character_name: str, voice_map: dict | None = None) -> dict:
    """Look up the voice config for a character name."""
    vm = voice_map or _DEFAULT_VOICE_MAP
    key = character_name.upper()
    if key in vm:
        return vm[key]
    # Fall back to gendered heuristics — callers should override for real characters
    return vm.get("DEFAULT_MALE", {"voice_name": "Fenrir"})


def synthesise_dialogue(
    scene: Scene,
    voice_map: dict | None = None,
    output_dir: Path = Path("output/audio/dialogue"),
) -> list[DialogueLine]:
    """
    Synthesise all dialogue lines in a scene using Gemini TTS.

    For scenes with multiple speakers, uses the multi-speaker config so that
    the TTS model can blend voices within a single API call.

    Args:
        scene:      A :class:`Scene` from the screenplay.
        voice_map:  Mapping of CHARACTER_NAME → TTS voice config dict.
                    Falls back to defaults if not provided.
        output_dir: Directory to save generated WAV files.

    Returns:
        List of :class:`DialogueLine` with file paths set.
    """
    client = get_client()

    dialogue_lines_raw = [
        line for line in scene.lines if line.type == "dialogue" and line.character
    ]
    if not dialogue_lines_raw:
        return []

    # Collect unique speakers
    speakers = list({line.character for line in dialogue_lines_raw if line.character})

    # Build multi-speaker voice configs
    speaker_voice_configs = []
    for speaker in speakers:
        vc = _voice_for_character(speaker, voice_map)
        speaker_voice_configs.append(
            types.SpeakerVoiceConfig(
                speaker=speaker,
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=vc.get("voice_name", "Fenrir")
                    )
                ),
            )
        )

    # Build the TTS prompt with speaker tags
    tts_text_parts = []
    for line in dialogue_lines_raw:
        tts_text_parts.append(f"{line.character}: {line.text}")
    full_script = "\n".join(tts_text_parts)

    response = client.models.generate_content(
        model=MODEL_TTS,
        contents=full_script,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=speaker_voice_configs
                )
            ),
        ),
    )

    pcm_bytes = response.candidates[0].content.parts[0].inline_data.data
    output_path = output_dir / f"scene_{scene.scene_number:04d}_dialogue.wav"
    _save_audio(pcm_bytes, output_path)

    # Return one DialogueLine per original line (all backed by the same file)
    results = []
    for i, line in enumerate(dialogue_lines_raw):
        results.append(
            DialogueLine(
                name=f"scene_{scene.scene_number:04d}_line_{i:03d}",
                description=f"{line.character}: {line.text[:60]}",
                file_path=str(output_path),
                scene_number=scene.scene_number,
                character=line.character or "",
                line_text=line.text,
                voice_config=_voice_for_character(line.character or "", voice_map),
            )
        )
    return results


# ── Full audio production pipeline ────────────────────────────────────────────

def produce_audio_package(
    screenplay: Screenplay,
    genre: str,
    voice_map: dict | None = None,
    output_dir: Path = Path("output/audio"),
    score_every_n_scenes: int = 5,
) -> AudioPackage:
    """
    Run the full audio production pipeline for a screenplay.

    Generates:
    - One score track per group of ``score_every_n_scenes`` scenes.
    - Dialogue synthesis for every scene with speaking roles.

    Args:
        screenplay:           Full :class:`Screenplay`.
        genre:                Film genre string for score prompts.
        voice_map:            Character → TTS voice config overrides.
        output_dir:           Root directory for all audio output.
        score_every_n_scenes: How many scenes to group per score track.

    Returns:
        :class:`AudioPackage` with all generated assets.
    """
    score_tracks: list[ScoreTrack] = []
    dialogue_lines: list[DialogueLine] = []

    # ── Score tracks ──────────────────────────────────────────────────────────
    scenes_by_act: dict[int, list[Scene]] = {}
    for scene in screenplay.scenes:
        scenes_by_act.setdefault(scene.act, []).append(scene)

    for act, scenes in sorted(scenes_by_act.items()):
        # Group scenes in chunks
        for chunk_idx in range(0, len(scenes), score_every_n_scenes):
            chunk = scenes[chunk_idx : chunk_idx + score_every_n_scenes]
            mood = chunk[0].emotional_beat
            scene_desc = "; ".join(s.description[:60] for s in chunk[:3])
            track = generate_score_track(
                scene_description=scene_desc,
                mood=mood,
                genre=genre,
                output_dir=output_dir / "score",
                track_name=f"act{act}_chunk{chunk_idx:03d}",
            )
            track.scene_numbers = [s.scene_number for s in chunk]
            score_tracks.append(track)

    # ── Dialogue synthesis ────────────────────────────────────────────────────
    for scene in screenplay.scenes:
        lines = synthesise_dialogue(
            scene=scene,
            voice_map=voice_map,
            output_dir=output_dir / "dialogue",
        )
        dialogue_lines.extend(lines)

    return AudioPackage(
        score_tracks=score_tracks,
        dialogue_lines=dialogue_lines,
        total_assets=len(score_tracks) + len(dialogue_lines),
    )

# Audio Production — Lyria 3 · Gemini TTS · Sentiment Analysis

Complete reference for all audio generation and analysis capabilities used by
the Agentic Cinema Audio Production Agent.

---

## Setup

```python
import os, base64, io, wave, re
import numpy as np
from IPython.display import Audio, Markdown, display
from google import genai
from google.genai import types

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION   = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

client = genai.Client(enterprise=True, project=PROJECT_ID, location=LOCATION)
```

---

## Models

| Model ID | Purpose |
|---|---|
| `lyria-3-pro-preview` | Full tracks up to 3 min with vocals, text prompt |
| `lyria-3-clip-preview` | 30-second clips, supports image + text prompts |
| `gemini-3.1-flash-tts-preview` | Text-to-Speech, single or multi-speaker |
| `gemini-3.6-flash` | Script writing, audio tag insertion, sentiment analysis |
| `gemini-2.5-flash` | Multimodal sentiment analysis (audio + text) |

All Lyria 3 and TTS audio is watermarked with [SynthID](https://deepmind.google/technologies/synthid/).

---

## 1. Lyria 3 — Music Generation

### Full track from text prompt

Generates tracks up to 3 minutes. Prompting tips:
- **Style/Genre:** classical, electronic, rock, jazz, hip-hop, cinematic, ambient, lo-fi
- **Vocals:** describe range and tone (e.g., "breathy Alto female vocal with reverb")
- **Instruments:** piano, synthesizer, acoustic guitar, drums, strings, flute

```python
MUSIC_MODEL = "lyria-3-pro-preview"

prompt = """
Sophisticated, rhythmic, and aspirational track with crisp 808 percussion,
digital plucks, and muted electric guitar rhythmic strums.
Include breathy, airy Alto female vocal textures with melodic, minimalist
oohs and aahs with heavy reverb and rhythmic delay.
"""

response = client.models.generate_content(
    model=MUSIC_MODEL,
    contents=prompt,
    config=types.GenerateContentConfig(
        response_modalities=["AUDIO", "TEXT"],
    ),
)

# Parse response: TEXT parts = lyrics/structure, AUDIO parts = music bytes
for part in response.parts:
    if part.text:
        display(Markdown(part.text))
    if part.inline_data:
        display(Audio(data=part.inline_data.data, autoplay=False))
```

### 30-second clip from an image

Up to 10 images can be supplied in a single request.

```python
CLIP_MODEL = "lyria-3-clip-preview"

with open("storyboard_panel.png", "rb") as f:
    image_bytes = f.read()

response = client.models.generate_content(
    model=CLIP_MODEL,
    contents=[
        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        "Generate an instrumental clip for this scene that starts quietly and builds tension.",
    ],
    config=types.GenerateContentConfig(
        response_modalities=["TEXT", "AUDIO"],
    ),
)
```

### Custom lyrics with genre instructions

```python
genre_and_lyrics = """
Genre: Upbeat acoustic Folk-Pop with warm acoustic guitars, a soft shaker rhythm,
and a friendly melodic vocal.

Lyrics:
The city sleeps beneath the silver screen,
A thousand stories no one's ever seen.
"""

response = client.models.generate_content(
    model=CLIP_MODEL,
    contents=[
        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        genre_and_lyrics,
    ],
    config=types.GenerateContentConfig(response_modalities=["TEXT", "AUDIO"]),
)
```

### Interactions API (multilingual streaming)

Supported languages: **English, German, Spanish, French, Hindi, Japanese, Korean, Portuguese**.

```python
# Non-streaming (returns complete response)
interaction = client.interactions.create(
    model=MUSIC_MODEL,
    input="Genera un tema pop",   # Spanish — auto-detected
)

# Streaming (yields text/lyrics as they arrive, then audio)
stream = client.interactions.create(
    model=MUSIC_MODEL,
    input="Generate a song about spending a day in Seoul in Korean.",
    stream=True,
)

for event in stream:
    if event.event_type == "content.delta":
        delta = event.delta if isinstance(event.delta, dict) else {}
        if "text" in delta:
            display(Markdown(delta["text"]))
        elif "data" in delta and "audio" in delta.get("mime_type", ""):
            display(Audio(data=base64.b64decode(delta["data"]), autoplay=False))
```

---

## 2. Gemini TTS — Text-to-Speech

### Single speaker with prebuilt voice

```python
TTS_MODEL = "gemini-3.1-flash-tts-preview"

def play_audio_pcm(response) -> None:
    """Play 24 kHz 16-bit mono PCM from a TTS response."""
    audio_bytes = response.candidates[0].content.parts[0].inline_data.data
    display(Audio(data=np.frombuffer(audio_bytes, dtype="<i2"), rate=24000))

response = client.models.generate_content(
    model=TTS_MODEL,
    contents="Welcome to Agentic Cinema. Lights, camera, action!",
    config=types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Fenrir")
            )
        ),
    ),
)
play_audio_pcm(response)
```

### Multilingual (auto-detected)

```python
response = client.models.generate_content(
    model=TTS_MODEL,
    contents="¡Hola! Bienvenido al estudio de cine.",
    config=types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Despina")
            )
        ),
    ),
)
```

### Multi-speaker (up to 2 speakers)

Speaker names in the config **must match** the names used in the prompt text.

```python
DIRECTOR_VOICE   = "Umbriel"
SCREENWRITER_VOICE = "Leda"

dialogue = """
Director: The third act needs more tension. What do you think?
Writer: Agreed. Let's add a reveal at the midpoint of the scene.
"""

response = client.models.generate_content(
    model=TTS_MODEL,
    contents=dialogue,
    config=types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    types.SpeakerVoiceConfig(
                        speaker="Director",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=DIRECTOR_VOICE,
                            )
                        ),
                    ),
                    types.SpeakerVoiceConfig(
                        speaker="Writer",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=SCREENWRITER_VOICE,
                            )
                        ),
                    ),
                ]
            )
        ),
    ),
)
play_audio_pcm(response)
```

### Audio tags — expressive delivery

Insert `[tag]` tokens directly before any phrase to control emotion and pacing.
Tags must be in **English** even if the transcript is in another language.

```python
tagged_script = """
[determination] Scene 47. Interior. The editing suite. Night.
[tension] The cut isn't working. We've been here for six hours.
[whispers] Wait — what if we play it in reverse?
[excitement] That's it! That's the whole film right there!
[laughs] I can't believe we almost missed it.
"""

response = client.models.generate_content(
    model=TTS_MODEL,
    contents=tagged_script,
    config=types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
            )
        ),
    ),
)
```

**Full audio tag library** (200+ tags):

```
[acceptance] [accomplishment] [admiration] [adoration] [affection]
[aggression] [agitation] [alarm] [amazement] [ambivalence] [amusement]
[anger] [annoyance] [anticipation] [anxiety] [apology] [appreciation]
[apprehension] [arrogance] [assertion] [assurance] [astonishment]
[awe] [awkwardness] [boredom] [caution] [certainty] [comfort]
[compassion] [concentration] [concern] [confidence] [confusion]
[contemplative] [contempt] [contentment] [conviction] [courage]
[curiosity] [decision] [defiance] [desire] [despair] [desperation]
[determination] [devotion] [directness] [disagreement] [disappointment]
[disapproval] [disbelief] [disdain] [disgust] [dismissive] [distress]
[doubt] [dread] [eagerness] [embarrassment] [empathy] [emphasis]
[enchantment] [enthusiasm] [excitement] [exhaustion] [fascination]
[fast] [fear] [focus] [fondness] [friendly] [frustration]
[gratitude] [grief] [guilt] [happy] [high energy] [hope] [horror]
[humor] [hurt] [incredulity] [indifference] [indignation] [interest]
[intrigue] [joy] [laughs] [long pause] [love] [low energy]
[melancholy] [negative] [nervousness] [neutral] [nostalgia]
[optimism] [pain] [panic] [passion] [pensive] [pessimism]
[playful] [pleading] [positive] [pride] [realization] [reflection]
[regret] [relaxation] [relief] [reminiscence] [resignation] [sadness]
[sarcasm] [satisfaction] [shock] [short pause] [skepticism]
[slow] [solemnity] [stress] [struggle] [success] [surprise]
[tension] [terror] [thinking] [thrill] [tiredness] [triumph]
[uncertainty] [urgency] [warning] [weariness] [whispers] [wisdom]
[wistful] [worry] [yearning]
```

### Auto-insert audio tags with Gemini

Use Gemini to intelligently add tags to any long script:

```python
GEMINI_MODEL = "gemini-3.6-flash"

screenplay_excerpt = """
The detective enters the rain-soaked alley. A figure steps from the shadows.
You shouldn't have come back. The detective smiles. I didn't have a choice.
"""

response = client.models.generate_content(
    model=GEMINI_MODEL,
    contents=f"""
    Insert audio tags from this list into the script.
    Place tags immediately before the phrase they influence.
    Match tags to the emotional arc. Don't overuse them.
    Available tags: [tension] [whispers] [determination] [fear] [confidence]
    [surprise] [curiosity] [neutral] [aggression] [relief] [laughs]
    Script: {screenplay_excerpt}
    """,
)
tagged_script = response.text
```

---

## 3. Multi-Speaker Podcast from a Document

Full pipeline: PDF → Gemini script → TTS multi-speaker WAV.

```python
import io, wave

def pcm_to_wav_bytes(pcm: bytes, sample_rate: int = 24000) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def generate_podcast_from_pdf(pdf_url: str) -> bytes:
    """
    1. Summarise a PDF into a two-speaker dialogue.
    2. Convert the dialogue to multi-speaker WAV.
    Returns raw WAV bytes.
    """
    from google.genai.types import (
        GenerateContentConfig, MultiSpeakerVoiceConfig, Part,
        PrebuiltVoiceConfig, SpeakerVoiceConfig, SpeechConfig, VoiceConfig,
    )

    # Step 1: Generate dialogue script
    script_response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            """The dialogue should be engaging and natural, each speaker contributing equally.
            Return the script in this format:
            R: [dialogue]
            S: [dialogue]
            Use the following document as source material:""",
            Part.from_uri(file_uri=pdf_url, mime_type="application/pdf"),
        ],
        config=GenerateContentConfig(
            system_instruction="You are a podcast writer. Generate a fun podcast-style dialogue between Speaker R and Speaker S.",
        ),
    )
    dialogue = script_response.text

    # Step 2: Convert to audio
    tts_response = client.models.generate_content(
        model="gemini-3.1-flash-tts-preview",
        contents=f"TTS the following conversation between speakers R & S: {dialogue}",
        config=GenerateContentConfig(
            speech_config=SpeechConfig(
                language_code="en-us",
                multi_speaker_voice_config=MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        SpeakerVoiceConfig(
                            speaker="R",
                            voice_config=VoiceConfig(
                                prebuilt_voice_config=PrebuiltVoiceConfig(voice_name="Kore")
                            ),
                        ),
                        SpeakerVoiceConfig(
                            speaker="S",
                            voice_config=VoiceConfig(
                                prebuilt_voice_config=PrebuiltVoiceConfig(voice_name="Achird")
                            ),
                        ),
                    ]
                ),
            ),
        ),
    )
    pcm = tts_response.candidates[0].content.parts[0].inline_data.data
    return pcm_to_wav_bytes(pcm)


# Usage: generate a "book discussion" podcast from the source material
wav_bytes = generate_podcast_from_pdf("https://arxiv.org/pdf/1706.03762")
display(Audio(wav_bytes))
```

---

## 4. Multimodal Sentiment Analysis

Compare sentiment from raw audio vs. text transcript — audio captures tone,
inflection, and non-verbal cues that text misses.

```python
from google.genai.types import GenerateContentConfig, Part

ANALYSIS_MODEL = "gemini-2.5-flash"

audio_part = Part.from_uri(
    file_uri="gs://bucket/dialogue_read.wav",
    mime_type="audio/wav",
)

# --- Pass 1: analyse the audio directly ---
audio_analysis = client.models.generate_content(
    model=ANALYSIS_MODEL,
    contents=[
        audio_part,
        "Provide a sentiment analysis of this conversation. Use Speaker A, Speaker B, etc.",
    ],
).text

# --- Pass 2: transcribe, then analyse the transcript ---
transcript = client.models.generate_content(
    model=ANALYSIS_MODEL,
    contents=[
        audio_part,
        "Transcribe this conversation. Use Speaker A, Speaker B, etc.",
    ],
).text

text_analysis = client.models.generate_content(
    model=ANALYSIS_MODEL,
    contents=f"Provide a sentiment analysis of this conversation.\nTranscript:\n{transcript}",
    config=GenerateContentConfig(response_modalities=["TEXT"]),
).text

# --- Pass 3: compare the two ---
comparison = client.models.generate_content(
    model=ANALYSIS_MODEL,
    contents=f"""
    Compare two sentiment analyses of the same audio conversation.
    One was based on the audio recording; one on a text transcript.
    Highlight where the audio analysis captured nuances the text missed.

    Audio analysis:
    {audio_analysis}

    Text analysis:
    {text_analysis}
    """,
    config=GenerateContentConfig(response_modalities=["TEXT"]),
).text

display(Markdown(comparison))
```

**Key insight:** Audio sentiment analysis detects sarcasm, hesitation, enthusiasm,
and emotional intensity that identical words read as plain text cannot convey.
For dialogue reads and actor feedback, always prefer the audio modality.

---

## Studio Pipeline — Audio Agent Workflow

```
[Screenplay scene]
       │
       ├──► Lyria 3 Pro ──────────────► Score / soundtrack (MP3/WAV)
       │      text prompt (mood, genre)
       │
       ├──► Gemini 3.6 Flash ──────────► Tagged dialogue script
       │      + audio tag insertion
       │
       ├──► Gemini TTS (multi-speaker) ► Cast read-through (WAV, 24kHz PCM)
       │
       └──► Gemini 2.5 Flash ──────────► Sentiment report
              audio analysis              (tone, delivery notes for director)
```

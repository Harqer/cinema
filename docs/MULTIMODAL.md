# Multimodal Capabilities — Gemini Reference

All the patterns the Agentic Cinema studio needs to feed Gemini text, images,
video, audio, PDFs, and code — in any combination.

---

## Setup (all examples below assume this)

```python
import os
from google import genai
from google.genai.types import (
    GenerateContentConfig, Part, CreateCachedContentConfig,
    ImageConfig, MediaResolution, VideoMetadata,
    ThinkingConfig, ThinkingLevel,
)

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION   = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

client = genai.Client(enterprise=True, project=PROJECT_ID, location=LOCATION)

MODEL_ID = "gemini-2.5-flash"   # swap for any model below
```

---

## Supported Models

| Model ID | Best For |
|---|---|
| `gemini-3.1-flash-lite` | Fast, cheap, standard video |
| `gemini-3.5-flash` | Complex or 1h+ video, good balance |
| `gemini-2.5-flash` | Multimodal reasoning, document QA |
| `gemini-2.5-pro` | Long/complex video, 54min+ transcription |
| `gemini-3.1-pro-preview` | Very complex video, max reasoning |
| `gemini-3-pro-image` | Image generation & editing (Nano Banana Pro) |

---

## Token Costs (quick reference)

| Input type | Tokens |
|---|---|
| Text word (common) | ~1 |
| Image (default high res) | 1 089 per image |
| Audio | 25 per second |
| Video frame (default 1 FPS) | 66 per frame |
| Video frame (high res) | 264 per frame |

---

## 1. Text

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        "What is the average weather in Mountain View, CA in mid-May?",
        "Considering the weather, suggest outfits for daytime and evening.",
    ],
)
print(response.text)
```

---

## 2. Documents / PDFs

Pass PDFs by bytes (local) or URI (GCS / HTTPS).

### By URI

```python
pdf = Part.from_uri(
    file_uri="gs://cloud-samples-data/generative-ai/pdf/invoice.pdf",
    mime_type="application/pdf",
)
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[pdf, "Summarize this document."],
)
```

### By bytes (local file)

```python
with open("script_draft.pdf", "rb") as f:
    file_bytes = f.read()

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        "Extract all character names and their first scene.",
        Part.from_bytes(data=file_bytes, mime_type="application/pdf"),
    ],
)
```

### Structured extraction with Pydantic schema

```python
from pydantic import BaseModel, Field
import json

class ScriptCharacter(BaseModel):
    name: str
    first_scene: str
    role: str

class ScriptEntities(BaseModel):
    characters: list[ScriptCharacter]
    locations: list[str]
    time_period: str

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        Part.from_bytes(data=file_bytes, mime_type="application/pdf"),
        "Extract all characters, locations, and the time period.",
    ],
    config=GenerateContentConfig(
        response_schema=ScriptEntities,
        response_mime_type="application/json",
    ),
)
entities: ScriptEntities = response.parsed
```

---

## 3. Images

### Single image

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        "Describe this location for a scout report.",
        Part.from_uri(
            file_uri="https://storage.googleapis.com/cloud-samples-data/generative-ai/image/living-room.png",
            mime_type="image/png",
        ),
    ],
)
```

### Multiple images (comparison / recommendation)

```python
art_urls = [
    "gs://cloud-samples-data/generative-ai/image/room-art-1.png",
    "gs://cloud-samples-data/generative-ai/image/room-art-2.png",
    "gs://cloud-samples-data/generative-ai/image/room-art-3.png",
]

contents = ["You are a production designer. Rank these concept art pieces for a noir film:"]
for i, url in enumerate(art_urls, 1):
    contents += [f"Option {i}:", Part.from_uri(file_uri=url, mime_type="image/png")]

response = client.models.generate_content(model=MODEL_ID, contents=contents)
```

---

## 4. Video

### From URL or GCS

```python
video = Part.from_uri(
    file_uri="gs://cloud-samples-data/generative-ai/video/behind_the_scenes_pixel.mp4",
    mime_type="video/mp4",
)
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        "Describe each scene in detail. Note lighting, pacing, and mood.",
        video,
    ],
)
```

### From YouTube

```python
video = Part.from_uri(
    file_uri="https://www.youtube.com/watch?v=gg7WjuFs8F4",
    mime_type="video/*",
)
```

### Video segment (start/end offsets)

```python
video_metadata = VideoMetadata(
    start_offset="120.0s",   # 2 min in
    end_offset="300.0s",     # 5 min in
)
video = Part(
    file_data={"file_uri": "gs://bucket/film.mp4", "mime_type": "video/mp4"},
    video_metadata=video_metadata,
)
```

### Custom frame rate & resolution

```python
video_metadata = VideoMetadata(fps=0.5)   # 1 frame every 2 seconds (static scenes)

config = GenerateContentConfig(
    media_resolution=MediaResolution.MEDIA_RESOLUTION_LOW,  # 66 tokens/frame, faster
    # media_resolution=MediaResolution.MEDIA_RESOLUTION_HIGH,  # 264 tokens/frame, more detail
)
```

---

## 5. Audio

```python
audio = Part.from_uri(
    file_uri="https://storage.googleapis.com/cloud-samples-data/generative-ai/audio/pixel.mp3",
    mime_type="audio/mpeg",
)

# Summarization
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[audio, "Provide a short summary and chapter titles."],
)

# Transcription with timestamps
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[audio, "Transcribe with timecode, speaker, caption."],
    config=GenerateContentConfig(
        max_output_tokens=8192,
        audio_timestamp=True,
    ),
)
```

---

## 6. Multimodal Video Transcription

The definitive pattern for speaker-diarized, timestamped video transcription in
a single request. Used by the Audio Production Agent.

```python
import pydantic
from google.genai.types import ThinkingConfig, ThinkingLevel

class Transcript(pydantic.BaseModel):
    start: str    # "MM:SS" or "H:MM:SS"
    text: str
    voice: int    # unique per distinct voice heard

class Speaker(pydantic.BaseModel):
    voice: int
    name: str
    company: str
    position: str
    role_in_video: str

class VideoTranscription(pydantic.BaseModel):
    task1_transcripts: list[Transcript] = pydantic.Field(default_factory=list)
    task2_speakers: list[Speaker] = pydantic.Field(default_factory=list)


TRANSCRIPTION_PROMPT = """
**Task 1 - Transcripts**
- Listen carefully to the video's audio.
- Assign each distinct voice heard a consistent, unique `voice` ID (1, 2, 3, etc.).
- Perform an exhaustive, verbatim speech-to-text transcription, including any intelligible background voice.
- If voices overlap, create separate transcript entries for each voice, logging the exact start time.
- Include `start` timecodes in MM:SS format.

**Task 2 - Speakers**
- For each `voice` ID from Task 1, extract available information about the corresponding speaker.
- Use visual and audio cues.
- Use `?` as the value for any unknown piece of information.
"""

def transcribe_video(video_uri: str, model: str = "gemini-2.5-pro") -> VideoTranscription:
    video = Part.from_uri(file_uri=video_uri, mime_type="video/*")
    response = client.models.generate_content(
        model=model,
        contents=[video, TRANSCRIPTION_PROMPT.strip()],
        config=GenerateContentConfig(
            temperature=0.0,
            top_p=0.0,
            seed=42,
            response_mime_type="application/json",
            response_schema=VideoTranscription,
            media_resolution=MediaResolution.MEDIA_RESOLUTION_LOW,
            thinking_config=ThinkingConfig(thinking_budget=128, include_thoughts=False),
        ),
    )
    return response.parsed if isinstance(response.parsed, VideoTranscription) else VideoTranscription()
```

**Model selection guide:**

| Video type | Recommended model |
|---|---|
| ≤ 10 min, simple | `gemini-3.1-flash-lite` |
| ≤ 1h, complex / multi-speaker | `gemini-3.5-flash` |
| 1h+, panel / documentary | `gemini-2.5-pro` |
| Very complex / dynamic edits | `gemini-3.1-pro-preview` |

---

## 7. Video Captioning (for asset metadata)

Used to generate rich searchable captions for the video asset library.

```python
CAPTION_SYSTEM_PROMPT = """
Generate a detailed video caption prioritizing motion, perspective, and nuanced descriptions.
Focus on: camera movement and angle, subject actions (vivid verbs + adverbs), scene composition,
lighting, subject appearance and body language, and any visible text.
Only return the caption. No preamble.
"""

def caption_video(video_uri: str) -> dict:
    contents = [
        Part.from_text(text="Caption this video"),
        Part.from_uri(file_uri=video_uri, mime_type="video/mp4"),
    ]
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=GenerateContentConfig(
            system_instruction=CAPTION_SYSTEM_PROMPT,
            temperature=0.7,
        ),
    )
    return {
        "caption": response.text,
        "prompt_tokens": response.usage_metadata.prompt_token_count,
        "output_tokens": response.usage_metadata.candidates_token_count,
    }
```

---

## 8. Image Generation (Storyboards & VFX)

Uses `gemini-3-pro-image` (Nano Banana Pro) for storyboard panels and concept art.

```python
IMAGE_MODEL = "gemini-3-pro-image"

from IPython.display import Image, display

def generate_storyboard_panel(
    scene_description: str,
    aspect_ratio: str = "16:9",
    size: str = "2K",
) -> bytes:
    """Generate a single storyboard panel from a scene description."""
    response = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=f"Cinematic storyboard panel: {scene_description}",
        config=GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=size,       # "1K", "2K", or "4K"
            ),
        ),
    )
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            return part.inline_data.data
    return b""


def generate_vfx_moodboard(prompt: str) -> bytes:
    """Generate a VFX mood board with search grounding for current visual trends."""
    from google.genai.types import Tool, GoogleSearch

    response = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=prompt,
        config=GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=ImageConfig(aspect_ratio="21:9"),
            tools=[Tool(google_search=GoogleSearch())],
        ),
    )
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            return part.inline_data.data
    return b""
```

### Supported aspect ratios

`1:1`, `3:2`, `2:3`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`

### Multi-turn image editing (iterative storyboarding)

```python
chat = client.chats.create(
    model=IMAGE_MODEL,
    config=GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
)

# Round 1: generate base panel
response = chat.send_message("Wide shot of a rain-soaked alley at night, neon reflections.")
image_data = next(p.inline_data.data for p in response.parts if p.inline_data)

# Round 2: refine
response = chat.send_message([
    Part.from_bytes(data=image_data, mime_type="image/png"),
    "Add a lone detective silhouette in the foreground walking away from camera.",
])
```

---

## 9. All Modalities at Once

Gemini can interleave text, image, video, and audio in a single `contents` list.

```python
# Example: cross-reference a video scene with a still image
video = Part.from_uri(
    file_uri="gs://cloud-samples-data/generative-ai/video/behind_the_scenes_pixel.mp4",
    mime_type="video/mp4",
)
image = Part.from_uri(
    file_uri="https://storage.googleapis.com/cloud-samples-data/generative-ai/image/a-man-and-a-dog.png",
    mime_type="image/png",
)
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        "Find the moment in the video that matches this image. Give a timestamp and explain the context.",
        video,
        image,
    ],
)
```

---

## 10. Context Caching (for repeated large inputs)

Use when the same book/script/codebase is queried multiple times. Provides
**90% discount** on cached input tokens.

```python
from google.genai.types import Content, CreateCachedContentConfig

# Cache a large video once
cached_content = client.caches.create(
    model=MODEL_ID,
    config=CreateCachedContentConfig(
        ttl="3600s",           # keep for 1 hour
        display_name="feature-film-rushes",
        contents=[
            Content(role="user", parts=[
                Part.from_uri(
                    file_uri="gs://bucket/feature-film-rushes.mp4",
                    mime_type="video/mp4",
                )
            ])
        ],
    ),
)
print(f"Cached {cached_content.usage_metadata.total_token_count:,} tokens")

cache_config = GenerateContentConfig(cached_content=cached_content.name)

# All subsequent queries reuse the cached tokens
queries = [
    "List every scene where the protagonist is alone.",
    "Find all exterior night shots.",
    "Identify the three most emotionally intense scenes.",
]
for q in queries:
    response = client.models.generate_content(
        model=MODEL_ID, contents=q, config=cache_config,
    )
    print(response.text)
```

**Key rule for implicit caching:** always put static data (video/PDF) **before**
the variable prompt — the prefix must match for a cache hit.

---

## 11. Codebase Analysis (context caching + gitingest)

For analyzing screenwriting tools, VFX pipelines, or any GitHub repo:

```python
from gitingest import ingest

_, code_index, code_text = ingest(
    "https://github.com/GoogleCloudPlatform/microservices-demo",
    exclude_patterns={"*.png", "*.jpg", "*.gif", "*.svg", ".git/"},
)

prompt = f"""
Context:
- File index:
{code_index}

- Full source:
{code_text}
"""

cached_content = client.caches.create(
    model=MODEL_ID,
    config=CreateCachedContentConfig(contents=prompt, ttl="3600s"),
)

# Query the codebase
response = client.models.generate_content(
    model=MODEL_ID,
    contents="Find the top 3 security issues in this codebase.",
    config=GenerateContentConfig(cached_content=cached_content.name),
)
```

---

## Agentic Cinema — Modality Mapping

| Studio Agent | Modality | Input | Output |
|---|---|---|---|
| Document Processing | PDF | Book PDF | Structured characters/locations |
| Script Development | Text | Story outline | Screenplay |
| Location Scout | Maps + Text | Scene descriptions | Real location suggestions |
| Research & Continuity | Text + Parallel | Scene + fact queries | Verified facts + sources |
| Production Design | Image generation | Scene descriptions | Storyboard panels, VFX moodboards |
| Audio Production | Audio | Dialogue text | Speech synthesis (TTS) |
| Video Transcription | Video + Audio | Raw footage | Speaker-diarized transcripts |
| Asset Captioning | Video | B-roll clips | Searchable captions |

"""Shared configuration and client initialisation."""

import os
from google import genai

# ── GCP project & location ────────────────────────────────────────────────────
PROJECT_ID: str = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION: str = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

# ── Model identifiers ─────────────────────────────────────────────────────────
MODEL_REASONING = "gemini-2.5-flash"          # script writing, orchestration
MODEL_LONG_CONTEXT = "gemini-2.5-pro"         # full-book ingestion
MODEL_TTS = "gemini-3.1-flash-tts-preview"    # dialogue synthesis
MODEL_IMAGE_GEN = "gemini-3-pro-image"        # storyboards / concept art
MODEL_LYRIA_TRACK = "lyria-3-pro-preview"     # full score tracks (3 min)
MODEL_LYRIA_CLIP = "lyria-3-clip-preview"     # short clips with image ref

# ── Parallel Web Search defaults ──────────────────────────────────────────────
PARALLEL_MODE_RESEARCH = "advanced"
PARALLEL_MODE_NEWS = "basic"
PARALLEL_FILM_DOMAINS = [
    "variety.com",
    "deadline.com",
    "hollywoodreporter.com",
    "boxofficemojo.com",
    "rottentomatoes.com",
    "imdb.com",
    "theguardian.com",
    "indiewire.com",
    "thewrap.com",
]

# ── Shared genai client (enterprise Vertex AI endpoint) ───────────────────────
def get_client() -> genai.Client:
    """Return a genai.Client pointed at the Vertex AI enterprise endpoint."""
    return genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )

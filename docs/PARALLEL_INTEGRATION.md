# Parallel Web Search — Integration Guide

Agentic Cinema submits to the **Parallel partner track**. This document is the
definitive reference for how and where Parallel Web Search is wired into the
agent network.

---

## What Parallel Provides

[Parallel Web Systems](https://parallel.ai) offers an LLM-optimized web search
API that gives Gemini access to live, publicly available web data from billions
of pages. Unlike Google Search grounding, Parallel is purpose-built for deep
multi-hop agent reasoning — making it ideal for the complex research chains
required in film production.

**Key strengths for our use case:**
- `advanced` mode for deep multi-hop research (script accuracy, adaptation rights context)
- Domain filtering — include only authoritative film industry sources
- Up-to-date data — no training cutoff lag for box office trends, crew availability, etc.
- Zero Data Retention option for sensitive IP (scripts, unreleased adaptations)

---

## Setup

### Option A: Google Cloud Marketplace (recommended)

Subscribe at:
[console.cloud.google.com/marketplace/product/parallel-web-systems-public/parallel-web-systems](https://console.cloud.google.com/marketplace/product/parallel-web-systems-public/parallel-web-systems)

No API key needed in code — billing flows through your GCP project.

### Option B: Bring Your Own API Key

Get a key at [platform.parallel.ai](https://platform.parallel.ai), then store it:

```bash
gcloud secrets create parallel-api-key --data-file=- <<< "YOUR_KEY_HERE"
```

Retrieve in code:

```python
from google.cloud import secretmanager

def get_secret(secret_id: str, project_id: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    return client.access_secret_version(name=name).payload.data.decode("utf-8")

PARALLEL_API_KEY = get_secret("parallel-api-key", PROJECT_ID)
```

---

## Installation

```bash
pip install --upgrade google-genai google-cloud-secret-manager
```

Set environment for Vertex AI enterprise endpoint:

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=global
export GOOGLE_GENAI_USE_ENTERPRISE=True
```

---

## Core Usage Pattern

```python
from google import genai
from google.genai import types

client = genai.Client()

def research_with_parallel(
    query: str,
    mode: str = "basic",         # "basic" or "advanced"
    include_domains: list = None,
    exclude_domains: list = None,
    max_results: int = 10,
) -> tuple[str, list]:
    """
    Ground a Gemini response with live Parallel web search.
    Returns (answer_text, grounding_chunks).
    """
    custom_configs = {
        "mode": mode,
        "max_results": max_results,
        "excerpts": {
            "max_chars_per_result": 30000,
            "max_chars_total": 100000,
        },
    }
    if include_domains:
        custom_configs["source_policy"] = {"include_domains": include_domains}
    if exclude_domains:
        custom_configs.setdefault("source_policy", {})["exclude_domains"] = exclude_domains

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query,
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    parallel_ai_search=types.ToolParallelAiSearch(
                        # Remove api_key line if using Marketplace subscription:
                        # api_key=PARALLEL_API_KEY,
                        custom_configs=custom_configs,
                    )
                )
            ],
        ),
    )

    chunks = response.candidates[0].grounding_metadata.grounding_chunks
    return response.text, chunks
```

---

## Integration Points in the Studio Pipeline

### 1. Script Research Agent — Film Market Grounding

Before writing a single scene, the script agent researches the competitive
landscape using Parallel in `advanced` mode:

```python
def research_comparable_films(book_title: str, genre: str) -> dict:
    """Find comparable films, their reception, and adaptation lessons."""
    query = f"""
    I am adapting the book "{book_title}" (genre: {genre}) into a film.
    What comparable book-to-film adaptations exist? What did they do well
    or poorly? What is the current market appetite for this genre?
    Provide specific box office data, critical reception, and release timing.
    """
    answer, sources = research_with_parallel(
        query=query,
        mode="advanced",
        include_domains=["variety.com", "deadline.com", "boxofficemojo.com",
                         "rottentomatoes.com", "imdb.com", "theguardian.com"],
        max_results=15,
    )
    return {"research": answer, "sources": [c.web.uri for c in sources if hasattr(c, "web")]}
```

### 2. Continuity & Fact Verification Agent

After each script draft, the continuity agent verifies historical and
technical accuracy using Parallel:

```python
def verify_scene_accuracy(scene_description: str, time_period: str) -> dict:
    """
    Verify that a scene's details are historically/technically accurate.
    E.g., "Is it accurate that a 1940s New York detective would carry a
    Colt 1911? What would the street lighting look like?"
    """
    query = f"""
    Verify the following scene details for a film set in {time_period}:
    {scene_description}
    Check for historical accuracy, anachronisms, and technical correctness.
    Cite specific sources.
    """
    answer, sources = research_with_parallel(
        query=query,
        mode="advanced",
        max_results=10,
    )
    return {"verdict": answer, "sources": sources}
```

### 3. Distribution Intelligence Agent

Near the end of production, the distribution agent researches the release
landscape:

```python
def analyze_release_window(genre: str, comparable_films: list[str]) -> dict:
    """Research optimal release timing and distribution strategy."""
    comparables = ", ".join(comparable_films)
    query = f"""
    For a {genre} film comparable to {comparables}:
    - What release windows (theatrical, streaming, hybrid) have performed best recently?
    - Which streaming platforms are currently acquiring this genre?
    - What are the current audience demographics for this genre?
    Provide data from the last 12 months.
    """
    answer, sources = research_with_parallel(
        query=query,
        mode="advanced",
        include_domains=["variety.com", "deadline.com", "hollywoodreporter.com",
                         "thewrap.com", "indiewire.com"],
        max_results=20,
    )
    return {"strategy": answer, "sources": sources}
```

### 4. Real-time Production News (Live Grounding)

For any agent that needs to check current crew/talent availability or
industry news:

```python
def get_industry_news(topic: str) -> str:
    """Get the latest film industry news on a topic."""
    answer, _ = research_with_parallel(
        query=f"Latest news about {topic} in the film industry",
        mode="basic",
        include_domains=["deadline.com", "variety.com", "hollywoodreporter.com"],
        max_results=5,
    )
    return answer
```

---

## Supported Models

The following Gemini models support Parallel grounding (use any):

| Model | Notes |
|---|---|
| `gemini-2.5-pro` | Best for complex multi-hop research |
| `gemini-2.5-flash` | Best performance/cost balance |
| `gemini-2.5-flash-lite` | Fastest, for simple lookups |
| `gemini-3.5-flash` | Latest generation |
| `gemini-3.1-pro-preview` | Preview — highest capability |
| `gemini-3.1-flash-lite` | Latest generation, cost-optimized |

---

## Grounding Response Structure

The `grounding_metadata` in every response contains cited web sources:

```python
response = client.models.generate_content(...)

# Iterate sources
for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
    print(f"Source: {chunk.web.title}")
    print(f"URL:    {chunk.web.uri}")
    print(f"Domain: {chunk.web.domain}")

# Grounding supports map text segments → source indices
for support in response.candidates[0].grounding_metadata.grounding_supports:
    print(f"Claim: '{support.segment.text}'")
    print(f"Supported by chunk indices: {support.grounding_chunk_indices}")
```

---

## Zero Data Retention (for sensitive IP)

If your adaptation involves unreleased IP or confidential script drafts:

1. Subscribe to the ZDR offering on Marketplace (separate listing)
2. Set the flag in your request:

```python
parallel_ai_search=types.ToolParallelAiSearch(
    custom_configs={"enable_zero_data_retention": True, ...}
)
```

---

## Quota & Cost Notes

- Default quota: **200 requests/minute**
- Each Gemini call may trigger multiple Parallel queries (fanout)
- To increase quota: contact your Google account team (Marketplace) or `support@parallel.ai` (BYOK)
- Billed separately: Gemini token consumption + Parallel query fees

See [parallel.ai/pricing](https://parallel.ai/pricing) or the
[Marketplace listing](https://console.cloud.google.com/marketplace/product/parallel-web-systems-public/parallel-web-systems)
for current rates.

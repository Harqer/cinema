"""Research Agent — Parallel Web Search integration for all film industry grounding.

This is the primary integration point for the Parallel partner track.
Every call that touches external web data goes through this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from google.genai import types

from cinema_studio.config import (
    MODEL_REASONING,
    PARALLEL_FILM_DOMAINS,
    PARALLEL_MODE_NEWS,
    PARALLEL_MODE_RESEARCH,
    get_client,
)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parallel_tool(
    mode: str = PARALLEL_MODE_RESEARCH,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    max_results: int = 10,
    api_key: str | None = None,
) -> types.Tool:
    """Build a ToolParallelAiSearch instance with the given config.

    Leave ``api_key`` as None when using the GCP Marketplace subscription
    (billing flows through the GCP project automatically).
    """
    custom_configs: dict = {
        "mode": mode,
        "max_results": max_results,
        "excerpts": {
            "max_chars_per_result": 30_000,
            "max_chars_total": 100_000,
        },
    }
    if include_domains or exclude_domains:
        source_policy: dict = {}
        if include_domains:
            source_policy["include_domains"] = include_domains
        if exclude_domains:
            source_policy["exclude_domains"] = exclude_domains
        custom_configs["source_policy"] = source_policy

    kwargs: dict = {"custom_configs": custom_configs}
    if api_key:
        kwargs["api_key"] = api_key

    return types.Tool(parallel_ai_search=types.ToolParallelAiSearch(**kwargs))


@dataclass
class ResearchResult:
    answer: str
    sources: list[dict] = field(default_factory=list)  # [{title, url, domain}]
    raw_chunks: list = field(default_factory=list)


def _run_search(
    query: str,
    model: str = MODEL_REASONING,
    mode: str = PARALLEL_MODE_RESEARCH,
    include_domains: list[str] | None = None,
    max_results: int = 10,
    api_key: str | None = None,
) -> ResearchResult:
    client = get_client()
    tool = _parallel_tool(
        mode=mode,
        include_domains=include_domains,
        max_results=max_results,
        api_key=api_key,
    )
    response = client.models.generate_content(
        model=model,
        contents=query,
        config=types.GenerateContentConfig(tools=[tool]),
    )

    chunks = []
    sources = []
    try:
        chunks = response.candidates[0].grounding_metadata.grounding_chunks or []
        for c in chunks:
            if hasattr(c, "web") and c.web:
                sources.append(
                    {
                        "title": getattr(c.web, "title", ""),
                        "url": getattr(c.web, "uri", ""),
                        "domain": getattr(c.web, "domain", ""),
                    }
                )
    except (AttributeError, IndexError):
        pass

    return ResearchResult(answer=response.text or "", sources=sources, raw_chunks=chunks)


# ── Public agent functions ────────────────────────────────────────────────────

def research_comparable_films(book_title: str, genre: str) -> ResearchResult:
    """
    Find comparable book-to-film adaptations, their reception, and market appetite.

    Used by: Script Development Agent (pre-writing market grounding).
    Parallel mode: advanced (multi-hop — follows references across sources).
    """
    query = (
        f'I am adapting the book "{book_title}" (genre: {genre}) into a feature film. '
        "What comparable book-to-film adaptations exist? "
        "What did they do well or poorly creatively and commercially? "
        "What is the current market appetite for this genre? "
        "Provide specific box office data, critical reception scores, and release timing. "
        "Focus on films released in the last 10 years where possible."
    )
    return _run_search(
        query=query,
        mode=PARALLEL_MODE_RESEARCH,
        include_domains=PARALLEL_FILM_DOMAINS,
        max_results=15,
    )


def verify_scene_accuracy(scene_description: str, time_period: str) -> ResearchResult:
    """
    Verify historical, cultural, and technical accuracy of a scene.

    Used by: Continuity Agent after each script draft.
    Parallel mode: advanced (needs to cross-reference multiple authoritative sources).
    """
    query = (
        f"Verify the following scene details for a film set in {time_period}:\n\n"
        f"{scene_description}\n\n"
        "Check for: historical accuracy, anachronisms, technical correctness, "
        "cultural authenticity. Flag any inaccuracies and suggest corrections. "
        "Cite specific sources for each claim."
    )
    return _run_search(
        query=query,
        mode=PARALLEL_MODE_RESEARCH,
        max_results=10,
    )


def analyze_release_strategy(genre: str, comparable_films: list[str]) -> ResearchResult:
    """
    Research optimal release timing, platform fit, and distribution strategy.

    Used by: Distribution Intelligence Agent near end of production.
    Parallel mode: advanced (box office trend analysis is multi-hop).
    """
    comps = ", ".join(comparable_films)
    query = (
        f"For a {genre} film comparable to {comps}:\n"
        "- What release windows (theatrical, streaming, hybrid) have performed best in the last 12 months?\n"
        "- Which streaming platforms are currently acquiring this genre?\n"
        "- What are the current audience demographics and size for this genre?\n"
        "- What marketing spend is typical?\n"
        "Provide data-backed recommendations."
    )
    return _run_search(
        query=query,
        mode=PARALLEL_MODE_RESEARCH,
        include_domains=[
            "variety.com",
            "deadline.com",
            "hollywoodreporter.com",
            "thewrap.com",
            "indiewire.com",
            "boxofficemojo.com",
        ],
        max_results=20,
    )


def get_industry_news(topic: str) -> ResearchResult:
    """
    Get the latest film industry news on a topic.

    Used by: any agent needing real-time industry context.
    Parallel mode: basic (simple recency lookup, no multi-hop needed).
    """
    query = f"Latest news about {topic} in the film and entertainment industry"
    return _run_search(
        query=query,
        mode=PARALLEL_MODE_NEWS,
        include_domains=["deadline.com", "variety.com", "hollywoodreporter.com"],
        max_results=5,
    )


def research_adaptation_rights(book_title: str, author: str) -> ResearchResult:
    """
    Check if adaptation rights have been optioned, sold, or are in production.

    Used by: Studio Orchestrator at project kickoff — avoids wasted work on
    already-adapted IP without a clear rights window.
    Parallel mode: advanced (rights status often buried in trade news).
    """
    query = (
        f'What is the current adaptation rights status for "{book_title}" by {author}? '
        "Has it been optioned, sold, or is there already a film or TV adaptation in development or released? "
        "Who holds the rights? Any recent deals?"
    )
    return _run_search(
        query=query,
        mode=PARALLEL_MODE_RESEARCH,
        include_domains=PARALLEL_FILM_DOMAINS,
        max_results=10,
    )

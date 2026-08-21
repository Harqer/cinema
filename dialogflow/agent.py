"""
dialogflow/agent.py
─────────────────────────────────────────────────────────────────────────────
Agent bootstrap — create or retrieve the "cinema" Dialogflow CX agent.

The ensure_agent() function is idempotent: if the agent already exists it
returns the existing resource without touching it.
"""

from __future__ import annotations

from dialogflow.client import DialogflowCXClient
from dialogflow.config import (
    AGENT_DEFAULT_LANGUAGE,
    AGENT_DISPLAY_NAME,
    AGENT_TIME_ZONE,
    LOCATION,
    PROJECT_ID,
    agent_parent,
)


def list_agents(client: DialogflowCXClient | None = None) -> list[dict]:
    """Return all agents in the project/location."""
    c = client or DialogflowCXClient()
    return c.list_all(agent_parent(), key="agents")


def find_agent(
    display_name: str = AGENT_DISPLAY_NAME,
    client: DialogflowCXClient | None = None,
) -> dict | None:
    """Return the agent with the given display name, or None if not found."""
    for agent in list_agents(client):
        if agent.get("displayName") == display_name:
            return agent
    return None


def create_agent(
    display_name: str = AGENT_DISPLAY_NAME,
    client: DialogflowCXClient | None = None,
) -> dict:
    """
    Create a new Dialogflow CX agent named ``display_name``.

    The agent is configured for Agentic Cinema:
    - Generative AI / playbook features enabled
    - Speech-to-text and TTS enabled
    - Vertex AI integration set to the cinema project
    """
    c = client or DialogflowCXClient()
    body = {
        "displayName": display_name,
        "defaultLanguageCode": AGENT_DEFAULT_LANGUAGE,
        "timeZone": AGENT_TIME_ZONE,
        "description": (
            "Agentic Cinema — AI-powered book-to-movie production studio. "
            "Handles script queries, location scouting, audio production, "
            "and distribution intelligence via natural language."
        ),
        "enableStackdriverLogging": True,
        "enableSpellCorrection": True,
        "speechToTextSettings": {
            "enableSpeechAdaptation": True,
        },
        "advancedSettings": {
            "loggingSettings": {
                "enableStackdriverLogging": True,
                "enableInteractionLogging": True,
            },
        },
        "genAppBuilderSettings": {
            "engine": (
                f"projects/{PROJECT_ID}/locations/{LOCATION}"
                f"/collections/default_collection/engines/{display_name}-engine"
            ),
        },
    }
    return c.post(agent_parent(), body=body)


def ensure_agent(
    display_name: str = AGENT_DISPLAY_NAME,
    client: DialogflowCXClient | None = None,
) -> tuple[dict, bool]:
    """
    Ensure the agent exists. Creates it if absent.

    Returns:
        (agent_dict, created) — created=True if newly created, False if already existed.
    """
    c = client or DialogflowCXClient()
    existing = find_agent(display_name, c)
    if existing:
        return existing, False
    created = create_agent(display_name, c)
    return created, True


def agent_id_from_name(resource_name: str) -> str:
    """Extract the bare agent ID from a full resource name."""
    # e.g. projects/cinema/locations/us-central1/agents/abc123 → abc123
    return resource_name.rstrip("/").split("/")[-1]

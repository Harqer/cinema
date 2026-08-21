"""
dialogflow/webhooks.py
─────────────────────────────────────────────────────────────────────────────
Webhook management — create and configure environment-specific webhooks for
the cinema agent.

Best practice (from the Dialogflow CX docs):
  - One webhook resource per environment.
  - Each webhook resource's URI points to the correct Cloud Run service for
    that environment.
  - This isolates production from dev/test so you never accidentally call
    the production backend during testing.
"""

from __future__ import annotations

from dialogflow.client import DialogflowCXClient
from dialogflow.config import (
    WEBHOOK_CINEMA_BACKEND,
    WEBHOOK_URLS,
    ENV_TESTING,
    ENV_DEVELOPMENT,
    ENV_PRODUCTION,
    webhooks_parent,
)


def list_webhooks(
    agent_id: str,
    client: DialogflowCXClient | None = None,
) -> list[dict]:
    """Return all webhooks defined on the agent."""
    c = client or DialogflowCXClient()
    return c.list_all(webhooks_parent(agent_id), key="webhooks")


def find_webhook(
    agent_id: str,
    display_name: str,
    client: DialogflowCXClient | None = None,
) -> dict | None:
    """Return the webhook with the given display name, or None."""
    for wh in list_webhooks(agent_id, client):
        if wh.get("displayName") == display_name:
            return wh
    return None


def create_webhook(
    agent_id: str,
    display_name: str,
    uri: str,
    timeout_seconds: int = 30,
    client: DialogflowCXClient | None = None,
) -> dict:
    """
    Create a webhook on the agent.

    Args:
        display_name:     Human-readable name, e.g. "cinema-backend".
        uri:              The Cloud Run / HTTPS endpoint URL.
        timeout_seconds:  Request timeout (max 30s for Dialogflow).
    """
    c = client or DialogflowCXClient()
    body = {
        "displayName": display_name,
        "genericWebService": {
            "uri": uri,
            "httpMethod": "POST",
        },
        "timeout": f"{timeout_seconds}s",
    }
    return c.post(webhooks_parent(agent_id), body=body)


def ensure_webhooks(
    agent_id: str,
    client: DialogflowCXClient | None = None,
) -> dict[str, dict]:
    """
    Create environment-specific webhooks for testing, development, and
    production if they do not already exist.  Idempotent.

    Creates three webhooks:
      - cinema-backend-testing
      - cinema-backend-development
      - cinema-backend-production

    Each points to the URL from config.WEBHOOK_URLS.

    Returns:
        Dict mapping env name → webhook resource.
    """
    c = client or DialogflowCXClient()
    results: dict[str, dict] = {}

    for env in [ENV_TESTING, ENV_DEVELOPMENT, ENV_PRODUCTION]:
        display_name = f"{WEBHOOK_CINEMA_BACKEND}-{env}"
        uri = WEBHOOK_URLS.get(env, "")

        existing = find_webhook(agent_id, display_name, c)
        if existing:
            results[env] = existing
        else:
            wh = create_webhook(
                agent_id=agent_id,
                display_name=display_name,
                uri=uri,
                client=c,
            )
            results[env] = wh

    return results

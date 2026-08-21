"""
dialogflow/flows.py
─────────────────────────────────────────────────────────────────────────────
Flow and Version management for the cinema Dialogflow CX agent.

Key operations:
  - list_flows          — list all flows in the agent
  - get_flow_id         — resolve a flow display name to its ID
  - create_version      — snapshot a flow draft as an immutable version
  - list_versions       — list all versions of a flow
  - get_version         — get a single version by ID
  - load_version_to_draft — restore a version as the current draft
  - compare_versions    — side-by-side diff two versions
  - delete_version      — delete a version (must not be referenced by an env)
"""

from __future__ import annotations

from dialogflow.client import DialogflowCXClient
from dialogflow.config import (
    flows_parent,
    versions_parent,
    version_name,
    flow_name,
)


# ── Flows ─────────────────────────────────────────────────────────────────────

def list_flows(agent_id: str, client: DialogflowCXClient | None = None) -> list[dict]:
    """Return all flows in the agent."""
    c = client or DialogflowCXClient()
    return c.list_all(flows_parent(agent_id), key="flows")


def get_flow_id(
    agent_id: str,
    display_name: str,
    client: DialogflowCXClient | None = None,
) -> str | None:
    """Resolve a flow display name to its bare flow ID."""
    for flow in list_flows(agent_id, client):
        if flow.get("displayName") == display_name:
            name: str = flow["name"]
            return name.split("/")[-1]
    return None


# ── Versions ──────────────────────────────────────────────────────────────────

def create_version(
    agent_id: str,
    flow_id: str,
    display_name: str,
    description: str = "",
    client: DialogflowCXClient | None = None,
) -> dict:
    """
    Snapshot the current draft of a flow as a named, immutable version.

    The version is not usable until its status becomes "READY" — check with
    ``get_version()``.  Creating a version triggers NLU training; this may
    take a few minutes.

    Args:
        agent_id:     Bare agent ID.
        flow_id:      Bare flow ID.
        display_name: Human-readable version name, e.g. "v1.2-production".
        description:  Optional description.

    Returns:
        Long-running operation dict — poll `name` field for completion.
    """
    c = client or DialogflowCXClient()
    body: dict = {"displayName": display_name}
    if description:
        body["description"] = description
    # Returns a long-running operation
    return c.post(versions_parent(agent_id, flow_id), body=body)


def list_versions(
    agent_id: str,
    flow_id: str,
    client: DialogflowCXClient | None = None,
) -> list[dict]:
    """Return all versions for a flow, sorted by creation time (newest first)."""
    c = client or DialogflowCXClient()
    versions = c.list_all(versions_parent(agent_id, flow_id), key="versions")
    return sorted(
        versions,
        key=lambda v: v.get("createTime", ""),
        reverse=True,
    )


def get_version(
    agent_id: str,
    flow_id: str,
    version_id: str,
    client: DialogflowCXClient | None = None,
) -> dict:
    """Fetch a single version by ID."""
    c = client or DialogflowCXClient()
    return c.get(version_name(agent_id, flow_id, version_id))


def load_version_to_draft(
    agent_id: str,
    flow_id: str,
    version_id: str,
    allow_override_agent_resources: bool = False,
    client: DialogflowCXClient | None = None,
) -> dict:
    """
    Restore a flow version as the current editable draft.

    This overwrites the live draft of the flow (and optionally agent-level
    resources like intents and entities).

    Args:
        allow_override_agent_resources:
            If True, agent-level resources (intents, entities, webhooks) are
            overwritten to match what they were when this version was created.
            Set to False if you only want the flow pages/routes restored.

    Returns:
        Long-running operation dict.
    """
    c = client or DialogflowCXClient()
    vname = version_name(agent_id, flow_id, version_id)
    return c.post(
        f"{vname}:load",
        body={"allowOverrideAgentResources": allow_override_agent_resources},
    )


def compare_versions(
    agent_id: str,
    flow_id: str,
    base_version_id: str,
    target_version_id: str,
    language_code: str = "en",
    client: DialogflowCXClient | None = None,
) -> dict:
    """
    Return a structured comparison between two flow versions (or draft).

    Use "0" as version_id to reference the current draft.

    Note: the API response is capped at 2 MB — use the REST API directly
    for very large flows.
    """
    c = client or DialogflowCXClient()
    # compareVersions is called on the base version resource
    base = version_name(agent_id, flow_id, base_version_id)
    target = version_name(agent_id, flow_id, target_version_id)
    return c.post(
        f"{base}:compareVersions",
        body={
            "targetVersion": target,
            "languageCode": language_code,
        },
    )


def delete_version(
    agent_id: str,
    flow_id: str,
    version_id: str,
    client: DialogflowCXClient | None = None,
) -> None:
    """
    Delete a flow version.

    The version must not be referenced by any environment before deletion.
    Remove it from all environments first or use update_environment().
    """
    c = client or DialogflowCXClient()
    c.delete(version_name(agent_id, flow_id, version_id))

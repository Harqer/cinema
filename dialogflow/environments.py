"""
dialogflow/environments.py
─────────────────────────────────────────────────────────────────────────────
Environment management for the cinema Dialogflow CX agent.

Environments:  draft (implicit), testing, development, production.

Key operations:
  - list_environments         — list all custom environments
  - get_environment           — get a single environment by ID or display name
  - create_environment        — create a new named environment
  - update_environment        — change which flow versions an env points to
  - delete_environment        — remove a custom environment
  - promote_version           — convenience: assign a version to an env
  - deploy_version_pipeline   — promote through testing → development → production
"""

from __future__ import annotations

from dialogflow.client import DialogflowCXClient
from dialogflow.config import (
    environments_parent,
    environment_name,
    WEBHOOK_URLS,
    ENV_TESTING,
    ENV_DEVELOPMENT,
    ENV_PRODUCTION,
    ENV_PIPELINE,
)


# ── Environments ──────────────────────────────────────────────────────────────

def list_environments(
    agent_id: str,
    client: DialogflowCXClient | None = None,
) -> list[dict]:
    """Return all custom environments for the agent."""
    c = client or DialogflowCXClient()
    return c.list_all(environments_parent(agent_id), key="environments")


def get_environment(
    agent_id: str,
    display_name: str,
    client: DialogflowCXClient | None = None,
) -> dict | None:
    """Find an environment by display name. Returns None if not found."""
    for env in list_environments(agent_id, client):
        if env.get("displayName") == display_name:
            return env
    return None


def create_environment(
    agent_id: str,
    display_name: str,
    description: str = "",
    flow_versions: list[dict] | None = None,
    webhook_config: dict | None = None,
    client: DialogflowCXClient | None = None,
) -> dict:
    """
    Create a custom environment.

    Args:
        agent_id:      Bare agent ID.
        display_name:  Environment name — "testing", "development", or "production".
        description:   Human-readable description.
        flow_versions: List of ``{"flow": "<flow-resource-name>",
                                  "version": "<version-resource-name>"}`` dicts.
                       If omitted the draft (version 0) is used for all flows.
        webhook_config: Optional webhook overrides — dict mapping webhook
                        resource names to environment-specific URIs.

    Returns:
        Long-running operation dict (environment deployment).
    """
    c = client or DialogflowCXClient()
    body: dict = {"displayName": display_name}

    if description:
        body["description"] = description

    if flow_versions:
        body["versionConfigs"] = flow_versions

    if webhook_config:
        body["webhookConfig"] = {"webhookOverrides": webhook_config}

    return c.post(environments_parent(agent_id), body=body)


def update_environment(
    agent_id: str,
    env_id: str,
    flow_versions: list[dict],
    webhook_config: dict | None = None,
    client: DialogflowCXClient | None = None,
) -> dict:
    """
    Update which flow versions an environment points to.

    This is the core "deploy" operation — call it to promote a new version
    into testing, development, or production.

    Args:
        agent_id:      Bare agent ID.
        env_id:        Bare environment ID (from the ``name`` field).
        flow_versions: New ``versionConfigs`` list — same format as create_environment.
        webhook_config: Optional updated webhook overrides.

    Returns:
        Long-running operation dict.
    """
    c = client or DialogflowCXClient()
    body: dict = {
        "name": environment_name(agent_id, env_id),
        "versionConfigs": flow_versions,
    }
    if webhook_config:
        body["webhookConfig"] = {"webhookOverrides": webhook_config}

    return c.patch(
        environment_name(agent_id, env_id),
        body=body,
        update_mask="versionConfigs,webhookConfig",
    )


def delete_environment(
    agent_id: str,
    env_id: str,
    client: DialogflowCXClient | None = None,
) -> None:
    """Delete a custom environment."""
    c = client or DialogflowCXClient()
    c.delete(environment_name(agent_id, env_id))


def promote_version(
    agent_id: str,
    flow_resource_name: str,
    version_resource_name: str,
    target_env_display_name: str,
    client: DialogflowCXClient | None = None,
) -> dict:
    """
    Point a named environment at a specific flow version.

    Convenience wrapper over update_environment — handles env lookup by
    display name automatically.

    Production environments also get the environment-specific webhook URL
    from config.WEBHOOK_URLS.

    Args:
        flow_resource_name:    Full flow resource name.
        version_resource_name: Full version resource name.
        target_env_display_name: "testing" | "development" | "production".

    Returns:
        Long-running operation dict from the update call.
    """
    c = client or DialogflowCXClient()

    env = get_environment(agent_id, target_env_display_name, c)
    if env is None:
        raise ValueError(
            f"Environment '{target_env_display_name}' not found. "
            "Create it first with create_environment()."
        )

    env_id = env["name"].split("/")[-1]

    flow_versions = [
        {
            "version": version_resource_name,
        }
    ]

    # Attach environment-specific webhook URL if configured
    webhook_url = WEBHOOK_URLS.get(target_env_display_name)
    webhook_config = None
    if webhook_url:
        # The override structure — keyed by webhook resource name
        webhook_config = {"genericWebServices": [{"uri": webhook_url}]}

    return update_environment(
        agent_id=agent_id,
        env_id=env_id,
        flow_versions=flow_versions,
        webhook_config=webhook_config,
        client=c,
    )


def bootstrap_environments(
    agent_id: str,
    client: DialogflowCXClient | None = None,
) -> dict[str, dict]:
    """
    Create the testing, development, and production environments if they
    do not already exist.  Idempotent — skips any that already exist.

    Returns:
        Dict mapping env display name → env resource (existing or newly created).
    """
    c = client or DialogflowCXClient()
    results: dict[str, dict] = {}

    for env_name in ENV_PIPELINE:
        existing = get_environment(agent_id, env_name, c)
        if existing:
            results[env_name] = existing
        else:
            op = create_environment(
                agent_id=agent_id,
                display_name=env_name,
                description=(
                    f"Agentic Cinema — {env_name} environment. "
                    "Webhook: " + WEBHOOK_URLS.get(env_name, "(not configured)")
                ),
                client=c,
            )
            results[env_name] = op

    return results


def deploy_version_pipeline(
    agent_id: str,
    flow_resource_name: str,
    version_resource_name: str,
    stop_at: str = ENV_PRODUCTION,
    client: DialogflowCXClient | None = None,
) -> dict[str, dict]:
    """
    Promote a version through the full environment pipeline:
    testing → development → production.

    Args:
        stop_at: Stop promotion at this environment (inclusive).
                 Default is "production" (full pipeline).

    Returns:
        Dict mapping env display name → operation result.
    """
    c = client or DialogflowCXClient()
    results: dict[str, dict] = {}

    for env_display in ENV_PIPELINE:
        op = promote_version(
            agent_id=agent_id,
            flow_resource_name=flow_resource_name,
            version_resource_name=version_resource_name,
            target_env_display_name=env_display,
            client=c,
        )
        results[env_display] = op
        if env_display == stop_at:
            break

    return results

"""
dialogflow/config.py
─────────────────────────────────────────────────────────────────────────────
Dialogflow CX configuration for the "cinema" project.

All resource IDs, regions, and environment names live here.
Secrets (API keys, webhook URLs) are pulled from environment variables
— never hard-coded.
"""

from __future__ import annotations

import os

# ── GCP project ───────────────────────────────────────────────────────────────
PROJECT_ID: str = os.environ.get("GOOGLE_CLOUD_PROJECT", "cinema")
LOCATION: str = os.environ.get("DIALOGFLOW_LOCATION", "us-central1")

# ── Agent ─────────────────────────────────────────────────────────────────────
AGENT_DISPLAY_NAME: str = "cinema"
AGENT_DEFAULT_LANGUAGE: str = "en"
AGENT_TIME_ZONE: str = "America/Los_Angeles"

# ── Environment names ─────────────────────────────────────────────────────────
ENV_DRAFT = "draft"          # the implicit default environment (always draft)
ENV_TESTING = "testing"
ENV_DEVELOPMENT = "development"
ENV_PRODUCTION = "production"

# Ordered promotion pipeline
ENV_PIPELINE = [ENV_TESTING, ENV_DEVELOPMENT, ENV_PRODUCTION]

# ── Core flow names ───────────────────────────────────────────────────────────
FLOW_DEFAULT = "Default Start Flow"      # auto-created by Dialogflow
FLOW_PRODUCTION = "Production Intake"
FLOW_SCRIPT_QUERY = "Script Query"
FLOW_LOCATION_QUERY = "Location Query"
FLOW_AUDIO_QUERY = "Audio Query"

# ── Webhook names ─────────────────────────────────────────────────────────────
WEBHOOK_CINEMA_BACKEND = "cinema-backend"

# Webhook URLs per environment — read from env vars so production URL
# is never committed to source control
WEBHOOK_URLS: dict[str, str] = {
    ENV_TESTING: os.environ.get(
        "WEBHOOK_URL_TESTING",
        "https://testing-cinema-backend.run.app/webhook",
    ),
    ENV_DEVELOPMENT: os.environ.get(
        "WEBHOOK_URL_DEVELOPMENT",
        "https://dev-cinema-backend.run.app/webhook",
    ),
    ENV_PRODUCTION: os.environ.get(
        "WEBHOOK_URL_PRODUCTION",
        "https://cinema-backend.run.app/webhook",
    ),
}

# ── Dialogflow CX REST base URL ───────────────────────────────────────────────
_API_BASE = "https://dialogflow.googleapis.com/v3"

def agent_parent() -> str:
    return f"projects/{PROJECT_ID}/locations/{LOCATION}"

def agent_name(agent_id: str) -> str:
    return f"{agent_parent()}/agents/{agent_id}"

def flows_parent(agent_id: str) -> str:
    return f"{agent_name(agent_id)}/flows"

def flow_name(agent_id: str, flow_id: str) -> str:
    return f"{flows_parent(agent_id)}/{flow_id}"

def versions_parent(agent_id: str, flow_id: str) -> str:
    return f"{flow_name(agent_id, flow_id)}/versions"

def version_name(agent_id: str, flow_id: str, version_id: str) -> str:
    return f"{versions_parent(agent_id, flow_id)}/{version_id}"

def environments_parent(agent_id: str) -> str:
    return f"{agent_name(agent_id)}/environments"

def environment_name(agent_id: str, env_id: str) -> str:
    return f"{environments_parent(agent_id)}/{env_id}"

def webhooks_parent(agent_id: str) -> str:
    return f"{agent_name(agent_id)}/webhooks"

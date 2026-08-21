#!/usr/bin/env python3
"""
dialogflow_cli.py — Dialogflow CX management CLI for the cinema project
─────────────────────────────────────────────────────────────────────────────

Usage
─────
  # Full bootstrap: create agent + environments + webhooks
  python dialogflow_cli.py bootstrap

  # Agent
  python dialogflow_cli.py agent list
  python dialogflow_cli.py agent info

  # Versions
  python dialogflow_cli.py version create --flow "Default Start Flow" --name "v1.0"
  python dialogflow_cli.py version list   --flow "Default Start Flow"
  python dialogflow_cli.py version load   --flow "Default Start Flow" --version-id <id>
  python dialogflow_cli.py version compare --flow "Default Start Flow" \\
      --base <id-or-0> --target <id>

  # Environments
  python dialogflow_cli.py env list
  python dialogflow_cli.py env create --name testing
  python dialogflow_cli.py env promote --flow "Default Start Flow" \\
      --version-id <id> --env testing
  python dialogflow_cli.py env pipeline --flow "Default Start Flow" \\
      --version-id <id> --stop-at production

  # Webhooks
  python dialogflow_cli.py webhook list
  python dialogflow_cli.py webhook bootstrap

Prerequisites
─────────────
  pip install google-auth requests
  gcloud auth application-default login
  export GOOGLE_CLOUD_PROJECT=cinema       # or set in .env
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

# ── Load .env if present (optional quality-of-life) ──────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _print(data: Any) -> None:
    """Pretty-print a dict or list as JSON."""
    print(json.dumps(data, indent=2, default=str))


def _agent_id() -> str:
    """Resolve the agent ID from the live API (creates agent if needed)."""
    from dialogflow.agent import ensure_agent, agent_id_from_name
    agent, created = ensure_agent()
    if created:
        print(f"✨  Created agent: {agent['displayName']}", file=sys.stderr)
    else:
        print(f"✓   Using agent: {agent['displayName']}", file=sys.stderr)
    return agent_id_from_name(agent["name"])


# ── Sub-command handlers ──────────────────────────────────────────────────────

def cmd_bootstrap(args: argparse.Namespace) -> None:
    """Full project bootstrap: agent + environments + webhooks."""
    from dialogflow.agent import ensure_agent, agent_id_from_name
    from dialogflow.environments import bootstrap_environments
    from dialogflow.webhooks import ensure_webhooks

    print("── Step 1/3: Agent ──────────────────────────────────────────")
    agent, created = ensure_agent()
    aid = agent_id_from_name(agent["name"])
    status = "created" if created else "already exists"
    print(f"  {status}: {agent['displayName']}  [{agent['name']}]")

    print("\n── Step 2/3: Environments ───────────────────────────────────")
    envs = bootstrap_environments(aid)
    for name, result in envs.items():
        rname = result.get("name", result.get("response", {}).get("name", "pending"))
        print(f"  {name}: {rname}")

    print("\n── Step 3/3: Webhooks ───────────────────────────────────────")
    webhooks = ensure_webhooks(aid)
    for env, wh in webhooks.items():
        uri = wh.get("genericWebService", {}).get("uri", "?")
        print(f"  {env}: {wh.get('displayName')}  →  {uri}")

    print("\n✅  Bootstrap complete.")


def cmd_agent_list(args: argparse.Namespace) -> None:
    from dialogflow.agent import list_agents
    _print(list_agents())


def cmd_agent_info(args: argparse.Namespace) -> None:
    from dialogflow.agent import ensure_agent
    agent, _ = ensure_agent()
    _print(agent)


def cmd_version_create(args: argparse.Namespace) -> None:
    from dialogflow.flows import get_flow_id, create_version
    aid = _agent_id()
    flow_id = get_flow_id(aid, args.flow)
    if flow_id is None:
        print(f"Error: flow '{args.flow}' not found.", file=sys.stderr)
        sys.exit(1)
    op = create_version(
        agent_id=aid,
        flow_id=flow_id,
        display_name=args.name,
        description=args.description or "",
    )
    print(f"✓  Version creation started (long-running operation):")
    _print(op)


def cmd_version_list(args: argparse.Namespace) -> None:
    from dialogflow.flows import get_flow_id, list_versions
    aid = _agent_id()
    flow_id = get_flow_id(aid, args.flow)
    if flow_id is None:
        print(f"Error: flow '{args.flow}' not found.", file=sys.stderr)
        sys.exit(1)
    versions = list_versions(aid, flow_id)
    if not versions:
        print("No versions found.")
        return
    # Formatted table
    print(f"\n{'Display Name':<30} {'Version ID':<20} {'Status':<12} {'Created'}")
    print("─" * 90)
    for v in versions:
        vid = v["name"].split("/")[-1]
        print(
            f"{v.get('displayName',''):<30} "
            f"{vid:<20} "
            f"{v.get('state',''):<12} "
            f"{v.get('createTime','')}"
        )


def cmd_version_load(args: argparse.Namespace) -> None:
    from dialogflow.flows import get_flow_id, load_version_to_draft
    aid = _agent_id()
    flow_id = get_flow_id(aid, args.flow)
    if flow_id is None:
        print(f"Error: flow '{args.flow}' not found.", file=sys.stderr)
        sys.exit(1)
    op = load_version_to_draft(
        agent_id=aid,
        flow_id=flow_id,
        version_id=args.version_id,
        allow_override_agent_resources=args.override_agent_resources,
    )
    print(f"✓  Load-to-draft operation started:")
    _print(op)


def cmd_version_compare(args: argparse.Namespace) -> None:
    from dialogflow.flows import get_flow_id, compare_versions
    aid = _agent_id()
    flow_id = get_flow_id(aid, args.flow)
    if flow_id is None:
        print(f"Error: flow '{args.flow}' not found.", file=sys.stderr)
        sys.exit(1)
    result = compare_versions(
        agent_id=aid,
        flow_id=flow_id,
        base_version_id=args.base,
        target_version_id=args.target,
    )
    _print(result)


def cmd_env_list(args: argparse.Namespace) -> None:
    from dialogflow.environments import list_environments
    from dialogflow.agent import ensure_agent, agent_id_from_name
    agent, _ = ensure_agent()
    aid = agent_id_from_name(agent["name"])
    envs = list_environments(aid)
    if not envs:
        print("No custom environments. (Draft always exists implicitly.)")
        return
    print(f"\n{'Display Name':<20} {'Environment ID':<40} {'Description'}")
    print("─" * 90)
    for e in envs:
        eid = e["name"].split("/")[-1]
        print(
            f"{e.get('displayName',''):<20} "
            f"{eid:<40} "
            f"{e.get('description','')[:50]}"
        )


def cmd_env_create(args: argparse.Namespace) -> None:
    from dialogflow.environments import create_environment
    aid = _agent_id()
    op = create_environment(
        agent_id=aid,
        display_name=args.name,
        description=args.description or f"Cinema {args.name} environment",
    )
    print(f"✓  Environment creation started:")
    _print(op)


def cmd_env_promote(args: argparse.Namespace) -> None:
    from dialogflow.flows import get_flow_id, flow_name as flow_rname, version_name
    from dialogflow.environments import promote_version
    from dialogflow.config import flows_parent
    aid = _agent_id()
    flow_id = get_flow_id(aid, args.flow)
    if flow_id is None:
        print(f"Error: flow '{args.flow}' not found.", file=sys.stderr)
        sys.exit(1)
    flow_res = flow_rname(aid, flow_id)
    version_res = version_name(aid, flow_id, args.version_id)
    op = promote_version(
        agent_id=aid,
        flow_resource_name=flow_res,
        version_resource_name=version_res,
        target_env_display_name=args.env,
    )
    print(f"✓  Promoted version {args.version_id} → {args.env}:")
    _print(op)


def cmd_env_pipeline(args: argparse.Namespace) -> None:
    from dialogflow.flows import get_flow_id, flow_name as flow_rname, version_name
    from dialogflow.environments import deploy_version_pipeline
    aid = _agent_id()
    flow_id = get_flow_id(aid, args.flow)
    if flow_id is None:
        print(f"Error: flow '{args.flow}' not found.", file=sys.stderr)
        sys.exit(1)
    flow_res = flow_rname(aid, flow_id)
    version_res = version_name(aid, flow_id, args.version_id)
    results = deploy_version_pipeline(
        agent_id=aid,
        flow_resource_name=flow_res,
        version_resource_name=version_res,
        stop_at=args.stop_at,
    )
    for env, op in results.items():
        print(f"  → {env}: {op.get('name', 'ok')}")


def cmd_webhook_list(args: argparse.Namespace) -> None:
    from dialogflow.webhooks import list_webhooks
    aid = _agent_id()
    webhooks = list_webhooks(aid)
    if not webhooks:
        print("No webhooks configured.")
        return
    print(f"\n{'Display Name':<35} {'URI'}")
    print("─" * 90)
    for wh in webhooks:
        uri = wh.get("genericWebService", {}).get("uri", "(service directory)")
        print(f"{wh.get('displayName',''):<35} {uri}")


def cmd_webhook_bootstrap(args: argparse.Namespace) -> None:
    from dialogflow.webhooks import ensure_webhooks
    aid = _agent_id()
    webhooks = ensure_webhooks(aid)
    for env, wh in webhooks.items():
        uri = wh.get("genericWebService", {}).get("uri", "?")
        status = "exists" if wh.get("name") else "created"
        print(f"  {env:<15} [{status}] {wh.get('displayName')}  →  {uri}")


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dialogflow_cli",
        description="Dialogflow CX management CLI — cinema project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── bootstrap ─────────────────────────────────────────────────────────────
    sub.add_parser("bootstrap", help="Full project bootstrap (agent + envs + webhooks)")

    # ── agent ─────────────────────────────────────────────────────────────────
    agent_p = sub.add_parser("agent", help="Agent management")
    agent_sub = agent_p.add_subparsers(dest="agent_cmd", required=True)
    agent_sub.add_parser("list", help="List all agents in the project")
    agent_sub.add_parser("info", help="Show the cinema agent details")

    # ── version ───────────────────────────────────────────────────────────────
    ver_p = sub.add_parser("version", help="Flow version management")
    ver_sub = ver_p.add_subparsers(dest="version_cmd", required=True)

    vc = ver_sub.add_parser("create", help="Snapshot current draft as a version")
    vc.add_argument("--flow", required=True, help="Flow display name")
    vc.add_argument("--name", required=True, help="Version display name, e.g. v1.2")
    vc.add_argument("--description", default="", help="Optional description")

    vl = ver_sub.add_parser("list", help="List versions of a flow")
    vl.add_argument("--flow", required=True, help="Flow display name")

    vload = ver_sub.add_parser("load", help="Load a version back to draft")
    vload.add_argument("--flow", required=True, help="Flow display name")
    vload.add_argument("--version-id", required=True, dest="version_id")
    vload.add_argument(
        "--override-agent-resources",
        action="store_true",
        dest="override_agent_resources",
        help="Also overwrite agent-level resources (intents, entities)",
    )

    vcmp = ver_sub.add_parser("compare", help="Side-by-side diff two versions")
    vcmp.add_argument("--flow", required=True)
    vcmp.add_argument("--base", required=True, help="Base version ID (use 0 for draft)")
    vcmp.add_argument("--target", required=True, help="Target version ID")

    # ── env ───────────────────────────────────────────────────────────────────
    env_p = sub.add_parser("env", help="Environment management")
    env_sub = env_p.add_subparsers(dest="env_cmd", required=True)

    env_sub.add_parser("list", help="List all environments")

    env_create = env_sub.add_parser("create", help="Create a custom environment")
    env_create.add_argument(
        "--name", required=True,
        choices=["testing", "development", "production"],
    )
    env_create.add_argument("--description", default="")

    env_promote = env_sub.add_parser("promote", help="Deploy a version to an environment")
    env_promote.add_argument("--flow", required=True)
    env_promote.add_argument("--version-id", required=True, dest="version_id")
    env_promote.add_argument(
        "--env", required=True,
        choices=["testing", "development", "production"],
    )

    env_pipe = env_sub.add_parser("pipeline", help="Promote a version through all environments")
    env_pipe.add_argument("--flow", required=True)
    env_pipe.add_argument("--version-id", required=True, dest="version_id")
    env_pipe.add_argument(
        "--stop-at", default="production", dest="stop_at",
        choices=["testing", "development", "production"],
    )

    # ── webhook ───────────────────────────────────────────────────────────────
    wh_p = sub.add_parser("webhook", help="Webhook management")
    wh_sub = wh_p.add_subparsers(dest="webhook_cmd", required=True)
    wh_sub.add_parser("list", help="List all webhooks")
    wh_sub.add_parser("bootstrap", help="Create environment-specific webhooks")

    return p


# ── Dispatch ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        ("bootstrap", None, None): cmd_bootstrap,
        ("agent", "list", None): cmd_agent_list,
        ("agent", "info", None): cmd_agent_info,
        ("version", "create", None): cmd_version_create,
        ("version", "list", None): cmd_version_list,
        ("version", "load", None): cmd_version_load,
        ("version", "compare", None): cmd_version_compare,
        ("env", "list", None): cmd_env_list,
        ("env", "create", None): cmd_env_create,
        ("env", "promote", None): cmd_env_promote,
        ("env", "pipeline", None): cmd_env_pipeline,
        ("webhook", "list", None): cmd_webhook_list,
        ("webhook", "bootstrap", None): cmd_webhook_bootstrap,
    }

    key = (
        args.command,
        getattr(args, "agent_cmd", None)
        or getattr(args, "version_cmd", None)
        or getattr(args, "env_cmd", None)
        or getattr(args, "webhook_cmd", None),
        None,
    )

    handler = dispatch.get(key)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except RuntimeError as exc:
        print(f"\n❌  {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()

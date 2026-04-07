"""APIM route update automation for the Teams service.

Usage:
    python -m teams_app.deploy route-teams   # Route /teams/* to port 8300
    python -m teams_app.deploy rollback      # Route /teams/* back to main app
    python -m teams_app.deploy verify        # Check routing is working
    python -m teams_app.deploy status        # Show current APIM config
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import os

logger = logging.getLogger(__name__)

# Azure resource details
RESOURCE_GROUP = "rg-docwain-dev"
APIM_SERVICE = "dhs-docwain-api"
API_ID = "docwain-api"
BACKEND_IP = "4.213.139.185"
TEAMS_PORT = 8300
MAIN_PORT = 8000


def _az(args: list) -> dict:
    """Run an az CLI command and return parsed JSON output."""
    cmd = ["az"] + args + ["--output", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("az command failed: %s\n%s", " ".join(cmd), result.stderr)
        raise RuntimeError(f"az CLI error: {result.stderr}")
    return json.loads(result.stdout) if result.stdout.strip() else {}


def route_teams():
    """Update APIM to route /teams/* requests to the Teams service on port 8300."""
    logger.info("Setting up APIM route: /teams/* -> port %d", TEAMS_PORT)

    policy_xml = f"""<policies>
    <inbound>
        <base />
        <set-backend-service base-url="http://{BACKEND_IP}:{TEAMS_PORT}" />
    </inbound>
    <backend>
        <base />
    </backend>
    <outbound>
        <base />
    </outbound>
    <on-error>
        <base />
    </on-error>
</policies>"""

    # Ensure the operation exists
    try:
        _az([
            "apim", "api", "operation", "show",
            "--resource-group", RESOURCE_GROUP,
            "--service-name", APIM_SERVICE,
            "--api-id", API_ID,
            "--operation-id", "teams-messages",
        ])
        logger.info("Operation teams-messages exists, updating policy...")
    except RuntimeError:
        logger.info("Creating APIM operation teams-messages...")
        _az([
            "apim", "api", "operation", "create",
            "--resource-group", RESOURCE_GROUP,
            "--service-name", APIM_SERVICE,
            "--api-id", API_ID,
            "--url-template", "/teams/messages",
            "--method", "POST",
            "--display-name", "Teams Bot Messages",
            "--operation-id", "teams-messages",
        ])

    # Apply the policy to route to port 8300
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(policy_xml)
        policy_file = f.name

    try:
        subprocess.run([
            "az", "apim", "api", "operation", "policy", "create",
            "--resource-group", RESOURCE_GROUP,
            "--service-name", APIM_SERVICE,
            "--api-id", API_ID,
            "--operation-id", "teams-messages",
            "--xml-file", policy_file,
        ], check=True)
        logger.info("APIM policy applied: /teams/messages -> port %d", TEAMS_PORT)
    finally:
        os.unlink(policy_file)

    print(f"Done. Teams traffic now routes to http://{BACKEND_IP}:{TEAMS_PORT}")


def rollback():
    """Remove the Teams routing policy — traffic falls back to the main app."""
    logger.info("Rolling back APIM route: /teams/* -> main app (port %d)", MAIN_PORT)

    try:
        subprocess.run([
            "az", "apim", "api", "operation", "policy", "delete",
            "--resource-group", RESOURCE_GROUP,
            "--service-name", APIM_SERVICE,
            "--api-id", API_ID,
            "--operation-id", "teams-messages",
            "--yes",
        ], check=True)
        logger.info("APIM policy removed. Teams traffic now goes to main app.")
    except subprocess.CalledProcessError as exc:
        logger.warning("Policy delete failed (may not exist): %s", exc)

    print(f"Done. Teams traffic now routes to main app at http://{BACKEND_IP}:{MAIN_PORT}")


def verify():
    """Verify the Teams service is reachable."""
    import httpx

    direct_url = f"http://localhost:{TEAMS_PORT}/health"
    try:
        resp = httpx.get(direct_url, timeout=5)
        print(f"Direct health check ({direct_url}): {resp.status_code}")
        print(json.dumps(resp.json(), indent=2))
    except Exception as exc:
        print(f"Direct health check FAILED: {exc}")
        return

    print("\nTeams service is running and healthy.")


def status():
    """Show current APIM routing configuration."""
    try:
        ops = _az([
            "apim", "api", "operation", "list",
            "--resource-group", RESOURCE_GROUP,
            "--service-name", APIM_SERVICE,
            "--api-id", API_ID,
        ])
        print("APIM Operations:")
        for op in ops:
            print(f"  {op.get('method', '?')} {op.get('urlTemplate', '?')} -- {op.get('displayName', '?')}")
    except Exception as exc:
        print(f"Failed to query APIM: {exc}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    commands = {
        "route-teams": route_teams,
        "rollback": rollback,
        "verify": verify,
        "status": status,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"Usage: python -m teams_app.deploy <{'|'.join(commands.keys())}>")
        sys.exit(1)

    commands[sys.argv[1]]()

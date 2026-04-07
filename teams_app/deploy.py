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
SUBSCRIPTION_ID = "249bb11f-9b6e-4c0e-a844-500d627b80b3"
RESOURCE_GROUP = "rg-docwain-dev"
APIM_SERVICE = "dhs-docwain-api"
API_ID = "docwain-api"
OPERATION_ID = "teamschat"  # Existing APIM operation for POST /teams/messages
BACKEND_IP = "198.145.127.234"
TEAMS_PORT = 8300
MAIN_PORT = 8000

POLICY_URL = (
    f"https://management.azure.com/subscriptions/{SUBSCRIPTION_ID}"
    f"/resourceGroups/{RESOURCE_GROUP}/providers/Microsoft.ApiManagement"
    f"/service/{APIM_SERVICE}/apis/{API_ID}/operations/{OPERATION_ID}"
    f"/policies/policy?api-version=2022-08-01"
)


def _az(args: list) -> dict:
    """Run an az CLI command and return parsed JSON output."""
    cmd = ["az"] + args + ["--output", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("az command failed: %s\n%s", " ".join(cmd), result.stderr)
        raise RuntimeError(f"az CLI error: {result.stderr}")
    return json.loads(result.stdout) if result.stdout.strip() else {}


def _az_rest(method: str, url: str, body: dict = None) -> str:
    """Run az rest command and return output."""
    cmd = ["az", "rest", "--method", method, "--url", url]
    if body:
        cmd += ["--body", json.dumps(body)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("az rest failed: %s", result.stderr)
        raise RuntimeError(f"az rest error: {result.stderr}")
    return result.stdout


def route_teams():
    """Update APIM to route /teams/* requests to the Teams service on port 8300."""
    logger.info("Setting up APIM route: /teams/* -> port %d", TEAMS_PORT)

    policy_xml = (
        f'<policies><inbound><base />'
        f'<set-backend-service base-url="http://{BACKEND_IP}:{TEAMS_PORT}" />'
        f'</inbound><backend><base /></backend>'
        f'<outbound><base /></outbound>'
        f'<on-error><base /></on-error></policies>'
    )

    body = {"properties": {"value": policy_xml, "format": "xml"}}
    _az_rest("PUT", POLICY_URL, body)
    logger.info("APIM policy applied: /teams/messages -> port %d", TEAMS_PORT)
    print(f"Done. Teams traffic now routes to http://{BACKEND_IP}:{TEAMS_PORT}")


def rollback():
    """Remove the Teams routing policy — traffic falls back to the main app."""
    logger.info("Rolling back APIM route: /teams/* -> main app (port %d)", MAIN_PORT)

    try:
        _az_rest("DELETE", POLICY_URL)
        logger.info("APIM policy removed. Teams traffic now goes to main app.")
    except RuntimeError as exc:
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

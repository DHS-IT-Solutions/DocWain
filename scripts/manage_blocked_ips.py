#!/usr/bin/env python3
"""CLI tool to manage the DocWain IP block list.

Usage:
    python scripts/manage_blocked_ips.py list
    python scripts/manage_blocked_ips.py block --ip 1.2.3.4
    python scripts/manage_blocked_ips.py unblock --ip 1.2.3.4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.config import Config


def _get_path() -> str:
    return getattr(Config.Security, "BLOCKED_IPS_FILE", "data/blocked_ips.json")


def _load() -> dict:
    p = Path(_get_path())
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _save(data: dict) -> None:
    p = Path(_get_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def cmd_list(_args):
    blocked = _load()
    if not blocked:
        print("No blocked IPs.")
        return
    print(f"{'IP Address':<20} {'Blocked At'}")
    print("-" * 50)
    for ip, ts in sorted(blocked.items()):
        print(f"{ip:<20} {ts}")
    print(f"\nTotal: {len(blocked)} blocked IPs")


def cmd_block(args):
    blocked = _load()
    from datetime import datetime, timezone
    blocked[args.ip] = datetime.now(timezone.utc).isoformat()
    _save(blocked)
    print(f"Blocked {args.ip}")


def cmd_unblock(args):
    blocked = _load()
    if args.ip in blocked:
        del blocked[args.ip]
        _save(blocked)
        print(f"Unblocked {args.ip}")
    else:
        print(f"{args.ip} is not in the block list")


def main():
    parser = argparse.ArgumentParser(description="Manage DocWain IP block list")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List all blocked IPs")

    block_p = sub.add_parser("block", help="Manually block an IP")
    block_p.add_argument("--ip", required=True, help="IP address to block")

    unblock_p = sub.add_parser("unblock", help="Unblock an IP")
    unblock_p.add_argument("--ip", required=True, help="IP address to unblock")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    {"list": cmd_list, "block": cmd_block, "unblock": cmd_unblock}[args.command](args)


if __name__ == "__main__":
    main()

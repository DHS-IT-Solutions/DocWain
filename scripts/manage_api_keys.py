#!/usr/bin/env python3
"""CLI tool to manage DocWain Standalone API keys."""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymongo import MongoClient

from src.api.config import Config
from src.api.standalone_auth import generate_api_key


def get_collection():
    client = MongoClient(Config.MongoDB.URI)
    db = client[Config.MongoDB.DB]
    collection_name = getattr(
        getattr(Config, "Standalone", None), "API_KEYS_COLLECTION", None
    ) or "api_keys"
    return db[collection_name]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_create(args):
    raw_key, key_hash = generate_api_key()
    key_prefix = raw_key[:10] + "..."

    doc = {
        "key_hash": key_hash,
        "key_prefix": key_prefix,
        "name": args.name,
        "created_at": datetime.now(tz=timezone.utc),
        "active": True,
        "permissions": ["process", "extract", "batch", "query"],
        "usage": {
            "total_requests": 0,
            "last_used": None,
            "requests_today": 0,
            "documents_processed": 0,
        },
    }
    if args.subscription_id:
        doc["subscription_id"] = args.subscription_id

    col = get_collection()
    col.insert_one(doc)

    print()
    print("API key created successfully.")
    print()
    print(f"  Name            : {args.name}")
    if args.subscription_id:
        print(f"  Subscription ID : {args.subscription_id}")
    print(f"  Key prefix      : {key_prefix}")
    print()
    print("  *** SAVE THIS KEY — it will NOT be shown again ***")
    print()
    print(f"  API Key: {raw_key}")
    print()


def cmd_list(args):
    col = get_collection()
    keys = list(col.find({}))

    if not keys:
        print("No API keys found.")
        return

    # Column widths
    col_widths = {
        "name": max(len("Name"), max(len(k.get("name", "") or "") for k in keys)),
        "prefix": max(len("Prefix"), max(len(k.get("key_prefix", "") or "") for k in keys)),
        "sub": max(len("Subscription"), max(len(k.get("subscription_id", "") or "") for k in keys)),
        "active": len("Active"),
        "requests": len("Requests"),
        "last_used": len("Last Used"),
    }

    header = (
        f"{'Name':<{col_widths['name']}}  "
        f"{'Prefix':<{col_widths['prefix']}}  "
        f"{'Subscription':<{col_widths['sub']}}  "
        f"{'Active':<{col_widths['active']}}  "
        f"{'Requests':>{col_widths['requests']}}  "
        f"Last Used"
    )
    separator = "-" * len(header)

    print()
    print(header)
    print(separator)

    for k in keys:
        name = k.get("name", "") or ""
        prefix = k.get("key_prefix", "") or ""
        sub = k.get("subscription_id", "") or ""
        active = "Yes" if k.get("active", False) else "No"

        usage = k.get("usage", {}) or {}
        # Support both nested usage dict and flat fields (legacy)
        total = usage.get("total_requests") if isinstance(usage, dict) else k.get("total_requests", 0)
        total = total or 0
        last_used_raw = usage.get("last_used") if isinstance(usage, dict) else k.get("last_used")
        if last_used_raw:
            if isinstance(last_used_raw, datetime):
                last_used = last_used_raw.strftime("%Y-%m-%d %H:%M UTC")
            else:
                last_used = str(last_used_raw)
        else:
            last_used = "Never"

        print(
            f"{name:<{col_widths['name']}}  "
            f"{prefix:<{col_widths['prefix']}}  "
            f"{sub:<{col_widths['sub']}}  "
            f"{active:<{col_widths['active']}}  "
            f"{total:>{col_widths['requests']}}  "
            f"{last_used}"
        )

    print()


def cmd_revoke(args):
    col = get_collection()
    result = col.update_one(
        {"key_prefix": {"$regex": f"^{args.prefix}"}},
        {"$set": {"active": False}},
    )
    if result.matched_count == 0:
        print(f"No key found with prefix starting with '{args.prefix}'.")
        sys.exit(1)
    print(f"API key with prefix '{args.prefix}' has been revoked.")


def cmd_reset_usage(args):
    col = get_collection()
    result = col.update_one(
        {"key_prefix": {"$regex": f"^{args.prefix}"}},
        {
            "$set": {
                "usage": {
                    "total_requests": 0,
                    "last_used": None,
                    "requests_today": 0,
                    "documents_processed": 0,
                },
                # Also reset flat fields if present (legacy)
                "total_requests": 0,
                "requests_today": 0,
                "documents_processed": 0,
                "last_used": None,
            }
        },
    )
    if result.matched_count == 0:
        print(f"No key found with prefix starting with '{args.prefix}'.")
        sys.exit(1)
    print(f"Usage counters reset for key with prefix '{args.prefix}'.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manage_api_keys.py",
        description="Manage DocWain Standalone API keys.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # create
    p_create = subparsers.add_parser("create", help="Create a new API key.")
    p_create.add_argument("--name", required=True, help="Human-readable name for this key (e.g. 'Partner X').")
    p_create.add_argument("--subscription-id", required=False, default=None, dest="subscription_id",
                          help="Optional subscription ID to associate with this key.")
    p_create.set_defaults(func=cmd_create)

    # list
    p_list = subparsers.add_parser("list", help="List all API keys.")
    p_list.set_defaults(func=cmd_list)

    # revoke
    p_revoke = subparsers.add_parser("revoke", help="Revoke an API key (set active=False).")
    p_revoke.add_argument("--prefix", required=True,
                          help="Key prefix (first characters before '...') to identify the key.")
    p_revoke.set_defaults(func=cmd_revoke)

    # reset-usage
    p_reset = subparsers.add_parser("reset-usage", help="Reset usage counters for an API key.")
    p_reset.add_argument("--prefix", required=True,
                         help="Key prefix (first characters before '...') to identify the key.")
    p_reset.set_defaults(func=cmd_reset_usage)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

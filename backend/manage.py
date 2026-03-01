#!/usr/bin/env python3
"""
Agent Management CLI

Thin CLI wrapper around AgentRegistryService.
All logic lives in services/agent_registry_service.py.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Manage agent database synchronization"
    )
    parser.add_argument(
        "action",
        choices=["sync-skills", "create-tables"],
        nargs="?",
        default="sync-skills",
        help="Action to perform. sync-skills (default) syncs SKILL.md files to the DB.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress verbose output"
    )

    args = parser.parse_args()

    # Import from the single source of truth
    from services.agent_registry_service import create_tables, sync_skill_entries

    if args.action == "create-tables":
        create_tables()
    elif args.action == "sync-skills":
        result = sync_skill_entries(verbose=not args.quiet)
        if result.get("errors"):
            sys.exit(1)


if __name__ == "__main__":
    main()

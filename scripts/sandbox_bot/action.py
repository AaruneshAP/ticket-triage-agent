"""CLI / GitHub Actions Entrypoint for Sandbox Triage Bot (Phase 10).

Reads GITHUB_EVENT_PATH (if running in GitHub Actions) or command-line arguments,
enforces the strict allow-list, triages the issue, and posts the draft comment.
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath("src"))

from triage.bot.sandbox_bot import assert_sandbox_write_allowed, triage_sandbox_issue

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run sandbox issue triage bot.")
    parser.add_argument("--issue-number", type=int, default=None, help="Issue number to triage.")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without posting comments.")
    args = parser.parse_args()

    issue_number = args.issue_number
    repo_name = os.getenv("GITHUB_REPOSITORY") or os.getenv("GITHUB_SANDBOX_REPO")

    # If triggered via GitHub Actions event payload
    event_path = os.getenv("GITHUB_EVENT_PATH")
    if not issue_number and event_path and os.path.exists(event_path):
        with open(event_path, "r", encoding="utf-8") as f:
            event = json.load(f)
        issue_number = event.get("issue", {}).get("number")
        repo_name = event.get("repository", {}).get("full_name", repo_name)
        logger.info(f"Loaded event payload: Issue #{issue_number} in {repo_name}")

    if not issue_number:
        logger.error("No issue number specified and GITHUB_EVENT_PATH contains no issue.")
        sys.exit(1)

    # 1. Assert Allow-list Guard
    logger.info(f"Checking allow-list for target repository: {repo_name}...")
    assert_sandbox_write_allowed(repo_name)
    logger.info("Allow-list verification passed.")

    # 2. Triage Issue
    logger.info(f"Triaging issue #{issue_number} in {repo_name} (dry_run={args.dry_run})...")
    res = triage_sandbox_issue(issue_number=issue_number, dry_run=args.dry_run)
    logger.info(f"Triage result status: {res.get('status')}")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

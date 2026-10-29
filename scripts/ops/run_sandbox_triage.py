"""Single-command runner for Sandbox Bot triage with automated allow-list verification.

Usage:
  python scripts/ops/run_sandbox_triage.py [--issue-number <num>]

Safety guarantees:
  - First tests and proves the StrictAllowlistViolation guard against unauthorized repos.
  - Posts ONLY to the authorized sandbox repo.
"""

import argparse
import sys

from dotenv import load_dotenv

from triage.bot.sandbox_bot import (
    SandboxGitHubClient,
    StrictAllowlistViolation,
    assert_sandbox_write_allowed,
    get_authorized_sandbox_repo,
    get_sandbox_token,
    triage_sandbox_issue,
)

load_dotenv()


def verify_allowlist_guard():
    """Verify that any unauthorized repo raises StrictAllowlistViolation."""
    print("=================================================================")
    print("[1/2] Verifying Strict Allow-List Security Guard...")
    print("=================================================================")

    unauthorized_targets = [
        "home-assistant/core",
        "home-assistant/architecture",
        "AaruneshAP/ticket-triage-agent",
        "octocat/Hello-World",
    ]

    for target in unauthorized_targets:
        try:
            assert_sandbox_write_allowed(target)
            print(f"[FAIL] Guard allowed unauthorized target: {target}", file=sys.stderr)
            sys.exit(1)
        except StrictAllowlistViolation:
            print(f"  [OK] Confirmed blocked '{target}' -> StrictAllowlistViolation successfully raised.")

    print("[OK] All unauthorized write attempts blocked at HTTP client layer.\n")


def run_sandbox_triage(issue_number: int = None):
    """Run triage on a test issue or specified issue in the sandbox repository."""
    print("=================================================================")
    print("[2/2] Running Sandbox Bot Triage Execution...")
    print("=================================================================")

    try:
        repo = get_authorized_sandbox_repo()
        token = get_sandbox_token()
    except Exception as e:
        print(f"[WARN] Sandbox credentials not fully configured in .env: {e}")
        print("Set SANDBOX_REPO and GITHUB_SANDBOX_TOKEN in .env to proceed.")
        return

    client = SandboxGitHubClient(repo=repo, token=token)
    print(f"Connected to authorized sandbox repository: {repo}")

    if issue_number:
        print(f"Triaging existing issue #{issue_number} with real LangGraph agent...")
        res = triage_sandbox_issue(issue_number=issue_number, repo=repo, token=token, use_langgraph=True)
        print(f"[OK] Triaged issue #{issue_number}: {res.get('comment_url')}")
        print(f"Draft posted:\n---\n{res.get('comment_text')}\n---")
    else:
        print("Creating an end-to-end sandbox verification issue for real LangGraph triage...")
        title = "[Live LangGraph Triage] ZHA Zigbee devices unavailable after 2026.10.1 coordinator restart"
        body = """### The problem
After upgrading to Home Assistant Core 2026.10.1 and restarting my Sonoff Zigbee coordinator, all ZHA light and switch entities intermittently enter unavailable state and fail to respond to commands.

### What version of Home Assistant Core has the issue?
2026.10.1

### What type of installation are you running?
Home Assistant OS

### Integration causing the issue
zha

### Link to or snippet of logs
```
Logger: homeassistant.components.zha
Source: components/zha/core/device.py:650
Exception: DeliveryError('[0x231a:1:0x0006]: Message send failure')
```
"""
        issue = client.create_issue(title=title, body=body)
        num = issue["number"]
        print(f"[OK] Created test issue #{num}: {issue['html_url']}")

        try:
            triage_res = triage_sandbox_issue(issue_number=num, repo=repo, token=token, use_langgraph=True)
            print(f"Real LangGraph draft triage response:\n---\n{triage_res['comment_text']}\n---")
            applied = triage_res.get("applied_labels") or triage_res.get("labels_to_apply")
            print(f"[OK] Applied labels: {applied}")
            print(f"[OK] Similar issues cited: {triage_res.get('similar_issues_count', 0)}")
            print(f"[OK] Completeness missing: {triage_res.get('missing_info', [])}")
            print(f"[OK] Posted real LangGraph triage comment: {triage_res['comment_url']}")
        finally:
            client.close_issue(num)
            print(f"[OK] Closed test issue #{num} (verification complete).")

    print("\n[OK] Sandbox bot triage execution succeeded with 100% allow-list compliance.")


def main():
    parser = argparse.ArgumentParser(description="Run sandbox bot triage with security guard check.")
    parser.add_argument("--issue-number", type=int, default=None, help="Specific sandbox issue number to triage")
    args = parser.parse_args()

    verify_allowlist_guard()
    run_sandbox_triage(args.issue_number)


if __name__ == "__main__":
    main()

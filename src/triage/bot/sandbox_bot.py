"""Sandbox GitHub Triage Bot with Strict Repository Allow-List Enforcement (Phase 10).

Constitution Enforcement:
- home-assistant/core is STRICTLY READ-ONLY.
- The ONLY repository permitted for GitHub API write calls (creating issues, comments, labels)
  is the explicitly designated sandbox repository in GITHUB_SANDBOX_REPO / SANDBOX_REPO.
- Every API mutation asserts this allow-list guard before sending the HTTP request.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from triage.agent.completeness import check_ticket_completeness_rules
from triage.agent.sandbox_guard import validate_sandbox_comment_no_resolution_time
from triage.parse.form_parser import parse_issue

load_dotenv()
logger = logging.getLogger(__name__)


class StrictAllowlistViolation(PermissionError):
    """Raised when an operation attempts to write to an unauthorized GitHub repository."""
    pass


def get_authorized_sandbox_repo() -> str:
    """Retrieve and validate the configured sandbox repository from environment."""
    repo = os.getenv("GITHUB_SANDBOX_REPO") or os.getenv("SANDBOX_REPO")
    if not repo:
        raise ValueError("Neither GITHUB_SANDBOX_REPO nor SANDBOX_REPO is configured in environment.")
    return repo.strip().lower()


def get_sandbox_token() -> str:
    """Retrieve the fine-grained sandbox personal access token."""
    token = os.getenv("GITHUB_SANDBOX_TOKEN") or os.getenv("SANDBOX_TOKEN")
    if not token:
        raise ValueError("Neither GITHUB_SANDBOX_TOKEN nor SANDBOX_TOKEN is configured in environment.")
    return token.strip()


def assert_sandbox_write_allowed(target_repo: str) -> None:
    """Enforce strict write allow-list guard.

    Raises:
        StrictAllowlistViolation: If target_repo does not match the configured sandbox repo.
    """
    authorized_repo = get_authorized_sandbox_repo()
    target_clean = target_repo.strip().lower()

    if target_clean != authorized_repo:
        raise StrictAllowlistViolation(
            f"SECURITY GUARD INTERVENTION: Write operation to repository '{target_repo}' is strictly forbidden. "
            f"Only the designated sandbox repository '{authorized_repo}' is permitted for bot modifications."
        )


class SandboxGitHubClient:
    """GitHub REST API Client bounded strictly by repository allow-list guards."""

    def __init__(self, repo: Optional[str] = None, token: Optional[str] = None):
        self.repo = (repo or get_authorized_sandbox_repo()).strip()
        self.token = token or get_sandbox_token()
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def get_issue(self, issue_number: int) -> Dict[str, Any]:
        """Fetch issue details (read operation)."""
        url = f"{self.base_url}/repos/{self.repo}/issues/{issue_number}"
        resp = requests.get(url, headers=self.headers, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def create_issue(self, title: str, body: str) -> Dict[str, Any]:
        """Create a new issue in the sandbox repo (guarded write operation)."""
        assert_sandbox_write_allowed(self.repo)
        url = f"{self.base_url}/repos/{self.repo}/issues"
        payload = {"title": title, "body": body}
        resp = requests.post(url, headers=self.headers, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def post_comment(self, issue_number: int, comment_body: str) -> Dict[str, Any]:
        """Post a comment on an issue (guarded write operation)."""
        assert_sandbox_write_allowed(self.repo)
        validate_sandbox_comment_no_resolution_time(comment_body)
        url = f"{self.base_url}/repos/{self.repo}/issues/{issue_number}/comments"
        payload = {"body": comment_body}
        resp = requests.post(url, headers=self.headers, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def add_labels(self, issue_number: int, labels: List[str]) -> List[Dict[str, Any]]:
        """Add labels to an issue (guarded write operation)."""
        assert_sandbox_write_allowed(self.repo)
        if not labels:
            return []
        url = f"{self.base_url}/repos/{self.repo}/issues/{issue_number}/labels"
        payload = {"labels": labels}
        resp = requests.post(url, headers=self.headers, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def close_issue(self, issue_number: int) -> Dict[str, Any]:
        """Close an issue (guarded write operation)."""
        assert_sandbox_write_allowed(self.repo)
        url = f"{self.base_url}/repos/{self.repo}/issues/{issue_number}"
        payload = {"state": "closed", "state_reason": "completed"}
        resp = requests.patch(url, headers=self.headers, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()


def generate_triage_draft_and_labels(parsed_issue: Dict[str, Any], raw_body: str, raw_title: str) -> Dict[str, Any]:
    """Generate the standardized triage draft response and recommended GitHub labels."""
    integration = parsed_issue.get("integration_in_form") or "core"
    has_tb = parsed_issue.get("has_traceback", False)
    has_lg = parsed_issue.get("has_logs", False)

    # 1. Evaluate Rule C completeness
    rule_reasons = check_ticket_completeness_rules(parsed_issue)
    is_incomplete = len(rule_reasons) > 0

    labels = []
    if integration and integration not in ["core", "unknown"]:
        labels.append(f"integration: {integration}")

    if is_incomplete:
        labels.append("needs-more-information")

    # 2. Construct factual, neutral draft comment
    lines = [
        "[AI draft, needs review]",
        "",
        "Thank you for reaching out. Based on automated triage analysis:",
        f"- **Identified Component**: `{integration}`",
    ]

    if is_incomplete:
        lines.append("- **Completeness Notice**: This report appears incomplete or lacks actionable diagnostics:")
        for r in rule_reasons:
            lines.append(f"  - {r}")
        lines.append("")
        lines.append("To help maintainers investigate, please provide:")
        if not has_lg:
            lines.append("- System diagnostics or relevant log output from Home Assistant Core (`Settings > System > Logs`).")
        if not has_tb:
            lines.append("- Detailed step-by-step reproduction steps and expected vs actual behavior.")
    else:
        lines.append("- **Status**: Issue contains diagnostic details and reproduction context.")
        if has_tb:
            lines.append("- **Diagnostics**: Python traceback detected in report.")
        elif has_lg:
            lines.append("- **Diagnostics**: System log snippets attached.")

    comment_text = "\n".join(lines)

    return {
        "is_incomplete": is_incomplete,
        "rule_reasons": rule_reasons,
        "integration": integration,
        "labels": labels,
        "comment_text": comment_text,
    }


def triage_sandbox_issue(
    issue_number: int,
    repo: Optional[str] = None,
    token: Optional[str] = None,
    dry_run: bool = False,
    use_langgraph: bool = True,
) -> Dict[str, Any]:
    """Execute triage pipeline for an issue in the sandbox repository."""
    client = SandboxGitHubClient(repo=repo, token=token)
    logger.info(f"Fetching issue #{issue_number} from {client.repo}...")
    issue = client.get_issue(issue_number)

    title = issue.get("title", "")
    body = issue.get("body", "") or ""

    parsed = parse_issue(body, title)
    labels = []
    similar_issues_count = 0
    missing_items = []

    if use_langgraph:
        try:
            from triage.agent.graph import TriageAgentGraph
            logger.info("Running real LangGraph agent (retrieval + completeness + LLM)...")
            agent = TriageAgentGraph()
            agent_res = agent.run({
                "number": issue_number,
                "title": title,
                "body": body,
                "created_at": issue.get("created_at"),
            })
            comment_text = agent_res.get("draft_reply") or ""
            missing_items = agent_res.get("missing_info", [])
            similar_tickets = agent_res.get("similar_tickets", [])
            similar_issues_count = len(similar_tickets)
            sugg_label = agent_res.get("suggested_label", "")

            # Formulate labels
            if sugg_label and sugg_label not in ["unlabeled", "unknown", "other"]:
                labels.append(sugg_label if sugg_label.startswith("integration:") else f"integration: {sugg_label}")
            elif parsed.get("integration_in_form") and parsed["integration_in_form"] not in ["core", "unknown"]:
                labels.append(f"integration: {parsed['integration_in_form']}")

            if missing_items:
                labels.append("needs-more-information")

            triage_result = {
                "integration": parsed.get("integration_in_form") or "core",
                "is_incomplete": len(missing_items) > 0,
                "labels": labels,
                "comment_text": comment_text,
                "similar_tickets": similar_tickets,
                "missing_info": missing_items,
            }
        except Exception as e:
            logger.warning(f"LangGraph execution fallback due to: {e}; using rule template draft.")
            triage_result = generate_triage_draft_and_labels(parsed, body, title)
    else:
        triage_result = generate_triage_draft_and_labels(parsed, body, title)

    result: Dict[str, Any] = {
        "repo": client.repo,
        "issue_number": issue_number,
        "dry_run": dry_run,
        "use_langgraph": use_langgraph,
        "integration": triage_result["integration"],
        "is_incomplete": triage_result["is_incomplete"],
        "labels_to_apply": triage_result["labels"],
        "comment_text": triage_result["comment_text"],
        "similar_issues_count": similar_issues_count,
        "missing_info": missing_items,
    }

    if dry_run:
        logger.info(f"[DRY RUN] Would post comment and labels to #{issue_number}:\n{triage_result['comment_text']}")
        result["status"] = "dry_run_success"
        return result

    # Guarded live writes
    logger.info(f"Posting triage comment to #{issue_number}...")
    comment_resp = client.post_comment(issue_number, triage_result["comment_text"])
    result["comment_id"] = comment_resp.get("id")
    result["comment_url"] = comment_resp.get("html_url")

    if triage_result["labels"]:
        logger.info(f"Applying labels {triage_result['labels']} to #{issue_number}...")
        try:
            labels_resp = client.add_labels(issue_number, triage_result["labels"])
            result["applied_labels"] = [lbl.get("name") for lbl in labels_resp]
        except Exception as e:
            logger.warning(f"Could not apply labels (they may need to be created first): {e}")
            result["label_warning"] = str(e)

    result["status"] = "live_triage_success"
    return result

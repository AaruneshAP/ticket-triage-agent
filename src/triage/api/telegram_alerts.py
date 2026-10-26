"""Telegram and n8n Alert Notification Dispatcher (Phase 9).

Dispatches real-time triage notifications for incoming GitHub issues:
1. Primary path: POST to n8n webhook endpoint (`http://localhost:5678/webhook/triage-alert`).
2. Fallback path: Direct HTTPS dispatch to Telegram Bot API (`/sendMessage`).
"""

import logging
import os
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/triage-alert")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def format_telegram_alert(ticket_data: Dict[str, Any]) -> str:
    """Format structured ticket triage data into a readable Telegram Markdown notification."""
    num = ticket_data.get("ticket_number", "N/A")
    title = str(ticket_data.get("title", "Untitled"))[:75]
    label = ticket_data.get("suggested_label", "unlabeled")
    conf = ticket_data.get("label_confidence")
    conf_str = f"({conf * 100:.0f}% conf)" if conf is not None else ""

    p_np = ticket_data.get("p_not_planned")
    p_np_str = f"{p_np * 100:.1f}%" if p_np is not None else "N/A"

    is_complete = ticket_data.get("is_complete", True)
    missing = ticket_data.get("missing_info", [])
    if isinstance(missing, list):
        missing_str = ", ".join(missing) if missing else "None"
    else:
        missing_str = str(missing)

    ttc = ticket_data.get("predicted_time_to_close", {})
    p50 = ttc.get("p50_days", 3.5)
    p90 = ttc.get("p90_days", 14.0)

    is_high_risk = (p_np and p_np >= 0.35) or not is_complete or (p90 and float(p90) >= 30.0)
    severity_header = "🚨 *[HIGH PRIORITY TRIAGE ALERT]*" if is_high_risk else "ℹ️ *[TRIAGE NOTIFICATION]*"

    lines = [
        severity_header,
        f"• *Ticket*: [#{num}](https://github.com/home-assistant/core/issues/{num})",
        f"• *Title*: `{title}`",
        f"• *Integration*: `{label}` {conf_str}",
        f"• *Not-Planned Risk*: *{p_np_str}*",
        f"• *Resolution Forecast*: P50: *{p50}d* | P90: *{p90}d*",
        f"• *Completeness*: {'✅ Complete' if is_complete else '⚠️ *Incomplete Report*'}",
    ]

    if not is_complete and missing_str != "None":
        lines.append(f"• *Missing Diagnostics*: {missing_str}")

    return "\n".join(lines)


def dispatch_direct_telegram(
    ticket_data: Dict[str, Any],
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Send alert directly to Telegram Bot API."""
    bot_token = token or os.getenv("TELEGRAM_BOT_TOKEN")
    target_chat = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not target_chat:
        logger.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not configured.")
        return {"status": "skipped", "reason": "Missing Telegram credentials"}

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    num = ticket_data.get("ticket_number", "")
    text = format_telegram_alert(ticket_data)

    payload = {
        "chat_id": str(target_chat),
        "text": text,
        "parse_mode": "Markdown",
        "reply_markup": {
            "inline_keyboard": [
                [
                    {
                        "text": "🐙 View on GitHub",
                        "url": f"https://github.com/home-assistant/core/issues/{num}",
                    }
                ]
            ]
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=8)
        data = resp.json()
        if data.get("ok"):
            logger.info(f"Direct Telegram alert dispatched for ticket #{num}")
            return {
                "status": "success",
                "channel": "telegram_direct",
                "message_id": data.get("result", {}).get("message_id"),
            }
        else:
            logger.warning(f"Telegram API responded with error: {data.get('description')}")
            return {"status": "error", "message": data.get("description")}
    except Exception as e:
        logger.error(f"Failed to reach Telegram API: {e}")
        return {"status": "error", "message": str(e)}


def dispatch_ticket_alert(
    ticket_data: Dict[str, Any],
    n8n_url: str = N8N_WEBHOOK_URL,
    allow_fallback: bool = True,
) -> Dict[str, Any]:
    """Dispatch ticket alert via n8n webhook with automatic direct Telegram fallback."""
    try:
        resp = requests.post(n8n_url, json=ticket_data, timeout=4)
        if resp.status_code in [200, 201]:
            logger.info(f"Alert delivered via n8n webhook: {n8n_url}")
            return {"status": "success", "channel": "n8n_webhook", "code": resp.status_code}
    except Exception as e:
        logger.debug(f"n8n webhook dispatch notice ({n8n_url}): {e}")

    if allow_fallback:
        return dispatch_direct_telegram(ticket_data)

    return {"status": "failed", "reason": "n8n webhook unreachable and fallback disabled"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_ticket = {
        "ticket_number": 177225,
        "title": "Tasmota switch integration disconnects after update",
        "suggested_label": "integration: tasmota",
        "label_confidence": 0.98,
        "p_not_planned": 0.382,
        "is_complete": False,
        "missing_info": ["logs", "reproduction_steps"],
        "predicted_time_to_close": {"p50_days": 4.2, "p90_days": 18.5},
    }
    print(dispatch_ticket_alert(sample_ticket))

"""
Turkish Agent — Daily Briefing
Agent reasons about what matters today. Not a template.
"""
from agent.core import run, store_memory
from datetime import datetime, timezone, timedelta
import urllib.request
import json
import os


TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def send_telegram(text):
    payload = json.dumps({
        "chat_id": int(TELEGRAM_CHAT_ID),
        "text": text,
        "parse_mode": "Markdown"
    }).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


BRIEFING_PROMPT = """Generate my morning briefing.

Check vacancies, requirements, deals, and matches.
Reason about what actually needs attention today — not just what exists.

Structure your response as:
1. What needs action TODAY (max 3 items, most urgent first)
2. Pipeline snapshot (key numbers only)
3. Best vacancy/requirement matches right now
4. One thing I should be aware of that I might be missing

Be direct. Dollar figures where relevant. No fluff."""


def generate_and_send():
    """Generate briefing via agent, send via Telegram, store for learning."""
    try:
        briefing = run(BRIEFING_PROMPT)

        # Send via Telegram
        send_telegram(briefing)

        # Store briefing for future learning
        from agent.core import db_insert
        db_insert("t_briefings", {
            "content": briefing,
            "delivered_via": "telegram",
            "reasoning": "Agent-generated morning briefing"
        })

        print(f"Briefing sent: {len(briefing)} chars")
        return True

    except Exception as e:
        print(f"Briefing error: {e}")
        # Send error via Telegram so Michael knows
        try:
            send_telegram(f"Turkish briefing error: {e}")
        except Exception:
            pass
        return False

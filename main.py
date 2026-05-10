"""
Turkish — Main application entry point.
Flask app + scheduler. Agent core is in agent.py.
"""
import os, json, logging, threading
from datetime import datetime, timezone, timedelta
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("turkish.main")

app = Flask(__name__)

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
BRIEFING_HOUR  = int(os.environ.get("BRIEFING_HOUR_UTC", "22"))  # 8am AEST

# ── Auth ──────────────────────────────────────────────────
def require_auth():
    auth = request.headers.get("Authorization", "")
    return auth == f"Bearer {ADMIN_PASSWORD}"

# ── Telegram webhook / polling ────────────────────────────
_tg_offset = 0

def poll_telegram():
    """Poll Telegram for new messages and respond via agent."""
    global _tg_offset
    updates = agent.tg_get_updates(offset=_tg_offset if _tg_offset else None)
    for update in updates:
        _tg_offset = update["update_id"] + 1
        msg = update.get("message", {})
        text = msg.get("text", "").strip()
        chat_id = str(msg.get("chat", {}).get("id", ""))
        if not text or not chat_id:
            continue
        # Only respond to Michael's chat
        if chat_id != agent.TG_CHAT_ID:
            log.warning("Message from unknown chat_id: %s", chat_id)
            continue
        log.info("Telegram message: %s", text[:80])
        try:
            response = agent.run_agent(text)
            # Telegram has 4096 char limit — split if needed
            for i in range(0, len(response), 4000):
                agent.tg_send(response[i:i+4000])
        except Exception as e:
            log.error("Agent error: %s", e)
            agent.tg_send(f"⚠️ Error: {e}")

# ── HTTP Routes ───────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok", "agent": "Turkish", "time": datetime.now(timezone(timedelta(hours=10))).isoformat()})

@app.route("/api/ask", methods=["POST"])
def api_ask():
    if not require_auth():
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json() or {}
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "message required"}), 400
    response = agent.run_agent(message)
    return jsonify({"response": response})

@app.route("/api/briefing/send", methods=["POST"])
def api_briefing():
    if not require_auth():
        return jsonify({"error": "unauthorized"}), 401
    threading.Thread(target=agent.send_daily_briefing, daemon=True).start()
    return jsonify({"success": True, "status": "generating"})

@app.route("/api/embed", methods=["POST"])
def api_embed():
    """Manually embed a piece of text into memory."""
    if not require_auth():
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json() or {}
    table = data.get("table", "t_memory")
    record_id = data.get("id", 0)
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "text required"}), 400
    agent.embed_and_store(table, record_id, text)
    return jsonify({"success": True})

@app.route("/api/memory", methods=["POST"])
def api_memory():
    """Store a fact in agent memory."""
    if not require_auth():
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json() or {}
    agent.store_memory(data.get("fact",""), data.get("context",""), source="api")
    return jsonify({"success": True})

@app.route("/api/ingest/email", methods=["POST"])
def api_ingest_email():
    """Ingest a new email — AI processes and stores it."""
    if not require_auth():
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json() or {}
    # Store raw email
    row = agent.sb_insert("t_emails", {
        "gmail_id": data.get("gmail_id"),
        "from_address": data.get("from"),
        "from_name": data.get("from_name"),
        "subject": data.get("subject"),
        "body": data.get("body"),
        "received_at": data.get("received_at"),
        "processed": False,
    })
    if row and len(row) > 0:
        email_id = row[0]["id"]
        # Let agent process it
        threading.Thread(
            target=process_email_async,
            args=(email_id, data),
            daemon=True
        ).start()
        return jsonify({"success": True, "id": email_id})
    return jsonify({"error": "insert failed"}), 500

def process_email_async(email_id, data):
    """Agent reads email, extracts meaning, stores it."""
    try:
        prompt = (
            f"New email received:\n"
            f"From: {data.get('from_name')} <{data.get('from')}>\n"
            f"Subject: {data.get('subject')}\n"
            f"Body: {data.get('body', '')[:2000]}\n\n"
            f"1. Summarise in one sentence.\n"
            f"2. Priority: High/Normal/Low\n"
            f"3. Action required? What?\n"
            f"4. Is this related to a property, deal, or requirement in our system? Which one?\n"
            f"Reply as JSON: {{summary, priority, action, action_required, related_to}}"
        )
        resp = agent.openai("/chat/completions", {
            "model": agent.OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "You are a commercial real estate assistant. Reply only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 300,
        })
        raw = resp["choices"][0]["message"]["content"]
        try:
            ai = json.loads(raw.strip().strip("```json").strip("```"))
        except:
            ai = {"summary": raw[:200], "priority": "Normal", "action_required": False}

        # Update email with AI analysis
        agent.sb("PATCH", f"/rest/v1/t_emails?id=eq.{email_id}", body={
            "ai_summary": ai.get("summary"),
            "ai_priority": ai.get("priority", "Normal"),
            "ai_action": ai.get("action"),
            "action_required": ai.get("action_required", False),
            "processed": True,
        })
        # Embed for semantic search
        text = f"Email from {data.get('from_name')}: {data.get('subject')}. {ai.get('summary')}"
        agent.embed_and_store("t_emails", email_id, text)
        log.info("Email %s processed: %s", email_id, ai.get("summary","")[:60])
    except Exception as e:
        log.error("Email processing error: %s", e)

# ── Scheduler ─────────────────────────────────────────────
scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(agent.send_daily_briefing, CronTrigger(hour=BRIEFING_HOUR, minute=0),
                  id="daily_briefing", misfire_grace_time=3600)
scheduler.add_job(poll_telegram, "interval", seconds=3,
                  id="telegram_poll", max_instances=1)
scheduler.start()
log.info("Turkish is awake. Briefing at %02d:00 UTC. Telegram polling active.", BRIEFING_HOUR)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


@app.route("/api/embed/all", methods=["POST"])
def api_embed_all():
    """Embed all records in all tables. Run once after data migration."""
    if not require_auth():
        return jsonify({"error": "unauthorized"}), 401
    threading.Thread(target=run_full_embed, daemon=True).start()
    return jsonify({"success": True, "status": "embedding started in background"})

def run_full_embed():
    import time
    tables = {
        "t_vacancies": lambda r: f"Vacancy: {r.get('address','')}, {r.get('suburb','')}. Size: {r.get('size_sqm','')}sqm. Rent: ${r.get('asking_rent_pa','')}pa. Status: {r.get('status','')}. Vacating: {r.get('vacating_tenant','')}.",
        "t_requirements": lambda r: f"Requirement: {r.get('company','')} needs {r.get('size_min','')}-{r.get('size_max','')}sqm in {r.get('preferred_location','')}. Budget: ${r.get('budget_pa','')}pa. Status: {r.get('status','')}.",
        "t_deals": lambda r: f"Deal: {r.get('tenant','')} at {r.get('address','')}. Size: {r.get('size_sqm','')}sqm. Rent: ${r.get('rent_pa','')}pa. Term: {r.get('term_years','')}yrs. Status: {r.get('status','')}.",
        "t_properties": lambda r: f"Property: {r.get('address','')}, {r.get('suburb','')}. Size: {r.get('size_sqm','')}sqm. Status: {r.get('status','')}. Landlord: {r.get('landlord','')}. Occupier: {r.get('occupier','')}.",
        "t_style_profile": lambda r: f"About Michael — {r.get('category','')}/{r.get('key','')}: {r.get('value','')}",
    }
    total = 0
    for table, text_fn in tables.items():
        rows = agent.sb_get(table, limit=500)
        for row in rows:
            try:
                text = text_fn(row)
                if text and len(text.strip()) > 5:
                    agent.embed_and_store(table, row['id'], text)
                    total += 1
                    time.sleep(0.05)
            except Exception as e:
                log.error("embed error %s %s: %s", table, row.get('id'), e)
        log.info("Embedded %s: %d rows", table, len(rows))
    log.info("Full embed complete: %d total", total)

def startup_embed_if_needed():
    """On startup, embed all data if embeddings table is empty."""
    import time
    time.sleep(5)  # Wait for app to be fully ready
    try:
        existing = agent.sb_get("t_embeddings", limit=1)
        if existing:
            log.info("Embeddings already exist (%d+), skipping startup embed", len(existing))
            return
        log.info("No embeddings found — running initial embed of all data")
        run_full_embed()
    except Exception as e:
        log.error("Startup embed check failed: %s", e)

# Run startup embed in background thread
threading.Thread(target=startup_embed_if_needed, daemon=True).start()

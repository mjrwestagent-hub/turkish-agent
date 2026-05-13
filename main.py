"""
Turkish — Main application entry point.
Flask + APScheduler. Agent core in agent.py.
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
BRIEFING_HOUR = int(os.environ.get("BRIEFING_HOUR_UTC", "22"))

def require_auth():
    return request.headers.get("Authorization","") == f"Bearer {agent.ADMIN_PASS}"

# ── Telegram polling ──────────────────────────────────────
_tg_offset = 0

def poll_telegram():
    global _tg_offset
    try:
        updates = agent.tg_get_updates(offset=_tg_offset if _tg_offset else None)
        for update in updates:
            _tg_offset = update["update_id"] + 1
            msg = update.get("message", {})
            text = msg.get("text", "").strip()
            chat_id = str(msg.get("chat", {}).get("id", ""))
            if not text or not chat_id:
                continue
            if chat_id != agent.TG_CHAT_ID:
                continue
            log.info("Telegram: %s", text[:80])
            if text.strip() == "/embed":
                agent.tg_send("Starting memory load...")
                threading.Thread(target=embed_all_records, daemon=True).start()
            elif text.strip() == "/status":
                rows = agent.sb_get("t_embeddings", limit=1)
                count = len(agent.sb_get("t_embeddings", limit=500))
                agent.tg_send(f"Embeddings in memory: {count}\nVacancies: {len(agent.sb_get(chr(116)+chr(95)+chr(118)+chr(97)+chr(99)+chr(97)+chr(110)+chr(99)+chr(105)+chr(101)+chr(115)))}\nRequirements: {len(agent.sb_get(chr(116)+chr(95)+chr(114)+chr(101)+chr(113)+chr(117)+chr(105)+chr(114)+chr(101)+chr(109)+chr(101)+chr(110)+chr(116)+chr(115)))}")
            else:
                threading.Thread(
                    target=lambda t=text: agent.tg_send(agent.run_agent(t)),
                    daemon=True
                ).start()
    except Exception as e:
        log.debug("Telegram poll error: %s", e)

# ── Routes ────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "agent": "Turkish",
        "time": datetime.now(timezone(timedelta(hours=10))).strftime("%Y-%m-%d %I:%M%p AEST"),
        "supabase": bool(agent.SUPABASE_URL),
        "openai": bool(agent.OPENAI_KEY),
        "telegram": bool(agent.TG_TOKEN),
    })

@app.route("/api/ask", methods=["POST"])
def api_ask():
    if not require_auth(): return jsonify({"error":"unauthorized"}), 401
    data = request.get_json() or {}
    msg = data.get("message","")
    if not msg: return jsonify({"error":"message required"}), 400
    response = agent.run_agent(msg)
    return jsonify({"response": response})

@app.route("/api/briefing/send", methods=["POST"])
def api_briefing():
    if not require_auth(): return jsonify({"error":"unauthorized"}), 401
    threading.Thread(target=agent.send_daily_briefing, daemon=True).start()
    return jsonify({"success": True})

@app.route("/api/embed_all", methods=["POST"])
def api_embed_all():
    if not require_auth(): return jsonify({"error":"unauthorized"}), 401
    threading.Thread(target=embed_all_records, daemon=True).start()
    return jsonify({"success": True, "status": "embedding started"})

@app.route("/api/memory", methods=["POST"])
def api_memory():
    if not require_auth(): return jsonify({"error":"unauthorized"}), 401
    data = request.get_json() or {}
    agent.store_memory(data.get("fact",""), data.get("context",""), source="api")
    return jsonify({"success": True})

@app.route("/api/ingest/email", methods=["POST"])
def api_ingest_email():
    if not require_auth(): return jsonify({"error":"unauthorized"}), 401
    data = request.get_json() or {}
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
        threading.Thread(target=process_email, args=(email_id, data), daemon=True).start()
        return jsonify({"success": True, "id": email_id})
    return jsonify({"error": "insert failed"}), 500

def process_email(email_id, data):
    try:
        prompt = (
            f"Email from {data.get('from_name')} <{data.get('from')}>\n"
            f"Subject: {data.get('subject')}\n"
            f"Body: {data.get('body','')[:2000]}\n\n"
            f"Reply ONLY with JSON: {{\"summary\":\"...\",\"priority\":\"High/Normal/Low\","
            f"\"action_required\":true/false,\"action\":\"...\"}}"
        )
        resp = agent.openai_call("/chat/completions", {
            "model": agent.OPENAI_MODEL,
            "messages": [
                {"role":"system","content":"You are a commercial real estate assistant. Reply only with valid JSON."},
                {"role":"user","content":prompt}
            ],
            "temperature": 0.1, "max_tokens": 200,
        })
        raw = resp["choices"][0]["message"]["content"].strip().strip("```json").strip("```")
        try:
            ai = json.loads(raw)
        except:
            ai = {"summary": raw[:200], "priority": "Normal", "action_required": False}
        agent.sb("PATCH", f"/rest/v1/t_emails?id=eq.{email_id}", body={
            "ai_summary": ai.get("summary"),
            "ai_priority": ai.get("priority","Normal"),
            "ai_action": ai.get("action"),
            "action_required": ai.get("action_required", False),
            "processed": True,
        })
        text = f"Email from {data.get('from_name')}: {data.get('subject')}. {ai.get('summary','')}"
        agent.embed_and_store("t_emails", email_id, text)
        log.info("Email %s processed", email_id)
    except Exception as e:
        log.error("Email processing error: %s", e)

def embed_all_records():
    """Embed all records into pgvector memory."""
    log.info("Starting bulk embed...")
    total = 0
    tables = {
        "t_vacancies": lambda r: f"Vacancy: {r.get('address','?')}, {r.get('suburb','?')}. Size: {r.get('size_sqm','?')}sqm. Rent: ${r.get('asking_rent_pa','?')}pa. Status: {r.get('status','?')}. Vacating: {r.get('vacating_tenant','?')}.",
        "t_requirements": lambda r: f"Requirement: {r.get('company','?')} needs {r.get('size_min','?')}-{r.get('size_max','?')}sqm in {r.get('preferred_location','?')}. Budget: ${r.get('budget_pa','?')}pa. Status: {r.get('status','?')}.",
        "t_deals": lambda r: f"Deal: {r.get('tenant','?')} at {r.get('address','?')}. Landlord: {r.get('landlord','?')}. Size: {r.get('size_sqm','?')}sqm. Rent: ${r.get('rent_pa','?')}pa. Term: {r.get('term_years','?')}yrs. Status: {r.get('status','?')}.",
        "t_properties": lambda r: f"Property: {r.get('address','?')}, {r.get('suburb','?')}. Size: {r.get('size_sqm','?')}sqm. Status: {r.get('status','?')}. Landlord: {r.get('landlord','?')}. Occupier: {r.get('occupier','?')}.",
    }
    for table, text_fn in tables.items():
        rows = agent.sb_get(table, limit=500)
        for r in rows:
            agent.embed_and_store(table, r["id"], text_fn(r))
            total += 1
    log.info("Bulk embed complete: %d records", total)
    agent.tg_send(f"Memory loaded: {total} records embedded into vector search.")

def check_gmail():
    """Poll Gmail, process new emails through agent, store in DB."""
    try:
        emails = agent.gmail_fetch_new(max_results=10)
        if not emails:
            return
        log.info("Gmail: %d new emails", len(emails))
        for email in emails:
            # Check if already processed
            existing = agent.sb_get("t_emails", {"gmail_id": f"eq.{email['gmail_id']}"})
            if existing:
                continue
            # Store and process
            row = agent.sb_insert("t_emails", {
                "gmail_id": email["gmail_id"],
                "from_address": email["from"],
                "subject": email["subject"],
                "body": email["body"],
                "received_at": email["date"],
                "processed": False,
            })
            if row and len(row) > 0:
                process_email(row[0]["id"], {
                    "from": email["from"],
                    "from_name": email["from"].split("<")[0].strip(),
                    "subject": email["subject"],
                    "body": email["body"],
                })
                agent.gmail_mark_read(email["gmail_id"])
    except Exception as e:
        log.error("check_gmail: %s", e)

# ── Scheduler ─────────────────────────────────────────────
scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(agent.send_daily_briefing, CronTrigger(hour=BRIEFING_HOUR, minute=0),
                  id="daily_briefing", misfire_grace_time=3600)
scheduler.add_job(poll_telegram, "interval", seconds=3, id="telegram_poll", max_instances=1)
scheduler.add_job(check_gmail, "interval", minutes=15, id="gmail_check", max_instances=1)
scheduler.start()

log.info("Turkish awake. Briefing at %02d:00 UTC. Telegram polling.", BRIEFING_HOUR)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

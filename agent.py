"""
Turkish — Personal Industrial Leasing Agent
Core agent: builds context from memory, reasons with GPT-4o, acts via tools.
"""
import os, json, logging
from datetime import datetime, timezone, timedelta
import urllib.request as ur
import urllib.parse as up

log = logging.getLogger("turkish")

# ── Config (safe defaults so import never crashes) ────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
OPENAI_KEY   = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
EMBED_MODEL  = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
TG_TOKEN     = os.environ.get("TELEGRAM_TOKEN", "")
TG_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
ADMIN_PASS   = os.environ.get("ADMIN_PASSWORD", "")

AEST = timezone(timedelta(hours=10))

# ── Supabase ──────────────────────────────────────────────
def sb(method, path, body=None, params=None):
    """Single Supabase HTTP call. Returns parsed JSON or None on error."""
    if not SUPABASE_URL:
        return None
    url = f"{SUPABASE_URL}{path}"
    if params:
        url += "?" + up.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    req = ur.Request(url, data=data, method=method, headers={
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    })
    try:
        with ur.urlopen(req) as r:
            raw = r.read()
            return json.loads(raw) if raw else []
    except ur.HTTPError as e:
        log.error("Supabase %s %s → %s", method, path, e.code)
        return None
    except Exception as e:
        log.error("Supabase error: %s", e)
        return None

def sb_get(table, filters=None, order=None, limit=200):
    params = {"select": "*", "limit": limit}
    if filters:
        params.update(filters)
    if order:
        params["order"] = order
    return sb("GET", f"/rest/v1/{table}", params=params) or []

def sb_insert(table, row):
    return sb("POST", f"/rest/v1/{table}", body=row)

def sb_rpc(fn, args):
    return sb("POST", f"/rest/v1/rpc/{fn}", body=args)

# ── OpenAI ────────────────────────────────────────────────
def openai_call(path, body):
    if not OPENAI_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    req = ur.Request(
        f"https://api.openai.com/v1{path}",
        data=json.dumps(body).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {OPENAI_KEY}",
            "Content-Type": "application/json",
        }
    )
    with ur.urlopen(req) as r:
        return json.loads(r.read())

def embed(text):
    """Get embedding vector for a piece of text."""
    resp = openai_call("/embeddings", {"model": EMBED_MODEL, "input": text[:8000]})
    return resp["data"][0]["embedding"]

def embed_and_store(table, record_id, text):
    """Embed text and store in t_embeddings."""
    if not text or len(text.strip()) < 10:
        return
    try:
        vector = embed(text)
        sb_insert("t_embeddings", {
            "source_table": table,
            "source_id": record_id,
            "text_chunk": text[:2000],
            "embedding": vector,
        })
    except Exception as e:
        log.error("embed_and_store error [%s %s]: %s", table, record_id, e)

# ── Memory ────────────────────────────────────────────────
def search_context(query, n=8):
    """Semantic search across all embedded knowledge."""
    try:
        vector = embed(query)
        results = sb_rpc("search_memory", {
            "query_embedding": vector,
            "match_count": n,
            "source_filter": None
        })
        if not results:
            return ""
        chunks = [f"[{r['source_table']}] {r['text_chunk']}" for r in results]
        return "\n\n".join(chunks)
    except Exception as e:
        log.error("search_context error: %s", e)
        return ""

def get_style_profile():
    """Load Michael's style profile."""
    rows = sb_get("t_style_profile")
    if not rows:
        return "Michael West, State Leader, MJR West Industrial, Melbourne"
    lines = [f"{r['category']}/{r['key']}: {r['value']}" for r in rows]
    return "\n".join(lines)

def store_memory(fact, context="", source="agent"):
    """Store a learned fact permanently."""
    try:
        row = sb_insert("t_memory", {
            "fact": fact,
            "context": context,
            "source": source,
            "confidence": 1.0,
            "active": True,
        })
        if row and len(row) > 0:
            embed_and_store("t_memory", row[0]["id"], f"{fact}. {context}")
    except Exception as e:
        log.error("store_memory error: %s", e)

# ── Agent Tools ───────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_vacancies",
            "description": "Get available properties for lease",
            "parameters": {"type": "object", "properties": {
                "status": {"type": "string", "default": "Available"},
            }}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_requirements",
            "description": "Get active tenant requirements — companies looking for space",
            "parameters": {"type": "object", "properties": {
                "status": {"type": "string", "default": "Active"},
                "priority": {"type": "string"},
            }}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_deals",
            "description": "Get deals — signed leases, negotiations, pipeline",
            "parameters": {"type": "object", "properties": {
                "status": {"type": "string"},
            }}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_properties",
            "description": "Get all properties in the portfolio",
            "parameters": {"type": "object", "properties": {
                "status": {"type": "string"},
            }}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Semantic search across all stored knowledge — emails, notes, memory, market data",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"},
                "n": {"type": "integer", "default": 6},
            }, "required": ["query"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_matches",
            "description": "Find vacancy-requirement matches by size and location",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_staff_summary",
            "description": "Get performance summary for a staff member",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string"},
            }, "required": ["name"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "store_learning",
            "description": "Store something important learned in this conversation for permanent memory",
            "parameters": {"type": "object", "properties": {
                "fact": {"type": "string"},
                "context": {"type": "string"},
            }, "required": ["fact"]}
        }
    },
]

def fmt_size(v):
    try: return f"{int(float(v)):,}sqm" if v else "?"
    except: return "?"

def fmt_rent(v):
    try: return f"${float(v):,.0f}pa" if v else "—"
    except: return "—"

def execute_tool(name, args):
    """Execute a tool and return result as string."""
    try:
        if name == "get_vacancies":
            rows = sb_get("t_vacancies", {"status": f"eq.{args.get('status','Available')}"}, order="size_sqm.desc")
            if not rows: return "No vacancies found."
            lines = [f"• {r.get('address','?')}, {r.get('suburb','?')} — {fmt_size(r.get('size_sqm'))} — {fmt_rent(r.get('asking_rent_pa'))}" for r in rows]
            return f"{len(rows)} vacancies:\n" + "\n".join(lines)

        elif name == "get_requirements":
            filters = {"status": f"eq.{args.get('status','Active')}"}
            if args.get("priority"): filters["priority"] = f"eq.{args['priority']}"
            rows = sb_get("t_requirements", filters)
            if not rows: return "No requirements found."
            lines = []
            for r in rows:
                sz = f"{fmt_size(r.get('size_min'))}-{fmt_size(r.get('size_max'))}"
                lines.append(f"• {r.get('company','?')} — {sz} — {r.get('preferred_location','?')} — last contact: {r.get('last_contact') or 'unknown'}")
            return f"{len(rows)} requirements:\n" + "\n".join(lines)

        elif name == "get_deals":
            filters = {}
            if args.get("status"): filters["status"] = f"eq.{args['status']}"
            rows = sb_get("t_deals", filters, order="created_at.desc")
            if not rows: return "No deals found."
            lines = [f"• {r.get('tenant','?')} @ {r.get('address','?')} — {fmt_size(r.get('size_sqm'))} — {fmt_rent(r.get('rent_pa'))} — {r.get('term_years','?')}yrs — {r.get('status','?')}" for r in rows]
            return f"{len(rows)} deals:\n" + "\n".join(lines)

        elif name == "get_properties":
            filters = {}
            if args.get("status"): filters["status"] = f"eq.{args['status']}"
            rows = sb_get("t_properties", filters)
            if not rows: return "No properties found."
            lines = [f"• {r.get('address','?')}, {r.get('suburb','?')} — {fmt_size(r.get('size_sqm'))} — {r.get('status','?')} — {r.get('landlord','?')}" for r in rows]
            return f"{len(rows)} properties:\n" + "\n".join(lines)

        elif name == "search_knowledge":
            return search_context(args["query"], args.get("n", 6)) or "No relevant knowledge found."

        elif name == "find_matches":
            vacancies = sb_get("t_vacancies", {"status": "eq.Available"}) or []
            requirements = sb_get("t_requirements", {"status": "eq.Active"}) or []
            matches = []
            for v in vacancies:
                vsz = float(v.get("size_sqm") or 0)
                if not vsz: continue
                for r in requirements:
                    rmin = float(r.get("size_min") or 0)
                    rmax = float(r.get("size_max") or rmin * 1.5)
                    if rmin and rmin * 0.7 <= vsz <= rmax * 1.3:
                        matches.append(f"• {v.get('address','?')} ({fmt_size(vsz)}) ↔ {r.get('company','?')} ({fmt_size(rmin)}-{fmt_size(rmax)})")
            return f"{len(matches)} matches:\n" + "\n".join(matches[:10]) if matches else "No size matches found."

        elif name == "get_staff_summary":
            name_q = args.get("name", "")
            rows = sb_get("t_staff")
            staff = [s for s in (rows or []) if name_q.lower() in s.get("name","").lower()]
            if not staff: return f"No staff found matching '{name_q}'"
            s = staff[0]
            target = float(s.get("target_pa") or 0)
            ytd = float(s.get("ytd_commission") or 0)
            pct = round(ytd / target * 100) if target else 0
            return f"{s['name']} — {s.get('role','')}\nTarget: ${target:,.0f}pa\nYTD: ${ytd:,.0f} ({pct}%)\nActive deals: {s.get('active_deals',0)}"

        elif name == "store_learning":
            store_memory(args["fact"], args.get("context",""), source="conversation")
            return "Stored in permanent memory."

        return f"Unknown tool: {name}"
    except Exception as e:
        log.error("Tool %s error: %s", name, e)
        return f"Tool error: {e}"

# ── Core Agent ────────────────────────────────────────────
def build_system_prompt():
    style = get_style_profile()
    now = datetime.now(AEST).strftime("%A %d %B %Y, %I:%M%p AEST")
    return f"""You are Turkish, a personal AI agent for Michael West.

Current time: {now}

Profile:
{style}

You have real-time access to Michael's business data via tools. Always use tools to get current data before answering questions about the business.

Behaviour:
- Direct and actionable. Never vague or padded.
- Surface what matters: commission risk, stale requirements, hot matches, urgent follow-ups.
- Learn from every conversation — store important facts permanently.
- For morning briefings: use all tools, reason about what needs action TODAY.
- For staff: pull their data, calculate against targets, identify risks.

You are not a chatbot. You are a commercial real estate agent's brain."""

def run_agent(user_message, conversation_history=None):
    """Core agent loop. Reasons with tools, returns response, saves to DB."""
    history = conversation_history or []

    if not history:
        recent = sb_get("t_conversations", order="created_at.desc", limit=20) or []
        for row in reversed(recent):
            history.append({"role": row["role"], "content": row["content"]})

    history.append({"role": "user", "content": user_message})
    messages = [{"role": "system", "content": build_system_prompt()}] + history

    answer = "I couldn't process that request."

    for _ in range(5):
        try:
            resp = openai_call("/chat/completions", {
                "model": OPENAI_MODEL,
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto",
                "temperature": 0.3,
                "max_tokens": 1500,
            })
        except Exception as e:
            log.error("OpenAI call failed: %s", e)
            answer = f"I had trouble connecting to my reasoning engine: {e}"
            break

        msg = resp["choices"][0]["message"]
        messages.append(msg)

        if not msg.get("tool_calls"):
            answer = msg.get("content", "")
            break

        for tc in msg["tool_calls"]:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except:
                args = {}
            result = execute_tool(fn, args)
            log.info("Tool %s → %d chars", fn, len(result))
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    sb_insert("t_conversations", {"role": "user", "content": user_message})
    sb_insert("t_conversations", {"role": "assistant", "content": answer})
    return answer

# ── Telegram ──────────────────────────────────────────────
def tg_send(text, chat_id=None):
    cid = chat_id or TG_CHAT_ID
    if not TG_TOKEN or not cid:
        log.error("Telegram not configured")
        return False
    # Split long messages
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    for chunk in chunks:
        try:
            payload = json.dumps({"chat_id": int(cid), "text": chunk, "parse_mode": "Markdown"}).encode()
            req = ur.Request(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with ur.urlopen(req) as r:
                result = json.loads(r.read())
            if not result.get("ok"):
                # Retry without markdown
                payload = json.dumps({"chat_id": int(cid), "text": chunk}).encode()
                req = ur.Request(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                    data=payload, headers={"Content-Type": "application/json"}, method="POST")
                with ur.urlopen(req) as r:
                    result = json.loads(r.read())
        except Exception as e:
            log.error("Telegram send error: %s", e)
            return False
    return True

def tg_get_updates(offset=None):
    params = {"timeout": 30}
    if offset:
        params["offset"] = offset
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates?" + up.urlencode(params)
        with ur.urlopen(url, timeout=35) as r:
            return json.loads(r.read()).get("result", [])
    except Exception as e:
        log.debug("Telegram poll: %s", e)
        return []

# ── Daily Briefing ────────────────────────────────────────
def send_daily_briefing():
    """Agent-generated briefing — reasons about what matters today."""
    log.info("Generating daily briefing")
    prompt = (
        "Generate my morning briefing. "
        "Use your tools to check vacancies, requirements, deals, and find matches. "
        "What needs my attention TODAY? "
        "What's urgent? What's stale (no contact in 30+ days)? Who should I call? "
        "What commission is at risk? Any strong matches I haven't followed up? "
        "Keep it tight — actionable bullets, no padding."
    )
    try:
        briefing = run_agent(prompt)
        tg_send(briefing)
        sb_insert("t_briefings", {
            "content": briefing,
            "reasoning": "Agent-generated morning briefing",
            "delivered_via": "Telegram",
            "delivered_at": datetime.now(timezone.utc).isoformat(),
        })
        log.info("Daily briefing delivered: %d chars", len(briefing))
    except Exception as e:
        log.error("Briefing failed: %s", e)
        tg_send(f"Briefing error: {e}")

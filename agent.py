"""
Turkish — Personal Industrial Leasing Agent
Core agent: builds context from memory, reasons with GPT-4o, acts via tools.
"""
import os, json, logging
from datetime import datetime, timezone, timedelta
import urllib.request as ur
import urllib.parse as up

log = logging.getLogger("turkish")

# ── Config ────────────────────────────────────────────────
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
OPENAI_KEY   = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
EMBED_MODEL  = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
TG_TOKEN     = os.environ.get("TELEGRAM_TOKEN", "")
TG_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

AEST = timezone(timedelta(hours=10))

# ── Supabase ──────────────────────────────────────────────
def sb(method, path, body=None, params=None):
    """Single Supabase HTTP call. Returns parsed JSON or raises."""
    url = f"{SUPABASE_URL}{path}"
    if params:
        url += "?" + up.urlencode(params)
    data = json.dumps(body).encode() if body else None
    req = ur.Request(url, data=data, method=method, headers={
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    })
    try:
        with ur.urlopen(req) as r:
            return json.loads(r.read()) if r.read else []
    except ur.HTTPError as e:
        log.error("Supabase %s %s → %s %s", method, path, e.code, e.read())
        return None

def sb_get(table, filters=None, order=None, limit=200):
    params = {"select": "*"}
    if filters:
        params.update(filters)
    if order:
        params["order"] = order
    if limit:
        params["limit"] = limit
    return sb("GET", f"/rest/v1/{table}", params=params) or []

def sb_insert(table, row):
    return sb("POST", f"/rest/v1/{table}", body=row)

def sb_rpc(fn, args):
    return sb("POST", f"/rest/v1/rpc/{fn}", body=args)

# ── OpenAI ────────────────────────────────────────────────
def openai(path, body):
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
    """Get embedding vector for text."""
    resp = openai("/embeddings", {"model": EMBED_MODEL, "input": text[:8000]})
    return resp["data"][0]["embedding"]

def embed_and_store(table, record_id, text):
    """Embed text and store in t_embeddings. Called after any insert."""
    if not text or len(text.strip()) < 10:
        return
    vector = embed(text)
    sb_insert("t_embeddings", {
        "source_table": table,
        "source_id": record_id,
        "text_chunk": text[:2000],
        "embedding": vector,
    })

# ── Memory ────────────────────────────────────────────────
def search_context(query, n=8):
    """Semantic search across all embedded knowledge."""
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

def get_style_profile():
    """Load Michael's style profile into a system prompt block."""
    rows = sb_get("t_style_profile")
    if not rows:
        return ""
    lines = []
    for r in rows:
        lines.append(f"{r['category']}/{r['key']}: {r['value']}")
    return "\n".join(lines)

def store_memory(fact, context="", source="agent"):
    """Store a learned fact in persistent memory."""
    row = sb_insert("t_memory", {
        "fact": fact,
        "context": context,
        "source": source,
        "confidence": 1.0,
        "active": True,
    })
    if row and len(row) > 0:
        embed_and_store("t_memory", row[0]["id"], f"{fact}. {context}")

# ── Agent Tools ───────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_vacancies",
            "description": "Get active property vacancies available for lease",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "default": "Available"},
                    "min_size": {"type": "number"},
                    "max_size": {"type": "number"},
                },
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_requirements",
            "description": "Get active tenant requirements — companies looking for space",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "default": "Active"},
                    "priority": {"type": "string"},
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_deals",
            "description": "Get deals — signed leases, negotiations in progress, pipeline",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Semantic search across all stored knowledge — emails, notes, memory, market data",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "n": {"type": "integer", "default": 6},
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_matches",
            "description": "Find vacancy-requirement matches based on size, location, budget",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_staff_summary",
            "description": "Get performance summary for a staff member",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "store_learning",
            "description": "Store something learned from this conversation for future use",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["fact"]
            }
        }
    },
]

def execute_tool(name, args):
    """Execute a tool call and return the result as a string."""
    if name == "get_vacancies":
        filters = {"status": f"eq.{args.get('status','Available')}"}
        rows = sb_get("t_vacancies", filters, order="size_sqm.desc")
        if not rows:
            return "No vacancies found."
        lines = []
        for v in rows:
            size = f"{int(v['size_sqm']):,}sqm" if v.get('size_sqm') else "?"
            rent = f"${v['asking_rent_pa']:,.0f}pa" if v.get('asking_rent_pa') else "—"
            lines.append(f"• {v.get('address','?')}, {v.get('suburb','?')} — {size} — {rent}")
        return f"{len(rows)} vacancies:\n" + "\n".join(lines)

    elif name == "get_requirements":
        filters = {"status": f"eq.{args.get('status','Active')}"}
        if args.get("priority"):
            filters["priority"] = f"eq.{args['priority']}"
        rows = sb_get("t_requirements", filters, order="created_at.desc")
        if not rows:
            return "No requirements found."
        lines = []
        for r in rows:
            sz_min = f"{int(r['size_min']):,}" if r.get('size_min') else "?"
            sz_max = f"{int(r['size_max']):,}" if r.get('size_max') else "?"
            lines.append(f"• {r.get('company','?')} — {sz_min}-{sz_max}sqm — {r.get('preferred_location','?')}")
        return f"{len(rows)} requirements:\n" + "\n".join(lines)

    elif name == "get_deals":
        filters = {}
        if args.get("status"):
            filters["status"] = f"eq.{args['status']}"
        rows = sb_get("t_deals", filters, order="created_at.desc")
        if not rows:
            return "No deals found."
        lines = []
        for d in rows:
            size = f"{int(d['size_sqm']):,}sqm" if d.get('size_sqm') else "?"
            rent = f"${d['rent_pa']:,.0f}pa" if d.get('rent_pa') else "—"
            lines.append(f"• {d.get('tenant','?')} @ {d.get('address','?')} — {size} — {rent} — {d.get('status','?')}")
        return f"{len(rows)} deals:\n" + "\n".join(lines)

    elif name == "search_knowledge":
        return search_context(args["query"], args.get("n", 6)) or "No relevant knowledge found."

    elif name == "find_matches":
        vacancies = sb_get("t_vacancies", {"status": "eq.Available"}) or []
        requirements = sb_get("t_requirements", {"status": "eq.Active"}) or []
        matches = []
        for v in vacancies:
            vsz = float(v.get("size_sqm") or 0)
            if not vsz:
                continue
            for r in requirements:
                rmin = float(r.get("size_min") or 0)
                rmax = float(r.get("size_max") or rmin * 1.5 or 0)
                if rmin and rmin * 0.7 <= vsz <= (rmax or rmin * 1.5) * 1.3:
                    matches.append(
                        f"• {v.get('address','?')} ({int(vsz):,}sqm) ↔ "
                        f"{r.get('company','?')} ({int(rmin):,}-{int(rmax):,}sqm)"
                    )
        return f"{len(matches)} potential matches:\n" + "\n".join(matches[:10]) if matches else "No size matches found."

    elif name == "get_staff_summary":
        name_filter = args.get("name", "")
        rows = sb_get("t_staff")
        staff = [s for s in rows if name_filter.lower() in s.get("name", "").lower()]
        if not staff:
            return f"No staff found matching '{name_filter}'"
        s = staff[0]
        target = s.get("target_pa") or 0
        ytd = s.get("ytd_commission") or 0
        pct = round(ytd / target * 100) if target else 0
        deals = sb_get("t_deals", {"tenant": f"ilike.*{name_filter}*"}) or []
        return (
            f"{s['name']} — {s.get('role','')}\n"
            f"Target: ${target:,.0f}pa\n"
            f"YTD Commission: ${ytd:,.0f} ({pct}%)\n"
            f"Active deals: {s.get('active_deals', 0)}"
        )

    elif name == "store_learning":
        store_memory(args["fact"], args.get("context", ""), source="conversation")
        return "Stored."

    return f"Unknown tool: {name}"

# ── Core Agent ────────────────────────────────────────────
def build_system_prompt():
    """Build the system prompt from style profile + current context."""
    style = get_style_profile()
    now = datetime.now(AEST).strftime("%A %d %B %Y, %I:%M%p AEST")
    return f"""You are Turkish, a personal AI agent for {style or 'a Melbourne industrial leasing agent'}.

Current time: {now}

You have deep knowledge of industrial property in Melbourne's West. You have access to real-time data about vacancies, requirements, deals, contacts and market intelligence via your tools.

Your job:
- Be direct and actionable. Never vague.
- Use your tools to retrieve real data before answering questions about the business.
- Learn from every conversation — store useful facts for later.
- Surface what matters: commission risk, stale requirements, hot matches, urgent follow-ups.
- For staff reviews: pull their data, calculate against target, identify risks.

Style:
{style}

You are not a chatbot. You are a commercial real estate agent's brain. Think like one."""

def run_agent(user_message, conversation_history=None):
    """
    Core agent loop. Takes a message, reasons with tools, returns response.
    Stores the conversation in t_conversations.
    """
    history = conversation_history or []

    # Load recent conversation context (last 20 messages)
    if not history:
        recent = sb_get("t_conversations", order="created_at.desc", limit=20) or []
        for row in reversed(recent):
            history.append({"role": row["role"], "content": row["content"]})

    # Add new user message
    history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": build_system_prompt()}] + history

    # Agentic loop — keep calling until no more tool calls
    for _ in range(5):  # max 5 tool rounds
        resp = openai("/chat/completions", {
            "model": OPENAI_MODEL,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0.3,
            "max_tokens": 1500,
        })

        msg = resp["choices"][0]["message"]
        messages.append(msg)

        if not msg.get("tool_calls"):
            # Final answer — no more tool calls
            answer = msg["content"]
            break

        # Execute tool calls
        for tc in msg["tool_calls"]:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except:
                args = {}
            result = execute_tool(fn, args)
            log.info("Tool: %s → %s chars", fn, len(result))
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })
    else:
        answer = "I couldn't complete that request."

    # Persist conversation
    sb_insert("t_conversations", {"role": "user", "content": user_message})
    sb_insert("t_conversations", {"role": "assistant", "content": answer})

    return answer

# ── Telegram ──────────────────────────────────────────────
def tg_send(text, chat_id=None):
    cid = chat_id or TG_CHAT_ID
    if not TG_TOKEN or not cid:
        log.error("Telegram not configured")
        return False
    payload = json.dumps({
        "chat_id": int(cid),
        "text": text,
        "parse_mode": "Markdown"
    }).encode()
    req = ur.Request(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with ur.urlopen(req) as r:
            result = json.loads(r.read())
        return result.get("ok", False)
    except Exception as e:
        log.error("Telegram send failed: %s", e)
        return False

def tg_get_updates(offset=None):
    params = {"timeout": 30}
    if offset:
        params["offset"] = offset
    url = f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates?" + up.urlencode(params)
    try:
        with ur.urlopen(url, timeout=35) as r:
            return json.loads(r.read()).get("result", [])
    except:
        return []

# ── Daily Briefing ────────────────────────────────────────
def send_daily_briefing():
    """Agent-generated briefing — reasons about what matters today."""
    log.info("Generating daily briefing")
    prompt = (
        "Generate my morning briefing. "
        "Use your tools to check: vacancies, requirements, deals, and find matches. "
        "Tell me what needs my attention TODAY. "
        "What's urgent? What's stale? Who should I call? "
        "What commission is at risk? "
        "Keep it tight — actionable bullets, no padding."
    )
    try:
        briefing = run_agent(prompt)
        ok = tg_send(briefing)
        sb_insert("t_briefings", {
            "content": briefing,
            "reasoning": "Agent-generated morning briefing",
            "delivered_via": "Telegram",
            "delivered_at": datetime.now(timezone.utc).isoformat(),
        })
        log.info("Daily briefing delivered: %s chars, telegram=%s", len(briefing), ok)
        return briefing
    except Exception as e:
        log.error("Briefing failed: %s", e)
        tg_send(f"⚠️ Briefing error: {e}")


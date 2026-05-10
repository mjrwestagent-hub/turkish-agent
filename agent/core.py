"""
Turkish Agent — Core
The brain. Reads from Supabase, reasons with GPT-4o, learns over time.
All data access goes through here. No hardcoded field names.
"""
import os
import json
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta

# ── Supabase ─────────────────────────────────────────────────────────────────

SUPA_URL = os.environ.get("SUPABASE_URL", "")
SUPA_KEY = os.environ.get("SUPABASE_KEY", "")

def sb(method, path, body=None, params=None):
    url = f"{SUPA_URL}/rest/v1/{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body else None
    headers = {
        "apikey": SUPA_KEY,
        "Authorization": f"Bearer {SUPA_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

def db_get(table, filters=None, limit=500):
    params = {"select": "*", "limit": limit}
    if filters:
        params.update(filters)
    return sb("GET", table, params=params)

def db_insert(table, row):
    return sb("POST", table, body=row)

def db_update(table, pk_val, updates):
    return sb("PATCH", f"{table}?id=eq.{pk_val}", body=updates)

# ── OpenAI ───────────────────────────────────────────────────────────────────

OPENAI_KEY  = os.environ.get("OPENAI_API_KEY", "")
AGENT_MODEL = "gpt-4o"
EMBED_MODEL = "text-embedding-3-small"

def _openai(endpoint, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"https://api.openai.com/v1/{endpoint}",
        data=data,
        headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

def embed(text):
    """Convert text to 1536-dim vector for semantic search."""
    r = _openai("embeddings", {"model": EMBED_MODEL, "input": text[:8000]})
    return r["data"][0]["embedding"]

def gpt(messages, tools=None, temperature=0.3):
    payload = {"model": AGENT_MODEL, "messages": messages, "temperature": temperature}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    r = _openai("chat/completions", payload)
    return r["choices"][0]["message"]

# ── Memory ───────────────────────────────────────────────────────────────────

def embed_record(source_table, source_id, text):
    """Embed any record and store. Called automatically on insert."""
    if not text or not text.strip():
        return
    try:
        vec = embed(text[:8000])
        # Remove old embedding if exists
        try:
            sb("DELETE", "t_embeddings",
               params={"source_table": f"eq.{source_table}", "source_id": f"eq.{source_id}"})
        except Exception:
            pass
        db_insert("t_embeddings", {
            "source_table": source_table,
            "source_id": source_id,
            "text_chunk": text[:2000],
            "embedding": vec
        })
    except Exception as e:
        print(f"embed_record error: {e}")

def store_memory(fact, context="", source="agent"):
    """Store a fact the agent has learned. Embedded immediately."""
    row = db_insert("t_memory", {"fact": fact, "context": context, "source": source})
    if row:
        rid = row[0].get("id", 0) if isinstance(row, list) else 0
        embed_record("t_memory", rid, f"{fact} {context}")

def semantic_search(query, limit=8):
    """Search knowledge base by meaning using pgvector."""
    try:
        q_vec = embed(query)
        return sb("POST", "rpc/search_embeddings", body={
            "query_embedding": q_vec,
            "match_count": limit
        })
    except Exception:
        # Fallback: keyword scan of memory
        rows = db_get("t_memory", limit=200)
        q = query.lower()
        return [r for r in rows if q in (r.get("fact","") + r.get("context","")).lower()][:limit]

# ── System Prompt ────────────────────────────────────────────────────────────

def build_system_prompt():
    """Build the agent's personality and context from the database."""
    try:
        profile = db_get("t_style_profile")
        p = {}
        for row in profile:
            p[f"{row['category']}.{row['key']}"] = row["value"]
    except Exception:
        p = {}

    aest = datetime.now(tz=timezone(timedelta(hours=10)))

    return f"""You are Turkish, a personal AI agent for {p.get('identity.name', 'Michael')}.

ROLE: {p.get('identity.role', 'State Leader, MJR West')}
MARKET: {p.get('identity.market', 'Melbourne industrial leasing')}
FOCUS: {p.get('identity.focus', 'Melbourne West industrial')}
TODAY: {aest.strftime('%A %d %B %Y, %I:%M%p AEST')}

COMMUNICATION:
{p.get('communication.style', 'Direct. No fluff.')}
Lead with what matters. Actions over descriptions. Numbers over narrative.

PRIORITIES:
{p.get('priorities.briefing_order', 'Commission risk, hot requirements, stale vacancies, matches')}

MARKET CONTEXT:
- Typical deal: {p.get('market.typical_deal_size', '5,000-25,000sqm')}
- Typical term: {p.get('market.typical_term', '3-7 years')}
- Commission: {p.get('market.commission_model', 'Percentage of first year rent')}

INSTRUCTIONS:
- Always use tools to retrieve data before answering. Never guess.
- When you learn something important, store it with store_fact.
- If something has changed or is notable, flag it explicitly.
- For briefings: reason about what needs action, not just what exists.
- You are learning over time. Every interaction makes you more useful."""

# ── Tools ────────────────────────────────────────────────────────────────────

TOOLS = [
    {"type": "function", "function": {
        "name": "get_vacancies",
        "description": "Get active property vacancies available to lease",
        "parameters": {"type": "object", "properties": {}, "required": []}
    }},
    {"type": "function", "function": {
        "name": "get_requirements",
        "description": "Get tenant requirements. Optionally filter by status.",
        "parameters": {"type": "object", "properties": {
            "status": {"type": "string", "description": "Active, Hot, On Hold, Closed"}
        }, "required": []}
    }},
    {"type": "function", "function": {
        "name": "get_deals",
        "description": "Get deals. Optionally filter by status.",
        "parameters": {"type": "object", "properties": {
            "status": {"type": "string"}
        }, "required": []}
    }},
    {"type": "function", "function": {
        "name": "get_properties",
        "description": "Get all properties in the portfolio",
        "parameters": {"type": "object", "properties": {}, "required": []}
    }},
    {"type": "function", "function": {
        "name": "get_contacts",
        "description": "Get contacts. Optionally filter by role (tenant, landlord, staff, prospect).",
        "parameters": {"type": "object", "properties": {
            "role": {"type": "string"}
        }, "required": []}
    }},
    {"type": "function", "function": {
        "name": "search_memory",
        "description": "Search everything in the knowledge base by meaning. Use for specific questions about people, properties, history.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to search for — natural language"}
        }, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "find_matches",
        "description": "Find size-matched vacancy/requirement pairs",
        "parameters": {"type": "object", "properties": {}, "required": []}
    }},
    {"type": "function", "function": {
        "name": "store_fact",
        "description": "Store an important fact or learning permanently in memory",
        "parameters": {"type": "object", "properties": {
            "fact": {"type": "string", "description": "The fact to remember"},
            "context": {"type": "string", "description": "Where this came from or why it matters"}
        }, "required": ["fact"]}
    }},
    {"type": "function", "function": {
        "name": "get_staff_summary",
        "description": "Get performance data for a staff member — deals, commission, pipeline",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}
        }, "required": ["name"]}
    }},
    {"type": "function", "function": {
        "name": "get_pipeline_value",
        "description": "Calculate total commission pipeline and breakdown",
        "parameters": {"type": "object", "properties": {}, "required": []}
    }},
]

def run_tool(name, args):
    """Execute a tool. Returns JSON string."""
    try:
        if name == "get_vacancies":
            rows = db_get("t_vacancies", {"status": "eq.Available"})
            return json.dumps(rows, default=str)

        elif name == "get_requirements":
            status = args.get("status", "Active")
            rows = db_get("t_requirements", {"status": f"eq.{status}"})
            return json.dumps(rows, default=str)

        elif name == "get_deals":
            filters = {}
            if args.get("status"):
                filters["status"] = f"eq.{args['status']}"
            rows = db_get("t_deals", filters)
            return json.dumps(rows, default=str)

        elif name == "get_properties":
            rows = db_get("t_properties")
            return json.dumps(rows, default=str)

        elif name == "get_contacts":
            filters = {}
            if args.get("role"):
                filters["role"] = f"eq.{args['role']}"
            rows = db_get("t_contacts", filters)
            return json.dumps(rows, default=str)

        elif name == "search_memory":
            results = semantic_search(args.get("query", ""))
            return json.dumps(results, default=str)

        elif name == "find_matches":
            vacs = db_get("t_vacancies", {"status": "eq.Available"})
            reqs = db_get("t_requirements", {"status": "eq.Active"})
            matches = []
            for v in vacs:
                vsz = float(v.get("size_sqm") or 0)
                if not vsz:
                    continue
                for r in reqs:
                    rmin = float(r.get("size_min") or 0)
                    rmax = float(r.get("size_max") or rmin * 1.5)
                    if rmin > 0 and rmin * 0.7 <= vsz <= rmax * 1.3:
                        matches.append({
                            "vacancy": v.get("address"),
                            "size_sqm": int(vsz),
                            "asking_rent_pa": v.get("asking_rent_pa"),
                            "requirement_company": r.get("company"),
                            "req_size_range": f"{int(rmin):,}-{int(rmax):,}sqm",
                            "req_rating": r.get("rating"),
                            "req_last_contact": r.get("last_contact"),
                        })
            return json.dumps(matches, default=str)

        elif name == "store_fact":
            store_memory(args.get("fact", ""), args.get("context", ""), "agent_observation")
            return "Stored."

        elif name == "get_staff_summary":
            staff_name = args.get("name", "")
            staff = db_get("t_staff")
            match = [s for s in staff if staff_name.lower() in s.get("name", "").lower()]
            deals = db_get("t_deals")
            # Associate deals with staff via notes or assigned field
            return json.dumps({"staff": match, "all_deals": deals}, default=str)

        elif name == "get_pipeline_value":
            deals = db_get("t_deals")
            active = [d for d in deals if d.get("status") not in ("Completed", "Dead")]
            total_rent = sum(float(d.get("rent_pa") or 0) for d in active)
            total_comm = sum(float(d.get("commission") or 0) for d in active)
            completed = [d for d in deals if d.get("status") == "Completed"]
            ytd_comm = sum(float(d.get("commission") or 0) for d in completed)
            return json.dumps({
                "active_deals": len(active),
                "pipeline_rent_pa": total_rent,
                "pipeline_commission": total_comm,
                "ytd_commission": ytd_comm,
                "deals_breakdown": active
            }, default=str)

    except Exception as e:
        return f"Tool error ({name}): {e}"

    return "Unknown tool"

# ── Agent Run ────────────────────────────────────────────────────────────────

def run(user_message, history=None):
    """
    Run the agent. Returns response text.
    history: list of prior {role, content} messages for conversation context.
    """
    messages = [{"role": "system", "content": build_system_prompt()}]
    if history:
        messages.extend(history[-10:])  # last 10 turns for context
    messages.append({"role": "user", "content": user_message})

    for _ in range(10):  # max tool iterations
        response = gpt(messages, tools=TOOLS)
        messages.append(response)

        if not response.get("tool_calls"):
            return response.get("content", "")

        for tc in response["tool_calls"]:
            result = run_tool(
                tc["function"]["name"],
                json.loads(tc["function"]["arguments"] or "{}")
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result
            })

    return "Reached tool limit. Ask something more specific."

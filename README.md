# Turkish — Personal Industrial Leasing Agent

## Architecture
- `agent.py` — core agent: memory, tools, OpenAI reasoning
- `main.py` — Flask API + Telegram polling + scheduler

## How it works
1. Everything that comes in (emails, calls, notes) gets embedded via pgvector
2. Every query retrieves relevant context via semantic search
3. GPT-4o reasons over real data + retrieved memory
4. Learns and stores facts permanently in Supabase

## Interfaces
- Telegram: primary conversation interface
- POST /api/ask — ask anything programmatically
- POST /api/briefing/send — trigger manual briefing
- POST /api/ingest/email — ingest a new email for AI processing
- POST /api/memory — store a fact manually
- GET /health — health check

## Memory layers
1. t_embeddings — vector search over all content
2. t_memory — explicit facts the agent has learned
3. t_style_profile — how Michael works and thinks
4. t_conversations — full conversation history

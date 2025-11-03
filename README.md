# AI Connector (Render + OpenAI + Turso) with KB (RAG) & Memory

Endpoints:
- `POST /migrate` → crée/altère les tables `chat_logs` (project_id) et `kb_chunks`.
- `POST /api/chat?stream=1|0` → proxy OpenAI (SSE ou JSON) + logs Turso + RAG + mémoire de conversation (50 derniers messages, configurables).
- `POST /kb/upsert` → ingère `{ doc_id, title, text, chunk_size? }` et stocke embeddings dans Turso.
- `POST /kb/clear` → supprime tous les chunks d'un `doc_id`.

Env vars (Render):
- `OPENAI_API_KEY` (obligatoire)
- `OPENAI_MODEL` (default: gpt-4o-mini)
- `EMBED_MODEL` (default: text-embedding-3-small)
- `TURSO_DATABASE_URL`, `TURSO_AUTH_TOKEN`
- `ALLOWED_ORIGINS` (ex: https://ton-site.com)
- `KB_TOP_K` (default: 5)
- `LLM_TIMEOUT_CONNECT` (10), `LLM_TIMEOUT_READ` (70)
- `MEMORY_ENABLED` (true/false), `MEMORY_MAX_TURNS` (50)

Frontend:
- POST `https://<TON-SERVICE>.onrender.com/api/chat?stream=1` avec body :
  `{ "project_id":"projet-alpha", "session_id":"...optional...", "messages":[{"role":"user","content":"..."}] }`
- Le backend renverra `session_id` dans la réponse JSON (non-stream).

Notes:
- La mémoire est scindée par `session_id`, et scorable par `project_id` si fourni.
- RAG: top-K cosine sur embeddings OpenAI stockés dans Turso.

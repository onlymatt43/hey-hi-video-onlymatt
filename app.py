\
import os, json, asyncio, uuid, math, struct
from typing import AsyncIterator, Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
from pydantic import BaseModel
from libsql_client import create_client
import numpy as np

# ==================== CONFIG ====================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL", "")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "")

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"

LLM_CONNECT_TIMEOUT = float(os.getenv("LLM_TIMEOUT_CONNECT", "10"))
LLM_READ_TIMEOUT = float(os.getenv("LLM_TIMEOUT_READ", "70"))
TOP_K = int(os.getenv("KB_TOP_K", "5"))

# Mémoire serveur
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "true").lower() == "true"
MEMORY_MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "50"))  # 50 derniers tours (user+assistant)

# ==================== APP & CORS ====================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["Authorization", "Content-Type", "Accept", "Cache-Control"],
)

# ==================== TURSO ====================
db = create_client(url=TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN) if TURSO_DATABASE_URL and TURSO_AUTH_TOKEN else None

SQL_CREATE_CHAT = """
CREATE TABLE IF NOT EXISTS chat_logs (
  id INTEGER PRIMARY KEY,
  session_id TEXT,
  provider TEXT,
  model TEXT,
  messages_json TEXT,
  response_json TEXT,
  tokens_in INTEGER,
  tokens_out INTEGER,
  created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
"""

SQL_CREATE_KB = """
CREATE TABLE IF NOT EXISTS kb_chunks (
  id INTEGER PRIMARY KEY,
  doc_id TEXT,
  title TEXT,
  chunk TEXT,
  embedding BLOB,
  created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_kb_doc ON kb_chunks(doc_id);
"""

SQL_ALTER_CHAT_ADD_PROJECT = "ALTER TABLE chat_logs ADD COLUMN project_id TEXT;"

async def db_exec(sql: str, params: list | None = None):
    if not db:
        return None
    return await asyncio.to_thread(lambda: db.execute(sql, params or []))

async def ensure_schema():
    await db_exec(SQL_CREATE_CHAT)
    await db_exec(SQL_CREATE_KB)
    # Ajout project_id si absent
    try:
        await db_exec(SQL_ALTER_CHAT_ADD_PROJECT)
    except Exception:
        pass  # déjà présent

async def save_chat(session_id: str, model: str, messages: list[dict],
                    response: dict | None, tokens_in: Optional[int], tokens_out: Optional[int],
                    project_id: Optional[str]):
    if not db:
        return
    try:
        await db_exec(
            "INSERT INTO chat_logs (session_id, provider, model, messages_json, response_json, tokens_in, tokens_out, project_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [session_id, "openai", model, json.dumps(messages, ensure_ascii=False),
             json.dumps(response, ensure_ascii=False) if response is not None else None,
             tokens_in, tokens_out, project_id]
        )
    except Exception as e:
        print("[Turso] insert error:", repr(e))

# ==================== SCHEMAS ====================
class Message(BaseModel):
    role: str
    content: str

class ChatBody(BaseModel):
    messages: list[Message]
    model: str | None = None
    session_id: str | None = None
    project_id: str | None = None  # pour scoper la mémoire/logs par projet

# ==================== UTILS (embeddings) ====================
def pack_vec(vec: list[float]) -> bytes:
    # store as float32 little-endian
    return b"".join(struct.pack("<f", float(x)) for x in vec)

def unpack_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4")

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": EMBED_MODEL, "input": texts}
    timeout = httpx.Timeout(connect=LLM_CONNECT_TIMEOUT, read=LLM_READ_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(OPENAI_EMBED_URL, headers=headers, json=payload)
        r.raise_for_status()
        d = r.json()
    return [item["embedding"] for item in d["data"]]

# ==================== MEMORY ====================
async def load_memory(session_id: Optional[str], project_id: Optional[str], max_turns: int) -> list[dict]:
    """
    Reconstruit les derniers tours user/assistant depuis chat_logs.
    Retourne une liste [{'role':'user'|'assistant','content':'...'}, ...] en ordre chronologique.
    """
    if not db or not session_id:
        return []
    if project_id:
        res = await db_exec(
            "SELECT messages_json, response_json FROM chat_logs WHERE session_id = ? AND (project_id = ? OR project_id IS NULL) ORDER BY id DESC LIMIT ?",
            [session_id, project_id, max_turns * 2]
        )
    else:
        res = await db_exec(
            "SELECT messages_json, response_json FROM chat_logs WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            [session_id, max_turns * 2]
        )
    rows = getattr(res, "rows", None) or res or []
    convo: list[dict] = []
    for row in rows:
        messages_json, response_json = row
        try:
            ms = json.loads(messages_json) if isinstance(messages_json, str) else messages_json
            resp = json.loads(response_json) if isinstance(response_json, str) else response_json
        except Exception:
            ms, resp = [], None

        # message user le plus récent de ce tour
        user_last = None
        for m in reversed(ms or []):
            if m.get("role") == "user":
                user_last = m.get("content")
                break
        if user_last:
            convo.append({"role": "user", "content": user_last})

        # réponse assistant
        assistant_text = None
        if isinstance(resp, dict):
            if "content" in resp:  # cas stream
                assistant_text = resp.get("content")
            else:
                try:
                    assistant_text = resp.get("choices", [{}])[0].get("message", {}).get("content")
                except Exception:
                    assistant_text = None
        if assistant_text:
            convo.append({"role": "assistant", "content": assistant_text})

    convo.reverse()
    return convo[-(max_turns*2):]

# ==================== HEALTH & MIGRATE ====================
@app.get("/", response_class=HTMLResponse)
async def root():
    # Petite page d'accueil pour éviter le 404
    return """
    <!doctype html>
    <html lang="fr">
    <head><meta charset="utf-8"><title>AI Connector</title></head>
    <body style="font-family:system-ui;max-width:720px;margin:40px auto;">
      <h1>AI Connector — Render + OpenAI + Turso</h1>
      <p>Service opérationnel.</p>
      <ul>
        <li>Healthcheck: <a href="/healthz">/healthz</a></li>
        <li>Chat API (POST): <code>/api/chat?stream=1</code></li>
        <li>Migrate (POST): <code>/migrate</code></li>
      </ul>
    </body>
    </html>
    """

@app.get("/favicon.ico")
async def favicon():
    # 204 pour ne rien renvoyer et éviter le 404
    return Response(status_code=204)

@app.get("/healthz")
async def healthz():
    kb_count = None
    if db:
        try:
            res = await db_exec("SELECT COUNT(*) FROM kb_chunks", [])
            rows = getattr(res, "rows", None) or res
            if rows:
                kb_count = rows[0][0] if isinstance(rows[0], (list, tuple)) else None
        except Exception:
            kb_count = None
    return {
        "status": "ok",
        "provider": "openai",
        "chat_model": CHAT_MODEL,
        "embed_model": EMBED_MODEL,
        "has_openai_key": bool(OPENAI_API_KEY),
        "has_turso": bool(db is not None),
        "kb_chunks": kb_count,
        "memory_enabled": MEMORY_ENABLED,
        "memory_max_turns": MEMORY_MAX_TURNS,
    }

@app.post("/migrate")
async def migrate():
    if not db:
        return JSONResponse(status_code=500, content={"code":"NO_TURSO","message":"TURSO_DATABASE_URL / TURSO_AUTH_TOKEN manquants"})
    await ensure_schema()
    return {"ok": True}

# ==================== KB UPSERT ====================
@app.post("/kb/upsert")
async def kb_upsert(payload: dict = Body(...)):
    """
    payload = { "doc_id": "doc-123", "title": "Mon doc", "text": "gros texte...", "chunk_size": 1000 }
    """
    if not OPENAI_API_KEY:
        return JSONResponse(status_code=500, content={"error": "Missing OPENAI_API_KEY"})
    if not db:
        return JSONResponse(status_code=500, content={"error": "Turso not configured"})
    await ensure_schema()

    doc_id: str = payload.get("doc_id") or str(uuid.uuid4())
    title: str = payload.get("title") or doc_id
    text: str = payload.get("text") or ""
    chunk_size: int = int(payload.get("chunk_size", 1000))
    if not text.strip():
        return JSONResponse(status_code=400, content={"error": "text is empty"})

    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    vectors = await embed_texts(chunks)

    inserted = 0
    for ch, vec in zip(chunks, vectors):
        await db_exec(
            "INSERT INTO kb_chunks (doc_id, title, chunk, embedding) VALUES (?, ?, ?, ?)",
            [doc_id, title, ch, b"".join(struct.pack("<f", float(x)) for x in vec)]
        )
        inserted += 1
    return {"ok": True, "doc_id": doc_id, "inserted": inserted}

@app.post("/kb/clear")
async def kb_clear(doc_id: str = Body(..., embed=True)):
    if not db:
        return JSONResponse(status_code=500, content={"error": "Turso not configured"})
    await db_exec("DELETE FROM kb_chunks WHERE doc_id = ?", [doc_id])
    return {"ok": True, "deleted_doc_id": doc_id}

# ==================== CHAT (with retrieval + memory) ====================
class _EmbedCache:
    # mini cache mémoire process-local pour éviter de recalculer trop souvent le même input
    cache: dict[str, list[float]] = {}
EMBED_CACHE = _EmbedCache()

async def _embed_cached(text: str) -> list[float]:
    key = text.strip()
    if key in EMBED_CACHE.cache:
        return EMBED_CACHE.cache[key]
    vec = (await embed_texts([key]))[0]
    EMBED_CACHE.cache[key] = vec
    return vec

@app.post("/api/chat")
async def chat(body: ChatBody, stream: int = Query(default=0)):
    if not OPENAI_API_KEY:
        return JSONResponse(status_code=500, content={"code": "MISSING_API_KEY", "message": "OPENAI_API_KEY manquante"})

    model = body.model or CHAT_MODEL
    messages = [m.model_dump() for m in body.messages]
    session_id = body.session_id or str(uuid.uuid4())
    project_id = body.project_id

    # -------- Mémoire : prépendre l'historique --------
    if MEMORY_ENABLED:
        past = await load_memory(session_id, project_id, MEMORY_MAX_TURNS)
        if past:
            messages = past + messages

    # -------- Retrieval (KB) --------
    user_query = ""
    for m in reversed(messages):
        if m["role"] == "user":
            user_query = m["content"]
            break

    context = ""
    try:
        if db and user_query.strip():
            q_vec = await _embed_cached(user_query)
            q = np.array(q_vec, dtype=np.float32)

            res = await db_exec("SELECT title, chunk, embedding FROM kb_chunks", [])
            rows = getattr(res, "rows", None) or res or []
            scored = []
            for row in rows:
                title, chunk, blob = row
                v = np.frombuffer(blob, dtype="<f4")
                s = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-8))
                scored.append((s, title, chunk))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:TOP_K]
            if top:
                context = "\n\n".join([f"[{t}] {c}" for _, t, c in top])

        if context:
            system_preface = (
                "Tu dois répondre en t'appuyant STRICTEMENT sur le contexte suivant. "
                "Si l'information est absente, dis-le.\n\n"
                f"=== CONTEXTE ===\n{context}\n=== FIN CONTEXTE ==="
            )
            messages = [{"role": "system", "content": system_preface}] + messages
    except Exception as e:
        print("[KB retrieval error]", repr(e))

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {"model": model, "messages": messages, "stream": bool(stream)}
    timeout = httpx.Timeout(connect=LLM_CONNECT_TIMEOUT, read=LLM_READ_TIMEOUT)

    # ---- Non-stream ----
    if not stream:
        tries, delay = 3, 0.5
        last_err = None
        for _ in range(tries):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    r = await client.post(OPENAI_CHAT_URL, headers=headers, json=payload)
                if r.status_code >= 400:
                    return JSONResponse(status_code=502, content={"code": "UPSTREAM_ERROR", "message": r.text})
                data = r.json()
                usage = data.get("usage", {}) or {}
                asyncio.create_task(save_chat(
                    session_id, model, messages, data,
                    usage.get("prompt_tokens"), usage.get("completion_tokens"),
                    project_id
                ))
                # renvoyer session_id pour que le front puisse le persister
                return {
                    "provider": "openai",
                    "model": model,
                    "choices": data.get("choices", []),
                    "usage": usage,
                    "session_id": session_id,
                }
            except Exception as e:
                last_err = e
                await asyncio.sleep(delay)
                delay *= 2
        raise HTTPException(status_code=502, detail=f"Upstream non-stream error: {last_err}")

    # ---- Streaming SSE ----
    async def sse_gen() -> AsyncIterator[str]:
        collected: list[str] = []
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", OPENAI_CHAT_URL, headers=headers, json=payload) as resp:
                    if resp.status_code >= 400:
                        txt = (await resp.aread()).decode("utf-8", "ignore")
                        yield f"data: {json.dumps({'error': resp.status_code, 'message': txt})}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data:"):
                            chunk = line[5:].strip()
                            if chunk and chunk != "[DONE]":
                                try:
                                    j = json.loads(chunk)
                                    piece = j.get("choices", [{}])[0].get("delta", {}).get("content")
                                    if piece:
                                        collected.append(piece)
                                except Exception:
                                    pass
                            yield line + "\n\n"
                    yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error':'STREAM_FAILED','message':str(e)})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if db is not None:
                response_json = {
                    "streamed": True,
                    "model": model,
                    "content": "".join(collected) if collected else None,
                }
                asyncio.create_task(save_chat(
                    session_id, model, messages, response_json, tokens_in=None, tokens_out=None, project_id=project_id
                ))

    return StreamingResponse(sse_gen(), media_type="text/event-stream")

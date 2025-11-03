import os, json, asyncio, uuid, struct, re
from typing import AsyncIterator, Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.requests import Request
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
TOP_K = int(os.getenv("KB_TOP_K", "0"))  # RAG OFF par défaut

# Mémoire serveur
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "true").lower() == "true"
MEMORY_MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "50"))  # 50 derniers tours (user+assistant)

# ==================== APP & CORS ====================
app = FastAPI()

# --- DEBUG GLOBAL: middleware pour voir les erreurs en JSON ---
@app.middleware("http")
async def catch_all(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        # => au lieu d'un "Internal Server Error" texte brut, JSON détaillé
        return JSONResponse({"error": "UNHANDLED_EXCEPTION", "detail": str(e)}, status_code=500)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["Authorization", "Content-Type", "Accept", "Cache-Control"],
)

# ---- TIMEOUTS & HEADERS helpers ----
HTTPX_TIMEOUT = httpx.Timeout(
    connect=LLM_CONNECT_TIMEOUT,
    read=LLM_READ_TIMEOUT,
    write=LLM_READ_TIMEOUT,
    pool=LLM_CONNECT_TIMEOUT,
)

def _headers_chat(accept_sse: bool = False) -> Dict[str, str]:
    h = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    if accept_sse:
        h["Accept"] = "text/event-stream"
    org = os.getenv("OPENAI_ORG_ID")
    proj = os.getenv("OPENAI_PROJECT_ID")
    if org:
        h["OpenAI-Organization"] = org
    if proj:
        h["OpenAI-Project"] = proj
    return h

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
  project_id TEXT,
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
            [
                session_id, "openai", model,
                json.dumps(messages, ensure_ascii=False),
                json.dumps(response, ensure_ascii=False) if response is not None else None,
                tokens_in, tokens_out, project_id
            ]
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
    headers = _headers_chat()
    payload = {"model": EMBED_MODEL, "input": texts}
    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT, http2=False) as c:
        r = await c.post(OPENAI_EMBED_URL, headers=headers, json=payload)
    if r.status_code >= 400:
        raise RuntimeError(f"EMBEDDINGS_UPSTREAM_ERROR {r.status_code}: {r.text[:300]}")
    d = r.json()
    return [item["embedding"] for item in d["data"]]

# ==================== MEMORY ====================
async def load_memory(session_id: Optional[str], project_id: Optional[str], max_turns: int) -> list[dict]:
    if not db or not session_id:
        return []
    limit = int(max_turns * 2)  # user+assistant
    if project_id:
        sql = f"""
            SELECT messages_json, response_json
            FROM chat_logs
            WHERE session_id = ? AND (project_id = ? OR project_id IS NULL)
            ORDER BY id DESC
            LIMIT {limit}
        """
        params = [session_id, project_id]
    else:
        sql = f"""
            SELECT messages_json, response_json
            FROM chat_logs
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT {limit}
        """
        params = [session_id]

    try:
        res = await db_exec(sql, params)
        rows = getattr(res, "rows", None) or res or []
    except Exception as e:
        print("[load_memory SQL error]", repr(e))
        return []

    convo: list[dict] = []
    for row in rows:
        messages_json, response_json = row
        try:
            ms = json.loads(messages_json) if isinstance(messages_json, str) else messages_json
            resp = json.loads(response_json) if isinstance(response_json, str) else response_json
        except Exception:
            ms, resp = [], None

        user_last = None
        for m in reversed(ms or []):
            if isinstance(m, dict) and m.get("role") == "user":
                user_last = m.get("content")
                break
        if user_last:
            convo.append({"role": "user", "content": user_last})

        assistant_text = None
        if isinstance(resp, dict):
            if "content" in resp:
                assistant_text = resp.get("content")
            else:
                try:
                    assistant_text = resp.get("choices", [{}])[0].get("message", {}).get("content")
                except Exception:
                    assistant_text = None
        if assistant_text:
            convo.append({"role": "assistant", "content": assistant_text})

    convo.reverse()
    return convo[-limit:]

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
        <li>Diag (GET): <code>/__diag</code></li>
      </ul>
    </body>
    </html>
    """

@app.get("/favicon.ico")
async def favicon():
    # 204 pour ne rien renvoyer et éviter le 404
    return Response(status_code=204)

@app.get("/__version")
async def __version():
    return {"version": "hardened-v1"}

@app.get("/__diag")
async def __diag():
    kb_count = None
    try:
        if db:
            res = await db_exec("SELECT COUNT(*) FROM kb_chunks", [])
            rows = getattr(res, "rows", None) or res or []
            kb_count = rows[0][0] if rows else None
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
        "kb_top_k": TOP_K,
        "allowed_origins": ALLOWED_ORIGINS,
    }

@app.get("/healthz")
async def healthz():
    return await __diag()

@app.post("/migrate")
async def migrate():
    if not db:
        return JSONResponse(status_code=500, content={"code":"NO_TURSO","message":"TURSO_DATABASE_URL / TURSO_AUTH_TOKEN manquants"})
    await ensure_schema()
    return {"ok": True}

# ==================== KB ENDPOINTS ====================
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
            [doc_id, title, ch, pack_vec(vec)]
        )
        inserted += 1
    return {"ok": True, "doc_id": doc_id, "inserted": inserted}

@app.post("/kb/clear")
async def kb_clear(doc_id: str = Body(..., embed=True)):
    if not db:
        return JSONResponse(status_code=500, content={"error": "Turso not configured"})
    await db_exec("DELETE FROM kb_chunks WHERE doc_id = ?", [doc_id])
    return {"ok": True, "deleted_doc_id": doc_id}

# ==================== MEMORY HINTS (name & preferences) ====================
NAME_PATTERNS = [
    r"\bmon nom est\s+([A-Za-zÀ-ÖØ-öø-ÿ\-\' ]{2,40})\b",
    r"\bje m'appelle\s+([A-Za-zÀ-ÖØ-öø-ÿ\-\' ]{2,40})\b",
    r"\bje me nomme\s+([A-Za-zÀ-ÖØ-öø-ÿ\-\' ]{2,40})\b",
]
def _extract_name_from_text(text: str) -> str | None:
    t = text.strip().strip(".!?")
    for pat in NAME_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            name = re.sub(r"\s+", " ", m.group(1)).strip(" '\"")
            if 1 <= len(name.split()) <= 4:
                return " ".join(s.capitalize() for s in name.split())
    return None

def extract_known_name(all_messages: list[dict]) -> str | None:
    name = None
    for m in all_messages:
        if m.get("role") == "user":
            cand = _extract_name_from_text(m.get("content", ""))
            if cand:
                name = cand
    return name

DRINK_PATTERNS = [
    r"\bje (?:préfère|prefere)\s+(le|la|les)?\s*([A-Za-zÀ-ÖØ-öø-ÿ\-\' ]{2,40})\b",
    r"\bma boisson préférée est\s+(le|la|les)?\s*([A-Za-zÀ-ÖØ-öø-ÿ\-\' ]{2,40})\b",
    r"\bma boisson preferee est\s+(le|la|les)?\s*([A-Za-zÀ-ÖØ-öø-ÿ\-\' ]{2,40})\b",
    r"\bma boisson (?:de choix|favorite)\s+est\s+(le|la|les)?\s*([A-Za-zÀ-ÖØ-öø-ÿ\-\' ]{2,40})\b",
]
GENERIC_PREF_PATTERNS = [
    r"\bmon ([A-Za-zÀ-ÖØ-öø-ÿ\-\' ]{2,20}) préféré\b(?: est| c'est)\s+(.*)$",
    r"\bma ([A-Za-zÀ-ÖØ-öø-ÿ\-\' ]{2,20}) préférée\b(?: est| c'est)\s+(.*)$",
    r"\bmes ([A-Za-zÀ-ÖØ-öø-ÿ\-\' ]{2,20}) préférés\b(?: sont| ce sont)\s+(.*)$",
    r"\bmes ([A-Za-zÀ-ÖØ-öø-ÿ\-\' ]{2,20}) préférées\b(?: sont| ce sont)\s+(.*)$",
]
STOP_TOKENS = r"[.!?]"

def _clean_noun_phrase(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(rf"{STOP_TOKENS}.*$", "", s)
    return s.strip(" '\"").lower()

def _extract_drink(text: str) -> str | None:
    t = text.strip()
    for pat in DRINK_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            drink = m.group(2) if m.lastindex and m.lastindex >= 2 else None
            if drink:
                drink = _clean_noun_phrase(drink)
                aliases = {
                    "un cafe": "café", "cafe": "café",
                    "the": "thé", "un the": "thé",
                }
                return aliases.get(drink, drink)
    m2 = re.search(r"\bje (?:préfère|prefere)\s+([A-Za-zÀ-ÖØ-öø-ÿ\-\' ]{2,40})\b", t, flags=re.IGNORECASE)
    if m2:
        cand = _clean_noun_phrase(m2.group(1))
        if 1 <= len(cand.split()) <= 4:
            return cand
    return None

def _extract_generic_pref(text: str) -> tuple[str, str] | None:
    t = text.strip()
    for pat in GENERIC_PREF_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            key = _clean_noun_phrase(m.group(1))
            val = _clean_noun_phrase(m.group(2))
            if 1 <= len(key.split()) <= 4 and 1 <= len(val.split()) <= 6:
                return (key, val)
    return None

def extract_known_prefs(all_messages: list[dict]) -> dict:
    prefs: dict[str, str] = {}
    for m in all_messages:
        if m.get("role") != "user":
            continue
        text = m.get("content", "")
        d = _extract_drink(text)
        if d:
            prefs["drink"] = d
        kv = _extract_generic_pref(text)
        if kv:
            k, v = kv
            prefs[k] = v
    return prefs

# ==================== RAG (optional, guarded) ====================
class _EmbedCache:
    cache: dict[str, list[float]] = {}
EMBED_CACHE = _EmbedCache()

async def _embed_cached(text: str) -> list[float]:
    key = text.strip()
    if key in EMBED_CACHE.cache:
        return EMBED_CACHE.cache[key]
    vec = (await embed_texts([key]))[0]
    EMBED_CACHE.cache[key] = vec
    return vec

def _similarity_search(q_vec: List[float], rows: List[Any], top_k: int) -> List[tuple]:
    q = np.array(q_vec, dtype=np.float32)
    scored: List[tuple] = []
    for row in rows:
        title, chunk, blob = row
        v = unpack_vec(blob)
        s = cosine(q, v)
        scored.append((s, title, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

# ==================== CHAT (hardened: memory + optional RAG) ====================
@app.post("/api/chat")
async def chat_with_memory(body: ChatBody, stream: int = Query(default=0)):
    if not OPENAI_API_KEY:
        return JSONResponse(status_code=500, content={"error":"MISSING_OPENAI_API_KEY"})

    model = body.model or CHAT_MODEL
    session_id = body.session_id or str(uuid.uuid4())
    project_id = body.project_id

    # On garde le tour courant "propre" pour stockage (DB) et on part de là pour construire la requête
    messages_current = [m.model_dump() for m in body.messages]
    messages = messages_current[:]

    # -------- Mémoire (prépend) --------
    past: list[dict] = []
    if MEMORY_ENABLED:
        try:
            past = await load_memory(session_id, project_id, MEMORY_MAX_TURNS)
            if past:
                system_preface = {
                    "role": "system",
                    "content": (
                        "Utilise l'historique ci-dessous pour répondre de façon cohérente. "
                        "Si l'utilisateur a partagé un fait (ex: son nom, préférences), "
                        "réutilise-le quand c'est pertinent."
                    )
                }
                messages = [system_preface] + past + messages
        except Exception as e:
            print("[memory] load error:", repr(e))

    # -------- Hints explicites à partir de la mémoire (nom + préférences) --------
    try:
        scan_messages = past + messages_current if isinstance(past, list) else messages_current
        known_name = extract_known_name(scan_messages)
        known_prefs = extract_known_prefs(scan_messages)  # ex: {'drink': 'café', 'couleur': 'bleu'}

        facts = []
        if known_name:
            facts.append(f"- Nom de l'utilisateur : « {known_name} ».")  # fait certain
        if "drink" in known_prefs:
            facts.append(f"- Boisson préférée : « {known_prefs['drink']} ».")  # fait certain

        # Hints génériques (on limite à 3 pour éviter un prompt trop long)
        extras = []
        for k, v in known_prefs.items():
            if k == "drink":
                continue
            if len(extras) < 3:
                extras.append(f"- {k.capitalize()} préféré(e) : « {v} ».")

        if facts or extras:
            messages = [{
                "role": "system",
                "content": (
                    "Règle d'or : tu DOIS utiliser les faits mémorisés ci-dessous pour répondre à toute "
                    "question qui s'y rapporte (même si la question demande plusieurs éléments à la fois). "
                    "Réponds explicitement avec les valeurs connues, sans dire que tu ne sais pas.\n\n"
                    "Faits mémorisés :\n"
                    + "\n".join(facts + extras) +
                    "\n\nExemples d'application :\n"
                    "- Si on te demande le nom de l'utilisateur, réponds exactement le nom connu.\n"
                    "- Si on te demande la boisson préférée, réponds exactement la boisson connue.\n"
                    "- Si on te demande les deux en même temps, donne les deux valeurs connues clairement."
                )
            }] + messages
    except Exception as e:
        print("[memory hints] error:", repr(e))

    # -------- RAG (optionnel) --------
    if TOP_K > 0 and db is not None:
        try:
            # prend la dernière question utilisateur
            user_query = ""
            for m in reversed(messages_current):
                if m.get("role") == "user":
                    user_query = m.get("content", "")
                    break
            if user_query.strip():
                q_vec = await _embed_cached(user_query)
                res = await db_exec("SELECT title, chunk, embedding FROM kb_chunks", [])
                rows = getattr(res, "rows", None) or res or []
                top = _similarity_search(q_vec, rows, TOP_K)
                if top:
                    context = "\n\n".join([f"[{t}] {c}" for _, t, c in top])
                    messages = [{
                        "role": "system",
                        "content": (
                            "Contexte (RAG) ci-dessous. Si l'information pertinente s'y trouve, "
                            "appuie ta réponse dessus. Sinon, réponds normalement.\n\n"
                            f"=== CONTEXTE ===\n{context}\n=== FIN CONTEXTE ==="
                        )
                    }] + messages
        except Exception as e:
            print("[RAG retrieval error]", repr(e))

    headers = _headers_chat(accept_sse=bool(stream))
    payload = {"model": model, "messages": messages, "stream": bool(stream)}

    # ---------- Non-stream ----------
    if not stream:
        try:
            async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT, http2=False) as client:
                r = await client.post(OPENAI_CHAT_URL, headers=headers, json=payload)

            if r.status_code >= 400:
                return JSONResponse(
                    status_code=r.status_code,
                    content={"provider":"openai","error":"UPSTREAM_ERROR","status":r.status_code,"body":r.text}
                )

            data = r.json()
            usage = data.get("usage", {}) or {}

            # Ecriture DB bloquante => mémoire dispo tout de suite au prochain tour
            try:
                await save_chat(
                    session_id, model, messages_current, data,
                    usage.get("prompt_tokens"), usage.get("completion_tokens"),
                    project_id
                )
            except Exception as e:
                print("[save_chat] error:", repr(e))

            return {
                "provider": "openai",
                "model": model,
                "choices": data.get("choices", []),
                "usage": usage,
                "session_id": session_id,
                "project_id": project_id,
            }

        except Exception as e:
            return JSONResponse(status_code=502, content={"error":"NONSTREAM_EXCEPTION","message":str(e)})

    # ---------- Streaming SSE ----------
    async def sse_gen() -> AsyncIterator[str]:
        collected: list[str] = []
        try:
            async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT, http2=False) as client:
                async with client.stream("POST", OPENAI_CHAT_URL, headers=headers, json=payload) as resp:
                    if resp.status_code >= 400:
                        txt = (await resp.aread()).decode("utf-8", "ignore")
                        yield f"data: {json.dumps({'error':'UPSTREAM_ERROR','status': resp.status_code, 'body': txt[:500]})}\n\n"
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
                try:
                    response_json = {
                        "streamed": True,
                        "model": model,
                        "content": "".join(collected) if collected else None,
                    }
                    await save_chat(session_id, model, messages_current, response_json, None, None, project_id)
                except Exception as e:
                    print("[save_chat stream] error:", repr(e))

    return StreamingResponse(sse_gen(), media_type="text/event-stream")

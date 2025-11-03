\
import os, json, asyncio, uuid, math, struct, re
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
TOP_K = int(os.getenv("KB_TOP_K", "5"))

# Mémoire serveur
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "true").lower() == "true"
MEMORY_MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "50"))  # 50 derniers tours (user+assistant)

# ---- TIMEOUTS & RETRIES (globaux) ----
HTTPX_TIMEOUT = httpx.Timeout(
    connect=float(os.getenv("LLM_TIMEOUT_CONNECT", "10")),
    read=float(os.getenv("LLM_TIMEOUT_READ", "70")),
    write=float(os.getenv("LLM_TIMEOUT_READ", "70")),
    pool=float(os.getenv("LLM_TIMEOUT_CONNECT", "10")),
)

def _headers_chat():
    h = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    org = os.getenv("OPENAI_ORG_ID")
    proj = os.getenv("OPENAI_PROJECT_ID")
    if org:  h["OpenAI-Organization"] = org
    if proj: h["OpenAI-Project"] = proj
    return h

# ==================== APP & CORS ====================
app = FastAPI()

# --- DEBUG GLOBAL: middleware pour voir les erreurs en JSON ---
@app.middleware("http")
async def catch_all(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        # => au lieu d'un "Internal Server Error" texte brut, tu verras {"error": "...", "detail": "..."}
        return JSONResponse({"error": "UNHANDLED_EXCEPTION", "detail": str(e)}, status_code=500)

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
    headers = _headers_chat()
    payload = {"model": EMBED_MODEL, "input": texts}
    async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT, http2=False) as c:
        r = await c.post(OPENAI_EMBED_URL, headers=headers, json=payload)
    if r.status_code >= 400:
        # on lève pour être catch plus haut sans planter le service
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
            if m.get("role") == "user":
                user_last = m.get("content"); break
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
      </ul>
    </body>
    </html>
    """

@app.get("/favicon.ico")
async def favicon():
    # 204 pour ne rien renvoyer et éviter le 404
    return Response(status_code=204)

# --- Version pour vérifier que le bon code est déployé ---
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
        "chat_model": CHAT_MODEL,
        "embed_model": EMBED_MODEL,
        "has_openai_key": bool(OPENAI_API_KEY),
        "has_turso": bool(db is not None),
        "memory_enabled": MEMORY_ENABLED,
        "memory_max_turns": MEMORY_MAX_TURNS,
        "kb_top_k": TOP_K,
        "kb_chunks": kb_count,
        "allowed_origins": ALLOWED_ORIGINS,
    }

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

# --- Extraction: NOM (déjà ajouté chez toi, je le laisse ici pour contexte) ---
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

# --- Extraction: PRÉFÉRENCES (boisson + générique “X préférée est Y”) ---
# Spécifique "boisson"
DRINK_PATTERNS = [
    r"\bje (?:préfère|prefere)\s+(le|la|les)?\s*([A-Za-zÀ-ÖØ-öø-ÿ\-\' ]{2,40})\b",
    r"\bma boisson préférée est\s*([A-Za-zÀ-ÖØ-öø-ÿ\-\' ]{2,40})\b",
    r"\bje bois souvent\s*([A-Za-zÀ-ÖØ-öø-ÿ\-\' ]{2,40})\b",
]
def _extract_drink_from_text(text: str) -> str | None:
    t = text.strip().strip(".!?")
    for pat in DRINK_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            drink = re.sub(r"\s+", " ", m.group(2)).strip(" '\"")
            return " ".join(s.capitalize() for s in drink.split())
    return None

def extract_known_prefs(all_messages: list[dict]) -> dict | None:
    prefs = {}
    for m in all_messages:
        if m.get("role") == "user":
            drink = _extract_drink_from_text(m.get("content", ""))
            if drink:
                prefs["drink"] = drink
    return prefs

# ==================== /api/chat HARDENED (mémoire ON, RAG OFF par défaut) ===
@app.post("/api/chat")
async def chat_with_memory(body: ChatBody, stream: int = Query(default=0)):
    model = body.model or CHAT_MODEL
    session_id = body.session_id
    project_id = body.project_id

    messages = [{"role": "user", "content": m.content} for m in body.messages]

    # --- 1. Enregistrement du tour de chat (dans tous les cas) ---
    tokens_in = sum(len(m.get("content", "").strip()) for m in messages)
    response = None
    try:
        # Si pas de session_id, on en crée un nouveau
        if not session_id:
            session_id = str(uuid.uuid4())

        # Si pas de project_id, on utilise un ID par défaut (ou on en crée un)
        if not project_id:
            project_id = "default-project-id"  # À remplacer par une logique de génération d'ID de projet si nécessaire

        # Enregistrement dans Turso (DB)
        await save_chat(session_id, model, messages, response, tokens_in, None, project_id)
    except Exception as e:
        print("[save_chat] error:", repr(e))

    # --- 2. Chargement de la mémoire (si activée) ---
    past = await load_memory(session_id, project_id, MEMORY_MAX_TURNS)

    # -------- Récupération des messages de mémoire (user + assistant) --------
    try:
        if past and len(past) > 0:
            # On prend les derniers tours de mémoire (user + assistant)
            messages_current = [{"role": "user", "content": m.get("content")} for m in past if m.get("role") == "user"][-MEMORY_MAX_TURNS:]
            messages_current += [{"role": "assistant", "content": m.get("content")} for m in past if m.get("role") == "assistant"][-MEMORY_MAX_TURNS:]

            # On limite à 4 tours max pour ne pas surcharger le prompt
            if len(messages_current) > 8:
                messages_current = messages_current[-8:]

            # Préfixe système avec rappel de contexte (optionnel)
            system_preface = {
                "role": "system",
                "content": (
                    "Tu es un assistant utile et amical. "
                    "N'oublie pas les détails importants de la conversation. "
                    "Si l'utilisateur a un nom ou une préférence de boisson, utilise ces informations pour personnaliser tes réponses."
                )
            }
            messages = [system_preface] + past + messages
        except Exception as e:
            print("[memory] load error:", repr(e))

    # -------- Hints explicites à partir de la mémoire (nom + préférences) --------
    try:
        # On scanne le passé + le tour courant pour extraire des faits
        scan_messages = past + messages_current if 'past' in locals() and isinstance(past, list) else messages_current
        known_name = extract_known_name(scan_messages)
        known_prefs = extract_known_prefs(scan_messages)  # ex: {'drink': 'café', 'couleur': 'bleu'}

        hint_lines = []
        if known_name:
            hint_lines.append(
                f"- Nom connu : « {known_name} ». Si l'utilisateur demande « Quel est mon nom ? », "
                f"réponds exactement « {known_name} »."
            )
        if "drink" in known_prefs:
            hint_lines.append(
                f"- Boisson préférée : « {known_prefs['drink']} ». "
                f"Si on te demande la boisson préférée de l'utilisateur, réponds exactement « {known_prefs['drink']} »."
            )

        # Hints génériques (si présents) — tu peux en garder 2-3 max pour rester concis
        extras = []
        for k, v in known_prefs.items():
            if k == "drink":
                continue
            # Limite les hints pour éviter un prompt trop long
            if len(extras) < 3:
                extras.append(f"- {k.capitalize()} préféré(e) : « {v} ».")

        if hint_lines or extras:
            messages = [{
                "role": "system",
                "content": (
                    "Mémoire session — Faits à appliquer strictement si pertinent :\n"
                    + ("\n".join(hint_lines + extras))
                )
            }] + messages
    except Exception as e:
        print("[memory hints] error:", repr(e))

    headers = _headers_chat()
    payload = {"model": model, "messages": messages, "stream": bool(stream)}
    try:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT, http2=False) as c:
            r = await c.post(OPENAI_CHAT_URL, headers=headers, json=payload)
        if r.status_code != 200:
            raise RuntimeError(f"OPENAI_API_ERROR {r.status_code}: {r.text}")
        response = r.json()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "OPENAI_API_ERROR", "detail": str(e)})

    # --- 3. Enregistrement de la réponse (dans tous les cas) ---
    try:
        if response and "choices" in response and len(response["choices"]) > 0:
            answer = response["choices"][0].get("message") or response["choices"][0].get("text")
            tokens_out = len(answer.get("content", "").strip()) if isinstance(answer, dict) else len(str(answer).strip())
            await save_chat(session_id, model, messages, response, tokens_in, tokens_out, project_id)
    except Exception as e:
        print("[save_chat response] error:", repr(e))

    # Réponse finale (stripped)
    answer = response["choices"][0].get("message", {}).get("content", "").strip() if response and "choices" in response and len(response["choices"]) > 0 else ""
    return {"id": session_id, "answer": answer, "model": model, "tokens_in": tokens_in, "tokens_out": tokens_out}

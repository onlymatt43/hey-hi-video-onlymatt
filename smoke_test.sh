#!/usr/bin/env bash
set -euo pipefail

BASE="${1:-http://localhost:10000}"
echo "Using base: $BASE"

echo "[1] Version & diag"
curl -s "$BASE/__version"; echo
curl -s "$BASE/__diag"; echo

echo "[2] Migrate (if Turso configured)"
curl -s -X POST "$BASE/migrate"; echo

echo "[3] Memory check (name + drink)"
SID="smoke-$$"
curl -s "$BASE/api/chat" -H "Content-Type: application/json"   -d "{"project_id":"proj","session_id":"$SID","messages":[{"role":"user","content":"Je m'appelle Mathieu et je préfère le café."}]}"; echo
sleep 2
curl -s "$BASE/api/chat" -H "Content-Type: application/json"   -d "{"project_id":"proj","session_id":"$SID","messages":[{"role":"user","content":"Quel est mon nom et quelle est ma boisson préférée ?"}]}"; echo

echo "[4] Stream test"
curl -N "$BASE/api/chat?stream=1" -H "Accept: text/event-stream" -H "Content-Type: application/json"   -d '{"messages":[{"role":"user","content":"Réponds en 3 mots."}]}' || true

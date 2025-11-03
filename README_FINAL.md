# FINAL BACKEND (hardened-v1)

## Deploy
1) Replace your backend files with this bundle.
2) Commit & push:
   git add app.py requirements.txt Dockerfile render.yaml README_FINAL.md smoke_test.sh
   git commit -m "final: hardened-v1 backend (drop-in)"
   git push
3) Render → Manual Deploy → Deploy latest commit

## Env (Render)
OPENAI_API_KEY=...
ALLOWED_ORIGINS=https://video.onlymatt.ca,https://onlymatt.ca,https://www.onlymatt.ca
MEMORY_ENABLED=true
MEMORY_MAX_TURNS=50
KB_TOP_K=0
# optional Turso
TURSO_DATABASE_URL=...
TURSO_AUTH_TOKEN=...
# optional OpenAI org/project
# OPENAI_ORG_ID=...
# OPENAI_PROJECT_ID=...

## Smoke test
bash smoke_test.sh https://YOUR-RENDER-URL.onrender.com

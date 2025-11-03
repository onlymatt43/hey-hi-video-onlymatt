FROM python:3.11-slim

# Evite la mise en cache de .pyc et active le mode non-bufferisé
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .

# ✅ Upgrade pip avant d'installer les dépendances
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 10000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000", "--no-server-header"]
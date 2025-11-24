FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*

# App code
COPY . .

EXPOSE 8000

ENV PYTHONPATH=/app
# Run Chainlit (prod: no autoreload)
CMD ["chainlit", "run", "src/clinical_chatbot/app.py", "--host", "0.0.0.0", "--port", "8000"]

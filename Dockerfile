FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LLM_POLICY_DIR=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN test -f core/router.lua && test -f core/llm_policy.lua

EXPOSE 8080

CMD ["python", "serve.py", "--config", "config.internal.lua", "--default-profile", "agent", "--host", "0.0.0.0", "--port", "8080"]

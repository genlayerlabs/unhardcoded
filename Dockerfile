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

# Default to the authenticated ingress. The router (serve.py) has no caller auth
# of its own and must only ever be reached through it, so every deployment that
# runs the router names that command explicitly (compose, k8s, CI).
CMD ["uvicorn", "auth_proxy:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]

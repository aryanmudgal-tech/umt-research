# The professor's web UI on Cloud Run: the agent, the engineering stack, and
# Playwright's headless Chromium for printing reports to PDF.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt \
 && python -m playwright install --with-deps --only-shell chromium \
 && rm -rf /var/lib/apt/lists/*

COPY agent/ agent/
COPY tools/ tools/
COPY web/ web/
COPY equations/ equations/

# One process: the one-run-at-a-time lock and the tracer live in memory.
# Cloud Run sets PORT; GEMINI_API_KEY and RUNS_BUCKET come from the service.
CMD exec uvicorn web.app:create_app --factory --host 0.0.0.0 --port "${PORT:-8080}" --workers 1

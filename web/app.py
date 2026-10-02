"""The professor's web UI: upload a brief, watch the run, read the report.

    .venv/bin/uvicorn web.app:create_app --factory --port 8080

POST /api/runs takes the brief and answers with a stream of newline-delimited
JSON: a "started" line, the narrator's stage updates as the run goes, pings
while nothing happens, and a final "result" line. The run lives inside that
request on purpose: Cloud Run gives an instance CPU only while a request is
open. The pipeline runs in a worker thread that also saves the run, so a
dropped connection does not lose a run that the instance goes on to finish.

There is no login, by decision. The safeguards are one run at a time and a
cap on runs per 24 hours.

Configuration, from the environment:
    RUNS_BUCKET    Cloud Storage bucket for runs; unset means a local folder
    RUNS_DIR       that local folder (default: web_runs/ in the repo)
    DAILY_CAP      runs allowed per 24 hours (default 30)
    GEMINI_API_KEY the paid-tier key, from Secret Manager on Cloud Run
"""

import json
import logging
import os
import queue
import re
import tempfile
import threading
import time
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent.config import load_api_key, retryable_error
from agent.orchestrator import refusal_reason
from agent.trace import tracer
from web.narrator import Narrator, plain_check_name
from web.storage import GCSStore, LocalStore, valid_run_id

log = logging.getLogger("web")

WEB_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEB_DIR.parent
STATIC_DIR = WEB_DIR / "static"
SAMPLE_BRIEF = WEB_DIR / "sample_brief.md"

MAX_BRIEF_BYTES = 200_000
PING_EVERY_S = 10
DAY_S = 24 * 3600

BUSY_MESSAGE = "The AI service is busy right now. Try again in a few minutes."
ERROR_MESSAGE = "Something went wrong on our side. The run was saved so it can be looked into."


# ------------------------------------------------------------------ helpers


def brief_title(text: str, filename: str) -> str:
    """The brief's first heading, without a leading "Brief:", else the file name."""
    for line in text.splitlines():
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            title = re.sub(r"^brief\s*:\s*", "", title, flags=re.IGNORECASE)
            if title:
                return title
    return Path(filename).stem or "Brief"


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "report"


def _sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


def outcome_message(outcome) -> tuple:
    """(status, plain sentence) for a finished RunOutcome.

    The sentence never repeats the status ("Verified", "Not verified"): the
    page shows the status as the heading and this as the line under it.
    """
    if outcome.code == 0:
        return "passed", "Every check passed and the independent check agrees."
    if outcome.code == 2:
        reason = refusal_reason(outcome.narrative) or (outcome.narrative or "").strip()[:400]
        return "refused", _sentence(reason or "the AI did not find a beam problem it could solve in this brief")
    gate = outcome.gate or {}
    if gate and not gate.get("passed", True):
        failed = sorted({plain_check_name(c["name"]) for c in gate.get("checks", []) if c.get("status") == "fail"})
        return "failed", "These checks failed: " + (", ".join(failed) or "see the report") + "."
    verdict = outcome.verdict or {}
    if verdict.get("model") is None:
        return "failed", "The independent check couldn't run because the AI service was busy."
    reason = (verdict.get("reasoning") or "").strip()
    return "failed", "The independent check disagreed." + (f" Its reason: {reason}" if reason else "")


def exception_message(exc) -> tuple:
    if retryable_error(exc) or "models failed" in str(exc):
        return "busy", BUSY_MESSAGE
    return "error", ERROR_MESSAGE


# ------------------------------------------------------------------ the runs


class Runs:
    """One run at a time: start it, stream it, save it."""

    def __init__(self, store, pipeline, render_pdf, daily_cap):
        self.store = store
        self.pipeline = pipeline
        self.render_pdf = render_pdf
        self.daily_cap = daily_cap
        self.lock = threading.Lock()

    def start(self, brief_text: str, title: str):
        if self.store.runs_since(DAY_S) >= self.daily_cap:
            raise HTTPException(
                429, f"The limit of {self.daily_cap} analyses in 24 hours is reached. Try again later."
            )
        if not self.lock.acquire(blocking=False):
            raise HTTPException(409, "Another analysis is running. Try again in a minute or two.")
        try:
            run_id = self.store.start_run(title)
            self.store.put(run_id, "brief.md", brief_text)
        except Exception:
            self.lock.release()
            raise
        updates = queue.Queue()
        threading.Thread(
            target=self._work, args=(run_id, brief_text, title, updates), daemon=True
        ).start()
        return run_id, updates

    def _work(self, run_id, brief_text, title, updates):
        narrator = Narrator()

        def on_event(event):
            for update in narrator.feed(event):
                updates.put({"kind": "stage", **update})

        started = time.monotonic()
        status, message = "error", ERROR_MESSAGE
        tracer.subscribe(on_event)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                narrative = ""
                try:
                    outcome = self.pipeline(brief_text, out)
                    status, message = outcome_message(outcome)
                    narrative = outcome.narrative or ""
                except Exception as exc:
                    log.exception("run %s failed", run_id)
                    status, message = exception_message(exc)
                tracer.unsubscribe(on_event)
                has_pdf = self._save(run_id, out, title)
                self.store.finish_run(
                    run_id,
                    status,
                    message=message,
                    narrative=narrative[:4000],
                    duration_s=round(time.monotonic() - started, 1),
                    has_report=(out / "report.md").is_file(),
                    has_pdf=has_pdf,
                )
        except Exception:
            log.exception("saving run %s failed", run_id)
            status, message = "error", ERROR_MESSAGE
        finally:
            tracer.unsubscribe(on_event)
            updates.put({"kind": "result", "run_id": run_id, "status": status, "message": message})
            updates.put(None)
            self.lock.release()

    def _save(self, run_id, out, title) -> bool:
        """Store the run's files; True if a PDF of the report was made."""
        for name in ("report.md", "trace.json", "trace.html"):
            path = out / name
            if path.is_file():
                self.store.put(run_id, name, path.read_bytes())
        report = out / "report.md"
        if not report.is_file():
            return False
        try:
            self.store.put(run_id, "report.pdf", self.render_pdf(report.read_text(), title))
            return True
        except Exception:
            log.exception("PDF for run %s failed", run_id)  # the page still shows the report
            return False


def _stream(run_id, title, updates):
    yield json.dumps({"kind": "started", "run_id": run_id, "title": title}) + "\n"
    while True:
        try:
            item = updates.get(timeout=PING_EVERY_S)
        except queue.Empty:
            yield json.dumps({"kind": "ping"}) + "\n"
            continue
        if item is None:
            return
        yield json.dumps(item) + "\n"


# --------------------------------------------------------------------- app


def _default_store():
    bucket = os.environ.get("RUNS_BUCKET")
    if bucket:
        return GCSStore(bucket)
    return LocalStore(Path(os.environ.get("RUNS_DIR") or REPO_ROOT / "web_runs"))


def create_app(store=None, pipeline=None, render_pdf=None, daily_cap=None, load_key=True) -> FastAPI:
    if load_key:
        load_api_key()
    if pipeline is None:
        from agent.run_phase1 import run_pipeline as pipeline
    if render_pdf is None:
        from web.pdf import render_pdf
    runs = Runs(
        store or _default_store(),
        pipeline,
        render_pdf,
        daily_cap if daily_cap is not None else int(os.environ.get("DAILY_CAP", "30")),
    )

    app = FastAPI(title="The 25-Meter Agent", docs_url=None, redoc_url=None)
    app.state.runs = runs
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def meta_or_404(run_id):
        meta = runs.store.read_meta(run_id) if valid_run_id(run_id) else None
        if meta is None:
            raise HTTPException(404, "No such run.")
        return meta

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/sample-brief.md", include_in_schema=False)
    def sample_brief():
        return FileResponse(
            SAMPLE_BRIEF, media_type="text/markdown", filename="sample-brief.md",
            content_disposition_type="attachment",
        )

    @app.post("/api/runs")
    async def create_run(brief: UploadFile = File(...)):
        if not (brief.filename or "").lower().endswith(".md"):
            raise HTTPException(400, "That isn't a brief file. Choose a .md file.")
        data = await brief.read(MAX_BRIEF_BYTES + 1)
        if len(data) > MAX_BRIEF_BYTES:
            raise HTTPException(413, "That file is too large for a brief (over 200 KB).")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(400, "That file isn't plain text. Save the brief as a .md text file.")
        if not text.strip():
            raise HTTPException(400, "That brief is empty.")
        title = brief_title(text, brief.filename)
        run_id, updates = runs.start(text, title)
        return StreamingResponse(
            _stream(run_id, title, updates),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/runs")
    def list_runs():
        return runs.store.list_runs()

    @app.get("/api/runs/{run_id}")
    def read_run(run_id: str):
        meta = meta_or_404(run_id)
        report = runs.store.get(run_id, "report.md")
        brief_md = runs.store.get(run_id, "brief.md")
        return JSONResponse({
            "meta": meta,
            "report_md": report.decode("utf-8") if report else None,
            "brief_md": brief_md.decode("utf-8") if brief_md else None,
        })

    def download(run_id, name, media_type, extension):
        meta = meta_or_404(run_id)
        data = runs.store.get(run_id, name)
        if data is None:
            raise HTTPException(404, "This run has no such file.")
        filename = f"{slug(meta.get('title', 'report'))}.{extension}"
        return Response(
            data, media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.patch("/api/runs/{run_id}")
    def rename_run(run_id: str, title: str = Body(..., embed=True)):
        meta_or_404(run_id)
        try:
            return runs.store.rename_run(run_id, title)
        except ValueError:
            raise HTTPException(400, "A run needs a name.")

    @app.delete("/api/runs/{run_id}", status_code=204)
    def delete_run(run_id: str):
        meta = meta_or_404(run_id)
        if meta.get("status") == "running":
            raise HTTPException(409, "This analysis is still running. Delete it once it finishes.")
        runs.store.delete_run(run_id)
        return Response(status_code=204)

    @app.get("/api/runs/{run_id}/report.pdf")
    def report_pdf(run_id: str):
        return download(run_id, "report.pdf", "application/pdf", "pdf")

    @app.get("/api/runs/{run_id}/report.md")
    def report_md(run_id: str):
        return download(run_id, "report.md", "text/markdown; charset=utf-8", "md")

    return app

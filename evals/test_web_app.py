"""Offline tests for the web server (web/app.py) with a fake pipeline.

The fake replays a recorded run of the soil brief through the real tracer, so
the narrator, the streaming, the storage and the downloads are all exercised
exactly as a live run would exercise them, without calling Gemini.
"""

import json
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.run_phase1 import RunOutcome
from agent.trace import tracer
from web.app import create_app
from web.storage import LocalStore

FIXTURE = json.loads((REPO_ROOT / "evals" / "fixtures" / "trace_soil_pass.json").read_text())
BRIEF = b"# Brief: 25 m beam on soil\n\nA beam resting on soil.\n"


def replaying_pipeline(code=0, narrative="", raise_exc=None, events=FIXTURE):
    def pipeline(brief_text, out_dir, brief_name="brief.md"):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tracer.reset()
        for e in events:
            tracer.emit(e["stage"], e["type"], e["title"], **e["data"])
        if raise_exc:
            raise raise_exc
        (out_dir / "trace.json").write_text(tracer.to_json())
        report = None
        if code != 2:
            report = out_dir / "report.md"
            report.write_text("# Phase 1 Report\n\n## Key results\n\n$$\nEI v'''' + k v = q\n$$\n")
        return RunOutcome(code=code, passed=code == 0, lines=[], report_path=report, narrative=narrative,
                          verdict={"refuted": code == 1, "reasoning": "The shear does not match.", "model": "v"},
                          gate={"passed": True, "checks": []})

    return pipeline


def make_client(tmp_path, pipeline=None, cap=30):
    store = LocalStore(tmp_path / "runs")
    app = create_app(
        store=store,
        pipeline=pipeline or replaying_pipeline(),
        render_pdf=lambda report_md, title: b"%PDF-fake " + title.encode(),
        daily_cap=cap,
        load_key=False,
    )
    return TestClient(app), store, app


def submit(client, content=BRIEF, name="brief.md"):
    with client.stream("POST", "/api/runs", files={"brief": (name, content, "text/markdown")}) as r:
        status = r.status_code
        lines = [json.loads(line) for line in r.iter_lines() if line.strip()] if status == 200 else r.read()
    return status, lines


# -------------------------------------------------------------- a full run


def test_a_run_streams_six_stages_then_the_result(tmp_path):
    client, store, _ = make_client(tmp_path)
    status, lines = submit(client)

    assert status == 200
    assert lines[0]["kind"] == "started" and lines[0]["title"] == "25 m beam on soil"
    stages = [l for l in lines if l["kind"] == "stage"]
    finished = {l["stage"] for l in stages if l["status"] == "done"}
    assert finished == {"reading", "modeling", "solving", "checking", "independent", "report"}
    result = lines[-1]
    assert result["kind"] == "result" and result["status"] == "passed"
    assert result["message"] == "Every check passed and the independent check agrees."


def test_a_run_is_saved_with_its_report_pdf_and_trace(tmp_path):
    client, store, _ = make_client(tmp_path)
    _, lines = submit(client)
    run_id = lines[0]["run_id"]

    assert store.get(run_id, "brief.md") == BRIEF
    assert store.get(run_id, "report.md").startswith(b"# Phase 1 Report")
    assert store.get(run_id, "report.pdf") == b"%PDF-fake 25 m beam on soil"
    assert store.get(run_id, "trace.json")
    meta = store.read_meta(run_id)
    assert meta["status"] == "passed" and meta["title"] == "25 m beam on soil"
    assert meta["duration_s"] >= 0
    assert meta["has_pdf"] is True


def test_a_failed_pdf_still_leaves_a_report(tmp_path):
    def broken_pdf(report_md, title):
        raise RuntimeError("chromium missing")

    store = LocalStore(tmp_path / "runs")
    app = create_app(store=store, pipeline=replaying_pipeline(), render_pdf=broken_pdf, daily_cap=30, load_key=False)
    _, lines = submit(TestClient(app))
    meta = store.read_meta(lines[0]["run_id"])
    assert lines[-1]["status"] == "passed"
    assert meta["has_pdf"] is False and meta["has_report"] is True


def test_past_runs_and_a_runs_report_can_be_read_back(tmp_path):
    client, _, _ = make_client(tmp_path)
    _, lines = submit(client)
    run_id = lines[0]["run_id"]

    runs = client.get("/api/runs").json()
    assert [r["id"] for r in runs] == [run_id]
    run = client.get(f"/api/runs/{run_id}").json()
    assert run["meta"]["status"] == "passed"
    assert run["report_md"].startswith("# Phase 1 Report")
    assert run["brief_md"].startswith("# Brief")


def test_downloads_come_as_attachments_named_after_the_brief(tmp_path):
    client, _, _ = make_client(tmp_path)
    _, lines = submit(client)
    run_id = lines[0]["run_id"]

    pdf = client.get(f"/api/runs/{run_id}/report.pdf")
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"
    assert 'filename="25-m-beam-on-soil.pdf"' in pdf.headers["content-disposition"]
    md = client.get(f"/api/runs/{run_id}/report.md")
    assert md.status_code == 200 and 'filename="25-m-beam-on-soil.md"' in md.headers["content-disposition"]


@pytest.mark.parametrize("path", ["/api/runs/../../etc", "/api/runs/20261002-000000-zzzzzz", "/api/runs/20261002-000000-abcdef/report.pdf"])
def test_unknown_or_malformed_runs_are_404(tmp_path, path):
    client, _, _ = make_client(tmp_path)
    assert client.get(path).status_code == 404


def test_the_sample_brief_is_served(tmp_path):
    client, _, _ = make_client(tmp_path)
    r = client.get("/sample-brief.md")
    assert r.status_code == 200 and "# Brief" in r.text and "attachment" in r.headers["content-disposition"]


# ---------------------------------------------------------- refused uploads


@pytest.mark.parametrize(
    "name, content, code, phrase",
    [
        ("brief.txt", BRIEF, 400, ".md"),
        ("brief.md", b"   \n", 400, "empty"),
        ("brief.md", b"#" * 200_001, 413, "too large"),
        ("brief.md", b"\xff\xfe\x00bad", 400, "text"),
    ],
)
def test_a_file_that_is_not_a_brief_is_refused_before_any_ai_call(tmp_path, name, content, code, phrase):
    calls = []

    def pipeline(*a, **k):
        calls.append(1)

    client, _, _ = make_client(tmp_path, pipeline=pipeline)
    status, body = submit(client, content, name)
    assert status == code
    assert phrase in json.loads(body)["detail"]
    assert calls == []


def test_a_second_run_waits_its_turn(tmp_path):
    client, _, app = make_client(tmp_path)
    app.state.runs.lock.acquire()
    try:
        status, body = submit(client)
    finally:
        app.state.runs.lock.release()
    assert status == 409
    assert "Another analysis is running" in json.loads(body)["detail"]


def test_the_daily_cap_is_enforced(tmp_path):
    client, store, _ = make_client(tmp_path, cap=1)
    assert submit(client)[0] == 200
    status, body = submit(client)
    assert status == 429
    assert "limit of 1" in json.loads(body)["detail"]


# ------------------------------------------------------- ways a run can end


def test_a_refused_brief_ends_with_the_agents_reason(tmp_path):
    pipeline = replaying_pipeline(code=2, narrative="CANNOT SOLVE: the equation has a third-derivative term.", events=[])
    client, store, _ = make_client(tmp_path, pipeline=pipeline)
    _, lines = submit(client)
    result = lines[-1]
    assert result["status"] == "refused"
    assert result["message"] == "The equation has a third-derivative term."
    assert store.get(lines[0]["run_id"], "report.pdf") is None


def test_a_refuted_run_is_not_verified_and_says_why(tmp_path):
    client, _, _ = make_client(tmp_path, pipeline=replaying_pipeline(code=1))
    result = submit(client)[1][-1]
    assert result["status"] == "failed"
    assert result["message"].startswith("The independent check disagreed.")
    assert "shear does not match" in result["message"]


def test_a_busy_ai_service_ends_with_a_plain_sentence(tmp_path):
    pipeline = replaying_pipeline(raise_exc=RuntimeError("all orchestrator models failed: 503 UNAVAILABLE"), events=[])
    client, store, _ = make_client(tmp_path, pipeline=pipeline)
    _, lines = submit(client)
    assert lines[-1]["status"] == "busy"
    assert "busy" in lines[-1]["message"]
    assert store.read_meta(lines[0]["run_id"])["status"] == "busy"


def test_a_bug_on_our_side_is_saved_as_an_error_not_a_crash(tmp_path):
    client, store, app = make_client(tmp_path, pipeline=replaying_pipeline(raise_exc=KeyError("material"), events=[]))
    _, lines = submit(client)
    assert lines[-1]["status"] == "error"
    assert "on our side" in lines[-1]["message"]
    assert not app.state.runs.lock.locked()  # the next run is not blocked


def test_the_run_does_not_leave_a_listener_on_the_tracer(tmp_path):
    before = len(tracer._subscribers)
    client, _, _ = make_client(tmp_path)
    submit(client)
    assert len(tracer._subscribers) == before


# ----------------------------------------------------- rename and delete


def test_a_run_can_be_renamed(tmp_path):
    client, store, _ = make_client(tmp_path)
    run_id = submit(client)[1][0]["run_id"]

    r = client.patch(f"/api/runs/{run_id}", json={"title": "Soil beam, k = 1e7"})
    assert r.status_code == 200 and r.json()["title"] == "Soil beam, k = 1e7"
    assert client.get("/api/runs").json()[0]["title"] == "Soil beam, k = 1e7"
    pdf = client.get(f"/api/runs/{run_id}/report.pdf")
    assert 'filename="soil-beam-k-1e7.pdf"' in pdf.headers["content-disposition"]


@pytest.mark.parametrize("body, code", [({"title": "   "}, 400), ({}, 422), ({"title": 5}, 422)])
def test_a_rename_needs_a_title(tmp_path, body, code):
    client, _, _ = make_client(tmp_path)
    run_id = submit(client)[1][0]["run_id"]
    assert client.patch(f"/api/runs/{run_id}", json=body).status_code == code


def test_a_run_can_be_deleted(tmp_path):
    client, store, _ = make_client(tmp_path)
    run_id = submit(client)[1][0]["run_id"]

    assert client.delete(f"/api/runs/{run_id}").status_code == 204
    assert client.get("/api/runs").json() == []
    assert client.get(f"/api/runs/{run_id}").status_code == 404
    assert client.get(f"/api/runs/{run_id}/report.pdf").status_code == 404
    assert client.delete(f"/api/runs/{run_id}").status_code == 404


def test_deleting_runs_does_not_reset_the_daily_cap(tmp_path):
    client, _, _ = make_client(tmp_path, cap=1)
    run_id = submit(client)[1][0]["run_id"]
    client.delete(f"/api/runs/{run_id}")
    assert submit(client)[0] == 429


def test_a_running_run_cannot_be_deleted(tmp_path):
    client, store, _ = make_client(tmp_path)
    run_id = store.start_run("still going")
    r = client.delete(f"/api/runs/{run_id}")
    assert r.status_code == 409 and "still running" in r.json()["detail"]
    assert store.read_meta(run_id) is not None


@pytest.mark.parametrize("method", ["patch", "delete"])
def test_unknown_runs_cannot_be_changed(tmp_path, method):
    client, _, _ = make_client(tmp_path)
    kwargs = {"json": {"title": "x"}} if method == "patch" else {}
    assert getattr(client, method)("/api/runs/20261002-000000-abcdef", **kwargs).status_code == 404

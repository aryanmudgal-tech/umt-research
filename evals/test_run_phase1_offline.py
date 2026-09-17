"""Offline tests for the Phase-1 CLI: argument parsing, wiring, event order.

Nothing here calls main(), an LLM or the network. The one event stream built
below is assembled from the real emitters (stage rules, the deterministic
gate, the verifier's verdict publisher, the run closer), so it is the same
shape a live run produces.
"""

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent import live_view, run_phase1, trace_html
from agent.recorder import solve_beam_3d
from agent.trace import tracer
from agent.verifier import _emit_verdict

# Same bridge-beam reference numbers as the rest of the offline suite.
L, E, G, I, Q = 25.0, 30e9, 12.5e9, 0.005, 30e3


def ss_udl_model(n_elem=4):
    return {
        "nodes": [{"id": f"N{k}", "x": L * k / n_elem} for k in range(n_elem + 1)],
        "material": {"E": E, "G": G},
        "section": {"A": 0.5, "Iy": 0.004, "Iz": I, "J": 0.001},
        "supports": {
            "N0": [True, True, True, True, False, False],
            f"N{n_elem}": [False, True, True, False, False, False],
        },
        "distributed_loads": [{"element": "all", "direction": "y", "w1": -Q, "w2": -Q}],
        "point_loads": [],
    }


# ------------------------------------------------------------------ argparse


def parse(*args):
    return run_phase1.build_parser().parse_args(list(args))


def test_defaults_need_no_arguments():
    args = parse()
    assert Path(args.brief) == run_phase1.DEFAULT_BRIEF
    assert Path(args.trace_html) == run_phase1.DEFAULT_TRACE_HTML
    assert args.plain is False
    assert args.no_html is False


def test_positional_brief_still_works():
    # `run_phase1.py somebrief.md` must keep meaning what it always meant
    assert parse("somebrief.md").brief == "somebrief.md"
    assert parse("evals/brief_25m.md", "--plain").brief == "evals/brief_25m.md"


def test_plain_and_no_html_flags():
    args = parse("--plain", "--no-html")
    assert args.plain is True
    assert args.no_html is True
    assert Path(args.trace_html) == run_phase1.DEFAULT_TRACE_HTML


def test_trace_html_override_in_either_order():
    for args in (
        parse("--trace-html", "/tmp/t.html", "brief.md"),
        parse("brief.md", "--trace-html", "/tmp/t.html"),
    ):
        assert args.trace_html == "/tmp/t.html"
        assert args.brief == "brief.md"


def test_unknown_flag_is_rejected():
    import pytest

    with pytest.raises(SystemExit):
        parse("--nope")


# -------------------------------------------------------------------- wiring


def test_wiring_functions_are_importable():
    for name in (
        "build_parser",
        "main",
        "run_orchestrator",
        "write_report",
        "write_trace_json",
        "write_trace_html",
        "attach_live_view",
        "deterministic_gate",
        "run_verifier",
        "tracer",
    ):
        assert hasattr(run_phase1, name), name


def test_stage_names_agree_across_the_three_modules():
    assert run_phase1.STAGES == live_view.STAGES
    assert run_phase1.STAGES == tuple(trace_html.STAGE_ORDER)


# --------------------------------------------------------------- event order


def build_stream(tmp_path):
    """Drive the real emitters through one full run's worth of events."""
    tracer.reset()
    started = time.monotonic()
    model = ss_udl_model()

    run_phase1._stage("brief")
    tracer.emit(
        "brief", "run_start", "brief: x.md", brief_path="x.md", brief_text="a beam"
    )

    run_phase1._stage("orchestrator")
    tracer.emit(
        "orchestrator", "model_attempt", "trying m", role="orchestrator", model="m"
    )
    result = solve_beam_3d(model)
    tracer.emit("orchestrator", "tool_call", "solve", name="solve_beam_3d", args=model)

    run_phase1._stage("gate")
    det = run_phase1.deterministic_gate(model, result)

    run_phase1._stage("verifier")
    _emit_verdict(
        {"refuted": False, "checks": [], "reasoning": "ok", "model": "verifier-model"}
    )

    run_phase1._stage("report")
    run_phase1._finish(
        det["passed"], tmp_path / "r.md", tmp_path / "t.html", tmp_path / "t.json", started
    )
    return tracer.events


def test_events_are_causally_ordered_and_every_stage_opens_with_stage_start(tmp_path):
    events = build_stream(tmp_path)

    assert [e["seq"] for e in events] == list(range(len(events)))
    times = [e["t"] for e in events]
    assert times == sorted(times)

    # each stage appears as one contiguous block, in pipeline order, and the
    # first event of every block is that stage's stage_start
    blocks = []
    for event in events:
        if not blocks or blocks[-1][0] != event["stage"]:
            blocks.append((event["stage"], []))
        blocks[-1][1].append(event)

    assert [stage for stage, _ in blocks] == list(run_phase1.STAGES)
    for stage, group in blocks:
        assert group[0]["type"] == "stage_start"
        assert group[0]["data"] == {"stage": stage}

    types_seen = [e["type"] for e in events]
    assert types_seen.index("run_start") < types_seen.index("gate_result")
    assert types_seen.index("gate_result") < types_seen.index("verdict")
    assert types_seen.index("verdict") < types_seen.index("run_end")
    assert types_seen[-1] == "run_end"


def test_finish_writes_both_trace_artifacts(tmp_path):
    events = build_stream(tmp_path)

    html = (tmp_path / "t.html").read_text()
    assert "<!DOCTYPE html>" in html
    assert "gate pass" in html

    dumped = json.loads((tmp_path / "t.json").read_text())
    assert dumped == events

    end = events[-1]["data"]
    assert end["passed"] is True
    assert end["trace_path"] == str(tmp_path / "t.html")
    assert isinstance(end["duration_s"], float)


def test_finish_without_html_points_the_trace_at_the_json(tmp_path):
    tracer.reset()
    run_phase1._finish(
        False, tmp_path / "r.md", None, tmp_path / "only.json", time.monotonic()
    )

    assert not (tmp_path / "only.html").exists()
    end = json.loads((tmp_path / "only.json").read_text())[-1]
    assert end["type"] == "run_end"
    assert end["data"]["trace_path"] == str(tmp_path / "only.json")


def test_summary_lines_name_the_report_and_both_traces():
    model = ss_udl_model()
    result = solve_beam_3d(model)
    det = {"passed": True, "checks": []}
    verdict = {"refuted": False, "model": "verifier-model"}

    lines = run_phase1._summary_lines(
        "orch-model", model, result, det, verdict, True,
        "results/phase1_report.md", "results/phase1_trace.html",
        "results/phase1_trace.json",
    )
    text = "\n".join(lines)
    assert text.startswith("Phase 1 summary")
    assert "orch-model" in text and "verifier-model" in text
    assert "GATE               : PASS" in text
    assert "results/phase1_trace.html" in text
    assert "results/phase1_trace.json" in text


def test_summary_lines_mark_a_skipped_html_trace():
    model = ss_udl_model()
    result = solve_beam_3d(model)
    lines = run_phase1._summary_lines(
        "orch-model", model, result, {"passed": False, "checks": []},
        {"refuted": True, "model": None}, False, "r.md", None, "t.json",
    )
    text = "\n".join(lines)
    assert "trace (html)       : skipped" in text
    assert "verifier model     : n/a" in text
    assert "GATE               : FAIL" in text

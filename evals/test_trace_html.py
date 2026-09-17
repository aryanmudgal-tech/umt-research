"""Offline tests for the standalone HTML trace viewer.

Synthetic events only: no API key, no network, no solver run.
"""

import copy
import html
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.trace_html import PRE_MAX_PX, write_trace_html

BRIEF = (
    "Analyze the professor's bridge beam: simply supported, span 25 m, "
    "E = 30 GPa, Iz = 0.005 m^4, uniform load q = 30 kN/m downward."
)
XSS = "<script>alert(1)</script>"

MODEL_DICT = {
    "nodes": [{"id": f"N{k}", "x": 25.0 * k / 4} for k in range(5)],
    "material": {"E": 30000000000.0, "G": 12500000000.0},
    "section": {"A": 0.5, "Iy": 0.004, "Iz": 0.005, "J": 0.001},
    "supports": {
        "N0": [True, True, True, True, False, False],
        "N4": [False, True, True, False, False, False],
    },
    "distributed_loads": [
        {"element": "all", "direction": "y", "w1": -30000.0, "w2": -30000.0}
    ],
    "point_loads": [],
}

GATE_CHECKS = [
    {"name": "invariant_equilibrium_forces", "passed": True, "detail": "sum Fy residual 1.2e-09 N"},
    {"name": "closed_form_midspan_deflection", "passed": True, "detail": "rel err 3.1e-13"},
    {"name": "pynite_agreement", "passed": True, "detail": "worst scaled mismatch 4.0e-13"},
]

VERIFIER_CHECKS = [
    {"name": "independent_pynite", "passed": True, "detail": "matches to 1e-12"},
    {"name": "span_reading", "passed": True, "detail": "25 m span read from the brief"},
]

REASONING = "Recomputed with PyNite and the textbook 5qL^4/384EI form; nothing to refute."


def _events():
    """One event of every type in the schema, in run order."""
    return [
        {"seq": 0, "t": 0.0, "stage": "brief", "type": "run_start",
         "title": "Run started", "data": {"brief_path": "evals/brief_25m.md", "brief_text": BRIEF}},
        {"seq": 1, "t": 0.01, "stage": "orchestrator", "type": "stage_start",
         "title": "Orchestrator", "data": {"stage": "orchestrator"}},
        {"seq": 2, "t": 0.02, "stage": "orchestrator", "type": "model_attempt",
         "title": "Trying gemini-3.8-flash", "data": {"role": "orchestrator", "model": "gemini-3.8-flash"}},
        {"seq": 3, "t": 0.9, "stage": "orchestrator", "type": "model_fallback",
         "title": "gemini-3.8-flash overloaded",
         "data": {"role": "orchestrator", "model": "gemini-3.8-flash", "error": "503 UNAVAILABLE"}},
        {"seq": 4, "t": 1.0, "stage": "orchestrator", "type": "llm_request",
         "title": "Prompting gemini-3.5-flash",
         "data": {"role": "orchestrator", "model": "gemini-3.5-flash",
                  "tool_names": ["solve_beam_3d", "closed_form_case"], "user_preview": BRIEF[:400]}},
        {"seq": 5, "t": 2.4, "stage": "orchestrator", "type": "llm_response",
         "title": "Gemini called solve_beam_3d",
         "data": {"role": "orchestrator", "model": "gemini-3.5-flash",
                  "text": f"Building the model. {XSS}",
                  "function_calls": [{"name": "solve_beam_3d", "args": {"model": MODEL_DICT}}]}},
        {"seq": 6, "t": 2.5, "stage": "orchestrator", "type": "tool_call",
         "title": "solve_beam_3d", "data": {"name": "solve_beam_3d", "args": {"model": MODEL_DICT}}},
        {"seq": 7, "t": 2.6, "stage": "orchestrator", "type": "tool_result",
         "title": "solve_beam_3d returned",
         "data": {"name": "solve_beam_3d", "summary": "midspan uy = -6.104e-02 m",
                  "result": {"max_abs": {"uy": 0.061035, "Mz": 2343750.0, "Vy": 375000.0}}}},
        {"seq": 8, "t": 3.0, "stage": "gate", "type": "stage_start",
         "title": "Deterministic gate", "data": {"stage": "gate"}},
        *[
            {"seq": 9 + i, "t": 3.1 + 0.1 * i, "stage": "gate", "type": "gate_check",
             "title": check["name"], "data": check}
            for i, check in enumerate(GATE_CHECKS)
        ],
        {"seq": 12, "t": 3.5, "stage": "gate", "type": "gate_result",
         "title": "Deterministic gate passed",
         "data": {"passed": True, "n_passed": len(GATE_CHECKS), "n_total": len(GATE_CHECKS)}},
        {"seq": 13, "t": 3.6, "stage": "verifier", "type": "model_attempt",
         "title": "Trying gemini-3-flash-preview", "data": {"role": "verifier", "model": "gemini-3-flash-preview"}},
        {"seq": 14, "t": 8.2, "stage": "verifier", "type": "verdict",
         "title": "Verifier did not refute",
         "data": {"refuted": False, "checks": VERIFIER_CHECKS, "reasoning": REASONING,
                  "model": "gemini-3-flash-preview"}},
        {"seq": 15, "t": 8.4, "stage": "report", "type": "run_end",
         "title": "GATE PASS",
         "data": {"passed": True, "report_path": "results/phase1_report.md",
                  "trace_path": "results/phase1_trace.html", "duration_s": 8.4}},
    ]


def _write(tmp_path, events=None, meta=None):
    out = write_trace_html(events if events is not None else _events(),
                           tmp_path / "trace.html", meta or {})
    return out, out.read_text(encoding="utf-8")


_PRE = re.compile(r'<pre class="json">(.*?)</pre>', re.DOTALL)


def _json_blocks(page):
    """Every rendered JSON block, unescaped, as lists of lines."""
    return [html.unescape(m).split("\n") for m in _PRE.findall(page)]


def _block_containing(page, needle):
    for block in _json_blocks(page):
        if any(needle in line for line in block):
            return block
    raise AssertionError(f"no JSON block contains {needle!r}")


def _events_with(args=None, result=None):
    """The standard event list, carrying a different solver payload."""
    events = _events()
    if args is not None:
        events[5]["data"]["function_calls"][0]["args"] = args
        events[6]["data"]["args"] = args
    if result is not None:
        events[7]["data"]["result"] = result
    return events


def test_file_is_written_and_self_contained(tmp_path):
    out, page = _write(tmp_path)

    assert out == tmp_path / "trace.html"
    assert out.exists() and page.startswith("<!DOCTYPE html>")
    assert "http://" not in page and "https://" not in page
    assert "<link" not in page and "src=" not in page
    assert "<style>" in page and "<script>" in page


def test_brief_and_models_and_duration_in_header(tmp_path):
    _, page = _write(tmp_path)

    assert html.escape(BRIEF, quote=True) in page  # apostrophes and all
    assert "span 25 m" in page
    assert "Phase 1 Trace" in page
    assert "gemini-3.5-flash" in page
    assert "gemini-3-flash-preview" in page
    assert "8.40 s" in page


def test_tool_call_shows_the_model_dict_numbers(tmp_path):
    _, page = _write(tmp_path)

    assert "solve_beam_3d" in page
    # the dict the orchestrator built, verbatim
    assert "30000000000.0" in page  # E in Pa
    assert "-30000.0" in page  # w in N/m
    assert "0.005" in page  # Iz in m^4
    assert '&quot;nodes&quot;' in page
    assert "midspan uy = -6.104e-02 m" in page


def test_gate_and_verifier_are_rendered(tmp_path):
    _, page = _write(tmp_path)

    for check in GATE_CHECKS:
        assert check["name"] in page
        assert check["detail"] in page
    for check in VERIFIER_CHECKS:
        assert check["name"] in page
    assert REASONING in page
    assert "independent agent" in page  # the verifier is marked as separate
    assert "3 of 3 checks passed" in page


def test_pass_banner(tmp_path):
    _, page = _write(tmp_path)
    assert "gate pass" in page
    assert "GATE PASS" in page


def test_fail_banner_when_refuted(tmp_path):
    events = _events()
    events[-2]["data"]["refuted"] = True
    events[-1]["data"]["passed"] = False
    _, page = _write(tmp_path, events)

    assert "gate fail" in page
    assert "GATE FAIL" in page
    assert "REFUTED" in page


def test_everything_is_html_escaped(tmp_path):
    events = _events()
    events[0]["data"]["brief_text"] = BRIEF + "\n" + XSS
    events[6]["data"]["args"]["note"] = XSS
    events[7]["data"]["summary"] = XSS
    _, page = _write(tmp_path, events)

    assert XSS not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_every_event_type_reaches_the_page(tmp_path):
    events = _events()
    _, page = _write(tmp_path, events)

    for event in events:
        assert event["type"] in page
        assert event["title"] in page


def test_supports_vector_is_one_line_not_twelve(tmp_path):
    _, page = _write(tmp_path)
    block = _block_containing(page, '"supports"')

    assert '"N0": [true, true, true, true, false, false]' in "\n".join(block)
    lone = [line for line in block if line.strip() in ("true,", "false,", "true", "false")]
    assert lone == []


def test_the_model_dict_reads_in_under_forty_lines(tmp_path, tool_call_args):
    """The real captured model dict, or a stand-in of the same shape."""
    _, page = _write(tmp_path, _events_with(args=tool_call_args))
    block = _block_containing(page, '"supports"')

    assert len(block) < 40, len(block)
    for key in tool_call_args["model"]:
        assert any(f'"{key}"' in line for line in block), key
    assert any("30000000000" in line for line in block)  # E in Pa, unrounded


def test_the_solver_result_shrinks_against_raw_json(tmp_path, tool_result_payload):
    import json as _json

    _, page = _write(tmp_path, _events_with(result=tool_result_payload))
    block = _block_containing(page, '"diagrams"')

    raw = _json.dumps(tool_result_payload, indent=2).count("\n") + 1
    assert len(block) < raw / 2, (len(block), raw)
    assert any('"max_abs"' in line for line in block)


def test_every_json_block_is_capped_by_css(tmp_path):
    _, page = _write(tmp_path)

    assert f"max-height: {PRE_MAX_PX}px" in page
    assert "overflow: auto" in page
    assert "overflow-x: auto" not in page  # the old uncapped rule is gone


def test_folded_content_stays_reachable(tmp_path, tool_call_args):
    # Force a fold: the captured run's mesh may be short enough to show whole.
    args = copy.deepcopy(tool_call_args)
    args["model"]["nodes"] = [{"id": f"N{k}", "x": 2.5 * k} for k in range(11)]
    _, page = _write(tmp_path, _events_with(args=args))
    compact = _block_containing(page, '"supports"')

    marker = [line for line in compact if "more items" in line]
    assert marker, "expected the nodes array to fold"
    assert '<details class="fold">' in page

    blocks = _json_blocks(page)
    full = blocks[blocks.index(compact) + 1]
    assert len(full) > len(compact)
    assert f"full JSON ({len(full)} lines)" in page
    # the node the compact view folded away is still in the page
    last_node = tool_call_args["model"]["nodes"][-1]["id"]
    assert any(last_node in line for line in full)


def test_nothing_is_folded_when_nothing_needs_it(tmp_path):
    small = [{"seq": 0, "t": 0.0, "stage": "orchestrator", "type": "tool_call",
              "title": "closed_form_case",
              "data": {"name": "closed_form_case",
                       "args": {"case": "ss_udl", "L": 25.0, "q": -30000.0}}}]
    _, page = _write(tmp_path, small)
    rendered = "\n".join("\n".join(b) for b in _json_blocks(page))

    assert "more items" not in rendered
    assert '<details class="fold">' not in page
    assert "full JSON" not in page


def test_the_highlighter_covers_compacted_text(tmp_path):
    _, page = _write(tmp_path)

    assert "j-fold" in page  # fold markers are styled, not tokenised as numbers
    assert "highlightLine" in page
    assert "raw.split('\\n').map(highlightLine)" in page
    assert 'querySelectorAll(\'pre.json\')' in page


def test_empty_and_unknown_events_do_not_crash(tmp_path):
    out = write_trace_html([], tmp_path / "empty.html", {})
    assert "No events recorded." in out.read_text(encoding="utf-8")

    odd = [{"seq": 0, "t": 1.0, "stage": "weird", "type": "mystery",
            "title": "Unknown event", "data": {"k": [1, 2, None, True]}}]
    _, page = _write(tmp_path, odd)
    assert "Unknown event" in page
    assert "gate incomplete" in page  # no verdict recorded -> no claim of a pass

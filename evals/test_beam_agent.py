"""Offline tests for the agent deployed on Agent Platform (beam_agent/).

The pipeline is faked, so these test the wiring around it: the brief reaches
the pipeline untouched, the report reaches the user word for word without a
model call, two analyses never overlap, and the deployment stays on the cheap
settings. No API calls.
"""

import asyncio
import json
import sys
import threading
import time
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.adk.models.llm_request import LlmRequest
from google.genai import types

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.run_phase1 import RunOutcome  # noqa: E402
from beam_agent import agent as beam  # noqa: E402

AGENT_DIR = REPO_ROOT / "beam_agent"
REPORT = (REPO_ROOT / "evals" / "fixtures" / "report_soil_pass.md").read_text()
GATE = {"passed": True, "checks": [{"name": "equation_equilibrium", "status": "pass", "passed": True, "detail": ""}]}
VERDICT = {"refuted": False, "checks": [], "reasoning": "", "model": "m"}


def fake_pipeline(seen=None):
    def run_pipeline(brief, out_dir):
        if seen is not None:
            seen.append(brief)
        report = Path(out_dir) / "report.md"
        report.write_text(REPORT)
        return RunOutcome(code=0, passed=True, lines=[], report_path=report, gate=GATE, verdict=VERDICT)

    return run_pipeline


def context(**state):
    return SimpleNamespace(state=dict(state))


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------- the agent


def test_the_agent_has_the_two_report_tools_and_the_hand_over():
    assert beam.root_agent.name == "umt_beam_engineer"
    names = sorted(tool.__name__ for tool in beam.root_agent.tools)
    assert names == ["analyze_brief", "show_full_report"]
    assert beam.root_agent.before_model_callback is beam.hand_over_report


def test_the_instruction_has_no_template_braces():
    # ADK fills {name} from session state: a stray brace would break every turn
    assert "{" not in beam.INSTRUCTION and "}" not in beam.INSTRUCTION


def test_the_instruction_forbids_numbers_that_did_not_come_from_a_result():
    assert "Never state a number that did not come from a result" in beam.INSTRUCTION
    assert "EXACTLY as the user wrote it" in beam.INSTRUCTION


# --------------------------------------------------------- analyze_brief


def test_the_brief_reaches_the_pipeline_untouched(monkeypatch):
    seen = []
    monkeypatch.setattr(beam, "run_pipeline", fake_pipeline(seen))
    brief = "# 25 m beam on soil\n\nE = 30 GPa, k = 1.0e7 N/m^2 ...\n"
    out = run(beam.analyze_brief(brief, context()))
    assert seen == [brief]
    assert out["verified"] is True


def test_the_summary_and_the_full_report_are_kept_for_the_hand_over(monkeypatch):
    monkeypatch.setattr(beam, "run_pipeline", fake_pipeline())
    ctx = context()
    out = run(beam.analyze_brief("brief", ctx))
    assert ctx.state[beam.SUMMARY_KEY] == out["summary"]
    assert out["summary"].startswith("**Verified.**")
    assert "| Midspan deflection | 3.015 mm downward | x = 12.5 m |" in out["summary"]
    assert ctx.state[beam.REPORT_KEY] == REPORT


def test_a_pipeline_that_raises_gives_a_sentence_not_a_crash(monkeypatch):
    def broken(brief, out_dir):
        raise RuntimeError("all orchestrator models failed: 503 UNAVAILABLE")

    monkeypatch.setattr(beam, "run_pipeline", broken)
    ctx = context()
    out = run(beam.analyze_brief("brief", ctx))
    assert out["verified"] is False
    assert out["summary"].startswith("**The analysis stopped before it finished.**")
    assert "503 UNAVAILABLE" in out["summary"]
    assert ctx.state[beam.SUMMARY_KEY] == out["summary"]


def test_two_analyses_never_overlap(monkeypatch):
    lock = threading.Lock()
    running, most = [0], [0]

    def slow(brief, out_dir):
        with lock:
            running[0] += 1
            most[0] = max(most[0], running[0])
        time.sleep(0.2)
        with lock:
            running[0] -= 1
        return fake_pipeline()(brief, out_dir)

    monkeypatch.setattr(beam, "run_pipeline", slow)

    async def both():
        return await asyncio.gather(
            beam.analyze_brief("a", context()), beam.analyze_brief("b", context())
        )

    outs = run(both())
    assert [o["verified"] for o in outs] == [True, True]
    assert most[0] == 1


def test_show_full_report_says_whether_there_is_one():
    assert beam.show_full_report(context()) == {"available": False}
    assert beam.show_full_report(context(**{beam.REPORT_KEY: "R"})) == {"available": True}


# ------------------------------------------------------------- hand over


def request_after(name):
    response = types.FunctionResponse(name=name, response={"ok": True})
    return LlmRequest(contents=[types.Content(role="user", parts=[types.Part(function_response=response)])])


@pytest.mark.parametrize(
    "tool, key", [("analyze_brief", "beam_summary"), ("show_full_report", "beam_report")]
)
def test_after_a_report_tool_the_text_is_handed_over_without_a_model_call(tool, key):
    ctx = context(beam_summary="SUMMARY 3.015 mm", beam_report="FULL REPORT")
    response = beam.hand_over_report(callback_context=ctx, llm_request=request_after(tool))
    assert response is not None
    assert response.content.parts[0].text == ctx.state[key]


def test_an_ordinary_turn_goes_to_the_model():
    ctx = context(beam_summary="S", beam_report="R")
    question = LlmRequest(contents=[types.Content(role="user", parts=[types.Part(text="Why is the shear so high?")])])
    assert beam.hand_over_report(callback_context=ctx, llm_request=question) is None
    assert beam.hand_over_report(callback_context=ctx, llm_request=LlmRequest(contents=[])) is None


def test_with_no_report_yet_the_model_answers():
    ctx = context()
    assert beam.hand_over_report(callback_context=ctx, llm_request=request_after("show_full_report")) is None


# ------------------------------------------------------------ deployment


def test_the_deployment_stays_on_the_cheap_settings():
    config = json.loads((AGENT_DIR / ".agent_engine_config.json").read_text())
    # Agent Platform's default is one instance kept on all the time, at 4 CPUs
    assert config["min_instances"] == 0
    assert config["max_instances"] == 1
    assert config["resource_limits"] == {"cpu": "1", "memory": "2Gi"}
    assert config["extra_packages"] == ["../agent", "../tools"]


def test_the_deployed_solver_is_the_tested_solver():
    pins = {}
    for line in (AGENT_DIR / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, pinned = line.split("==")
            pins[name.split("[")[0].lower()] = pinned
    for package in ("numpy", "scipy", "sympy", "mpmath", "PyNiteFEA", "google-genai", "google-api-core"):
        assert pins[package.lower()] == version(package), package

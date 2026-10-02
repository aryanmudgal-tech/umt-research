"""Offline tests for web/story.py: how the answer was reached, told to a civil engineer.

The professor sees a run's trace as seven numbered steps built from the
recorded events. These tests build the story from a fresh soil run (the demo
pipeline: recorded Gemini turns, real solve and gate), from a standard-beam
run assembled the same way the pipeline records one, and from runs that
stopped early. No API calls.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.gates import deterministic_gate
from agent.recorder import to_plain
from agent.trace import tracer
from tools.fem.solver import solve_beam_3d
from web.demo import demo_pipeline
from web.story import build_story

PASSED = {"status": "passed", "message": "Every check passed and the independent check agrees.", "duration_s": 27.4}


@pytest.fixture(scope="module")
def soil_events():
    brief = (REPO_ROOT / "web" / "sample_brief.md").read_text()
    with tempfile.TemporaryDirectory() as tmp:
        demo_pipeline(speed=0)(brief, Path(tmp))
        return json.loads((Path(tmp) / "trace.json").read_text())


@pytest.fixture(scope="module")
def soil(soil_events):
    return build_story(soil_events, PASSED)


def step(story, key):
    return next(s for s in story["steps"] if s["key"] == key)


def item(step_, start):
    return next(i for i in step_["items"] if i["text"].startswith(start))


def test_the_story_has_seven_numbered_steps_in_order(soil):
    assert [s["key"] for s in soil["steps"]] == ["reading", "equation", "model", "solving", "checking", "independent", "report"]
    assert all(s["status"] == "done" for s in soil["steps"])
    assert soil["outcome"]["status"] == "passed"


def test_reading_says_which_ai_read_it_and_what_it_found(soil):
    reading = step(soil, "reading")
    assert "gemini-3.8-flash" in reading["summary"]
    assert "performs no calculation" in reading["summary"]
    texts = [i["text"] for i in reading["items"]]
    assert "Span: 25 m" in texts and "Subgrade modulus: 1 × 10⁷ N/m²" in texts


def test_the_equation_is_typeset_and_each_term_explained(soil):
    equation = step(soil, "equation")
    assert "Beam on elastic foundation" in equation["summary"]
    assert "v''''" in equation["tex"] and "= q" in equation["tex"]
    texts = [i["text"] for i in equation["items"]]
    assert "Flexural rigidity EI = 1.5 × 10⁸ N·m²" in texts
    assert "Modulus of subgrade reaction k = 1 × 10⁷ N/m² (Winkler foundation)" in texts
    assert "Distributed load q = 30 kN/m, downward" in texts


def test_the_model_names_its_pieces_and_supports(soil):
    model = step(soil, "model")
    assert model["summary"] == (
        "Discretized into 20 Hermite cubic beam elements of 1.25 m: 21 nodes, 42 degrees of freedom."
    )
    assert item(model, "Pin at x = 0 m")["detail"] == "Restrains deflection; rotation free."
    assert item(model, "Roller at x = 25 m")


def test_the_solution_was_the_solvers_and_gives_the_answers(soil):
    solving = step(soil, "solving")
    assert "the AI model played no part" in solving["summary"]
    assert "K·u = F (42 degrees of freedom)" in solving["summary"]
    texts = [i["text"] for i in solving["items"]]
    assert "Midspan deflection: 3.015 mm downward" in texts
    assert any(t.startswith("Maximum bending moment: 37.5 kN·m at x = 2.1") for t in texts)
    assert (
        "Load sharing: the foundation carries 666.5 kN of the 750 kN applied load; the supports carry 83.5 kN."
        in texts
    )


def test_each_check_says_what_it_tests_and_shows_its_evidence(soil):
    checking = step(soil, "checking")
    balance = item(checking, "Global vertical equilibrium")
    assert balance["status"] == "pass"
    assert balance["evidence"] == "Supports 83.5 kN + foundation 666.5 kN = 750 kN = applied load."
    mms = item(checking, "Convergence study")
    assert mms["status"] == "pass"
    assert mms["evidence"] == (
        "Meshes of 20, 40 and 80 elements: the error fell 16× per halving of the element size, "
        "an observed order of 4.00 against the theoretical 4."
    )
    assert item(checking, "Residual of the stiffness equations")["evidence"].endswith("zero to machine precision.")
    sign = item(checking, "Deflection sense")
    assert sign["status"] == "skipped" and "foundation" in sign["evidence"]
    textbook = item(checking, "Closed-form solution")
    assert textbook["status"] == "skipped" and "no closed-form solution" in textbook["evidence"]
    assert "never counted as a pass" in checking["summary"]


def test_the_independent_check_says_what_it_did(soil):
    independent = step(soil, "independent")
    assert "gemini-3-flash-preview" in independent["summary"]
    assert "collocation on the strong form" in independent["summary"]
    assert any(i["status"] == "pass" for i in independent["items"])
    assert independent["verdict"] == "Result confirmed: no discrepancy found."


def test_the_report_step_gives_the_time(soil):
    assert "27 s" in step(soil, "report")["summary"]


# ------------------------------------------------------- the standard path


def test_a_standard_beam_story_uses_the_textbook_equation_and_pynite():
    L, Q = 25.0, 30e3
    model = {
        "nodes": [{"id": f"N{k}", "x": L * k / 4} for k in range(5)],
        "material": {"E": 30e9, "G": 12.5e9},
        "section": {"A": 0.5, "Iy": 0.004, "Iz": 0.005, "J": 0.001},
        "supports": {"N0": [True, True, True, True, False, False], "N4": [False, True, True, False, False, False]},
        "distributed_loads": [{"element": "all", "direction": "y", "w1": -Q, "w2": -Q}],
        "point_loads": [],
    }
    tracer.reset()
    tracer.emit("orchestrator", "stage_start", "stage", stage="orchestrator")
    tracer.emit("orchestrator", "model_attempt", "m", role="orchestrator", model="gemini-3.8-flash")
    tracer.emit("orchestrator", "tool_call", "c", name="solve_beam_3d", args={"model": model})
    result = to_plain(solve_beam_3d(model))
    tracer.emit("orchestrator", "tool_result", "r", name="solve_beam_3d", result=result, summary="")
    tracer.emit("gate", "stage_start", "stage", stage="gate")
    deterministic_gate(model, result)
    tracer.emit("verifier", "stage_start", "stage", stage="verifier")
    tracer.emit("verifier", "tool_call", "c", name="pynite_crosscheck", args={})
    tracer.emit("verifier", "verdict", "v", refuted=False, checks=[], reasoning="", model="gemini-3-flash-preview")
    story = build_story(tracer.events, PASSED)

    assert step(story, "equation")["summary"].startswith("Euler–Bernoulli beam equation")
    assert "3D frame elements" in step(story, "model")["summary"]
    assert "PyNite" in step(story, "independent")["summary"]
    balance = item(step(story, "checking"), "Global vertical equilibrium")
    assert balance["evidence"] == "Supports 750 kN = applied load."
    assert item(step(story, "checking"), "Independent FEM program (PyNite)")["status"] == "pass"


# ------------------------------------------------------ runs that stopped


def test_a_refused_brief_stops_after_reading_with_the_reason():
    events = [
        {"stage": "orchestrator", "type": "stage_start", "t": 0, "data": {"stage": "orchestrator"}},
        {"stage": "orchestrator", "type": "model_attempt", "t": 0, "data": {"role": "orchestrator", "model": "gemini-3.8-flash"}},
        {"stage": "report", "type": "run_end", "t": 6, "data": {"passed": False}},
    ]
    story = build_story(events, {"status": "refused", "message": "The equation has a third-derivative term."})
    assert [s["key"] for s in story["steps"]] == ["reading", "stopped"]
    assert story["steps"][1]["summary"] == "The equation has a third-derivative term."


def test_a_busy_model_is_mentioned_and_a_busy_run_stops_plainly():
    events = [
        {"stage": "orchestrator", "type": "stage_start", "t": 0, "data": {"stage": "orchestrator"}},
        {"stage": "orchestrator", "type": "model_attempt", "t": 0, "data": {"role": "orchestrator", "model": "m1"}},
        {"stage": "orchestrator", "type": "model_fallback", "t": 1, "data": {"role": "orchestrator", "model": "m1", "error": "503"}},
        {"stage": "orchestrator", "type": "model_attempt", "t": 1, "data": {"role": "orchestrator", "model": "m2"}},
    ]
    story = build_story(events, {"status": "busy", "message": "The AI service is busy right now. Try again in a few minutes."})
    assert "m1 was busy" in story["steps"][0]["summary"] and "m2" in story["steps"][0]["summary"]
    assert story["steps"][-1]["key"] == "stopped"


def test_the_story_speaks_engineering_not_software(soil):
    words = json.dumps(soil).lower()
    for term in ("json", "tool call", "api", "solve_with_equation", "solve_beam_3d", "orchestrator", "pipeline", "llm"):
        assert term not in words, term

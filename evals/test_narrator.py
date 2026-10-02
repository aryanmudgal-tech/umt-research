"""Offline tests for web/narrator.py: raw run events in, the professor's view out.

The wording is a civil engineer's (elements, degrees of freedom, subgrade
modulus), never software's (tool calls, JSON, function names).

The page shows six stages in plain English. The narrator turns the tracer's
events into stage updates, so these tests replay a real recorded run and a
handful of scripted ones (busy model, refused equation, failed check,
refuting verifier, no solver call) and check what the professor would read.
No API calls.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from web.narrator import STAGES, Narrator, plain_check_name

FIXTURE = REPO_ROOT / "evals" / "fixtures" / "trace_soil_pass.json"


def replay(events):
    narrator = Narrator()
    updates = []
    for event in events:
        updates.extend(narrator.feed(event))
    return narrator, updates


def last(updates, stage):
    return [u for u in updates if u["stage"] == stage][-1]


def ev(stage, type, /, **data):
    return {"stage": stage, "type": type, "title": type, "data": data}


@pytest.fixture(scope="module")
def soil_run():
    return replay(json.loads(FIXTURE.read_text()))


# ------------------------------------------------------- a real passing run


def test_every_stage_ends_done_in_order(soil_run):
    _, updates = soil_run
    assert [s for s in STAGES] == ["reading", "modeling", "solving", "checking", "independent", "report"]
    for stage in STAGES:
        assert last(updates, stage)["status"] == "done", stage
    first_active = [next(i for i, u in enumerate(updates) if u["stage"] == s) for s in STAGES]
    assert first_active == sorted(first_active)


def test_reading_lists_the_facts_it_found(soil_run):
    _, updates = soil_run
    reading = last(updates, "reading")
    assert "Span: 25 m" in reading["facts"]
    assert "Supports: pin and roller" in reading["facts"]
    assert "Load: 30 kN/m downward" in reading["facts"]
    assert "Subgrade modulus: 1 × 10⁷ N/m²" in reading["facts"]


def test_the_model_and_the_scene_follow_the_brief(soil_run):
    _, updates = soil_run
    modeling = last(updates, "modeling")
    assert "20 Hermite cubic beam elements of 1.25 m: 21 nodes, 42 degrees of freedom" in modeling["explanation"]
    scene = modeling["scene"]
    assert scene["soil"] is True and scene["axial"] is False and scene["tapered"] is False
    assert scene["elements"] == 20


def test_solving_reports_the_real_numbers(soil_run):
    _, updates = soil_run
    solving = last(updates, "solving")
    assert "3.015 mm" in solving["explanation"]
    assert "37.5 kN·m" in solving["explanation"]
    assert solving["scene"]["sag_mm"] == pytest.approx(3.015, rel=1e-3)


def test_checking_and_the_independent_check_say_what_happened(soil_run):
    _, updates = soil_run
    assert "6 of 6" in last(updates, "checking")["headline"]
    assert "skipped" in last(updates, "checking")["explanation"]
    assert "agrees" in last(updates, "independent")["headline"]


def test_the_run_end_is_reported(soil_run):
    narrator, _ = soil_run
    assert narrator.finished and narrator.passed is True


# ---------------------------------------------------------- scripted runs

SOLVE_ARGS = {
    "model": {
        "nodes": [{"id": "N0", "x": 0}, {"id": "N1", "x": 12.5}, {"id": "N2", "x": 25}],
        "supports": {"N0": [True, True, True, True, False, False], "N2": [False, True, True, False, False, False]},
    },
    "equation": {"coeffs": {"v4": "E*I", "v2": "P"}, "rhs": "q", "params": {"E": 30e9, "I": 0.005, "P": 2e6, "q": -30e3}},
}


def test_a_busy_model_shows_as_a_note_not_a_failure():
    _, updates = replay([
        ev("orchestrator", "stage_start", stage="orchestrator"),
        ev("orchestrator", "model_attempt", role="orchestrator", model="m1"),
        ev("orchestrator", "model_fallback", role="orchestrator", model="m1", error="503 UNAVAILABLE"),
    ])
    reading = last(updates, "reading")
    assert reading["status"] == "active"
    assert "busy" in reading["note"]


def test_an_axial_force_brief_gets_the_thrust_scene():
    _, updates = replay([
        ev("orchestrator", "stage_start", stage="orchestrator"),
        ev("orchestrator", "tool_call", name="solve_with_equation", args=SOLVE_ARGS),
    ])
    assert last(updates, "modeling")["scene"]["axial"] is True
    assert "Axial force: 2 MN compression" in last(updates, "reading")["facts"]


def test_a_refused_equation_shows_the_solvers_reason_while_the_ai_decides():
    _, updates = replay([
        ev("orchestrator", "stage_start", stage="orchestrator"),
        ev("orchestrator", "tool_call", name="solve_with_equation", args=SOLVE_ARGS),
        ev("orchestrator", "tool_result", name="solve_with_equation",
           result={"error": "EquationError: coefficient 'v3' is not supported: a third-derivative term", "solved": False}),
    ])
    solving = last(updates, "solving")
    assert solving["status"] == "active"
    assert "third-derivative" in solving["note"]


def test_no_solver_call_marks_the_reading_stage_failed():
    narrator, updates = replay([
        ev("orchestrator", "stage_start", stage="orchestrator"),
        ev("report", "stage_start", stage="report"),
        ev("report", "run_end", passed=False),
    ])
    assert last(updates, "reading")["status"] == "failed"
    assert narrator.finished and narrator.passed is False


def test_a_failed_check_is_named_in_plain_words():
    _, updates = replay([
        ev("gate", "stage_start", stage="gate"),
        ev("gate", "gate_check", name="equation_equilibrium", passed=False, status="fail", detail="residual 1e-3"),
        ev("gate", "gate_check", name="equation_mms", passed=True, status="pass", detail=""),
        ev("gate", "gate_result", passed=False, n_passed=1, n_failed=1, n_skipped=0, n_total=2),
    ])
    checking = last(updates, "checking")
    assert checking["status"] == "failed"
    assert "Global vertical equilibrium" in checking["explanation"]


def test_a_refuting_verifier_gives_its_reason():
    _, updates = replay([
        ev("verifier", "stage_start", stage="verifier"),
        ev("verifier", "verdict", refuted=True, checks=[], reasoning="The reported shear does not match the reaction.", model="v"),
    ])
    independent = last(updates, "independent")
    assert independent["status"] == "failed"
    assert "shear does not match" in independent["explanation"]


@pytest.mark.parametrize(
    "name, plain",
    [
        ("invariant_equilibrium_forces", "Global vertical equilibrium"),
        ("equation_mms", "Convergence study (manufactured solution)"),
        ("closed_form_max_moment", "Closed-form solution"),
        ("some_new_check", "Some new check"),
    ],
)
def test_check_names_read_as_english(name, plain):
    assert plain_check_name(name) == plain

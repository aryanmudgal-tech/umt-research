"""Two things the gate must never do: bless a corrupted result, or count a skip.

The gate exists because nothing downstream can tell a right number from a
plausible one. These tests attack it from both sides.

CORRUPTION. The three ways a result can be wrong that matter are the three a
model could produce: every deflection off by a factor, one node's value edited,
and the reactions edited. All three are run against the professor's own
Euler-Bernoulli spec AND against a foundation spec, because the two are checked
by different arithmetic and the Euler-Bernoulli half is the one with no
foundation term to give a corrupted deflection away.

SKIPS. A check that did not apply verified nothing, so a run with one skipped
check must read "N passed, 1 skipped" on every surface the professor sees, and
the skipped row must never look like a pass.

Offline and deterministic: no LLM, no network.
"""

import copy
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent import live_view, run_phase1, trace_html  # noqa: E402
from agent.gates import (  # noqa: E402
    FAIL,
    SKIPPED,
    check_status,
    deterministic_gate,
    equation_checks,
    tally,
)
from tools.fem.equation import equilibrium_residual, solve_equation_beam  # noqa: E402

SPAN, N_ELEM = 25.0, 8


def spec(name):
    return json.loads((REPO_ROOT / "equations" / f"{name}.json").read_text())


EULER_BERNOULLI = spec("euler_bernoulli")
FOUNDATION = spec("elastic_foundation")


def equation_model(n_elem=N_ELEM):
    return {
        "nodes": [{"id": f"N{k}", "x": SPAN * k / n_elem} for k in range(n_elem + 1)],
        "supports": {
            "N0": [True, True, True, True, False, False],
            f"N{n_elem}": [False, True, True, False, False, False],
        },
        "point_loads": [],
    }


@pytest.fixture(scope="module")
def runs():
    """One clean solve per spec, shared: the MMS check is the slow part."""
    model = equation_model()
    return {
        "euler_bernoulli": (model, EULER_BERNOULLI, solve_equation_beam(model, EULER_BERNOULLI)),
        "elastic_foundation": (model, FOUNDATION, solve_equation_beam(model, FOUNDATION)),
    }


# ------------------------------------------------------------- the corruptions


def scale_every_deflection(result):
    for node in result["displacements"].values():
        node["v"] *= 1.05
        node["slope"] *= 1.05


def edit_one_node(result):
    node = f"N{N_ELEM // 2}"  # midspan, where the deflection is largest
    result["displacements"][node]["v"] *= 1.05


def edit_the_reactions(result):
    for forces in result["reactions"].values():
        for dof in forces:
            forces[dof] *= 1.05


CORRUPTIONS = [
    pytest.param(scale_every_deflection, id="every-deflection-scaled"),
    pytest.param(edit_one_node, id="one-node-edited"),
    pytest.param(edit_the_reactions, id="reactions-edited"),
]
SPECS = ["euler_bernoulli", "elastic_foundation"]


@pytest.mark.parametrize("spec_name", SPECS)
def test_the_clean_result_passes_the_gate(runs, spec_name):
    model, equation, result = runs[spec_name]
    gate = deterministic_gate(model, result, equation)
    assert gate["passed"], [c for c in gate["checks"] if check_status(c) == FAIL]


@pytest.mark.parametrize("spec_name", SPECS)
@pytest.mark.parametrize("corrupt", CORRUPTIONS)
def test_a_corrupted_result_fails_the_gate(runs, spec_name, corrupt):
    model, equation, clean = runs[spec_name]
    tampered = copy.deepcopy(clean)
    corrupt(tampered)

    gate = deterministic_gate(model, tampered, equation)
    failed = [c for c in gate["checks"] if check_status(c) == FAIL]

    assert not gate["passed"], "a corrupted result passed the gate"
    assert failed, "the gate failed without naming a single check"


@pytest.mark.parametrize("spec_name", SPECS)
@pytest.mark.parametrize("corrupt", CORRUPTIONS)
def test_the_failure_says_what_disagrees_and_by_how_much(runs, spec_name, corrupt):
    """A failure the professor cannot act on is barely better than no failure."""
    model, equation, clean = runs[spec_name]
    tampered = copy.deepcopy(clean)
    corrupt(tampered)

    failed = [
        c for c in equation_checks(model, equation, tampered) if check_status(c) == FAIL
    ]
    assert failed
    for check in failed:
        assert re.search(r"\d\.\d+e[+-]\d+", check["detail"]), check
        assert "tolerance" in check["detail"] or "imply" in check["detail"], check


def test_a_corrupted_deflection_is_caught_even_where_equilibrium_cannot_see_it():
    """Why equation_solution_residual exists, pinned so it cannot be dropped.

    For a spec with no a0 and no a1 term, rows 0 and 2 of every element matrix
    sum to zero, so the w = 1 equilibrium balance is identically independent of
    the displacements: the residual does not move when they are all scaled.
    Before the solution-residual check, that made every deflection in an
    Euler-Bernoulli run ungated.
    """
    model = equation_model()
    clean = solve_equation_beam(model, EULER_BERNOULLI)
    tampered = copy.deepcopy(clean)
    scale_every_deflection(tampered)

    before = equilibrium_residual(model, EULER_BERNOULLI, clean)
    after = equilibrium_residual(model, EULER_BERNOULLI, tampered)
    assert after == pytest.approx(before, abs=1e-12), "premise changed; re-read this test"

    checks = {c["name"]: c for c in equation_checks(model, EULER_BERNOULLI, tampered)}
    assert check_status(checks["equation_solution_residual"]) == FAIL


def test_an_edited_headline_number_is_caught_on_its_own():
    """max_abs is what the report headlines and the narrative quotes."""
    model = equation_model()
    tampered = copy.deepcopy(solve_equation_beam(model, FOUNDATION))
    tampered["max_abs"]["v"] *= 0.9

    checks = {c["name"]: c for c in equation_checks(model, FOUNDATION, tampered)}
    assert check_status(checks["equation_samples_consistent"]) == FAIL
    assert "max_abs" in checks["equation_samples_consistent"]["detail"]


# --------------------------------------------------- a skip is not a pass


@pytest.fixture(scope="module")
def skipped_run(runs):
    """A real gate result with at least one skipped check, and its tally."""
    model, equation, result = runs["elastic_foundation"]
    gate = deterministic_gate(model, result, equation)
    skipped = [c for c in gate["checks"] if check_status(c) == SKIPPED]
    assert skipped, "this spec is meant to skip the closed form"
    return gate, skipped


def test_the_tally_splits_the_skips_out_of_the_passes(skipped_run):
    gate, skipped = skipped_run
    counts = gate["tally"]

    assert counts == tally(gate["checks"])
    assert counts["skipped"] == len(skipped)
    assert counts["passed"] + counts["skipped"] + counts["failed"] == counts["total"]
    assert counts["passed"] == counts["total"] - counts["skipped"]


def test_the_gate_still_passes_but_says_what_it_did_not_check(skipped_run):
    gate, skipped = skipped_run
    assert gate["passed"], "a skipped check must not fail the gate"
    for check in skipped:
        assert check["detail"].strip(), f"{check['name']} skipped without saying why"


def gate_events(gate):
    counts = gate["tally"]
    events = [
        {
            "seq": n,
            "t": 0.1 * n,
            "stage": "gate",
            "type": "gate_check",
            "title": f"{check['name']}: {check['status'].upper()}",
            "data": dict(check),
        }
        for n, check in enumerate(gate["checks"], start=1)
    ]
    events.append(
        {
            "seq": len(events) + 1,
            "t": 9.9,
            "stage": "gate",
            "type": "gate_result",
            "title": "deterministic gate",
            "data": {
                "passed": gate["passed"],
                "n_passed": counts["passed"],
                "n_skipped": counts["skipped"],
                "n_failed": counts["failed"],
                "n_total": counts["total"],
            },
        }
    )
    return events


def test_the_live_view_shows_the_same_split_and_marks_the_skip(skipped_run, capsys):
    gate, skipped = skipped_run
    counts = gate["tally"]
    view = live_view.LiveView(console=live_view.Console(force_terminal=False, width=240))
    for event in gate_events(gate):
        view.handle(event)
    out = capsys.readouterr().out

    assert f"{counts['passed']}/{counts['total']} checks passed" in out
    assert f"{counts['skipped']} skipped" in out
    for check in skipped:
        line = next(l for l in out.splitlines() if check["name"] in l)
        assert "SKIP" in line and "PASS" not in line


def test_the_html_trace_shows_the_same_split_and_pills_the_skip(skipped_run, tmp_path):
    gate, skipped = skipped_run
    counts = gate["tally"]
    page = tmp_path / "trace.html"
    trace_html.write_trace_html(gate_events(gate), page, {"passed": True, "duration_s": 1.0})
    html = page.read_text()

    assert f"{counts['passed']} of {counts['total']} checks passed" in html
    assert f"{counts['skipped']} skipped" in html
    assert '<span class="pill warn">SKIPPED</span>' in html
    for check in skipped:
        assert check["name"] in html


def test_the_report_shows_the_same_split_and_rows_the_skip(skipped_run, runs, tmp_path):
    gate, skipped = skipped_run
    model, equation, result = runs["elastic_foundation"]
    counts = gate["tally"]

    original = run_phase1.REPORT_PATH
    run_phase1.REPORT_PATH = tmp_path / "report.md"
    try:
        path = run_phase1.write_report(
            "brief",
            "offline-test-model",
            model,
            result,
            gate,
            {"refuted": False, "checks": [], "reasoning": "", "model": "verifier"},
            True,
            equation,
        )
        report = path.read_text()
    finally:
        run_phase1.REPORT_PATH = original

    section = report.split("## Deterministic gate", 1)[1]
    assert f"{counts['passed']} of {counts['total']} checks passed" in section
    assert f"{counts['skipped']} skipped" in section
    for check in skipped:
        row = next(line for line in section.splitlines() if check["name"] in line)
        assert "| SKIPPED |" in row
        assert "PASS" not in row


def test_every_surface_agrees_on_the_numbers(skipped_run, runs, tmp_path):
    """One run, four surfaces, one set of numbers - the point of the exercise."""
    gate, _ = skipped_run
    counts = gate["tally"]
    assert counts["skipped"] >= 1

    page = tmp_path / "trace.html"
    trace_html.write_trace_html(gate_events(gate), page, {"passed": True, "duration_s": 1.0})
    html_counts = re.search(r"(\d+) of (\d+) checks passed, (\d+) skipped", page.read_text())

    assert html_counts
    assert [int(g) for g in html_counts.groups()] == [
        counts["passed"],
        counts["total"],
        counts["skipped"],
    ]

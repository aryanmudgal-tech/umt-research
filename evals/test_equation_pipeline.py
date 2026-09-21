"""The changeable equation, end to end: recorder -> gate -> report.

Every test here is OFFLINE and deterministic. No LLM is called, no network is
touched: the orchestrator agent is built but never run, and the solvers, the
gate and the report writer are exercised directly.

What these tests are defending:

1. the harness can tell WHICH solver produced the numbers it is gating;
2. a check that did not apply reports SKIPPED and is not counted as a pass —
   the gate used to report a skipped closed form as a pass, which is the one
   way a green gate can be a lie;
3. a run under the professor's own equation is still verified, by manufactured
   solutions and by equilibrium in the form that equation implies, and a
   tampered result still fails;
4. the report says which equation produced its numbers, and stays readable in
   a Markdown viewer while doing it.
"""

import copy
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent import live_view, run_phase1, trace_html
from agent.gates import FAIL, PASS, SKIPPED, check_status, deterministic_gate, tally
from agent.orchestrator import _INSTRUCTION, build_orchestrator
from agent.recorder import (
    EQUATION_TOOL,
    BEAM_TOOL,
    recorder,
    solve_beam_3d,
    solve_with_equation,
)
from tools.fem.solver import solve_beam_3d as raw_solve_beam_3d

L, E, G, I, Q, K_SOIL = 25.0, 30e9, 12.5e9, 0.005, 30e3, 1.0e7

BRIEF = """# Brief: 25 m beam on soil

- Simply supported, span 25 m, q = 30 kN/m downward, resting on ground with a
  modulus of subgrade reaction k = 1.0e7 N/m per metre of span.
"""

VERDICT = {
    "refuted": False,
    "checks": [{"name": "moment", "passed": True, "detail": "recomputed M = qL²/8"}],
    "reasoning": "recomputed; nothing to refute",
    "model": "verifier-model",
}


def equation_model(n_elem=8):
    """Bending-only model: the equation solver reads nodes, supports, point loads."""
    return {
        "nodes": [{"id": f"N{k}", "x": L * k / n_elem} for k in range(n_elem + 1)],
        "supports": {
            "N0": [True, True, True, True, False, False],
            f"N{n_elem}": [False, True, True, False, False, False],
        },
        "point_loads": [],
    }


def beam_model(n_elem=4):
    """The standard 3D model, for the solve_beam_3d half of these tests."""
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


def spec(name):
    return json.loads((REPO_ROOT / "equations" / f"{name}.json").read_text())


FOUNDATION = spec("elastic_foundation")
EULER_BERNOULLI = spec("euler_bernoulli")


@pytest.fixture(scope="module")
def foundation_run():
    """One solve of the foundation spec, shared: the MMS gate is the slow part."""
    model = equation_model()
    result = solve_with_equation(model, FOUNDATION)
    return model, result


@pytest.fixture(scope="module")
def foundation_gate(foundation_run):
    model, result = foundation_run
    return deterministic_gate(model, result, FOUNDATION)


# ------------------------------------------------------- the recorder


def test_the_recorder_records_both_tools():
    recorder.reset()
    solve_beam_3d(beam_model())
    solve_with_equation(equation_model(), FOUNDATION)

    assert [call.tool for call in recorder.calls] == [BEAM_TOOL, EQUATION_TOOL]
    beam_call, equation_call = recorder.calls
    assert beam_call.equation is None and beam_call.is_equation is False
    assert equation_call.equation == FOUNDATION and equation_call.is_equation is True
    recorder.reset()


def test_a_recorded_call_still_unpacks_as_model_and_result():
    # the harness has always read `model_dict, result = recorder.last`
    recorder.reset()
    solve_with_equation(equation_model(), FOUNDATION)
    model_dict, result = recorder.last
    assert model_dict == recorder.last.model
    assert result["max_abs"]["v"] > 0
    assert recorder.last[0] is model_dict and recorder.last[1] is result
    recorder.reset()


def test_the_recorder_deep_copies_the_equation_it_was_given():
    recorder.reset()
    mutable = copy.deepcopy(FOUNDATION)
    solve_with_equation(equation_model(), mutable)
    mutable["params"]["k"] = 0.0
    assert recorder.last.equation["params"]["k"] == K_SOIL
    recorder.reset()


def test_the_orchestrator_is_offered_both_solvers():
    agent = build_orchestrator("gemini-2.0-flash")  # built, never run
    names = {getattr(tool, "__name__", "") for tool in agent.tools}
    assert {BEAM_TOOL, EQUATION_TOOL} <= names


@pytest.mark.parametrize(
    "phrase",
    [
        EQUATION_TOOL,
        "a4*v'''' + a2*v'' + a1*v' + a0*v = f(x)",  # the sign convention, spelled out
        "DOWNWARD NEGATIVE",
        '"v4": "E*I", "v2": "0", "v1": "0", "v0": "k"',  # the worked example
        "elastic foundation",
        "VARIES along the span",
    ],
)
def test_the_instruction_carries_the_equation_contract(phrase):
    assert phrase in _INSTRUCTION


# --------------------------------------- skipped is not passed (honesty bug)


def not_a_textbook_case():
    """A standard beam the closed-form reference does not cover: a cantilever."""
    model = beam_model()
    model["supports"] = {"N0": [True, True, True, True, True, True]}
    return model


def test_an_inapplicable_closed_form_reports_skipped_not_pass():
    model = not_a_textbook_case()
    gate = deterministic_gate(model, raw_solve_beam_3d(model))
    closed_form = next(c for c in gate["checks"] if c["name"] == "closed_form_reference")

    assert check_status(closed_form) == SKIPPED
    assert closed_form["status"] == SKIPPED
    assert gate["passed"], "a skipped check must not fail the gate either"


def test_a_skipped_check_is_counted_apart_from_the_passes():
    model = not_a_textbook_case()
    gate = deterministic_gate(model, raw_solve_beam_3d(model))
    counts = gate["tally"]

    assert counts["skipped"] == 1
    assert counts["passed"] == counts["total"] - 1
    assert counts["passed"] + counts["skipped"] + counts["failed"] == counts["total"]


def test_the_tally_never_reads_a_skipped_check_as_a_pass():
    checks = [
        {"name": "a", "status": PASS, "passed": True, "detail": ""},
        {"name": "b", "status": SKIPPED, "passed": True, "detail": ""},
        {"name": "c", "status": FAIL, "passed": False, "detail": ""},
        {"name": "old", "passed": True, "detail": ""},  # written before statuses
    ]
    assert tally(checks) == {"passed": 2, "skipped": 1, "failed": 1, "total": 4}


def test_the_live_view_prints_skip_for_a_skipped_check(capsys):
    view = live_view.LiveView(console=live_view.Console(force_terminal=False, width=200))
    view.handle(
        {
            "stage": "gate",
            "type": "gate_check",
            "title": "closed_form_reference: SKIPPED",
            "data": {
                "name": "closed_form_reference",
                "status": SKIPPED,
                "passed": True,
                "detail": "no textbook case",
            },
        }
    )
    out = capsys.readouterr().out
    assert "SKIP" in out
    assert "PASS" not in out


def test_the_html_trace_pills_a_skipped_check_as_skipped(tmp_path):
    events = [
        {
            "seq": 1,
            "t": 0.1,
            "stage": "gate",
            "type": "gate_check",
            "title": "closed_form_reference: SKIPPED",
            "data": {
                "name": "closed_form_reference",
                "status": SKIPPED,
                "passed": True,
                "detail": "no textbook case",
            },
        },
        {
            "seq": 2,
            "t": 0.2,
            "stage": "gate",
            "type": "gate_result",
            "title": "deterministic gate",
            "data": {"passed": True, "n_passed": 3, "n_skipped": 1, "n_total": 4},
        },
    ]
    page = (tmp_path / "trace.html")
    trace_html.write_trace_html(events, page, {"passed": True, "duration_s": 1.0})
    html = page.read_text()

    assert '<span class="pill warn">SKIPPED</span>' in html
    assert "closed_form_reference" in html
    assert "3 of 4 checks passed, 1 skipped" in html


# --------------------------------------------- the gate on a custom equation


def test_the_foundation_gate_runs_mms_and_equilibrium(foundation_gate):
    by_name = {c["name"]: c for c in foundation_gate["checks"]}

    assert foundation_gate["passed"], [
        c for c in foundation_gate["checks"] if check_status(c) == FAIL
    ]
    assert check_status(by_name["equation_mms"]) == PASS
    assert check_status(by_name["equation_equilibrium"]) == PASS
    assert check_status(by_name["equation_stiffness_symmetry"]) == PASS
    assert check_status(by_name["equation_support_conditions"]) == PASS
    # MMS ran on THIS mesh, not some default one
    assert "meshes (8, 16, 32) elements" in by_name["equation_mms"]["detail"]


def test_the_foundation_gate_skips_the_closed_form_rather_than_passing_it(foundation_gate):
    closed_form = next(
        c for c in foundation_gate["checks"] if c["name"] == "closed_form_reference"
    )
    assert check_status(closed_form) == SKIPPED
    assert "manufactured solutions" in closed_form["detail"]
    # A skipped check is counted as skipped and never as passed. Counting the
    # skips rather than naming a number keeps this honest when a check is added.
    skipped = [c["name"] for c in foundation_gate["checks"] if check_status(c) == SKIPPED]
    assert "closed_form_reference" in skipped
    assert foundation_gate["tally"]["skipped"] == len(skipped)
    assert foundation_gate["tally"]["passed"] == len(foundation_gate["checks"]) - len(skipped)


def test_equilibrium_is_the_one_a_naive_reaction_sum_would_get_wrong(foundation_run):
    """The foundation carries most of the load, so sum(R) + applied is far from zero."""
    _model, result = foundation_run
    reactions = sum(r.get("F", 0.0) for r in result["reactions"].values())
    applied = Q * L
    assert abs(reactions - applied) / applied > 0.5  # the naive check would "fail"


def test_a_spec_that_reduces_to_euler_bernoulli_gets_the_closed_form_too():
    model = equation_model()
    result = solve_with_equation(model, EULER_BERNOULLI)
    gate = deterministic_gate(model, result, EULER_BERNOULLI)
    names = {c["name"]: check_status(c) for c in gate["checks"]}

    assert gate["passed"]
    assert names["closed_form_midspan_deflection"] == PASS
    assert names["closed_form_max_moment"] == PASS
    assert names["closed_form_end_shear"] == PASS
    assert gate["tally"]["skipped"] == 0
    recorder.reset()


def test_a_beam_column_skips_only_the_closed_form():
    model = equation_model()
    beam_column = spec("beam_column")
    result = solve_with_equation(model, beam_column)
    gate = deterministic_gate(model, result, beam_column)
    statuses = {c["name"]: check_status(c) for c in gate["checks"]}

    assert gate["passed"]
    assert statuses["equation_mms"] == PASS
    assert statuses["closed_form_reference"] == SKIPPED
    recorder.reset()


def test_stiffness_symmetry_is_skipped_with_a_reason_when_a1_is_nonzero():
    model = equation_model()
    unsymmetric = {
        "label": "beam with a first-derivative term",
        "coeffs": {"v4": "E*I", "v1": "c"},
        "rhs": "q",
        "params": {"E": E, "I": I, "c": 5.0e4, "q": -Q},
    }
    gate = deterministic_gate(
        model, solve_with_equation(model, unsymmetric), unsymmetric
    )
    symmetry = next(
        c for c in gate["checks"] if c["name"] == "equation_stiffness_symmetry"
    )

    assert check_status(symmetry) == SKIPPED
    assert "a1" in symmetry["detail"] and "unsymmetric" in symmetry["detail"]
    recorder.reset()


def test_a_tampered_result_fails_the_equation_gate(foundation_run):
    model, result = foundation_run
    tampered = copy.deepcopy(result)
    for node in tampered["displacements"].values():
        node["v"] *= 1.05
        node["slope"] *= 1.05

    gate = deterministic_gate(model, tampered, FOUNDATION)
    failed = [c["name"] for c in gate["checks"] if check_status(c) == FAIL]

    assert not gate["passed"]
    assert "equation_equilibrium" in failed


def test_a_tampered_euler_bernoulli_result_fails_too():
    model = equation_model()
    result = solve_with_equation(model, EULER_BERNOULLI)
    tampered = copy.deepcopy(result)
    for node in tampered["displacements"].values():
        node["v"] *= 1.05
        node["slope"] *= 1.05
    for key in tampered["max_abs"]:
        tampered["max_abs"][key] *= 1.05

    gate = deterministic_gate(model, tampered, EULER_BERNOULLI)
    assert not gate["passed"]
    recorder.reset()


# ------------------------------------------------------------- the report


@pytest.fixture(scope="module")
def equation_report(tmp_path_factory, foundation_run, foundation_gate):
    model, result = foundation_run
    path = tmp_path_factory.mktemp("report") / "report.md"
    original = run_phase1.REPORT_PATH
    run_phase1.REPORT_PATH = path
    try:
        run_phase1.write_report(
            BRIEF, "orchestrator-model", model, result, foundation_gate, VERDICT,
            True, FOUNDATION,
        )
    finally:
        run_phase1.REPORT_PATH = original
    return path.read_text()


def test_the_report_has_a_governing_equation_section(equation_report):
    assert "## Governing equation" in equation_report
    assert FOUNDATION["label"] in equation_report
    # replaced, not appended: the canned derivation section is gone entirely
    assert "## Galerkin derivation" not in equation_report.splitlines()


def test_the_governing_equation_section_shows_symbols_and_parameters(equation_report):
    section = equation_report.split("## Governing equation", 1)[1].split("## Model", 1)[0]
    assert "E I" in section and "v''''" in section  # the equation in symbols
    assert "downward negative" in section.lower()
    for name, value in FOUNDATION["params"].items():
        assert f"| `{name}` |" in section, name
        assert f"{value:.6e}" in section, name


def test_the_report_carries_the_derivation_for_this_equation(equation_report):
    assert "### Derivation for this equation" in equation_report
    # sympy integrated these for THIS spec; the canned document never mentions k
    assert "foundation" in equation_report.lower()
    assert "Hermite" in equation_report


def test_the_report_names_the_solver_that_produced_its_numbers(equation_report):
    assert f"- Solver: {EQUATION_TOOL}" in equation_report


def test_the_results_table_uses_the_equation_solver_names(equation_report):
    results = equation_report.split("## Results", 1)[1].split("## Deterministic", 1)[0]
    assert "Max deflection (m)" in results
    assert "Max bending moment (N*m)" in results
    assert "Max shear (N)" in results


def test_the_report_renders_a_skipped_check_as_skipped(equation_report):
    gate = equation_report.split("## Deterministic gate", 1)[1]
    row = next(line for line in gate.splitlines() if "closed_form_reference" in line)
    assert "| SKIPPED |" in row
    assert "PASS" not in row
    counted = re.search(r"(\d+) of (\d+) checks passed, (\d+) skipped", gate)
    assert counted, gate
    passed, total, skipped = (int(g) for g in counted.groups())
    assert skipped >= 1  # the closed form above is one of them
    assert passed + skipped == total  # a skip is never folded into the passes


# ------------------------------------ the report still renders as Markdown
#
# Same rules evals/test_report_markdown.py pins for the standard report; an
# equation report embeds a different generated document, so it has to obey
# them on its own.


def unescaped_pipes(line):
    return len(re.findall(r"(?<!\\)\|", line))


def test_the_equation_report_has_exactly_one_title(equation_report):
    titles = [line for line in equation_report.splitlines() if re.match(r"#\s", line)]
    assert titles == ["# Phase 1 Report - The 25-Meter Agent"]


def test_the_equation_reports_math_stays_balanced(equation_report):
    lines = equation_report.splitlines()
    marks = [i for i, line in enumerate(lines) if "$$" in line]
    assert marks and len(marks) % 2 == 0
    assert all(lines[i].strip() == "$$" for i in marks), "a $$ shares its line with text"
    for open_, close in zip(marks[::2], marks[1::2]):
        body = lines[open_ + 1 : close]
        assert len(body) == 1, f"equation at line {open_} spans {len(body)} lines"
        assert not re.match(r"\s*([-+*>]|\d+[.)])\s", body[0]), body[0][:60]


def test_every_table_row_in_the_equation_report_keeps_its_columns(equation_report):
    tables, current = [], []
    for line in equation_report.splitlines():
        if line.startswith("|"):
            current.append(line)
        elif current:
            tables.append(current)
            current = []
    assert tables
    for table in tables:
        header = unescaped_pipes(table[0])
        for row in table:
            assert unescaped_pipes(row) == header, row


def test_the_embedded_derivation_sits_under_its_report_section(equation_report):
    headings = [line for line in equation_report.splitlines() if re.match(r"#{1,6}\s", line)]
    assert any(h.startswith("#### Galerkin derivation:") for h in headings)
    assert any(h.startswith("##### 1. The equation you gave") for h in headings)


def test_the_model_block_is_still_the_first_json_block(equation_report):
    block = equation_report.split("```json\n", 1)[1].split("\n```", 1)[0]
    assert json.loads(block) == json.loads(json.dumps(equation_model()))


# ------------------------------------------- what the README promises him


def readme_text():
    return (REPO_ROOT / "README.md").read_text(encoding="utf-8")


def test_the_readme_documents_the_three_routes():
    text = readme_text()
    assert "## Changing the equation" in text
    assert "equations/README.md" in text  # route 3 points at the full contract
    for route in ("in the brief", "equations/", "your own JSON"):
        assert route in text, route


def test_the_readme_snippet_is_a_real_spec_that_solves():
    """A README example that no longer parses is worse than no example."""
    blocks = re.findall(r"```json\n(.*?)\n```", readme_text(), re.DOTALL)
    assert blocks, "the README lost its worked spec"
    example = json.loads(blocks[0])
    result = solve_with_equation(equation_model(), example)
    assert result["max_abs"]["v"] > 0
    recorder.reset()


def test_the_readme_says_what_is_out_of_scope():
    text = readme_text().lower()
    for topic in ("out of scope", "v3", "non-linear"):
        assert topic in text, topic

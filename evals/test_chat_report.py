"""Offline tests for the chat answer (agent/chat_report.py).

The answer is cut from the report the pipeline wrote, so the tests hold it to
that report: its key results must be copied line for line, and it must say
plainly when a run is not verified. No API calls.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.chat_report import chat_report, equation_line, section  # noqa: E402
from agent.run_phase1 import RunOutcome  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPORT = (FIXTURES / "report_soil_pass.md").read_text()

SOIL = {
    "label": "Beam on elastic foundation",
    "coeffs": {"v4": "E*I", "v2": "0", "v1": "0", "v0": "k"},
    "rhs": "q",
    "params": {"E": 30e9, "I": 0.005, "k": 1.0e7, "q": -30e3},
}


def check(name, status, detail="detail"):
    return {"name": name, "status": status, "passed": status != "fail", "detail": detail}


GATE = {
    "passed": True,
    "checks": [
        check("equation_equilibrium", "pass"),
        check("equation_solution_residual", "pass"),
        check("equation_deflection_sign", "skipped"),
        check("equation_mms", "pass"),
    ],
}
VERDICT = {
    "refuted": False,
    "checks": [
        {
            "name": "Midspan deflection check",
            "passed": True,
            "detail": "FEM value -0.00301473 m vs independent check -0.00301475 m (diff < 0.001%).",
        }
    ],
    "reasoning": "All values agree.",
    "model": "gemini-3-flash-preview",
}


def outcome(**over):
    fields = dict(code=0, passed=True, lines=[], gate=GATE, verdict=VERDICT, equation=SOIL)
    fields.update(over)
    return RunOutcome(**fields)


def test_the_key_results_are_copied_line_for_line():
    answer = chat_report(outcome(), REPORT)
    key = section(REPORT, "Key results")
    assert key.startswith("## Key results")
    for line in key.splitlines():
        assert line in answer
    assert "| Midspan deflection | 3.015 mm downward | x = 12.5 m |" in answer


def test_a_verified_run_says_so_first():
    assert chat_report(outcome(), REPORT).startswith("**Verified.**")


def test_the_checks_are_named_as_an_engineer_would():
    answer = chat_report(outcome(), REPORT)
    assert "| Global vertical equilibrium | Passed |" in answer
    assert "Not applicable" in answer
    assert "3 of 4 passed, 1 not applicable to this beam." in answer
    assert "equation_mms" not in answer


def test_the_equation_is_plain_text_with_its_parameters():
    answer = chat_report(outcome(), REPORT)
    assert "E·I·v'''' + k·v = q" in answer
    assert "E = 3e+10, I = 0.005, k = 1e+07, q = -30000" in answer
    assert "$$" not in answer and "```" not in answer


def test_a_coefficient_that_varies_along_the_span_is_in_divergence_form():
    tapered = {"coeffs": {"v4": "E*I0*(1 + x/L)", "v2": "P"}, "rhs": "q"}
    assert equation_line(tapered) == "(E·I0·(1 + x/L)·v'')'' + P·v'' = q"
    assert equation_line({"coeffs": {"v4": "E*I*exp(-ds*ms/R)"}, "rhs": "q"}) == "E·I·exp(-ds·ms/R)·v'''' = q"


def test_the_standard_beam_has_no_equation_section():
    assert "## Governing equation" not in chat_report(outcome(equation=None), REPORT)


def test_a_failed_check_is_named_with_its_reason():
    gate = {"passed": False, "checks": GATE["checks"][:1] + [check("equation_mms", "fail", "observed order 0.4")]}
    answer = chat_report(outcome(passed=False, code=1, gate=gate), REPORT)
    assert answer.startswith("**Not verified:** 1 engineering check failed.")
    assert "| Convergence study (manufactured solution) | FAILED |" in answer
    assert "- Convergence study (manufactured solution): observed order 0.4" in answer


def test_a_refuted_review_is_reported_with_its_reasoning():
    verdict = {**VERDICT, "refuted": True, "reasoning": "Moment is 20 % high."}
    answer = chat_report(outcome(passed=False, code=1, verdict=verdict), REPORT)
    assert "the independent review found a problem" in answer
    assert "Its reasoning: Moment is 20 % high." in answer


def test_a_run_with_no_analysis_gives_the_reason():
    answer = chat_report(
        RunOutcome(code=2, passed=False, lines=[], narrative="CANNOT SOLVE: the equation has a v''' term.")
    )
    assert answer.startswith("**No analysis was run.**")
    assert "v''' term" in answer


def test_it_always_offers_the_full_report():
    assert chat_report(outcome(), REPORT).rstrip().endswith("the detail of every check.")

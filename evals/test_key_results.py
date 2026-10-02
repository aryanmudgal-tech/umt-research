"""Offline tests for the report's Key results: the answers the brief asked for.

The results table carried peak values only, so a brief asking for the midspan
deflection and the support shear forces got neither by name: the midspan value
was in the verifier's prose and the reactions only as a total inside a gate
line. key_results() pulls them out of the solver's own output, for both
solvers, with where each peak occurs. No API calls.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent import run_phase1
from agent.key_results import key_results, key_results_markdown
from tools.fem.equation import deflection_at, solve_equation_beam
from tools.fem.solver import solve_beam_3d

L, E, G, I, Q = 25.0, 30e9, 12.5e9, 0.005, 30e3
PINNED = [True, True, True, True, False, False]
ROLLER = [False, True, True, False, False, False]
WINKLER = {"coeffs": {"v4": "E*I", "v0": "k"}, "rhs": "q", "params": {"E": E, "I": I, "k": 1.0e7, "q": -Q}}
PRISMATIC = {"coeffs": {"v4": "E*I"}, "rhs": "q", "params": {"E": E, "I": I, "q": -Q}}


def equation_model(n):
    return {
        "nodes": [{"id": f"N{k}", "x": L * k / n} for k in range(n + 1)],
        "supports": {"N0": PINNED, f"N{n}": ROLLER},
        "point_loads": [],
    }


def beam_3d(n):
    return {
        "nodes": [{"id": f"N{k}", "x": L * k / n} for k in range(n + 1)],
        "material": {"E": E, "G": G},
        "section": {"A": 0.5, "Iy": 0.004, "Iz": I, "J": 0.001},
        "supports": {"N0": PINNED, f"N{n}": ROLLER},
        "distributed_loads": [{"element": "all", "direction": "y", "w1": -Q, "w2": -Q}],
        "point_loads": [],
    }


# ------------------------------------------------------------ equation path


def test_the_soil_beam_answers_what_its_brief_asked():
    model = equation_model(20)
    k = key_results(model, solve_equation_beam(model, WINKLER), WINKLER)

    assert k["midspan"]["x"] == pytest.approx(12.5)
    assert k["midspan"]["deflection"] == pytest.approx(-0.00301475, rel=1e-4)
    assert k["max_deflection"]["value"] == pytest.approx(0.00319736, rel=1e-4)
    assert k["max_deflection"]["x"] == pytest.approx(6.52, abs=0.05)
    assert k["max_moment"]["value"] == pytest.approx(37491.5, rel=1e-3)
    assert k["max_moment"]["x"] == pytest.approx(2.19, abs=0.05)
    assert [(s["node"], s["kind"]) for s in k["supports"]] == [("N0", "pin"), ("N20", "roller")]
    for s in k["supports"]:
        assert s["vertical"] == pytest.approx(41761.3, rel=1e-3)
        assert s["moment"] is None


def test_midspan_is_found_even_with_no_node_there():
    model = equation_model(7)  # 25/7: no node at 12.5 m
    result = solve_equation_beam(model, PRISMATIC)
    exact = -5 * Q * L**4 / (384 * E * I)
    assert deflection_at(model, result["displacements"], 12.5) == pytest.approx(exact, rel=1e-3)
    assert key_results(model, result, PRISMATIC)["midspan"]["deflection"] == pytest.approx(exact, rel=1e-3)


def test_deflection_at_a_node_is_that_nodes_value_exactly():
    model = equation_model(8)
    result = solve_equation_beam(model, WINKLER)
    for node in model["nodes"]:
        assert deflection_at(model, result["displacements"], node["x"]) == result["displacements"][node["id"]]["v"]


# ----------------------------------------------------------- standard path


def test_the_professors_beam_answers_what_its_brief_asked():
    model = beam_3d(4)
    k = key_results(model, solve_beam_3d(model))

    assert k["midspan"]["deflection"] == pytest.approx(-5 * Q * L**4 / (384 * E * I), rel=1e-9)
    assert k["max_moment"]["value"] == pytest.approx(Q * L**2 / 8, rel=1e-9)
    assert k["max_moment"]["x"] == pytest.approx(12.5)
    assert [s["vertical"] for s in k["supports"]] == pytest.approx([Q * L / 2] * 2)


def test_a_fixed_end_reports_its_moment():
    model = beam_3d(4)
    model["supports"] = {"N0": [True] * 6}
    model["distributed_loads"][0]["w1"] = model["distributed_loads"][0]["w2"] = -Q
    k = key_results(model, solve_beam_3d(model))

    (support,) = k["supports"]
    assert support["kind"] == "fixed"
    assert support["vertical"] == pytest.approx(Q * L)
    assert abs(support["moment"]) == pytest.approx(Q * L**2 / 2, rel=1e-9)


# ------------------------------------------------------------- the markdown


def test_the_section_reads_in_engineering_units_and_plain_directions():
    model = equation_model(20)
    md = key_results_markdown(key_results(model, solve_equation_beam(model, WINKLER), WINKLER))

    assert md.startswith("## Key results")
    assert "| Midspan deflection | 3.015 mm downward | x = 12.5 m |" in md
    assert "| Maximum bending moment | 37.5 kN·m |" in md
    assert "| Reaction at N0 (pin, x = 0 m) | 41.77 kN upward |" in md


def test_the_report_opens_with_the_key_results(tmp_path, monkeypatch):
    model = equation_model(20)
    result = solve_equation_beam(model, WINKLER)
    det = {"passed": True, "checks": [], "tally": {"passed": 0, "failed": 0, "skipped": 0, "total": 0}}
    verdict = {"refuted": False, "checks": [], "reasoning": "", "model": "v"}
    path = run_phase1.write_report("# Brief\n\nSoil.", "m", model, result, det, verdict, True, WINKLER, path=tmp_path / "r.md")

    text = path.read_text()
    assert text.index("## Key results") < text.index("## Brief") < text.index("## Governing equation")

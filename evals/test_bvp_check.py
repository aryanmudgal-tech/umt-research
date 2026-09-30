"""Offline tests for the independent equation check (tools/fem/bvp_check.py).

The check solves the strong form by collocation, a different method from the
Galerkin FEM, so these tests hold it to exact answers where they exist and to
the FEM where they do not. No API calls.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.fem.bvp_check import solve_equation_bvp
from tools.fem.equation import solve_equation_beam

L, E, I, Q = 25.0, 30e9, 0.005, 30e3
PINNED = [True, True, True, True, False, False]
ROLLER = [False, True, True, False, False, False]
FIXED = [True] * 6

WINKLER = {
    "coeffs": {"v4": "E*I", "v0": "k"},
    "rhs": "q",
    "params": {"E": E, "I": I, "k": 1.0e7, "q": -Q},
}
TAPERED = {
    "coeffs": {"v4": "E*I0*(1 + x/L)"},
    "rhs": "q",
    "params": {"E": E, "I0": 0.0025, "L": L, "q": -Q},
}
BEAM_COLUMN = {
    "coeffs": {"v4": "E*I", "v2": "P"},
    "rhs": "q",
    "params": {"E": E, "I": I, "P": 2.0e6, "q": -Q},
}
PLAIN = {"coeffs": {"v4": "E*I"}, "rhs": "0", "params": {"E": E, "I": I}}


def beam(n, left=PINNED, right=ROLLER, point_loads=()):
    return {
        "nodes": [{"id": f"N{k}", "x": L * k / n} for k in range(n + 1)],
        "supports": {k: v for k, v in (("N0", left), (f"N{n}", right)) if v is not None},
        "point_loads": list(point_loads),
    }


# ---------------------------------------------- against the exact solution

# EI v'''' + k v = q, simply supported, solved in closed form to 40 digits
EXACT_WINKLER = {
    "mid_v": -0.00301474950651241,
    "max_v": 0.00319735917808695,
    "max_v_x": 6.5233065,
    "max_M": 37491.5304186555,
    "max_M_x": 2.1875574,
    "R": 41761.3301574325,
}


def test_winkler_beam_matches_the_exact_solution():
    r = solve_equation_bvp(beam(20), WINKLER)

    assert r["applicable"] and r["converged"]
    assert r["midspan"]["v"] == pytest.approx(EXACT_WINKLER["mid_v"], rel=1e-6)
    assert r["max_abs"]["v"]["value"] == pytest.approx(EXACT_WINKLER["max_v"], rel=1e-6)
    assert r["max_abs"]["v"]["x"] == pytest.approx(EXACT_WINKLER["max_v_x"], abs=0.02)
    assert r["max_abs"]["moment"]["value"] == pytest.approx(EXACT_WINKLER["max_M"], rel=1e-5)
    assert r["max_abs"]["moment"]["x"] == pytest.approx(EXACT_WINKLER["max_M_x"], abs=0.02)
    # the largest shear is at a support, and equals that support's reaction
    assert r["max_abs"]["shear"]["value"] == pytest.approx(EXACT_WINKLER["R"], rel=1e-6)
    for node in ("N0", "N20"):
        assert r["reactions"][node]["F"] == pytest.approx(EXACT_WINKLER["R"], rel=1e-6)


def test_cantilever_tip_load_matches_the_textbook():
    P = -10e3  # downward, at the free tip
    model = beam(10, left=FIXED, right=None, point_loads=[{"node": "N10", "dof": "FY", "value": P}])
    r = solve_equation_bvp(model, PLAIN)

    assert r["converged"]
    assert r["deflection_at_nodes"]["N10"] == pytest.approx(P * L**3 / (3 * E * I), rel=1e-8)
    assert r["reactions"]["N0"]["F"] == pytest.approx(-P, rel=1e-8)
    assert abs(r["reactions"]["N0"]["M"]) == pytest.approx(abs(P) * L, rel=1e-8)
    assert "N10" not in r["reactions"]


# ------------------------------------------------ against the Galerkin FEM


@pytest.mark.parametrize(
    "spec, left, right, loads",
    [
        (TAPERED, PINNED, ROLLER, []),
        (BEAM_COLUMN, PINNED, ROLLER, []),
        (WINKLER, FIXED, ROLLER, []),
        (PLAIN, FIXED, None, [{"node": "N80", "dof": "MZ", "value": 5e4}]),
        (PLAIN, FIXED, None, [{"node": "N80", "dof": "FY", "value": -8e3}]),
    ],
    ids=["tapered", "beam-column", "winkler-propped", "tip-moment", "tip-force"],
)
def test_agrees_with_the_fem_node_by_node_including_reaction_signs(spec, left, right, loads):
    # 80 elements: with a foundation or a taper the FEM is not exact at the nodes,
    # and at 40 its own O(h^4) error (2e-6) is larger than this tolerance
    model = beam(80, left=left, right=right, point_loads=loads)
    fem = solve_equation_beam(model, spec)
    bvp = solve_equation_bvp(model, spec)

    assert bvp["converged"]
    peak = max(abs(d["v"]) for d in fem["displacements"].values())
    for node, d in fem["displacements"].items():
        assert bvp["deflection_at_nodes"][node] == pytest.approx(d["v"], abs=1e-6 * peak)
    assert set(bvp["reactions"]) == set(fem["reactions"])
    for node, forces in fem["reactions"].items():
        for key, value in forces.items():
            scale = max(abs(x) for f in fem["reactions"].values() for x in f.values())
            assert bvp["reactions"][node][key] == pytest.approx(value, abs=1e-5 * scale)


# ------------------------------------------------------------ out of scope


def test_an_interior_support_is_reported_not_applicable():
    model = beam(10)
    model["supports"]["N5"] = ROLLER
    r = solve_equation_bvp(model, WINKLER)
    assert r["applicable"] is False
    assert "N5" in r["reason"]


def test_an_interior_point_load_is_reported_not_applicable():
    model = beam(10, point_loads=[{"node": "N5", "dof": "FY", "value": -1e4}])
    r = solve_equation_bvp(model, WINKLER)
    assert r["applicable"] is False
    assert "N5" in r["reason"]


def test_an_unsupported_beam_does_not_raise():
    # free at both ends with no foundation: no static solution exists
    r = solve_equation_bvp(beam(10, left=None, right=None), {**PLAIN, "rhs": "-1e3"})
    assert r["applicable"] is True
    assert r["converged"] is False

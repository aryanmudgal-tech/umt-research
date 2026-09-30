"""Moment and shear on the equation path must be the beam's, not the cubic's.

They used to be read off each element's own cubic: moment = a4 v'' and shear =
(a4 v'')'. On a Hermite cubic that makes shear a constant per element and
moment a straight line, so on the 25 m beam on soil, at the 20 elements the
orchestrator chose, the reported maximum shear was 25.5 kN against a support
reaction of 41.8 kN in the same report, the moment at the pinned ends was
3.2 kN*m instead of zero, both jumped at every node, and the maximum moment
was 2.7 % high (20 % at 10 elements).

They are now recovered by equilibrium: each element's end forces are
k_e d - f_e, and moment and shear are carried across the element by the
equation itself. What is pinned here:

- the soil beam's peaks against its exact solution, at the orchestrator's
  meshes;
- the identities the old method broke: support shear equals the reaction, a
  pin carries no moment, and nothing jumps at a node that carries no load;
- the jumps that SHOULD be there, under a point force and a point moment;
- the professor's own beam, where the recovery is exact on any mesh;
- a taper and a beam-column against the independent collocation solve.

Offline and deterministic: no LLM, no network.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.fem.bvp_check import solve_equation_bvp  # noqa: E402
from tools.fem.equation import solve_equation_beam  # noqa: E402

L, E, I, Q = 25.0, 30e9, 0.005, -30e3
PINNED = [True, True, True, True, False, False]
ROLLER = [False, True, True, False, False, False]

WINKLER = {
    "coeffs": {"v4": "E*I", "v0": "k"},
    "rhs": "q",
    "params": {"E": E, "I": I, "k": 1.0e7, "q": Q},
}
PRISMATIC = {"coeffs": {"v4": "E*I"}, "rhs": "q", "params": {"E": E, "I": I, "q": Q}}
TAPERED = {
    "coeffs": {"v4": "E*I0*(1 + x/L)"},
    "rhs": "q",
    "params": {"E": E, "I0": 0.0025, "L": L, "q": Q},
}
SQRT_TAPER = {
    "coeffs": {"v4": "E*I0*sqrt(1 + x/L)"},
    "rhs": "q",
    "params": {"E": E, "I0": I, "L": L, "q": Q},
}
BEAM_COLUMN = {
    "coeffs": {"v4": "E*I", "v2": "P"},
    "rhs": "q",
    "params": {"E": E, "I": I, "P": 2.0e6, "q": Q},
}

# EI v'''' + k v = q, simply supported, in closed form to 40 digits
EXACT_WINKLER_MAX_M = 37491.5304186555
EXACT_WINKLER_R = 41761.3301574325


def beam(n, point_loads=()):
    return {
        "nodes": [{"id": f"N{k}", "x": L * k / n} for k in range(n + 1)],
        "supports": {"N0": PINNED, f"N{n}": ROLLER},
        "point_loads": list(point_loads),
    }


def at(samples, x, side):
    """The sample at x from the element on that side of it ("left" or "right")."""
    hits = [p for p in samples if abs(p["x"] - x) < 1e-9]
    assert len(hits) == 2, f"expected x = {x} to end one element and start the next"
    return hits[0] if side == "left" else hits[1]


# ------------------------------------------------ the soil beam, at real meshes


@pytest.mark.parametrize("n, tol", [(10, 1e-2), (20, 1e-3), (40, 1e-4)])
def test_winkler_peak_moment_matches_the_exact_solution(n, tol):
    # the old reading was 20 % high at 10 elements and 2.7 % high at 20
    peak = solve_equation_beam(beam(n), WINKLER)["max_abs"]["moment"]
    assert peak == pytest.approx(EXACT_WINKLER_MAX_M, rel=tol)


@pytest.mark.parametrize("n", [10, 20])
def test_winkler_peak_shear_is_the_support_shear(n):
    # the old reading was 67 % low at 10 elements and 39 % low at 20
    r = solve_equation_beam(beam(n), WINKLER)
    assert r["max_abs"]["shear"] == pytest.approx(r["reactions"]["N0"]["F"], rel=1e-12)
    assert r["max_abs"]["shear"] == pytest.approx(EXACT_WINKLER_R, rel=5e-3)


# ------------------------------------------ the identities the old reading broke


@pytest.mark.parametrize(
    "spec", [WINKLER, TAPERED, SQRT_TAPER, BEAM_COLUMN], ids=["winkler", "taper", "sqrt-taper", "beam-column"]
)
def test_ends_carry_the_reaction_and_no_moment(spec):
    n = 12
    r = solve_equation_beam(beam(n), spec)
    first, last = r["samples"][0], r["samples"][-1]
    P = spec["params"].get("P", 0.0)
    scale = max(abs(p["moment"]) for p in r["samples"])

    # the vertical force at a support is shear + P v', and it is the reaction
    assert first["shear"] + P * first["slope"] == pytest.approx(r["reactions"]["N0"]["F"], rel=1e-9)
    assert -(last["shear"] + P * last["slope"]) == pytest.approx(r["reactions"][f"N{n}"]["F"], rel=1e-9)
    # a pin or a roller cannot hold a moment
    assert abs(first["moment"]) < 1e-9 * scale
    assert abs(last["moment"]) < 1e-9 * scale


@pytest.mark.parametrize("spec", [WINKLER, TAPERED, BEAM_COLUMN], ids=["winkler", "taper", "beam-column"])
def test_nothing_jumps_at_a_node_that_carries_no_load(spec):
    n = 8
    r = solve_equation_beam(beam(n), spec)
    m_scale = max(abs(p["moment"]) for p in r["samples"])
    v_scale = max(abs(p["shear"]) for p in r["samples"])
    for k in range(1, n):
        left, right = at(r["samples"], L * k / n, "left"), at(r["samples"], L * k / n, "right")
        assert right["moment"] == pytest.approx(left["moment"], abs=1e-9 * m_scale)
        assert right["shear"] == pytest.approx(left["shear"], abs=1e-9 * v_scale)


def test_a_point_force_steps_the_shear_and_a_point_moment_steps_the_moment():
    P, M = -50e3, 80e3
    r = solve_equation_beam(
        beam(8, [{"node": "N2", "dof": "FY", "value": P}, {"node": "N6", "dof": "MZ", "value": M}]),
        WINKLER,
    )
    s = r["samples"]
    assert at(s, L * 2 / 8, "right")["shear"] - at(s, L * 2 / 8, "left")["shear"] == pytest.approx(P, rel=1e-9)
    assert at(s, L * 6 / 8, "right")["moment"] - at(s, L * 6 / 8, "left")["moment"] == pytest.approx(-M, rel=1e-9)
    # and each leaves the other field alone
    m_scale = max(abs(p["moment"]) for p in s)
    assert at(s, L * 2 / 8, "right")["moment"] == pytest.approx(at(s, L * 2 / 8, "left")["moment"], abs=1e-9 * m_scale)


# ------------------------------------------------------ the professor's own beam


@pytest.mark.parametrize("n", [1, 3, 4])
def test_the_professors_beam_is_exact_on_any_mesh(n):
    """Uniform load, no foundation: shear is linear and moment a parabola, exactly."""
    r = solve_equation_beam(beam(n), PRISMATIC)
    q = abs(Q)
    for p in r["samples"]:
        x = p["x"]
        assert abs(p["moment"]) == pytest.approx(q * x * (L - x) / 2, abs=1e-9 * q * L**2 / 8)
        assert abs(p["shear"]) == pytest.approx(abs(q * (L / 2 - x)), abs=1e-9 * q * L / 2)
    assert r["max_abs"]["moment"] == pytest.approx(q * L**2 / 8, rel=1e-9)
    assert r["max_abs"]["shear"] == pytest.approx(q * L / 2, rel=1e-12)


def test_no_sample_exceeds_the_reported_maximum():
    r = solve_equation_beam(beam(7), WINKLER)
    for key in ("v", "slope", "moment", "shear"):
        assert max(abs(p[key]) for p in r["samples"]) <= r["max_abs"][key]


# ------------------------------------------- against the independent solve


@pytest.mark.parametrize(
    "spec", [TAPERED, SQRT_TAPER, BEAM_COLUMN], ids=["taper", "sqrt-taper", "beam-column"]
)
def test_peaks_agree_with_the_independent_collocation_solve(spec):
    model = beam(20)
    fem = solve_equation_beam(model, spec)["max_abs"]
    ref = solve_equation_bvp(model, spec)["max_abs"]
    assert fem["moment"] == pytest.approx(ref["moment"]["value"], rel=1e-4)
    assert fem["shear"] == pytest.approx(ref["shear"]["value"], rel=1e-4)

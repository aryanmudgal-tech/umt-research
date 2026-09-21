"""Physics of the equation-driven solver: signs, stability, and loud failure.

These are regression tests for defects found by attacking the solver rather
than by reading it. Each one produced a plausible number before the fix:

- A beam-column at or past the Euler load, a negative v4, and a v4 that goes
  negative over part of the span all returned a smooth deflection pointing the
  WRONG WAY, and the gate passed it. The free stiffness matrix is indefinite in
  every one of those cases, so the "solution" is an unstable equilibrium.
- The gate divided a FEM magnitude by a SIGNED closed-form reference, so a
  negative EI made every relative error negative and every closed-form check
  pass unconditionally.
- A non-finite parameter produced a result full of nan that still looked like a
  result.

Offline and deterministic: no LLM, no network.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.gates import (  # noqa: E402
    FAIL,
    PASS,
    SKIPPED,
    check_status,
    deterministic_gate,
)
from tools.fem.equation import EquationError, solve_equation_beam  # noqa: E402

E_VAL, I_VAL, Q_VAL, SPAN = 30e9, 0.005, 30e3, 25.0
EI = E_VAL * I_VAL
P_EULER = np.pi**2 * EI / SPAN**2

PINNED = [True, True, True, True, False, False]
ROLLER = [False, True, True, False, False, False]
FIXED = [True] * 6


def beam(n_elements=8, span=SPAN, supports="ss", point_loads=None, x0=0.0):
    xs = np.linspace(x0, x0 + span, n_elements + 1)
    nodes = [{"id": f"N{k}", "x": float(x)} for k, x in enumerate(xs)]
    if supports == "ss":
        held = {"N0": PINNED, f"N{n_elements}": ROLLER}
    elif supports == "cantilever":
        held = {"N0": FIXED}
    else:
        held = supports
    return {"nodes": nodes, "supports": held, "point_loads": point_loads or []}


def euler_bernoulli(q=-Q_VAL):
    return {"coeffs": {"v4": "E*I"}, "rhs": "q", "params": {"E": E_VAL, "I": I_VAL, "q": q}}


def beam_column(P):
    return {
        "coeffs": {"v4": "E*I", "v2": "P"},
        "rhs": "q",
        "params": {"E": E_VAL, "I": I_VAL, "P": P, "q": -Q_VAL},
    }


# ------------------------------------------------------------ sign convention


def test_a_downward_load_deflects_the_beam_downward():
    result = solve_equation_beam(beam(8), euler_bernoulli())
    values = [p["v"] for p in result["samples"]]
    assert max(values) <= 0.0
    assert min(values) == pytest.approx(-5 * Q_VAL * SPAN**4 / (384 * EI), rel=1e-9)


def test_an_upward_load_mirrors_the_downward_one():
    down = solve_equation_beam(beam(8), euler_bernoulli(q=-Q_VAL))
    up = solve_equation_beam(beam(8), euler_bernoulli(q=+Q_VAL))
    for node, d in down["displacements"].items():
        assert up["displacements"][node]["v"] == pytest.approx(-d["v"], rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("n", [4, 8, 40])
def test_moment_and_shear_recovery_match_the_textbook(n):
    """M = qL^2/8 at midspan, 0 at the supports, and V changes sign at midspan.

    Moment converges as O(h^2) and shear as O(h) because the element cubic has a
    constant third derivative, so the tolerances follow the mesh rather than
    pretending to machine precision.
    """
    result = solve_equation_beam(beam(n), euler_bernoulli())
    samples = sorted(result["samples"], key=lambda p: p["x"])
    midspan = min(samples, key=lambda p: abs(p["x"] - SPAN / 2))

    peak_moment = Q_VAL * SPAN**2 / 8
    assert midspan["moment"] == pytest.approx(peak_moment, rel=4.0 / n**2)
    # sagging positive: a downward load bends the beam into a smile
    assert midspan["moment"] > 0

    for end in (samples[0], samples[-1]):
        assert abs(end["moment"]) < peak_moment * 4.0 / n**2

    end_shear = Q_VAL * SPAN / 2
    assert samples[0]["shear"] == pytest.approx(end_shear, rel=2.0 / n)
    assert samples[-1]["shear"] == pytest.approx(-end_shear, rel=2.0 / n)

    # shear is constant per element, so it steps through zero at midspan
    left = [p["shear"] for p in samples if p["x"] < SPAN / 2 - 1e-9]
    right = [p["shear"] for p in samples if p["x"] > SPAN / 2 + 1e-9]
    assert min(left) > 0 and max(right) < 0


# ------------------------------------------------------------- the beam-column


@pytest.mark.parametrize("P", [-2.0e6, -1.0e5, 1.0e5, 1.0e6])
def test_axial_force_softens_in_compression_and_stiffens_in_tension(P):
    """A positive a2 is compression and must make the beam deflect MORE.

    The amplification is the textbook 1/(1 - P/P_euler), which pins the size as
    well as the direction: a sign error in the -a2 N'^T N' term would move the
    deflection the wrong way, and a factor error would miss the amplifier.
    """
    base = min(p["v"] for p in solve_equation_beam(beam(20), beam_column(0.0))["samples"])
    got = min(p["v"] for p in solve_equation_beam(beam(20), beam_column(P))["samples"])

    if P > 0:
        assert got < base  # compression: further down
    else:
        assert got > base  # tension: pulled straighter

    assert got == pytest.approx(base / (1 - P / P_EULER), rel=5e-3)


def test_a_compressive_load_past_euler_buckling_is_refused_not_answered():
    """Past P_cr the linear solution deflects UPWARD under a downward load.

    It used to be returned, and the whole gate passed it. There is no stable
    equilibrium to report, so the solver must refuse.
    """
    with pytest.raises(EquationError, match="no stable equilibrium"):
        solve_equation_beam(beam(20), beam_column(1.5 * P_EULER))


def test_just_below_euler_buckling_still_solves():
    """The refusal has to land at the buckling load, not before it."""
    result = solve_equation_beam(beam(20), beam_column(0.9 * P_EULER))
    assert max(p["v"] for p in result["samples"]) <= 0.0


# --------------------------------------------------------- elastic foundation


def test_a_long_beam_on_soil_matches_the_classical_infinite_beam():
    """Hetenyi: v(0) = -P*beta/(2k) with beta = (k/4EI)^(1/4) under a point load.

    The span is 12/beta, so the end effects have decayed by exp(-6); the mesh
    resolves the decay length 1/beta about 13 times over.
    """
    k, load, n = 1.0e7, 1.0e5, 400
    beta = (k / (4 * EI)) ** 0.25
    span = 12.0 / beta

    xs = np.linspace(-span / 2, span / 2, n + 1)
    middle = f"N{n // 2}"
    model = {
        "nodes": [{"id": f"N{i}", "x": float(x)} for i, x in enumerate(xs)],
        "supports": {},  # the soil alone holds the beam up
        "point_loads": [{"node": middle, "dof": "FY", "value": -load}],
    }
    spec = {"coeffs": {"v4": "E*I", "v0": "k"}, "rhs": "0",
            "params": {"E": E_VAL, "I": I_VAL, "k": k}}
    result = solve_equation_beam(model, spec)

    under_load = -load * beta / (2 * k)
    assert result["displacements"][middle]["v"] == pytest.approx(under_load, rel=1e-3)

    # and the decay, one eighth of a wavelength away from the load
    x_probe = np.pi / (4 * beta)
    node = min(model["nodes"], key=lambda nd: abs(nd["x"] - x_probe))
    b = beta * node["x"]
    classical = under_load * np.exp(-b) * (np.cos(b) + np.sin(b))
    assert result["displacements"][node["id"]]["v"] == pytest.approx(classical, rel=1e-3)


# -------------------------------------------------------- variable coefficients


def tapered(reversed_=False):
    coefficient = "E*I0*(2 - x/L)" if reversed_ else "E*I0*(1 + x/L)"
    return {
        "coeffs": {"v4": coefficient},
        "rhs": "q",
        "params": {"E": E_VAL, "I0": 0.0025, "L": SPAN, "q": -Q_VAL},
    }


def test_a_tapered_beam_converges_to_the_same_answer_on_4_and_40_elements():
    coarse = solve_equation_beam(beam(4), tapered())
    fine = solve_equation_beam(beam(40), tapered())
    # x = 12.5 is a node of both meshes
    assert coarse["displacements"]["N2"]["v"] == pytest.approx(
        fine["displacements"]["N20"]["v"], rel=2e-3
    )
    assert fine["displacements"]["N20"]["v"] < 0


def test_reversing_the_taper_mirrors_the_deflection_profile():
    """The same beam turned end for end must deflect the same way, mirrored.

    This is what fails when the x-dependent integration uses the local
    coordinate instead of the element's true global interval.
    """
    forward = solve_equation_beam(beam(40), tapered())
    backward = solve_equation_beam(beam(40), tapered(reversed_=True))
    peak = max(abs(d["v"]) for d in forward["displacements"].values())
    for k in range(41):
        assert forward["displacements"][f"N{k}"]["v"] == pytest.approx(
            backward["displacements"][f"N{40 - k}"]["v"], abs=1e-9 * peak
        )


def test_an_x_dependent_coefficient_is_integrated_over_the_true_global_interval():
    """Shift the same beam to x in [100, 125] and it must give the same answer."""
    here = solve_equation_beam(beam(40), tapered())
    shifted_spec = {
        "coeffs": {"v4": "E*I0*(1 + (x - 100)/L)"},
        "rhs": "q",
        "params": {"E": E_VAL, "I0": 0.0025, "L": SPAN, "q": -Q_VAL},
    }
    there = solve_equation_beam(beam(40, x0=100.0), shifted_spec)
    peak = max(abs(d["v"]) for d in here["displacements"].values())
    for k in range(41):
        assert there["displacements"][f"N{k}"]["v"] == pytest.approx(
            here["displacements"][f"N{k}"]["v"], abs=1e-9 * peak
        )


# ------------------------------------------------------------- loud failure


@pytest.mark.parametrize(
    "label, supports",
    [
        ("no supports at all", {}),
        ("one pin only", {"N0": PINNED}),
        ("one roller only", {"N0": ROLLER}),
        ("slope held at both ends but no deflection", {
            "N0": [False] * 5 + [True], "N4": [False] * 5 + [True]}),
    ],
)
def test_a_support_pattern_that_leaves_a_mechanism_fails_loudly(label, supports):
    with pytest.raises(EquationError, match="no stable equilibrium"):
        solve_equation_beam(beam(4, supports=supports), euler_bernoulli())


@pytest.mark.parametrize(
    "label, v4",
    [
        ("negative everywhere", "-E*I"),
        ("negative over the right half", "E*I*(1 - 2*x/L)"),
        ("zero over the right half", "E*I*Max(0, 1 - 2*x/L)"),
    ],
)
def test_a_v4_that_is_not_positive_along_the_whole_span_fails_loudly(label, v4):
    """A bending stiffness that is zero or negative anywhere is not a beam.

    Each of these used to return a deflection, and the first of them scored a
    clean 7 of 7 on the gate while the beam rose under a downward load.
    """
    spec = {"coeffs": {"v4": v4}, "rhs": "q",
            "params": {"E": E_VAL, "I": I_VAL, "L": SPAN, "q": -Q_VAL}}
    with pytest.raises(EquationError, match="no stable equilibrium"):
        solve_equation_beam(beam(8), spec)


def test_a_taper_that_only_touches_zero_at_the_very_end_still_solves():
    """The refusal must be about the span, not about a single point of it."""
    spec = {"coeffs": {"v4": "E*I*(1 - x/L)"}, "rhs": "q",
            "params": {"E": E_VAL, "I": I_VAL, "L": SPAN, "q": -Q_VAL}}
    result = solve_equation_beam(beam(8), spec)
    assert max(p["v"] for p in result["samples"]) <= 0.0


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_a_non_finite_parameter_is_refused_rather_than_returned_as_nan(value):
    spec = {"coeffs": {"v4": "E*I"}, "rhs": "q",
            "params": {"E": value, "I": I_VAL, "q": -Q_VAL}}
    with pytest.raises(EquationError, match="finite"):
        solve_equation_beam(beam(4), spec)


def test_a_model_with_one_node_says_so():
    model = {"nodes": [{"id": "A", "x": 0.0}], "supports": {"A": FIXED}, "point_loads": []}
    with pytest.raises(EquationError, match="at least two"):
        solve_equation_beam(model, euler_bernoulli())


@pytest.mark.parametrize("scale", [1e-12, 1e12])
def test_extreme_parameter_scales_still_give_the_textbook_answer(scale):
    spec = {"coeffs": {"v4": "E*I"}, "rhs": "q",
            "params": {"E": E_VAL * scale, "I": I_VAL, "q": -Q_VAL}}
    result = solve_equation_beam(beam(8), spec)
    exact = -5 * Q_VAL * SPAN**4 / (384 * EI * scale)
    assert min(p["v"] for p in result["samples"]) == pytest.approx(exact, rel=1e-9)


# ------------------------------------------------------------------- the gate


def test_the_gate_refuses_a_closed_form_reference_that_is_not_a_positive_magnitude():
    """A signed reference in the denominator made every closed-form check pass.

    closed_form() returns positive magnitudes, and the check divided by that
    value. A negative EI made it negative, the relative error came out negative,
    and "-2.0 < tolerance" passed a beam deflecting the wrong way.
    """
    from agent.gates import _reference_check

    good = _reference_check("closed_form_midspan_deflection", 1.0, 1.0, 1e-6)
    assert check_status(good) == PASS

    for bad_reference in (-1.0, 0.0, float("nan")):
        check = _reference_check("closed_form_midspan_deflection", 1.0, bad_reference, 1e-6)
        assert check_status(check) == FAIL
        assert "not a positive magnitude" in check["detail"]


def _sign_check(model, spec):
    gate = deterministic_gate(model, solve_equation_beam(model, spec), spec)
    return next(c for c in gate["checks"] if c["name"] == "equation_deflection_sign")


@pytest.mark.parametrize("q", [-Q_VAL, +Q_VAL])
def test_the_gate_checks_that_the_beam_deflected_the_way_its_load_pushes(q):
    """The convention, stated where the professor reads it.

    It cannot know what he meant by the sign of q — only whether the answer is
    consistent with the equation he wrote. Both directions are checked so the
    rule is symmetric rather than a one-sided assertion that happens to hold.
    """
    check = _sign_check(beam(8), euler_bernoulli(q=q))
    assert check_status(check) == PASS
    assert ("v <= 0" if q < 0 else "v >= 0") in check["detail"]


def test_the_sign_check_fails_a_result_that_moves_against_its_load():
    """Feed the check a deflection of the wrong sign and it must bite."""
    model = beam(8)
    spec = euler_bernoulli(q=-Q_VAL)
    result = solve_equation_beam(model, spec)
    flipped = {
        **result,
        "samples": [{**p, "v": -p["v"]} for p in result["samples"]],
    }
    gate = deterministic_gate(model, flipped, spec)
    check = next(c for c in gate["checks"] if c["name"] == "equation_deflection_sign")
    assert check_status(check) == FAIL
    assert not gate["passed"]


@pytest.mark.parametrize(
    "label, coeffs, params",
    [
        ("foundation", {"v4": "E*I", "v0": "k"}, {"k": 1.0e7}),
        ("axial force", {"v4": "E*I", "v2": "P"}, {"P": 1.0e6}),
        ("first-derivative term", {"v4": "E*I", "v1": "c"}, {"c": 5.0e4}),
    ],
)
def test_the_sign_check_skips_where_lifting_is_physical(label, coeffs, params):
    """Outside pure bending, a beam may legitimately lift away from its own load.

    A Winkler foundation does it either side of a point load. Claiming a pass
    here would be a statement about physics the check cannot make, so it skips
    and says why — and the gate counts skips apart from passes.
    """
    spec = {"coeffs": coeffs, "rhs": "q",
            "params": {"E": E_VAL, "I": I_VAL, "q": -Q_VAL, **params}}
    check = _sign_check(beam(8), spec)
    assert check_status(check) == SKIPPED
    assert "not pure bending" in check["detail"]


# ---------------------------- the non-self-adjoint half of the equation family


def test_an_a1_spec_is_held_to_coercivity_even_though_it_has_no_energy():
    """A spec with an a1 term gets no energy test, so it needs the general one.

    u^T K u = u^T sym(K) u, so positive definiteness of the symmetric part is
    the energy minimum when the spec is self-adjoint and Lax-Milgram coercivity
    when it is not. Before that covered both halves, this spec returned a beam
    rising under a downward load and the whole gate passed it.
    """
    destabilised = {
        "coeffs": {"v4": "E*I", "v1": "c", "v0": "k0"},
        "rhs": "q",
        "params": {"E": E_VAL, "I": I_VAL, "c": 5.0e4, "k0": -1.0e7, "q": -Q_VAL},
    }
    with pytest.raises(EquationError, match="no stable equilibrium"):
        solve_equation_beam(beam(8), destabilised)


def test_a_modest_a1_term_is_still_solved():
    """The coercivity test must not refuse the whole non-self-adjoint family."""
    spec = {"coeffs": {"v4": "E*I", "v1": "c"}, "rhs": "q",
            "params": {"E": E_VAL, "I": I_VAL, "c": 5.0e4, "q": -Q_VAL}}
    result = solve_equation_beam(beam(8), spec)
    assert min(p["v"] for p in result["samples"]) < 0

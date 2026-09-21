"""Checks of the symbolic equation solver: matrices, buckling, and four specs.

Offline and deterministic. The element matrices are compared SYMBOLICALLY with
the textbook forms, so a sign or a factor cannot hide behind a lucky number.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as sla
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.fem.equation as equation_module  # noqa: E402
from tools.fem.equation import (  # noqa: E402
    EquationError,
    assemble_equation,
    element_matrices,
    equilibrium_residual,
    equilibrium_terms,
    parse_spec,
    solve_equation_beam,
)
from tools.fem.solver import solve_beam_3d  # noqa: E402

E_VAL, I_VAL, Q_VAL, SPAN = 30e9, 0.005, 30e3, 25.0

L, E, I, K, P, Q = sp.symbols("L E I k P q", positive=True)

PINNED = [True, True, True, True, False, False]
ROLLER = [False, True, True, False, False, False]
FIXED = [True] * 6


def beam(n_elements, span=SPAN, supports="ss", point_loads=None):
    """Uniformly meshed beam model with n_elements elements."""
    xs = np.linspace(0.0, span, n_elements + 1)
    nodes = [{"id": f"N{k}", "x": float(x)} for k, x in enumerate(xs)]
    if supports == "ss":
        held = {"N0": PINNED, f"N{n_elements}": ROLLER}
    elif supports == "cantilever":
        held = {"N0": FIXED}
    else:
        held = {}
    return {"nodes": nodes, "supports": held, "point_loads": point_loads or []}


def bending_only(L_sym=L):
    """The pure a4 = EI element matrix, used to isolate the other terms."""
    return (
        E
        * I
        / L_sym**3
        * sp.Matrix(
            [
                [12, 6 * L_sym, -12, 6 * L_sym],
                [6 * L_sym, 4 * L_sym**2, -6 * L_sym, 2 * L_sym**2],
                [-12, -6 * L_sym, 12, -6 * L_sym],
                [6 * L_sym, 2 * L_sym**2, -6 * L_sym, 4 * L_sym**2],
            ]
        )
    )


def zero(matrix):
    return sp.simplify(matrix) == sp.zeros(*matrix.shape)


def test_euler_bernoulli_matrix_is_the_classic_one():
    k_e, f_e = element_matrices({"coeffs": {"v4": "E*I"}, "params": {"E": E, "I": I}}, L)
    assert zero(k_e - bending_only())
    assert zero(f_e)


def test_consistent_load_vector_is_the_classic_one():
    spec = {"coeffs": {"v4": "E*I"}, "rhs": "q", "params": {"E": E, "I": I, "q": Q}}
    _, f_e = element_matrices(spec, L)
    assert zero(f_e - sp.Matrix([L * Q / 2, L**2 * Q / 12, L * Q / 2, -(L**2) * Q / 12]))


def test_elastic_foundation_matrix_is_the_textbook_one():
    # v4 may not be zero, so the a0 block is isolated by removing the a4 block.
    spec = {"coeffs": {"v4": "E*I", "v0": "k"}, "params": {"E": E, "I": I, "k": K}}
    k_e, _ = element_matrices(spec, L)
    textbook = (
        K
        * L
        / 420
        * sp.Matrix(
            [
                [156, 22 * L, 54, -13 * L],
                [22 * L, 4 * L**2, 13 * L, -3 * L**2],
                [54, 13 * L, 156, -22 * L],
                [-13 * L, -3 * L**2, -22 * L, 4 * L**2],
            ]
        )
    )
    assert zero(k_e - bending_only() - textbook)


def test_geometric_stiffness_matrix_is_the_standard_one():
    spec = {"coeffs": {"v4": "E*I", "v2": "P"}, "params": {"E": E, "I": I, "P": P}}
    k_e, _ = element_matrices(spec, L)
    textbook = (
        -P
        / (30 * L)
        * sp.Matrix(
            [
                [36, 3 * L, -36, 3 * L],
                [3 * L, 4 * L**2, -3 * L, -(L**2)],
                [-36, -3 * L, 36, -3 * L],
                [3 * L, -(L**2), -3 * L, 4 * L**2],
            ]
        )
    )
    assert zero(k_e - bending_only() - textbook)


def _free_indices(model, dof_map):
    held = {dof_map[("N0", "v")], dof_map[(f"N{len(model['nodes']) - 1}", "v")]}
    return [g for g in range(2 * len(model["nodes"])) if g not in held]


def test_euler_buckling_pins_down_the_a2_sign():
    """P_cr = pi^2 EI / L^2 for a simply supported beam-column, 10 elements.

    A positive a2 is compression, so it must SOFTEN the beam: the assembled
    stiffness has to lose positive definiteness as P grows past P_cr. If the
    sign of the -a2 N'^T N' term were flipped, no positive P would be critical.
    """
    model = beam(10)
    params = {"E": E_VAL, "I": I_VAL}
    K_bend, _, dof_map = assemble_equation(model, {"coeffs": {"v4": "E*I"}, "params": params})
    K_unit, _, _ = assemble_equation(model, {"coeffs": {"v4": "E*I", "v2": "1"}, "params": params})
    K_geo = K_unit - K_bend  # the a2 = 1 contribution alone

    free = _free_indices(model, dof_map)
    eigenvalues = sla.eigh(
        K_bend[np.ix_(free, free)], -K_geo[np.ix_(free, free)], eigvals_only=True
    )
    p_cr = min(v for v in eigenvalues if v > 0)
    euler = np.pi**2 * E_VAL * I_VAL / SPAN**2
    assert p_cr == pytest.approx(euler, rel=5e-3)

    # and the real user-facing spec: below P_cr stable, above it not
    for factor, stable in ((0.9, True), (1.1, False)):
        spec = {"coeffs": {"v4": "E*I", "v2": "P"}, "params": {**params, "P": factor * euler}}
        K_p, _, _ = assemble_equation(model, spec)
        smallest = min(np.linalg.eigvalsh(K_p[np.ix_(free, free)]))
        assert bool(smallest > 0) is stable


def test_professor_beam_matches_closed_form_and_the_existing_solver():
    model = beam(4)
    spec = {
        "label": "Euler-Bernoulli beam",
        "coeffs": {"v4": "E*I", "v2": "0", "v1": "0", "v0": "0"},
        "rhs": "q",
        "params": {"E": E_VAL, "I": I_VAL, "q": -Q_VAL},
    }
    result = solve_equation_beam(model, spec)

    midspan = -5 * Q_VAL * SPAN**4 / (384 * E_VAL * I_VAL)
    assert result["displacements"]["N2"]["v"] == pytest.approx(midspan, rel=1e-9)
    assert result["reactions"]["N0"]["F"] == pytest.approx(Q_VAL * SPAN / 2, rel=1e-9)
    assert len(result["samples"]) >= 21
    assert abs(equilibrium_residual(model, spec, result)) < 1e-9

    reference = solve_beam_3d(
        {
            "nodes": model["nodes"],
            "material": {"E": E_VAL, "G": E_VAL / 2.4},
            "section": {"A": 1.0, "Iy": I_VAL, "Iz": I_VAL, "J": 0.001},
            "supports": model["supports"],
            "distributed_loads": [{"element": "all", "direction": "y", "w1": -Q_VAL, "w2": -Q_VAL}],
            "point_loads": [],
        }
    )
    for node in model["nodes"]:
        got = result["displacements"][node["id"]]
        want = reference["displacements"][node["id"]]
        assert got["v"] == pytest.approx(want["uy"], rel=1e-9, abs=1e-15)
        assert got["slope"] == pytest.approx(want["rz"], rel=1e-9, abs=1e-15)


def test_beam_on_elastic_foundation_matches_the_infinite_beam():
    """Hetenyi: an infinite beam on a foundation of modulus k under a point load P

    deflects y(x) = (P*beta / 2k) * exp(-beta*x) * (cos beta*x + sin beta*x) with
    beta = (k / 4EI)^(1/4), so directly under the load y = P*beta / (2k). Here
    beta*span = 30, and end effects decay like exp(-beta*x), so a free-free beam
    of this length is "infinite" to about 1e-13. The mesh must resolve the decay
    length 1/beta = 1.97 m; convergence is checked by halving h.
    """
    k_found, load, span = 40e6, 100e3, 60.0
    beta = (k_found / (4 * E_VAL * I_VAL)) ** 0.25
    closed_form = -load * beta / (2 * k_found)
    spec = {
        "coeffs": {"v4": "E*I", "v0": "k"},
        "params": {"E": E_VAL, "I": I_VAL, "k": k_found},
    }

    errors = []
    for n_elements in (120, 240):
        centre = f"N{n_elements // 2}"
        model = beam(
            n_elements,
            span=span,
            supports="free",
            point_loads=[{"node": centre, "dof": "FY", "value": -load}],
        )
        result = solve_equation_beam(model, spec)
        errors.append(abs(result["displacements"][centre]["v"] / closed_form - 1))
        assert abs(equilibrium_residual(model, spec, result)) < 1e-9

    assert errors[0] < 1e-4
    assert errors[1] < errors[0] / 8  # fourth-order convergence


def test_elastic_foundation_carries_load_that_the_reactions_do_not():
    """With a0 != 0, sum(reactions) + applied load is NOT zero, but the residual is."""
    model = beam(20)
    spec = {
        "coeffs": {"v4": "E*I", "v0": "k"},
        "rhs": "q",
        "params": {"E": E_VAL, "I": I_VAL, "k": 5e6, "q": -Q_VAL},
    }
    result = solve_equation_beam(model, spec)
    terms = equilibrium_terms(model, spec, result)

    naive = terms["reactions"] + terms["distributed"]
    assert abs(naive) > 0.5 * Q_VAL * SPAN  # the foundation carries most of the load
    assert terms["carried"] == pytest.approx(naive, rel=1e-9)
    assert abs(equilibrium_residual(model, spec, result)) < 1e-9


def test_first_derivative_term_is_in_the_equilibrium_balance():
    # a cantilever keeps v' one-signed, so integral(a1 v') cannot cancel itself
    model = beam(20, supports="cantilever")
    spec = {
        "coeffs": {"v4": "E*I", "v1": "c"},
        "rhs": "q",
        "params": {"E": E_VAL, "I": I_VAL, "c": 5e5, "q": -Q_VAL},
    }
    result = solve_equation_beam(model, spec)
    terms = equilibrium_terms(model, spec, result)
    assert abs(terms["carried"]) > 0.1 * Q_VAL * SPAN
    assert abs(equilibrium_residual(model, spec, result)) < 1e-9


def _tapered_midspan(n_elements):
    spec = {
        "label": "tapered beam",
        "coeffs": {"v4": "E*I0*(1 + x/L)"},
        "rhs": "q",
        "params": {"E": E_VAL, "I0": I_VAL, "L": SPAN, "q": -Q_VAL},
    }
    model = beam(n_elements)
    result = solve_equation_beam(model, spec)
    assert abs(equilibrium_residual(model, spec, result)) < 1e-9
    return result["displacements"][f"N{n_elements // 2}"]["v"]


def test_tapered_beam_lies_between_the_two_uniform_beams():
    # EI runs from E*I0 at x=0 to 2*E*I0 at x=L, so the beam is stiffer than the
    # uniform E*I0 beam everywhere but softer than the uniform 2*E*I0 one.
    soft = -5 * Q_VAL * SPAN**4 / (384 * E_VAL * I_VAL)
    stiff = soft / 2
    tapered = _tapered_midspan(16)
    assert soft < tapered < stiff


def test_tapered_beam_converges_under_mesh_refinement():
    values = [_tapered_midspan(n) for n in (8, 16, 32)]
    coarse = abs(values[1] - values[0])
    fine = abs(values[2] - values[1])
    assert fine < coarse / 4


def test_x_dependence_is_part_of_the_cache_key():
    uniform = parse_spec({"coeffs": {"v4": "E*I"}, "params": {"E": E_VAL, "I": I_VAL}})
    tapered = parse_spec(
        {"coeffs": {"v4": "E*I*(1 + x/L)"}, "params": {"E": E_VAL, "I": I_VAL, "L": SPAN}}
    )
    assert uniform.key != tapered.key
    assert not uniform.is_x_dependent and tapered.is_x_dependent


def test_symbolic_derivation_happens_once_per_equation():
    equation_module._SYMBOLIC_CACHE.clear()
    equation_module._NUMERIC_CACHE.clear()
    spec = {"coeffs": {"v4": "E*I"}, "rhs": "q", "params": {"E": E_VAL, "I": I_VAL, "q": -Q_VAL}}
    result = solve_equation_beam(beam(100), spec)
    assert len(equation_module._SYMBOLIC_CACHE) == 1  # not once per element
    assert result["displacements"]["N50"]["v"] == pytest.approx(
        -5 * Q_VAL * SPAN**4 / (384 * E_VAL * I_VAL), rel=1e-9
    )


@pytest.mark.parametrize(
    "spec, message",
    [
        ({"coeffs": {"v4": "E*I", "v3": "c"}, "params": {"E": 1, "I": 1, "c": 1}}, "v3"),
        ({"coeffs": {"v4": "E*I"}, "params": {"E": 1}}, "I"),
        ({"coeffs": {"v4": "0"}, "params": {}}, "identically zero"),
        ({"coeffs": {"v4": "E*I", "v5": "1"}, "params": {"E": 1, "I": 1}}, "unknown"),
        ({"coeffs": {"v4": "E*(("}, "params": {"E": 1}}, "could not parse"),
        ({"coeffs": {"v4": "E*I"}, "rhs": "w", "params": {"E": 1, "I": 1}}, "w"),
        ({"rhs": "q", "params": {"q": 1}}, "coeffs"),
    ],
)
def test_bad_specs_are_rejected_with_a_clear_message(spec, message):
    with pytest.raises(EquationError) as exc:
        parse_spec(spec)
    assert message in str(exc.value)


def test_imaginary_unit_and_euler_number_are_not_special():
    # sympy's default namespace would read "E*I" as e * sqrt(-1)
    parsed = parse_spec({"coeffs": {"v4": "E*I"}, "params": {"E": 2.0, "I": 3.0}})
    assert parsed.coeffs["v4"] == 6.0
    assert parsed.coeffs["v4"].is_real


def test_solving_needs_numeric_params():
    spec = {"coeffs": {"v4": "E*I"}, "params": {"E": E, "I": I_VAL}}
    with pytest.raises(EquationError, match="no numeric value"):
        solve_equation_beam(beam(4), spec)


def test_out_of_plane_point_load_is_refused():
    model = beam(4, point_loads=[{"node": "N2", "dof": "FX", "value": 1e3}])
    spec = {"coeffs": {"v4": "E*I"}, "params": {"E": E_VAL, "I": I_VAL}}
    with pytest.raises(EquationError, match="bending DOFs"):
        solve_equation_beam(model, spec)


def test_moment_sign_is_sagging_positive_under_downward_load():
    model = beam(16)
    spec = {"coeffs": {"v4": "E*I"}, "rhs": "q", "params": {"E": E_VAL, "I": I_VAL, "q": -Q_VAL}}
    result = solve_equation_beam(model, spec)
    midspan = [s for s in result["samples"] if s["x"] == pytest.approx(SPAN / 2)]
    assert midspan and all(s["moment"] > 0 for s in midspan)
    assert result["max_abs"]["moment"] == pytest.approx(Q_VAL * SPAN**2 / 8, rel=5e-3)
    assert min(s["v"] for s in result["samples"]) < 0

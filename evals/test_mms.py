"""Method-of-manufactured-solutions checks on the equation-driven solver.

The point of these tests is not that MMS passes — it is that MMS FAILS when the
element matrices are wrong. A verifier that always passes verifies nothing, so
two of these tests mutate the element matrix and demand a failure.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.fem import mms  # noqa: E402

E = 30e9
I = 0.005
L = 25.0
Q = -30e3  # downward negative

SPECS = {
    "euler-bernoulli": {
        "label": "Euler-Bernoulli beam",
        "coeffs": {"v4": "E*I", "v2": "0", "v1": "0", "v0": "0"},
        "rhs": "q",
        "params": {"E": E, "I": I, "q": Q},
    },
    "elastic foundation": {
        "label": "beam on elastic foundation",
        "coeffs": {"v4": "E*I", "v0": "k"},
        "rhs": "q",
        "params": {"E": E, "I": I, "k": 5e6, "q": Q},
    },
    "beam-column": {
        "label": "beam-column, axial compression",
        "coeffs": {"v4": "E*I", "v2": "P"},
        "rhs": "q",
        "params": {"E": E, "I": I, "P": 2.0e5, "q": Q},
    },
    "tapered": {
        "label": "tapered beam",
        "coeffs": {"v4": "E*I0*(1 + x/L)"},
        "rhs": "q",
        "params": {"E": E, "I0": I, "L": L, "q": Q},
    },
}


def _model(supports, xs=(0.0, L / 2, L)):
    return {
        "nodes": [{"id": f"N{k}", "x": float(x)} for k, x in enumerate(xs)],
        "material": {"E": E, "G": E / 2.4},
        "section": {"A": 0.5, "Iy": I, "Iz": I, "J": 0.001},
        "supports": supports,
        "distributed_loads": [{"element": "all", "direction": "y", "w1": Q, "w2": Q}],
        "point_loads": [],
    }


def simply_supported():
    return _model(
        {
            "N0": [True, True, True, True, False, False],
            "N2": [False, True, True, False, False, False],
        }
    )


def cantilever():
    return _model({"N0": [True] * 6})


def _mutate(monkeypatch, wrap):
    """Bug the element matrices for one test.

    _numeric_element is the seam the assembly actually goes through: it turns a
    parsed equation into the callable that hands out (k_e, f_e) per element, so
    wrapping it is the closest thing to shipping a wrong element matrix.
    """
    from tools.fem import equation

    monkeypatch.setattr(equation, "_numeric_element", wrap(equation._numeric_element))


@pytest.mark.parametrize("name", sorted(SPECS))
def test_mms_passes_for_every_equation_in_the_family(name):
    result = mms.mms_check(simply_supported(), SPECS[name])
    assert result["passed"], result["detail"]
    assert result["errors"][-1] <= 1e-3
    assert "simply supported" in result["detail"]
    assert "sin" in result["v_star"]


def test_mms_picks_and_reports_a_cantilever_solution():
    result = mms.mms_check(cantilever(), SPECS["elastic foundation"])
    assert result["passed"], result["detail"]
    assert "cantilever" in result["detail"]
    assert result["rate"] >= 2.5


@pytest.mark.parametrize(
    "supports, expected",
    [
        ({"N0": [True] * 6, "N2": [True] * 6}, "clamped-clamped"),
        (
            {"N0": [True] * 6, "N2": [False, True, True, False, False, False]},
            "propped cantilever",
        ),
        ({"N2": [True] * 6}, "cantilever fixed at the right end"),
    ],
)
def test_every_built_in_support_case_converges(supports, expected):
    result = mms.mms_check(_model(supports), SPECS["tapered"])
    assert result["passed"], result["detail"]
    assert expected in result["detail"]


def test_a_tapered_beam_is_the_divergence_form_not_a4_times_v4():
    """An x-dependent a4 makes (a4 v'')'' and a4 v'''' different equations.

    The weak form integrates by parts twice, so the elements solve the first.
    Manufacturing the load from the second leaves an error that stops falling —
    the signature of a discretization that converges to the wrong function — and
    that is what would happen to every tapered beam if mms.py used a4 v''''.
    """
    from tools.fem import equation

    model = simply_supported()
    v_star, _ = mms.choose_manufactured_solution(model)
    a4 = mms._coefficients(SPECS["tapered"])["v4"]
    naive = sp.sstr(a4 * sp.diff(v_star, mms.X, 4), full_prec=True)
    exact = sp.lambdify(mms.X, v_star, "math")

    errors = []
    for n in (4, 16):
        refined = mms._refined_model(model, n)
        result = equation.solve_equation_beam(refined, {**SPECS["tapered"], "rhs": naive})
        errors.append(
            max(
                abs(result["displacements"][node["id"]]["v"] - exact(node["x"]))
                for node in refined["nodes"]
            )
        )
    assert errors[1] > 0.5 * errors[0]  # measured: a 3.2% error that never shrinks


def test_errors_fall_with_refinement():
    result = mms.mms_check(simply_supported(), SPECS["tapered"])
    assert result["errors"] == sorted(result["errors"], reverse=True)
    assert result["rate"] == pytest.approx(4.0, abs=0.6)


def test_mutation_scaling_one_entry_by_5_percent_fails_the_check(monkeypatch):
    def wrap(original):
        def mutated(eq):
            evaluate = original(eq)

            def scaled(L, x0):
                k, f = evaluate(L, x0)
                k = np.array(k, dtype=float)
                k[0, 0] *= 1.05
                return k, f

            return scaled

        return mutated

    _mutate(monkeypatch, wrap)
    result = mms.mms_check(simply_supported(), SPECS["euler-bernoulli"])
    assert not result["passed"], result["detail"]
    assert result["errors"][-1] > 1e-2  # measured ~1e0: nowhere near converging


def test_mutation_flipping_the_a0_sign_fails_the_check(monkeypatch):
    from tools.fem.equation import ParsedEquation

    def wrap(original):
        def mutated(eq):
            return original(
                ParsedEquation(
                    label=eq.label,
                    coeffs={**eq.coeffs, "v0": -eq.coeffs["v0"]},
                    rhs=eq.rhs,
                    params=eq.params,
                )
            )

        return mutated

    _mutate(monkeypatch, wrap)
    result = mms.mms_check(simply_supported(), SPECS["elastic foundation"])
    assert not result["passed"], result["detail"]


def test_unmutated_run_still_passes_after_monkeypatching_is_undone():
    result = mms.mms_check(simply_supported(), SPECS["elastic foundation"])
    assert result["passed"], result["detail"]


@pytest.mark.parametrize("name", sorted(SPECS))
def test_manufactured_load_is_exact(name):
    spec = SPECS[name]
    v_star, _ = mms.choose_manufactured_solution(simply_supported())
    load = mms.manufactured_load(spec, v_star)
    assert sp.simplify(mms.residual({**spec, "rhs": load}, v_star)) == 0


def test_manufactured_load_of_the_professors_beam_is_the_textbook_udl():
    # v'''' of the exact simply-supported UDL shape gives back q
    spec = SPECS["euler-bernoulli"]
    x = mms.X
    v = Q / (24 * E * I) * (x**4 - 2 * L * x**3 + L**3 * x)
    assert float(mms.manufactured_load(spec, v)) == pytest.approx(Q, rel=1e-12)


def test_parameters_named_E_and_I_are_not_eulers_number_and_sqrt_minus_one():
    load = mms.manufactured_load(SPECS["euler-bernoulli"], "x**4")
    assert load.is_real
    assert float(load) == pytest.approx(24 * E * I, rel=1e-12)


def test_v3_is_rejected():
    spec = {**SPECS["euler-bernoulli"], "coeffs": {"v4": "E*I", "v3": "c"}}
    with pytest.raises(ValueError, match="v3"):
        mms.manufactured_load(spec, "x**4")


def test_a_missing_parameter_is_named():
    spec = {"coeffs": {"v4": "E*I"}, "rhs": "0", "params": {"E": E}}
    with pytest.raises(ValueError, match=r"params.*for: I$"):
        mms.manufactured_load(spec, "x**4")


def test_a_manufactured_solution_that_breaks_the_support_conditions_is_refused():
    with pytest.raises(ValueError, match="cantilever|slope|deflection"):
        mms.mms_check(
            cantilever(), SPECS["euler-bernoulli"], v_expr=sp.sin(sp.pi * mms.X / L)
        )


def test_supports_with_no_built_in_solution_say_so():
    pinned_one_end = _model({"N0": [True, True, True, True, False, False]})
    with pytest.raises(ValueError, match="no built-in manufactured solution"):
        mms.choose_manufactured_solution(pinned_one_end)

"""A square-root taper is a real beam, so it has to solve — and say how.

Two things are being pinned here.

SOLVABILITY. `E*I0*sqrt(1 + x/L)` and `E*I0*(1 + x/L)**0.5` are the same
equation spelled two ways. Integrated symbolically the first ran for over five
minutes with no output and the second raised ValueError('Non-suitable
parameters') from inside sympy's hypergeometric machinery. Both now go to
Gauss-Legendre quadrature, in well under a second, and — being the same
equation — they must return the same numbers.

HONESTY. Quadrature buys that speed by giving up exactness, so a numerically
integrated element must never be able to pass itself off as an exact one. Every
result says which method ran and why; the professor's own beam still says
"symbolic"; and where the two methods overlap they are held to agreeing to
1e-12, which is what makes the quadrature answer trustworthy in the cases where
no symbolic answer exists.

The method is chosen by STRUCTURE, never by a timeout: a timeout needs a signal
handler or a subprocess, and an ADK tool may not run on the main thread.

Offline and deterministic: sympy and numpy only, no LLM, no network.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.fem import equation as equation_module  # noqa: E402
from tools.fem import mms  # noqa: E402
from tools.fem.derivation import (  # noqa: E402
    equation_derivation_markdown,
    integration_method,
    validate_equation_spec,
)
from tools.fem.equation import (  # noqa: E402
    QUADRATURE_POINTS,
    EquationError,
    element_matrices,
    integration_plan,
    parse_spec,
    solve_equation_beam,
)

SPAN, N_ELEM = 25.0, 4
E_VAL, I0_VAL, Q_VAL = 30e9, 0.0025, -30e3

# The two spellings from the bug report, and the plain beam for contrast.
SQRT = "E*I0*sqrt(1 + x/L)"
POW_HALF = "E*I0*(1 + x/L)**0.5"
POLY_TAPER = "E*I0*(1 + x/L)"

SOLVE_BUDGET_S = 10.0  # the bug was "no answer at all", so this is deliberately loose


def beam(n_elem=N_ELEM):
    return {
        "nodes": [{"id": f"N{k}", "x": SPAN * k / n_elem} for k in range(n_elem + 1)],
        "supports": {
            "N0": [True, True, True, True, False, False],
            f"N{n_elem}": [False, True, True, False, False, False],
        },
        "point_loads": [],
    }


def taper(v4, label="taper"):
    return {
        "label": label,
        "coeffs": {"v4": v4, "v2": "0", "v1": "0", "v0": "0"},
        "rhs": "q",
        "params": {"E": E_VAL, "I0": I0_VAL, "L": SPAN, "q": Q_VAL},
    }


PLAIN_BEAM = {
    "label": "Euler-Bernoulli beam",
    "coeffs": {"v4": "E*I", "v2": "0", "v1": "0", "v0": "0"},
    "rhs": "q",
    "params": {"E": E_VAL, "I": 0.005, "q": Q_VAL},
}


def cold(spec):
    """Solve with every per-equation cache cleared, so a timing is a real one."""
    equation_module._INTEGRATION_CACHE.clear()
    equation_module._SYMBOLIC_CACHE.clear()
    equation_module._NUMERIC_CACHE.clear()
    start = time.perf_counter()
    result = solve_equation_beam(beam(), spec)
    return result, time.perf_counter() - start


def peak_values(result):
    return np.array([result["max_abs"][k] for k in ("v", "slope", "moment", "shear")])


# ------------------------------------------------- the two spellings of one taper


@pytest.mark.parametrize("v4", [SQRT, POW_HALF], ids=["sqrt()", "**0.5"])
def test_a_square_root_taper_solves_quickly_instead_of_hanging(v4):
    """Before: sqrt() ran past 300 s, **0.5 raised from sympy. Both under 10 s now."""
    result, elapsed = cold(taper(v4))

    assert elapsed < SOLVE_BUDGET_S, f"{v4} took {elapsed:.1f} s"
    assert result["integration"]["method"] == "quadrature"
    assert result["max_abs"]["v"] > 0


def test_the_two_spellings_are_the_same_equation_and_give_the_same_answer():
    """sqrt(u) and u**0.5 differ only in how they are typed."""
    by_sqrt, _ = cold(taper(SQRT))
    by_power, _ = cold(taper(POW_HALF))

    assert by_sqrt["integration"] == by_power["integration"]
    np.testing.assert_allclose(peak_values(by_power), peak_values(by_sqrt), rtol=1e-10)
    for node, dofs in by_sqrt["displacements"].items():
        for name, value in dofs.items():
            other = by_power["displacements"][node][name]
            assert other == pytest.approx(value, rel=1e-10, abs=1e-18), f"{node}.{name}"


# ------------------------------------------- quadrature against the exact integral


def test_the_two_methods_agree_entry_by_entry_on_a_polynomial_taper():
    """The overlap case: where a closed form exists, quadrature must reproduce it.

    A polynomial taper is integrated symbolically, so this compares the matrix
    the solver ships against the one the quadrature rule would have produced —
    the evidence that the quadrature answer can be trusted for the equations
    where no closed form exists.
    """
    spec = taper(POLY_TAPER, "polynomial taper")
    length, x_left = SPAN / N_ELEM, SPAN / 2  # off-origin, so x-dependence shows

    assert integration_plan(spec)["method"] == "symbolic"
    k_sym, f_sym = element_matrices(spec, length, x_left)
    k_sym = np.array(k_sym, dtype=float)
    f_sym = np.array(f_sym, dtype=float).reshape(4)

    by_quadrature = equation_module._quadrature_element(parse_spec(spec))
    k_quad, f_quad = by_quadrature(length, x_left)

    np.testing.assert_allclose(k_quad, k_sym, rtol=1e-12)
    np.testing.assert_allclose(f_quad, f_sym, rtol=1e-12)


def test_quadrature_matches_sympys_exact_integral_for_a_square_root_taper():
    """The taper that has no practical closed form, checked entry by entry anyway.

    sympy cannot integrate the whole 4x4 in useful time, but it integrates ONE
    entry of it quickly. That is enough to pin the quadrature rule against an
    exact result on the very coefficient that provoked this work.
    """
    spec = taper(SQRT)
    length, x_left = SPAN / N_ELEM, SPAN / 2
    eq = parse_spec(spec)

    xi = sp.Symbol("xi", real=True)
    N = [expr.subs({equation_module._EL: length}) for expr in equation_module._hermite()]
    d2 = [sp.diff(expr.subs({equation_module._XI: xi}), xi, 2) for expr in N]
    a4 = eq.coeffs["v4"].subs({equation_module.X: x_left + xi})

    k_quad, _ = equation_module._quadrature_element(eq)(length, x_left)
    for row, col in ((0, 0), (0, 1), (1, 3)):
        exact = float(sp.integrate(a4 * d2[row] * d2[col], (xi, 0, length)))
        assert k_quad[row, col] == pytest.approx(exact, rel=1e-12), f"k_e[{row},{col}]"


# ------------------------------------------------------ the professor's own beam


def test_the_plain_beam_is_still_integrated_symbolically_and_still_exact():
    """5qL^4/384EI, and the method that produced it says so."""
    result = solve_equation_beam(beam(8), PLAIN_BEAM)

    assert result["integration"]["method"] == "symbolic"
    assert result["integration"]["points"] is None

    E, I, q = PLAIN_BEAM["params"]["E"], PLAIN_BEAM["params"]["I"], abs(Q_VAL)
    expected = 5 * q * SPAN**4 / (384 * E * I)
    assert result["displacements"]["N4"]["v"] == pytest.approx(-expected, rel=1e-9)


# ------------------------------------------------------ what the result admits to


@pytest.mark.parametrize(
    "spec, method, wanted",
    [
        (PLAIN_BEAM, "symbolic", "polynomial in x"),
        (taper(POLY_TAPER), "symbolic", "polynomial in x"),
        (taper(SQRT), "quadrature", "coefficient 'v4' is not polynomial in x"),
        (taper(POW_HALF), "quadrature", "coefficient 'v4' is not polynomial in x"),
        (
            {**taper("E*I0"), "rhs": "q*sqrt(1 + x/L)"},
            "quadrature",
            "the right-hand side is not polynomial in x",
        ),
    ],
    ids=["plain", "poly-taper", "sqrt()", "**0.5", "non-poly-rhs"],
)
def test_every_result_says_how_it_was_integrated_and_why(spec, method, wanted):
    record = solve_equation_beam(beam(), spec)["integration"]

    assert record["method"] == method
    assert wanted in record["reason"], record["reason"]
    assert record["points"] == (None if method == "symbolic" else QUADRATURE_POINTS)


def test_the_integration_record_cannot_be_edited_through_a_result():
    """Two solves of one equation must not be able to contaminate each other."""
    first = solve_equation_beam(beam(), taper(SQRT))
    first["integration"]["method"] = "symbolic"

    second = solve_equation_beam(beam(), taper(SQRT))
    assert second["integration"]["method"] == "quadrature"


def test_integration_plan_answers_without_solving():
    assert integration_plan(taper(SQRT)) == solve_equation_beam(beam(), taper(SQRT))[
        "integration"
    ]


# --------------------------------------------------- the convergence self-check


@pytest.mark.parametrize(
    "v4",
    [
        "E*I0*(0.5 + sqrt(x/L))",  # infinite slope at x = 0, an element boundary
        "E*I0*(1 + 1/(1e-4 + (x/L)**2))",  # a pole just off the real axis
    ],
    ids=["endpoint-singularity", "near-pole"],
)
def test_a_coefficient_too_sharp_to_integrate_is_refused_by_name(v4):
    """A quietly inaccurate stiffness matrix is worse than an error."""
    with pytest.raises(EquationError) as exc:
        solve_equation_beam(beam(), taper(v4))

    message = str(exc.value)
    assert "v4" in message
    assert f"{QUADRATURE_POINTS}-point" in message and f"{2 * QUADRATURE_POINTS}-point" in message


def test_a_smooth_taper_passes_the_self_check_on_every_element():
    """The self-check must not be so tight that legitimate tapers trip it."""
    for v4 in (SQRT, "E*I0*exp(x/L)", "E*I0*(2 + sin(3*x/L))"):
        result = solve_equation_beam(beam(8), taper(v4))
        assert result["integration"]["method"] == "quadrature"


# ------------------------------------------------------------------------- MMS


def test_mms_verifies_the_quadrature_path_on_a_square_root_taper():
    """The general verification has to keep working when the method changes."""
    result = mms.mms_check(beam(), taper(SQRT), refinements=(4, 8, 16))

    assert result["passed"], result["detail"]
    assert result["errors"][-1] <= 1e-3


def _mutate(monkeypatch, wrap):
    monkeypatch.setattr(
        equation_module, "_element_evaluator", wrap(equation_module._element_evaluator)
    )


def _scale_one_entry(original):
    def mutated(eq):
        evaluate = original(eq)

        def scaled(L, x0):
            k, f = evaluate(L, x0)
            k = np.array(k, dtype=float)
            k[0, 0] *= 1.05
            return k, f

        return scaled

    return mutated


def test_mms_still_catches_a_wrong_element_matrix_on_the_quadrature_path(monkeypatch):
    """Otherwise the sqrt taper would be fast and unverified, which is worse."""
    _mutate(monkeypatch, _scale_one_entry)
    result = mms.mms_check(beam(), taper(SQRT), refinements=(4, 8, 16))

    assert not result["passed"], result["detail"]


# A polynomial v*: it vanishes and has zero curvature at both ends, so it is
# admissible for a simply supported beam, and the load it manufactures is a
# polynomial too - which keeps this case on the SYMBOLIC path. The sine v* that
# mms picks by default would not, so without this the symbolic path would lose
# its manufactured-solution cover entirely.
POLYNOMIAL_V_STAR = "0.01*((x/25)  - 2*(x/25)**3 + (x/25)**4)"


def test_mms_verifies_the_symbolic_path_with_a_polynomial_manufactured_solution():
    result = mms.mms_check(beam(), PLAIN_BEAM, v_expr=POLYNOMIAL_V_STAR, refinements=(4, 8, 16))

    assert result["passed"], result["detail"]
    assert integration_plan(PLAIN_BEAM)["method"] == "symbolic"


def test_mms_still_catches_a_wrong_element_matrix_on_the_symbolic_path(monkeypatch):
    _mutate(monkeypatch, _scale_one_entry)
    result = mms.mms_check(beam(), PLAIN_BEAM, v_expr=POLYNOMIAL_V_STAR, refinements=(4, 8, 16))

    assert not result["passed"], result["detail"]


# --------------------------------------------------------------- the document


def test_the_derivation_document_for_a_square_root_taper_is_generated_quickly():
    start = time.perf_counter()
    md = equation_derivation_markdown(taper(SQRT, "Square-root taper"))
    elapsed = time.perf_counter() - start

    assert elapsed < SOLVE_BUDGET_S, f"the document took {elapsed:.1f} s"
    assert f"{QUADRATURE_POINTS}-point Gauss-Legendre quadrature" in md
    assert "numerical, not closed-form" in md
    assert f"{2 * QUADRATURE_POINTS} points" in md  # the self-check is stated too


def test_the_derivation_document_for_a_polynomial_spec_still_shows_closed_forms():
    md = equation_derivation_markdown(taper(POLY_TAPER, "Polynomial taper"))

    assert "symbolically" in md
    assert "Gauss-Legendre" not in md
    assert r"\mathbf{k}_e = \left[\begin{matrix}" in md


@pytest.mark.parametrize(
    "spec",
    [PLAIN_BEAM, taper(POLY_TAPER), taper(SQRT), taper(POW_HALF), taper("E*I0*exp(x/L)")],
    ids=["plain", "poly-taper", "sqrt()", "**0.5", "exp"],
)
def test_the_document_and_the_solver_agree_on_the_integration_method(spec):
    """They decide it independently, so a document cannot claim a method that
    never ran. Same reason validate_equation_spec mirrors parse_spec."""
    method, _why = integration_method(validate_equation_spec(spec))

    assert method == integration_plan(spec)["method"]


def test_the_document_obeys_the_report_markdown_rules_on_the_quadrature_path():
    """Same rules evals/test_report_markdown.py holds every equation to."""
    lines = equation_derivation_markdown(taper(SQRT)).splitlines()
    marks = [i for i, line in enumerate(lines) if "$$" in line]

    assert marks and len(marks) % 2 == 0
    assert all(lines[i].strip() == "$$" for i in marks), "a $$ shares its line with text"
    for open_, close in zip(marks[::2], marks[1::2]):
        assert close - open_ == 2, f"equation at line {open_} spans {close - open_ - 1} lines"
        assert not lines[open_ + 1].lstrip().startswith(("-", "+", "*", ">"))
        assert open_ == 0 or lines[open_ - 1] == ""
        assert close == len(lines) - 1 or lines[close + 1] == ""

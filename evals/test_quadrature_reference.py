"""The quadrature answer, checked against a solution this file derives itself.

A hang is loud. A numerically integrated stiffness matrix that is quietly a few
percent wrong is not: the report still prints a deflection, the gate still sees
equilibrium balance, and nothing says the number is bad. So the quadrature path
is not trusted here because it agrees with itself — it is checked against a
deflection computed WITHOUT any finite element, any Hermite cubic or any
Gauss-Legendre rule.

The reference comes from the equation rather than from the code under test. The
weak form integrates $a_4 v''''$ by parts twice, so what the solver actually
solves for a taper is the self-adjoint $(EI(x)\\,v'')'' = q$. Writing
$M = EI v''$ turns that into $M'' = q$, and for a simply supported span both end
moments are zero, so

    M(x) = q (x^2 - L x) / 2

exactly — the statically determinate bending moment, the SAME for every taper,
which is worth stating because it means the reference below cannot have
inherited the solver's idea of how EI varies. Then $v'' = M/EI$ integrates twice
under v(0) = v(L) = 0:

    v(x) = int_0^x (x - s) g(s) ds + C x,   g = M/EI,
    C    = -(1/L) int_0^L (L - s) g(s) ds

and those two integrals are done by scipy's adaptive Gauss-Kronrod quad, which
shares no code, no nodes and no order with the fixed 16-point Gauss-Legendre
rule in tools/fem/equation.py. A second reference solves the same problem again
as a four-equation boundary value problem with scipy.integrate.solve_bvp, and
the two agree to 1e-15, so neither of them is being taken on trust either.

Offline and deterministic: numpy and scipy only, no LLM, no network.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad, solve_bvp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.fem import equation as equation_module  # noqa: E402
from tools.fem.equation import solve_equation_beam  # noqa: E402

SPAN, E_VAL, I0_VAL, Q_VAL = 25.0, 30e9, 0.0025, -30e3

CONSTANT = "E*I0"
LINEAR = "E*I0*(1 + x/L)"
SQRT = "E*I0*sqrt(1 + x/L)"

# The same three tapers as callables, written out by hand rather than lambdified
# from the spec, so the reference does not go through the solver's parser.
EI_OF = {
    CONSTANT: lambda x: E_VAL * I0_VAL * np.ones_like(np.asarray(x, dtype=float)),
    LINEAR: lambda x: E_VAL * I0_VAL * (1.0 + np.asarray(x, dtype=float) / SPAN),
    SQRT: lambda x: E_VAL * I0_VAL * np.sqrt(1.0 + np.asarray(x, dtype=float) / SPAN),
}


def beam(n_elem):
    """A simply supported 25 m span on n_elem equal elements."""
    return {
        "nodes": [{"id": f"N{k}", "x": SPAN * k / n_elem} for k in range(n_elem + 1)],
        "supports": {
            "N0": [True, True, True, True, False, False],
            f"N{n_elem}": [False, True, True, False, False, False],
        },
        "point_loads": [],
    }


def taper(v4):
    return {
        "label": "reference taper",
        "coeffs": {"v4": v4, "v2": "0", "v1": "0", "v0": "0"},
        "rhs": "q",
        "params": {"E": E_VAL, "I0": I0_VAL, "L": SPAN, "q": Q_VAL},
    }


# ------------------------------------------------------- the reference itself


def reference_deflection(EI, xs):
    """v(x) for (EI(x) v'')'' = q, simply supported, by adaptive quadrature.

    Args:
        EI: callable x -> bending stiffness, in N*m^2.
        xs: the stations to evaluate at, in m.

    Returns:
        ndarray of deflections in m, downward negative, as the pipeline signs them.
    """
    moment = lambda s: Q_VAL * (s * s - SPAN * s) / 2.0  # noqa: E731
    curvature = lambda s: moment(s) / float(EI(s))  # noqa: E731

    def integrate(fn, upper):
        with warnings.catch_warnings():  # a roundoff notice at 1e-14 is not news
            warnings.simplefilter("ignore")
            return quad(fn, 0.0, upper, limit=400, epsabs=1e-14, epsrel=1e-14)[0]

    slope_at_origin = -integrate(lambda s: (SPAN - s) * curvature(s), SPAN) / SPAN
    return np.array(
        [
            0.0 if x <= 0.0 else integrate(lambda s: (x - s) * curvature(s), x)
            + slope_at_origin * x
            for x in np.atleast_1d(xs)
        ]
    )


def reference_deflection_bvp(EI, xs):
    """The same deflection again, as a BVP in y = [v, v', M, M'] — a second opinion."""

    def rhs(x, y):
        return np.vstack([y[1], y[2] / EI(x), y[3], np.full_like(x, Q_VAL)])

    def boundary(ya, yb):
        return np.array([ya[0], yb[0], ya[2], yb[2]])  # v and M zero at both ends

    mesh = np.linspace(0.0, SPAN, 2001)
    sol = solve_bvp(rhs, boundary, mesh, np.zeros((4, mesh.size)), tol=1e-12,
                    max_nodes=200000)
    assert sol.status == 0, sol.message
    return sol.sol(np.atleast_1d(xs))[0]


def fem_deflection(v4, n_elem):
    """(stations, deflections, the integration record) from one solver run."""
    result = solve_equation_beam(beam(n_elem), taper(v4))
    xs = np.array([p["x"] for p in result["samples"]])
    vs = np.array([p["v"] for p in result["samples"]])
    return xs, vs, result


def relative_error(v4, n_elem):
    xs, vs, _ = fem_deflection(v4, n_elem)
    reference = reference_deflection(EI_OF[v4], xs)
    return float(np.max(np.abs(vs - reference)) / np.max(np.abs(reference)))


# --------------------------------------------------- the two references agree


@pytest.mark.parametrize("v4", [CONSTANT, LINEAR, SQRT], ids=["constant", "linear", "sqrt"])
def test_the_two_independent_references_agree_with_each_other(v4):
    """Neither reference is taken on trust: they are derived differently."""
    xs = np.linspace(0.0, SPAN, 21)
    by_quadrature = reference_deflection(EI_OF[v4], xs)
    by_bvp = reference_deflection_bvp(EI_OF[v4], xs)

    scale = np.max(np.abs(by_quadrature))
    assert np.max(np.abs(by_quadrature - by_bvp)) / scale < 1e-10


# ------------------------------------ the solver against the reference, by method


@pytest.mark.parametrize(
    "v4, method",
    [(LINEAR, "symbolic"), (SQRT, "quadrature")],
    ids=["linear-symbolic", "sqrt-quadrature"],
)
def test_the_solver_converges_to_the_independent_reference(v4, method):
    """A taper solved by either method has to land on the same beam.

    The linear taper is here as the control: it goes through the symbolic path,
    so an error in the REFERENCE would show up on it too and could not be
    mistaken for an error in the quadrature rule.
    """
    assert equation_module.integration_plan(taper(v4))["method"] == method

    errors = {n: relative_error(v4, n) for n in (4, 16, 64)}

    assert errors[4] < 5e-3, errors
    assert errors[16] < 2e-5, errors
    assert errors[64] < 1e-7, errors
    # Hermite cubics interpolating a smoothly varying curvature: O(h^4). Two
    # quarterings of h is a factor of 256 in theory; 100 leaves room for the
    # sampling not landing on the same stations at every refinement.
    assert errors[4] / errors[64] > 100, errors


def test_the_quadrature_taper_is_as_accurate_as_the_symbolic_one():
    """Not merely "close": no worse than the closed-form path on the same mesh.

    If the 16-point rule were leaking error into the element matrices, the sqrt
    taper would sit at a different error level from the linear one, whose
    matrices are exact expressions.
    """
    for n_elem in (8, 32):
        symbolic = relative_error(LINEAR, n_elem)
        numeric = relative_error(SQRT, n_elem)
        assert numeric < 3 * symbolic, (n_elem, symbolic, numeric)


# ------------------------------------ what the quadrature rule itself contributes


@pytest.mark.parametrize("v4", [SQRT, "E*I0*exp(x/L)", "E*I0*log(2 + x/L)"])
def test_raising_the_order_does_not_move_the_answer(v4, monkeypatch):
    """So the error left against the reference is the MESH, not the integration.

    16 points and 64 points on the same mesh have to give the same displacements
    to near machine precision. They do, which is what licenses reading the
    remaining O(h^4) gap as discretization error and nothing else.
    """
    at_16 = solve_equation_beam(beam(8), taper(v4))["displacements"]

    monkeypatch.setattr(equation_module, "QUADRATURE_POINTS", 64)
    at_64 = solve_equation_beam(beam(8), taper(v4))["displacements"]

    for node, dofs in at_16.items():
        for name, value in dofs.items():
            assert at_64[node][name] == pytest.approx(value, rel=1e-12, abs=1e-18)


# ------------------------------- a check the closed-form gate has to skip on a taper


def moment_error(v4, n_elem):
    """Largest departure of the sampled moment from q(x^2 - Lx)/2, relative to qL^2/8."""
    result = solve_equation_beam(beam(n_elem), taper(v4))
    xs = np.array([p["x"] for p in result["samples"]])
    sampled = np.array([p["moment"] for p in result["samples"]])
    exact = Q_VAL * (xs**2 - SPAN * xs) / 2.0
    return float(np.max(np.abs(sampled - exact)) / (abs(Q_VAL) * SPAN**2 / 8.0))


@pytest.mark.parametrize(
    "v4", [CONSTANT, LINEAR, SQRT, "E*I0*exp(x/L)", "E*I0*(1 + x/L)**1.5"]
)
def test_the_moment_converges_to_the_statically_determinate_one_for_every_taper(v4):
    """M(x) = q(x^2 - Lx)/2, whatever EI does along the span, because M'' = q fixes it.

    The gate's closed-form checks skip a tapered beam, so this is the one
    external number a taper can still be held to, and it is independent of the
    element matrices: they can only get it wrong, never define it. Moment is
    EI v'', two derivatives down from the displacement, so it converges at
    O(h^2) rather than O(h^4) and needs a mesh before it means anything.
    """
    coarse, fine = moment_error(v4, 16), moment_error(v4, 64)

    assert fine < 1e-3, (coarse, fine)
    # O(h^2) over a 4x refinement is a factor of 16; every taper here lands
    # between 15 and 16, so 12 is a floor rather than a fitted threshold.
    assert coarse / fine > 12, (coarse, fine)


def test_the_moment_of_a_quadrature_taper_is_as_good_as_a_symbolic_one():
    """The same statement as above, made as a comparison rather than a tolerance."""
    assert moment_error(SQRT, 64) < 2 * moment_error(CONSTANT, 64)

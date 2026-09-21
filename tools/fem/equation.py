"""Galerkin FEM for a user-supplied 4th-order beam ODE.

The governing equation is DATA, not code. An equation spec is a plain dict::

    {"label": "Euler-Bernoulli beam",
     "coeffs": {"v4": "E*I", "v2": "0", "v1": "0", "v0": "0"},
     "rhs": "q",
     "params": {"E": 30e9, "I": 0.005, "q": -30e3}}

which means the residual form

    a4 v'''' + a2 v'' + a1 v' + a0 v = f(x)

with v(x) the transverse deflection, x along the beam, SI units, and DOWNWARD
NEGATIVE. Coefficients may depend on x (tapered beam); v3 is not supported.

Multiplying the residual by a test function w, integrating the v'''' term by
parts twice and the v'' term once, gives the element matrices

    k_e = integral( a4 N''^T N'' - a2 N'^T N' + a1 N^T N' + a0 N^T N ) dx
    f_e = integral( f N^T ) dx

over the four Hermite cubics N of tools/fem/derivation.py. Every entry is
produced by sympy at call time; no matrix is typed in.

Two consequences of that weak form are worth stating because the code cannot:
integrating twice turns an x-dependent a4 into the self-adjoint operator
(a4 v'')'', which is the physically correct tapered-beam equation rather than
a4 v''''; and the -a2 N'^T N' sign makes a POSITIVE a2 (compression) soften the
beam, so a simply supported beam-column goes singular at P = pi^2 EI / L^2.

That last sentence is also why solving is not merely a matrix inverse. With no
a1 term the weak form is an energy, and a static equilibrium is the shape that
MINIMIZES it — which is exactly the free stiffness matrix being positive
definite. Past the buckling load, or with a4 non-positive anywhere along the
span, that matrix is indefinite: the linear system still has one unique
solution, but it is an unstable equilibrium and it points the wrong way, so a
beam under a downward load comes back deflecting upward. Those systems are
refused here rather than reported. See _solve_free.

What a result calls its maximum is the maximum of the SOLUTION, not of the
sample list. The samples are a readable series for plots and the report, and a
peak falling between two of them would otherwise never be reported — and
refining the mesh would not rescue it, because the samples keep landing at the
same relative position inside each element. Deflection and slope are
polynomials on the element, so their extrema are solved for exactly; moment and
shear carry a4(x), which need not be a polynomial, so they are scanned densely
instead. See _peaks.

How the element integrals are computed is decided by the STRUCTURE of the spec,
never by a timeout: a timeout needs a signal handler or a subprocess, and this
module runs as an agent tool that may not be on the main thread. A spec whose
coefficients and rhs are all polynomials in x is integrated symbolically, which
is exact and leaves the derivation document a closed form. Anything else — a
square-root taper, an exponential, a trigonometric coefficient — is integrated
with fixed-order Gauss-Legendre quadrature, which costs the same few
microseconds whatever the coefficient is, rather than sending sympy into an
integral it may never finish. Which of the two ran, and why, is reported in the
result's "integration" record: a numerically integrated element must never be
able to pass itself off as exact. See _element_rule.
"""

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import scipy.linalg as sla
import sympy as sp

from tools.fem.safe_expr import ExpressionError, clean_label, parse_expression

COEFF_KEYS = ("v4", "v2", "v1", "v0")

X = sp.Symbol("x")  # the beam coordinate, the one name a spec may not rebind

# Underscored so a spec may carry params of its own named L or x0.
_EL = sp.Symbol("_L", positive=True)  # element length
_X0 = sp.Symbol("_x0", real=True)  # global x at the element's left node
_XI = sp.Symbol("_xi", real=True)  # local coordinate, 0 <= _xi <= _L

_DOF_OF_LOAD = {"FY": "v", "MZ": "slope"}
_SUPPORT_FLAG = {"v": 1, "slope": 5}  # index into the 6-flag support list


class EquationError(ValueError):
    """An equation spec is malformed, incomplete, or outside the supported family."""


@dataclass
class ParsedEquation:
    """A validated equation spec with its parameters already substituted.

    coeffs maps each of v4/v2/v1/v0 to a sympy expression in x (and in whatever
    symbols the params themselves carry); rhs is f(x); key identifies the
    symbolic derivation for caching.
    """

    label: str
    coeffs: dict
    rhs: sp.Expr
    params: dict
    key: tuple = field(init=False, repr=False)

    def __post_init__(self):
        # The key is the expressions themselves, so x-dependence is part of it.
        self.key = tuple(sp.srepr(self.coeffs[k]) for k in COEFF_KEYS) + (sp.srepr(self.rhs),)

    @property
    def is_x_dependent(self) -> bool:
        """True if any coefficient or the right-hand side varies along the beam."""
        return any(X in e.free_symbols for e in (*self.coeffs.values(), self.rhs))

    def free_symbols(self) -> set:
        """Symbols still unresolved after substitution (empty for a numeric spec)."""
        symbols = set().union(*(e.free_symbols for e in (*self.coeffs.values(), self.rhs)))
        return symbols - {X}


def _parse(text, what):
    """Parse one spec expression under tools.fem.safe_expr's whitelist grammar."""
    if isinstance(text, (np.integer, np.floating)):
        text = float(text)
    try:
        return parse_expression(text, local_dict={"x": X})
    except ExpressionError as exc:
        raise EquationError(f"could not parse {what}: {text!r} ({exc})") from exc


def parse_spec(spec: dict) -> ParsedEquation:
    """Validate an equation spec and substitute its params.

    Args:
        spec: dict with optional "label", a "coeffs" dict over v4/v2/v1/v0
            (sympy-parseable strings in x and in the param names), an optional
            "rhs" f(x) defaulting to 0, and a "params" dict of values.

    Returns:
        ParsedEquation with params substituted into every expression.

    Raises:
        EquationError: unknown coefficient keys, a v3 term, an identically zero
            v4, an unparseable expression, or a symbol no param supplies.
    """
    if isinstance(spec, ParsedEquation):
        return spec
    if not isinstance(spec, dict):
        raise EquationError(f"equation spec must be a dict, got {type(spec).__name__}")

    coeffs = spec.get("coeffs")
    if not isinstance(coeffs, dict):
        raise EquationError("equation spec needs a 'coeffs' dict over " + ", ".join(COEFF_KEYS))
    unknown = [k for k in coeffs if k not in COEFF_KEYS]
    if "v3" in unknown:
        raise EquationError(
            "coefficient 'v3' is not supported: a third-derivative term has no "
            "symmetric Galerkin form on Hermite C1 elements. Supported keys: "
            + ", ".join(COEFF_KEYS)
        )
    if unknown:
        raise EquationError(
            f"unknown coefficient key(s) {sorted(unknown)}; supported keys: "
            + ", ".join(COEFF_KEYS)
        )

    params = spec.get("params") or {}
    if not isinstance(params, dict):
        raise EquationError(f"'params' must be a dict, got {type(params).__name__}")
    subs = {sp.Symbol(name): _parse(value, f"params[{name!r}]") for name, value in params.items()}

    raw = {k: _parse(coeffs.get(k, 0), f"coeffs[{k!r}]") for k in COEFF_KEYS}
    raw["rhs"] = _parse(spec.get("rhs", 0), "rhs")

    supplied = {s.name for s in subs}
    for name, expr in raw.items():
        missing = sorted(s.name for s in expr.free_symbols if s != X and s.name not in supplied)
        if missing:
            where = "rhs" if name == "rhs" else f"coeffs[{name!r}]"
            raise EquationError(
                f"{where} = {expr} references {missing} but params supplies "
                f"{sorted(supplied) or 'nothing'}"
            )

    done = {name: expr.subs(subs) for name, expr in raw.items()}
    for name, expr in done.items():
        # inf and nan propagate through every integral and every solve and come
        # out the far end as a result full of nan that still looks like a result.
        if expr.has(sp.nan, sp.oo, -sp.oo, sp.zoo):
            where = "rhs" if name == "rhs" else f"coeffs[{name!r}]"
            raise EquationError(
                f"{where} evaluates to {expr}, which is not a finite number; "
                "every parameter must be finite"
            )
    if sp.simplify(done["v4"]) == 0:
        raise EquationError(
            "coefficient 'v4' is identically zero; the Hermite C1 element needs a "
            "fourth-order term. Drop to a lower-order formulation instead."
        )

    return ParsedEquation(
        label=clean_label(spec.get("label")),
        coeffs={k: done[k] for k in COEFF_KEYS},
        rhs=done["rhs"],
        params=dict(params),
    )


@lru_cache(maxsize=None)
def _hermite():
    """The four Hermite cubics in (_xi, _L), derived from the nodal conditions."""
    v1, t1, v2, t2 = sp.symbols("v_1 theta_1 v_2 theta_2")
    a = sp.symbols("a_0:4")
    v = sum(a[k] * _XI**k for k in range(4))
    sol = sp.solve(
        [
            v.subs(_XI, 0) - v1,
            v.diff(_XI).subs(_XI, 0) - t1,
            v.subs(_XI, _EL) - v2,
            v.diff(_XI).subs(_XI, _EL) - t2,
        ],
        a,
    )
    vx = sp.expand(v.subs(sol))
    return sp.Matrix([sp.factor(vx.coeff(d)) for d in (v1, t1, v2, t2)])


@lru_cache(maxsize=None)
def _hermite_numeric():
    """(N, N', N'', N''') as fast callables of (xi, L) returning length-4 lists."""
    N = _hermite()
    return tuple(sp.lambdify((_XI, _EL), list(N.diff(_XI, d)), "numpy") for d in range(4))


@lru_cache(maxsize=None)
def _hermite_powers():
    """Callable L -> (4, 4) array C, row k holding each shape function's xi**k term.

    The deflection on an element is the cubic sum_j d_j N_j(xi), so C(L) @ d is
    that cubic's coefficients in ascending powers of xi. Used only to LOCATE a
    stationary point: the value there is always read back through the shape
    functions themselves, which stay exact next to a node, where the cubic's
    own coefficients cancel against each other.
    """
    N = _hermite()
    powers = [[sp.expand(N[j]).coeff(_XI, k) for j in range(4)] for k in range(4)]
    return sp.lambdify(_EL, sp.Matrix(powers), "numpy")


def _tidy(expr):
    try:
        return sp.factor(sp.expand(expr))
    except Exception:  # an exotic spec integrates to something factor() dislikes
        return expr


_SYMBOLIC_CACHE = {}


def _symbolic_element(eq: ParsedEquation):
    """(k_e, f_e) symbolic in (_L, _x0), integrated once per distinct equation."""
    if eq.key in _SYMBOLIC_CACHE:
        return _SYMBOLIC_CACHE[eq.key]

    N = _hermite()
    N1, N2 = N.diff(_XI), N.diff(_XI, 2)
    glob = {X: _X0 + _XI}  # the element's true interval, in the global coordinate
    a4, a2, a1, a0 = (eq.coeffs[k].subs(glob) for k in COEFF_KEYS)
    f = eq.rhs.subs(glob)

    k_integrand = a4 * N2 * N2.T - a2 * N1 * N1.T + a1 * N * N1.T + a0 * N * N.T
    f_integrand = f * N

    def over_element(expr):
        return _tidy(sp.integrate(sp.expand(expr), (_XI, 0, _EL)))

    k_e = sp.Matrix(4, 4, lambda r, c: over_element(k_integrand[r, c]))
    f_e = sp.Matrix(4, 1, lambda r, _: over_element(f_integrand[r]))

    _SYMBOLIC_CACHE[eq.key] = (k_e, f_e)
    return k_e, f_e


_NUMERIC_CACHE = {}


def _numeric_element(eq: ParsedEquation):
    """Callable (L, x0) -> (k_e, f_e) as numpy arrays, for meshes of any size."""
    if eq.key not in _NUMERIC_CACHE:
        k_e, f_e = _symbolic_element(eq)
        _NUMERIC_CACHE[eq.key] = sp.lambdify((_EL, _X0), (k_e, f_e), "numpy")
    fn = _NUMERIC_CACHE[eq.key]

    def evaluate(L, x0):
        k, f = fn(float(L), float(x0))
        return np.asarray(k, dtype=float), np.asarray(f, dtype=float).reshape(4)

    return evaluate


# --------------------------------------------------------------- quadrature

# Gauss-Legendre points per element when a coefficient is not a polynomial.
# An n-point rule is exact for a polynomial of degree 2n-1. Over one element
# the integrands are the Hermite cubics against each other: the v4 integrand
# (N'')^T N'' is degree 2 in xi, the v2 integrand N'^T N' is degree 4, the v1
# integrand N^T N' is degree 5 and the v0 integrand N^T N is degree 6 — so
# degree 6 is the worst case a CONSTANT coefficient produces, and the load
# term f N^T is degree 3. 16 points are exact to degree 31, which still leaves
# room for a polynomial coefficient of degree 25 riding on top of the worst of
# those: every polynomial spec this module would otherwise integrate
# symbolically is reproduced exactly, with room to spare, for 16 evaluations
# of each coefficient per element.
QUADRATURE_POINTS = 16

# The doubled-order self-check must agree to this, relative to the largest
# entry of the same term. Gauss-Legendre converges far past it on a smooth
# coefficient; one with a corner, a pole or an endpoint singularity does not.
QUADRATURE_TOL = 1e-10

_FIELD_NAMES = {**{k: f"coefficient {k!r}" for k in COEFF_KEYS}, "rhs": "the right-hand side"}


@lru_cache(maxsize=None)
def _leggauss(n):
    """(nodes, weights) of the n-point Gauss-Legendre rule on [-1, 1]."""
    return np.polynomial.legendre.leggauss(n)


def _is_polynomial_in_x(expr) -> bool:
    """True when expr is a polynomial in x; a constant counts, at degree zero."""
    if X not in expr.free_symbols:
        return True
    try:
        sp.Poly(expr, X)
    except Exception:
        # sympy signals "not a polynomial in this generator" through several
        # exception types (PolynomialError, GeneratorsNeeded, CoercionFailed).
        # They all mean the same thing here, and none of them is worth telling
        # the professor about: the answer is simply "integrate it numerically".
        return False
    return True


def _non_polynomial(eq: ParsedEquation) -> list:
    """Spec fields that are not polynomials in x, named as the spec names them."""
    fields = [(key, eq.coeffs[key]) for key in COEFF_KEYS] + [("rhs", eq.rhs)]
    return [_FIELD_NAMES[key] for key, expr in fields if not _is_polynomial_in_x(expr)]


def _along_x(expr):
    """expr as a callable x -> ndarray, broadcasting a constant over the points."""
    fn = sp.lambdify(X, expr, "numpy")

    def evaluate(x):
        return np.broadcast_to(np.asarray(fn(x), dtype=float), np.shape(x))

    return evaluate


def _shape_values(fn, xi, L):
    """The four Hermite values at every quadrature point, as a (4, n) array.

    lambdify returns a bare float for a shape function that does not depend on
    xi, so each row is broadcast before stacking.
    """
    return np.array(
        [np.broadcast_to(np.asarray(row, dtype=float), xi.shape) for row in fn(xi, L)]
    )


def _quadrature_terms(eq: ParsedEquation):
    """Callable (L, x0, n) -> (per-coefficient 4x4 blocks, f_e) by n-point quadrature."""
    a_fn = {key: _along_x(eq.coeffs[key]) for key in COEFF_KEYS}
    f_fn = _along_x(eq.rhs)
    N_fn, N1_fn, N2_fn, _ = _hermite_numeric()

    def blocks(L, x0, n):
        nodes, weights = _leggauss(n)
        xi = 0.5 * L * (nodes + 1.0)  # element-local coordinate, 0 <= xi <= L
        weight = 0.5 * L * weights  # dxi = (L/2) dt maps [-1, 1] onto [0, L]
        x = x0 + xi  # the coefficients are written in the global x
        N = _shape_values(N_fn, xi, L)
        N1 = _shape_values(N1_fn, xi, L)
        N2 = _shape_values(N2_fn, xi, L)
        a = {key: a_fn[key](x) for key in COEFF_KEYS}
        terms = {
            # the same four integrands as _symbolic_element, sign for sign
            "v4": (N2 * (weight * a["v4"])) @ N2.T,
            "v2": -(N1 * (weight * a["v2"])) @ N1.T,
            "v1": (N * (weight * a["v1"])) @ N1.T,
            "v0": (N * (weight * a["v0"])) @ N.T,
        }
        return terms, N @ (weight * f_fn(x))

    return blocks


def _quadrature_element(eq: ParsedEquation, points=None):
    """Callable (L, x0) -> (k_e, f_e), integrated numerically and self-checked.

    Every element is integrated twice, at `points` and again at 2 * points, and
    the two are compared term by term. Gauss-Legendre converges very fast on a
    coefficient that is smooth across the element and slowly on one with a
    corner, a pole or an endpoint singularity inside it, so a term that has not
    settled by the doubled order was not accurate at the lower one either. That
    case raises, naming the coefficient: a stiffness matrix nobody can tell is
    inaccurate is worse than an error. The matrix returned is the `points` one,
    which is the order the result reports.

    The order defaults to QUADRATURE_POINTS read HERE, not bound as a default
    argument: a default argument freezes the module constant at import time,
    so raising or lowering it afterwards would change the order a result
    claims without changing the order that ran. See _integration_record.
    """
    points = QUADRATURE_POINTS if points is None else int(points)
    blocks = _quadrature_terms(eq)

    def evaluate(L, x0):
        L, x0 = float(L), float(x0)
        coarse_k, coarse_f = blocks(L, x0, points)
        fine_k, fine_f = blocks(L, x0, 2 * points)

        for key in (*COEFF_KEYS, "rhs"):
            coarse = coarse_f if key == "rhs" else coarse_k[key]
            fine = fine_f if key == "rhs" else fine_k[key]
            scale = float(max(np.max(np.abs(coarse)), np.max(np.abs(fine))))
            if scale == 0.0:  # the term is absent, so there is nothing to settle
                continue
            # written so a nan difference fails the test rather than passing it
            error = float(np.max(np.abs(fine - coarse)) / scale)
            if not error <= QUADRATURE_TOL:
                raise EquationError(
                    f"{_FIELD_NAMES[key]} cannot be integrated reliably over the "
                    f"element of length {L:g} starting at x = {x0:g}: a "
                    f"{points}-point Gauss-Legendre rule and a {2 * points}-point "
                    f"rule differ by {error:.3e} relative, against a required "
                    f"{QUADRATURE_TOL:g}. Fixed-order quadrature converges only "
                    "on a coefficient that is smooth across the element, so this "
                    "one most likely has a corner, a pole or an endpoint "
                    "singularity inside it. Refine the mesh so that feature lands "
                    "on a node, or write the coefficient as a smoother expression "
                    "in x."
                )

        k_e = np.zeros((4, 4))
        for term in coarse_k.values():
            k_e += term
        return k_e, coarse_f

    return evaluate


# ----------------------------------------------------------- choosing between

_INTEGRATION_CACHE = {}  # eq.key -> (method, reason); the ORDER is never cached

_SYMBOLIC_REASON = (
    "every coefficient and the right-hand side is a polynomial in x, "
    "so sympy integrates each element entry exactly"
)


def _integration_choice(eq: ParsedEquation) -> tuple:
    """(method, reason) for this equation's element integrals. Decided once.

    The choice is structural — see the module docstring. A polynomial spec is
    integrated symbolically; anything else goes straight to quadrature without
    sympy being asked for an integral it may never return from. A symbolic
    attempt that raises for ANY reason falls back to quadrature too, so a sympy
    internal error like 'Non-suitable parameters' cannot reach the caller.
    """
    if eq.key in _INTEGRATION_CACHE:
        return _INTEGRATION_CACHE[eq.key]

    loose = _non_polynomial(eq)
    if loose:
        choice = (
            "quadrature",
            f"{loose[0]} is not polynomial in x, so the element integrals have no "
            "closed form sympy can be relied on to finish",
        )
    elif eq.free_symbols():
        # Symbolic work: the spec still carries unresolved parameters, so there
        # is nothing to evaluate numerically and nothing to solve either (see
        # _require_numeric). Only element_matrices gets this far.
        choice = ("symbolic", _SYMBOLIC_REASON)
    else:
        try:
            _numeric_element(eq)(1.0, 0.0)  # flush out a printer or eval error now
        except Exception as exc:  # noqa: BLE001 - any failure at all means quadrature
            choice = (
                "quadrature",
                "every coefficient is polynomial in x, but symbolic integration "
                f"failed ({type(exc).__name__}: {str(exc)[:160]}), so the element "
                "integrals fall back to quadrature",
            )
        else:
            choice = ("symbolic", _SYMBOLIC_REASON)

    _INTEGRATION_CACHE[eq.key] = choice
    return choice


def _integration_record(eq: ParsedEquation) -> dict:
    """The record a result carries: {"method", "points", "reason"}.

    A fresh dict every call, and the quadrature ORDER is read now rather than
    cached with the choice, because the record is a claim about what will run:
    _quadrature_element reads the same constant at the same moment, so the two
    cannot drift apart even if QUADRATURE_POINTS is changed between solves.
    """
    method, reason = _integration_choice(eq)
    return {
        "method": method,
        "points": None if method == "symbolic" else QUADRATURE_POINTS,
        "reason": reason,
    }


def _element_evaluator(eq: ParsedEquation):
    """The callable (L, x0) -> (k_e, f_e) this equation's assembly uses.

    The one seam every assembly goes through, whichever method was chosen, and
    so the one place to bug in order to test that a wrong element matrix is
    caught (evals/test_mms.py does exactly that). Nothing is cached here: both
    branches cache what is expensive — the symbolic integration and the
    lambdified matrices — and a cached callable would quietly outlive a test's
    monkeypatch.
    """
    if _integration_record(eq)["method"] == "symbolic":
        return _numeric_element(eq)
    return _quadrature_element(eq)


def _element_rule(eq: ParsedEquation):
    """(evaluate(L, x0) -> (k_e, f_e), integration record) for this equation."""
    return _element_evaluator(eq), _integration_record(eq)


def integration_plan(spec: dict) -> dict:
    """How this equation's element integrals are computed, and why.

    The same decision solve_equation_beam makes and records, exposed so a
    derivation document or a report can state the method without solving. The
    first call for a polynomial spec pays for the symbolic integration; every
    call after that is free, because the choice is cached per equation.

    Args:
        spec: equation spec dict (or an already-parsed ParsedEquation).

    Returns:
        dict with "method" ("symbolic" or "quadrature"), "points" (the
        Gauss-Legendre order, or None when symbolic) and "reason", a sentence
        saying why that method was chosen.
    """
    return dict(_element_rule(parse_spec(spec))[1])


def element_matrices(spec: dict, L, x0=0):
    """Element stiffness and load vector for one element.

    Integrated symbolically for a polynomial spec, and by Gauss-Legendre
    quadrature otherwise — in which case L and x0 must both be numbers, since
    a numerical rule has nothing to say about a symbolic element length.

    Args:
        spec: equation spec dict (or an already-parsed ParsedEquation).
        L: element length, a number or a sympy Symbol for symbolic work.
        x0: global x at the element's left node; only matters when a
            coefficient or the right-hand side depends on x.

    Returns:
        (k_e, f_e): a 4x4 and a 4x1 sympy Matrix over the DOFs
        [v_i, slope_i, v_j, slope_j].

    Raises:
        EquationError: the spec is malformed, or it needs quadrature while L
            or x0 is symbolic.
    """
    eq = parse_spec(spec)
    plan = _element_rule(eq)[1]
    if plan["method"] == "symbolic":
        k_e, f_e = _symbolic_element(eq)
        subs = {_EL: sp.sympify(L), _X0: sp.sympify(x0)}
        return k_e.subs(subs), f_e.subs(subs)

    length, left = sp.sympify(L), sp.sympify(x0)
    if not (length.is_number and left.is_number):
        raise EquationError(
            f"this equation is integrated by {plan['points']}-point quadrature "
            f"({plan['reason']}), which needs a numeric element length and "
            f"position; got L = {L} and x0 = {x0}. Element matrices symbolic in "
            "the element length exist only for a spec that is polynomial in x."
        )
    k_e, f_e = _element_rule(eq)[0](float(length), float(left))
    return sp.Matrix(k_e), sp.Matrix(f_e.reshape(4, 1))


def _elements(model):
    """Element list, defaulting to consecutive-node connectivity (as solver.py)."""
    elems = model.get("elements")
    if elems:
        return elems
    nodes = model["nodes"]
    if len(nodes) < 2:
        raise EquationError(
            f"the model has {len(nodes)} node(s); a beam needs at least two, "
            "so that there is one element to integrate over"
        )
    return [
        {"id": f"E{k + 1}", "i": nodes[k]["id"], "j": nodes[k + 1]["id"]}
        for k in range(len(nodes) - 1)
    ]


def _dof_indices(e, dof_map):
    """The four global DOF indices of one element, in Hermite order."""
    return [
        dof_map[(e["i"], "v")],
        dof_map[(e["i"], "slope")],
        dof_map[(e["j"], "v")],
        dof_map[(e["j"], "slope")],
    ]


def _require_numeric(eq: ParsedEquation):
    loose = sorted(s.name for s in eq.free_symbols())
    if loose:
        raise EquationError(f"cannot solve: {loose} have no numeric value in params")


def assemble_equation(model: dict, equation: dict):
    """Assemble the bending-only global system for one equation spec.

    The distributed load comes from the equation's rhs; any "distributed_loads"
    in the model are ignored. Only FY and MZ point loads act on these DOFs.

    Args:
        model: beam model dict (nodes, supports, point_loads; elements optional).
        equation: equation spec dict.

    Returns:
        (K, F, dof_map): K before supports are applied, F the load vector, and
        dof_map = {(node_id, "v" | "slope"): global index}.
    """
    eq = parse_spec(equation)
    _require_numeric(eq)

    nodes = model["nodes"]
    x_of = {n["id"]: float(n["x"]) for n in nodes}
    dof_map = {
        (n["id"], name): 2 * k + d
        for k, n in enumerate(nodes)
        for d, name in enumerate(("v", "slope"))
    }

    n_dof = 2 * len(nodes)
    K = np.zeros((n_dof, n_dof))
    F = np.zeros(n_dof)
    evaluate = _element_rule(eq)[0]

    for e in _elements(model):
        L = x_of[e["j"]] - x_of[e["i"]]
        if L <= 0:
            raise EquationError(f"element {e['id']!r} has non-positive length {L}")
        k_e, f_e = evaluate(L, x_of[e["i"]])
        idx = _dof_indices(e, dof_map)
        K[np.ix_(idx, idx)] += k_e
        F[idx] += f_e

    for pl in model.get("point_loads") or []:
        if (pl["node"], "v") not in dof_map:
            raise EquationError(f"point load references unknown node {pl['node']!r}")
        name = _DOF_OF_LOAD.get(pl["dof"])
        if name is None:
            if pl["value"]:
                raise EquationError(
                    f"point load {pl['dof']!r} on node {pl['node']!r} is outside the "
                    "bending DOFs this equation solver carries (FY, MZ)"
                )
            continue
        F[dof_map[(pl["node"], name)]] += float(pl["value"])

    return K, F, dof_map


def _restrained(model, dof_map):
    """Global indices held by the supports, via flag[1] = v and flag[5] = slope."""
    held = []
    for node, flags in (model.get("supports") or {}).items():
        if (node, "v") not in dof_map:
            raise EquationError(f"support references unknown node {node!r}")
        if len(flags) < 6:
            raise EquationError(f"support {node!r} needs 6 flags, got {len(flags)}")
        for name, k in _SUPPORT_FLAG.items():
            if flags[k]:
                held.append(dof_map[(node, name)])
    return sorted(held)


def _a4_callables(eq: ParsedEquation):
    """(a4(x), a4'(x)) as numpy callables, refusing a v4 that has a corner in it.

    Shear is (a4 v'')' , so reading it off the element interpolation needs a4
    to be differentiable. A coefficient built from Abs, Min, Max or sign is
    not: sympy leaves an unevaluated Derivative that lambdify cannot print, and
    the professor would get a printer traceback from deep inside sympy instead
    of a sentence about his equation.
    """
    a4 = eq.coeffs["v4"]
    try:
        return sp.lambdify(X, a4, "numpy"), sp.lambdify(X, sp.diff(a4, X), "numpy")
    except Exception as exc:
        raise EquationError(
            f"coefficient 'v4' = {a4} cannot be differentiated along the span, "
            "so the shear it implies cannot be evaluated. Functions with a "
            "corner in them - Abs, Min, Max, sign - are not usable in v4; "
            f"write the variation as a smooth expression in x instead ({exc})"
        ) from exc


def _samples(model, eq, u, dof_map):
    """Sample v, slope, moment and shear along each element's own cubic."""
    x_of = {n["id"]: float(n["x"]) for n in model["nodes"]}
    elems = sorted(_elements(model), key=lambda e: x_of[e["i"]])
    N, N1, N2, N3 = _hermite_numeric()
    a4_fn, da4_fn = _a4_callables(eq)

    n_per = max(3, math.ceil(20 / len(elems)) + 1)  # >= 21 samples in total
    points = []
    for e in elems:
        L = x_of[e["j"]] - x_of[e["i"]]
        d = u[_dof_indices(e, dof_map)]
        for s in np.linspace(0.0, L, n_per):
            x = x_of[e["i"]] + s
            v2 = float(np.dot(N2(s, L), d))
            v3 = float(np.dot(N3(s, L), d))
            EI = float(a4_fn(x))
            points.append(
                {
                    "x": float(x),
                    "v": float(np.dot(N(s, L), d)),
                    "slope": float(np.dot(N1(s, L), d)),
                    "moment": EI * v2,
                    "shear": float(da4_fn(x)) * v2 + EI * v3,
                }
            )
    return points


# Points per element for the dense scan that reports peak moment and shear.
# Deflection and slope are polynomials on the element, so their extrema are
# solved for exactly; moment and shear carry a4(x), which need not be a
# polynomial at all, so the only way to find their peak is to look. 51 points
# is 50 intervals: the scan brackets an extremum to within h/100, and a smooth
# field is flat at its peak, so the height it misses by falls as the square of
# the spacing - of order 1/2500 of the element's own O(h^2) moment error.
# Looking more finely than the interpolation is right would only cost time.
PEAK_SCAN_POINTS = 51

# A stationary point this far (relative to the element length) from a node is
# treated as being at the node. At a billionth of an element the height between
# the two differs by about 1e-18 of the deflection - a thousand times below the
# last bit of a double - so nothing is given up. See _stationary_points.
EDGE_MARGIN = 1e-9


def _stationary_points(coeffs, L):
    """Where a polynomial with these ascending coefficients is flat inside (0, L).

    A root within EDGE_MARGIN of an end IS that end as far as a double is
    concerned: the polynomial is flat there, so the two heights differ by less
    than an ulp, while the shape functions in their factored form are only good
    to a few ulp that close to a node. Both ends are candidates in their own
    right and are read off exactly, so dropping such a root loses nothing and
    keeps a peak that sits on a node equal to the nodal value bit for bit.
    """
    derivative = np.polyder(np.asarray(coeffs, dtype=float)[::-1])
    if not derivative.size or not np.any(derivative):
        return []
    edge = EDGE_MARGIN * L
    inside = []
    for root in np.roots(derivative):
        real = float(root.real)
        if abs(root.imag) <= 1e-9 * (1.0 + abs(real)) and edge < real < L - edge:
            inside.append(real)
    return inside


def _peaks(model, eq, u, dof_map):
    """The largest |v|, |slope|, |moment| and |shear| ANYWHERE on the solution.

    Not the largest sample. A peak that falls between two samples is invisible
    to the sample list, and refining the mesh does not rescue it: the samples
    keep landing at the same relative position inside each element, so the
    reported maximum stalls while the solution underneath it goes on
    converging.

    Deflection is a cubic on each element and slope is its derivative, so their
    extrema are located exactly, as the roots of a quadratic and of a line.
    Moment and shear carry a4(x), which need not be polynomial, so those two
    are scanned at PEAK_SCAN_POINTS per element instead. Every candidate is
    read back through the same shape functions the samples use, which is what
    keeps a peak sitting on a node equal to that node's value to the last bit.
    """
    x_of = {n["id"]: float(n["x"]) for n in model["nodes"]}
    elems = sorted(_elements(model), key=lambda e: x_of[e["i"]])
    N, N1, N2, N3 = _hermite_numeric()
    a4_fn, da4_fn = _a4_callables(eq)
    powers_of = _hermite_powers()

    peaks = {key: 0.0 for key in ("v", "slope", "moment", "shear")}
    for e in elems:
        L = x_of[e["j"]] - x_of[e["i"]]
        d = u[_dof_indices(e, dof_map)]
        cubic = np.asarray(powers_of(L), dtype=float) @ d  # ascending powers of xi
        slope_poly = np.polyder(cubic[::-1])[::-1]

        for key, basis, coeffs in (("v", N, cubic), ("slope", N1, slope_poly)):
            for s in (0.0, L, *_stationary_points(coeffs, L)):
                peaks[key] = max(peaks[key], abs(float(np.dot(basis(s, L), d))))

        for s in np.linspace(0.0, L, PEAK_SCAN_POINTS):
            x = x_of[e["i"]] + s
            v2 = float(np.dot(N2(s, L), d))
            v3 = float(np.dot(N3(s, L), d))
            EI = float(a4_fn(x))
            peaks["moment"] = max(peaks["moment"], abs(EI * v2))
            peaks["shear"] = max(peaks["shear"], abs(float(da4_fn(x)) * v2 + EI * v3))
    return peaks


def _restate(model, equation, displacements):
    """(eq, u, dof_map) for a caller-supplied set of nodal displacements."""
    eq = parse_spec(equation)
    _require_numeric(eq)
    nodes = model["nodes"]
    dof_map = {
        (n["id"], name): 2 * k + d
        for k, n in enumerate(nodes)
        for d, name in enumerate(("v", "slope"))
    }
    u = np.zeros(2 * len(nodes))
    for node, dofs in displacements.items():
        for name in ("v", "slope"):
            if (node, name) in dof_map:
                u[dof_map[(node, name)]] = float(dofs[name])
    return eq, u, dof_map


def peak_values(model: dict, equation: dict, displacements: dict) -> dict:
    """The extrema a given set of nodal displacements implies, without solving.

    The same four numbers solve_equation_beam reports as "max_abs", exposed so
    a caller holding a result can re-derive its headline figures and compare.
    Like sample_solution it re-derives rather than re-solves: the displacements
    are the input.

    Args:
        model: the beam model dict (nodes, elements, supports).
        equation: the equation spec dict, whose v4 sets moment and shear.
        displacements: {node_id: {"v", "slope"}}, as a result carries them.

    Returns:
        {"v", "slope", "moment", "shear"} -> the largest absolute value each
        reaches anywhere along the beam, which need not be at a node and need
        not be at any sampled point.

    Raises:
        EquationError: the spec is malformed or not fully numeric.
        KeyError: displacements is missing a node the model names.
    """
    return _peaks(model, *_restate(model, equation, displacements))


def sample_solution(model: dict, equation: dict, displacements: dict) -> list:
    """The samples a given set of nodal displacements implies, without solving.

    Same interpolation solve_equation_beam uses for its own samples, exposed so
    a caller holding a result can ask what its displacements imply and compare.
    It re-derives, it does not re-solve: the displacements are the input.

    Args:
        model: the beam model dict (nodes, elements, supports).
        equation: the equation spec dict, whose v4 sets moment and shear.
        displacements: {node_id: {"v", "slope"}}, as a result carries them.

    Returns:
        the same list of {"x", "v", "slope", "moment", "shear"} dicts.

    Raises:
        EquationError: the spec is malformed or not fully numeric.
        KeyError: displacements is missing a node the model names.
    """
    return _samples(model, *_restate(model, equation, displacements))


def _solve_free(Kff, Ff, eq: ParsedEquation):
    """Solve the free-DOF system, refusing anything that is not a stable equilibrium.

    The test is that the SYMMETRIC PART of Kff is positive definite, because
    u^T K u = u^T sym(K) u: that one condition is the energy minimum when the
    spec is self-adjoint (no a1 term) and the coercivity that Lax-Milgram asks
    for when it is not, so both halves of the equation family are held to it.
    A Cholesky factorization is therefore the physics test here, not merely a
    fast solver. It fails for an unrestrained mechanism, for an a4 that is zero
    or negative over part of the span, for a compressive a2 at or past this
    beam's buckling load, and for an a1 large enough to swamp the bending term.
    Every one of those still has a unique linear solution, and every one of them
    returns a deflection pointing the wrong way, so they are refused rather than
    reported: a number no one can tell is wrong is worse than an error.

    Refusing is right even though the system is solvable, because nothing
    downstream could catch it. Manufactured solutions verify the element
    matrices against a load MMS derives itself, and those matrices are correct
    here — it is the equation that has no stable answer.
    """
    if Kff.shape[0] == 0:
        return np.zeros(0)

    symmetric_part = (Kff + Kff.T) / 2
    try:
        factor = sla.cho_factor(symmetric_part, lower=True)
    except (np.linalg.LinAlgError, ValueError) as exc:
        eigenvalues = np.linalg.eigvalsh(symmetric_part)
        raise EquationError(
            "no stable equilibrium: the symmetric part of the free stiffness "
            f"matrix is not positive definite (smallest eigenvalue "
            f"{eigenvalues.min():.4e}, largest {eigenvalues.max():.4e}). The "
            "linear system may still have a unique solution, but it is an "
            "unstable equilibrium and the deflection points the wrong way. Usual "
            "causes: the supports leave a mechanism; coefficient v4 is zero or "
            "negative somewhere along the span; a compressive v2 is at or past "
            "the buckling load of this beam; or v1 is large enough to overwhelm "
            "the bending term."
        ) from exc

    if sp.simplify(eq.coeffs["v1"]) == 0:
        u = sla.cho_solve(factor, Ff)  # Kff is its own symmetric part
    else:
        try:
            u = np.linalg.solve(Kff, Ff)
        except np.linalg.LinAlgError as exc:
            raise EquationError("structure is unstable: singular stiffness matrix") from exc

    if not np.all(np.isfinite(u)):
        raise EquationError(
            "the solve produced non-finite displacements; the stiffness matrix is "
            "numerically degenerate for this equation and mesh"
        )
    return u


def solve_equation_beam(model: dict, equation: dict) -> dict:
    """Solve a beam model under a caller-supplied governing equation.

    The equation spec sets the ODE (see the module docstring); the model sets
    geometry, supports and point loads. The distributed load is the equation's
    rhs, so any "distributed_loads" in the model are ignored.

    Args:
        model: dict with "nodes" [{"id", "x"}], optional "elements"
            [{"id", "i", "j"}], "supports" {node_id: 6 flags, index 1 holding
            deflection and index 5 holding slope}, and "point_loads"
            [{"node", "dof": "FY" | "MZ", "value"}].
        equation: equation spec dict with coeffs v4/v2/v1/v0, rhs and params.

    Returns:
        dict with "displacements" {node_id: {"v", "slope"}}, "reactions" at
        restrained DOFs only {node_id: {"F", "M"}}, "samples" of at least 21
        points {"x", "v", "slope", "moment", "shear"}, "max_abs" — the largest
        absolute value each of those four reaches ANYWHERE along the beam,
        which is in general not one of the samples — and "integration",
        {"method", "points", "reason"}, saying whether the element entries are
        exact or numerical, and why.

    Raises:
        EquationError: the spec is malformed, the free system is not a stable
            equilibrium (see _solve_free), or a coefficient is too sharp for
            the quadrature rule to integrate reliably (see _quadrature_element).
    """
    eq = parse_spec(equation)
    integration = _element_rule(eq)[1]
    K, F, dof_map = assemble_equation(model, eq)

    held = set(_restrained(model, dof_map))
    free = [g for g in range(K.shape[0]) if g not in held]

    u = np.zeros(K.shape[0])
    u[free] = _solve_free(K[np.ix_(free, free)], F[free], eq)

    R = K @ u - F  # reactions recovered from the full, unmodified system

    displacements = {
        n["id"]: {
            "v": float(u[dof_map[(n["id"], "v")]]),
            "slope": float(u[dof_map[(n["id"], "slope")]]),
        }
        for n in model["nodes"]
    }
    reactions = {}
    for node, flags in (model.get("supports") or {}).items():
        out = {}
        if flags[_SUPPORT_FLAG["v"]]:
            out["F"] = float(R[dof_map[(node, "v")]])
        if flags[_SUPPORT_FLAG["slope"]]:
            out["M"] = float(R[dof_map[(node, "slope")]])
        if out:
            reactions[node] = out

    samples = _samples(model, eq, u, dof_map)
    max_abs = _peaks(model, eq, u, dof_map)

    return {
        "displacements": displacements,
        "reactions": reactions,
        "samples": samples,
        "max_abs": max_abs,
        # copied so a caller holding the result cannot edit the cached record
        "integration": dict(integration),
    }


def equilibrium_terms(model: dict, equation: dict, result: dict) -> dict:
    """The signed force terms whose sum is zero for any spec in the family.

    Taking w = 1 in the weak form kills the a4 and a2 terms (w'' = w' = 0) and
    leaves vertical equilibrium of the whole beam:

        support reactions + point loads + integral(f) - integral(a0 v + a1 v') = 0

    The last term is the load the foundation and the a1 term carry, which is why
    reactions alone do not balance the applied load once a0 or a1 is nonzero.

    Args:
        model: the beam model that was solved.
        equation: the equation spec that was solved.
        result: the dict returned by solve_equation_beam.

    Returns:
        dict of "reactions", "point_loads", "distributed" (integral of f) and
        "carried" (integral of a0 v + a1 v'), all in N, plus their "residual"
        and the "scale" it is judged against.
    """
    eq = parse_spec(equation)
    _require_numeric(eq)
    x_of = {n["id"]: float(n["x"]) for n in model["nodes"]}
    evaluate = _element_rule(eq)[0]

    distributed = 0.0
    carried = 0.0
    for e in _elements(model):
        L = x_of[e["j"]] - x_of[e["i"]]
        k_e, f_e = evaluate(L, x_of[e["i"]])
        d = np.array(
            [
                result["displacements"][e["i"]]["v"],
                result["displacements"][e["i"]]["slope"],
                result["displacements"][e["j"]]["v"],
                result["displacements"][e["j"]]["slope"],
            ]
        )
        # rows 0 and 2 sum to the w = 1 test function, since N1 + N3 == 1.
        distributed += float(f_e[0] + f_e[2])
        carried += float((k_e[0, :] + k_e[2, :]) @ d)

    reactions = float(sum(r.get("F", 0.0) for r in result["reactions"].values()))
    point_loads = float(
        sum(pl["value"] for pl in (model.get("point_loads") or []) if pl["dof"] == "FY")
    )

    terms = {
        "reactions": reactions,
        "point_loads": point_loads,
        "distributed": distributed,
        "carried": carried,
    }
    terms["scale"] = max(1.0, *(abs(v) for v in terms.values()))
    terms["residual"] = reactions + point_loads + distributed - carried
    return terms


def equilibrium_residual(model: dict, equation: dict, result: dict) -> float:
    """Dimensionless vertical-equilibrium residual, valid for any spec.

    Normalized by the largest force term so one gate tolerance (say 1e-9) works
    across the whole equation family. See equilibrium_terms for the breakdown.

    Args:
        model: the beam model that was solved.
        equation: the equation spec that was solved.
        result: the dict returned by solve_equation_beam.

    Returns:
        signed residual divided by the largest force term in the balance.
    """
    terms = equilibrium_terms(model, equation, result)
    return terms["residual"] / terms["scale"]


solve_equation = solve_equation_beam  # the name tools/fem/mms.py imports

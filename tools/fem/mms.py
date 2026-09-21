"""Method of manufactured solutions: verification for an equation nobody has seen.

A closed-form check only works for equations someone has already solved. MMS
works for any equation in the family: pick a smooth v*(x), push it through the
differential operator to get the load f(x) that makes v* the exact solution,
solve that load with the FEM on a sequence of meshes, and watch the nodal error
fall at the rate the elements promise. If the element matrices are wrong, the
discrete solution converges to something other than v* and the error stops
falling — which is what makes changing the equation safe.

Operator form. The weak form the elements integrate,

    B(w, v) = integral [ a4 w'' v'' - a2 w' v' + a1 w v' + a0 w v ] dx,

is the Galerkin statement of

    (a4 v'')'' + (a2 v')' + a1 v' + a0 v = f(x).

For constant coefficients that is exactly the spec's a4 v'''' + a2 v'' + a1 v'
+ a0 v. When a coefficient varies with x (a tapered beam, soil stiffness that
changes along the span) the two differ, and the divergence form above is both
what the element matrices discretize and the physically correct variable-EI
beam, so it is the residual used here.

This module derives the residual and parses the spec itself rather than reusing
tools/fem/equation.py, which it imports only to run a solve: a verifier that
shares its arithmetic with the code under test cannot catch that arithmetic
being wrong.

Units: SI. Deterministic and offline: sympy does the calculus, no LLM, no network.
"""

import math

import numpy as np
import sympy as sp

from tools.fem.safe_expr import ExpressionError, parse_expression

X = sp.Symbol("x")

_COEFF_KEYS = ("v4", "v2", "v1", "v0")
_AMPLITUDE = 0.01  # peak of v* in metres; errors are relative, so this only sets scale
_RATE_MIN = 2.5
# Finest relative nodal error. Deliberately loose: a wrong element matrix
# converges to the wrong function, which shows up as a collapsed rate and errors
# of 1e-2 and up, so this only has to exclude the grossly inaccurate.
_TOL = 1e-3
_EXACT = 1e-10  # below this the scheme is nodally exact and the rate is meaningless
_ZERO = 1e-9  # relative tolerance for "this boundary term vanishes"


def _parse(text, params):
    """Sympify a spec string in x with the parameters bound and substituted.

    Parsed by tools.fem.safe_expr, the same whitelist grammar the solver uses,
    so the verification cannot accept an expression the solver would refuse -
    nor the other way round. That whitelist is also why 'E' and 'I' are not
    sympy's Euler number and imaginary unit here: they stay free symbols until
    params supplies them, and any name params does not supply is reported as
    missing rather than silently turning the stiffness complex.
    """
    if isinstance(text, sp.Basic):
        expr = text
    else:
        local = {name: sp.Symbol(name) for name in params}
        local["x"] = X
        try:
            expr = parse_expression(text, local_dict=local)
        except ExpressionError as exc:
            raise ValueError(f"could not parse {text!r}: {exc}") from exc
    expr = sp.sympify(expr).subs(
        {
            sp.Symbol(name): _parse(value, {}) if isinstance(value, str) else value
            for name, value in params.items()
        }
    )
    unknown = expr.free_symbols - {X}
    if unknown:
        names = ", ".join(sorted(str(s) for s in unknown))
        raise ValueError(f"no value in spec['params'] for: {names}")
    return expr


def _coefficients(spec):
    """{'v4': expr, 'v2': expr, 'v1': expr, 'v0': expr} in x, params substituted."""
    coeffs = spec.get("coeffs") or {}
    if "v3" in coeffs:
        raise ValueError(
            "v3 is not supported: the Hermite weak form integrates v'''' by parts "
            "twice and v'' once, which leaves no place for a third derivative"
        )
    params = spec.get("params") or {}
    return {key: _parse(coeffs.get(key, "0"), params) for key in _COEFF_KEYS}


def _operator(coeffs, v_expr):
    """(a4 v'')'' + (a2 v')' + a1 v' + a0 v — see the module docstring."""
    a4, a2, a1, a0 = (coeffs[key] for key in _COEFF_KEYS)
    return (
        sp.diff(a4 * sp.diff(v_expr, X, 2), X, 2)
        + sp.diff(a2 * sp.diff(v_expr, X), X)
        + a1 * sp.diff(v_expr, X)
        + a0 * v_expr
    )


def manufactured_load(spec, v_expr):
    """The load f(x) that makes v_expr the exact solution of the spec's equation.

    Args:
        spec: equation spec dict (coeffs, params); its rhs is ignored.
        v_expr: the manufactured deflection v*(x), a sympy expression or a
            sympy-parseable string in x.

    Returns:
        A sympy expression in x with the spec's parameters substituted, ready to
        use as the spec's rhs.
    """
    params = spec.get("params") or {}
    return sp.simplify(_operator(_coefficients(spec), _parse(v_expr, params)))


def residual(spec, v_expr):
    """The spec's equation as a residual: operator applied to v_expr, minus rhs.

    Zero for every x exactly when v_expr solves the equation.
    """
    params = spec.get("params") or {}
    return _operator(_coefficients(spec), _parse(v_expr, params)) - _parse(
        spec.get("rhs", "0"), params
    )


def _span(model):
    xs = [float(node["x"]) for node in model["nodes"]]
    return min(xs), max(xs), max(xs) - min(xs)


def _held(model, dof):
    """Sorted x of the nodes whose DOF (1 = deflection, 5 = slope) is restrained."""
    x_of = {node["id"]: float(node["x"]) for node in model["nodes"]}
    return sorted(x_of[nid] for nid, flags in model["supports"].items() if flags[dof])


def choose_manufactured_solution(model):
    """Pick a v*(x) that satisfies this model's support conditions.

    The manufactured solution has to be admissible for the model's supports: it
    must vanish where the deflection is held and have zero slope where the slope
    is held, and at a free end it must leave the natural (moment, shear)
    boundary terms zero, because the weak form sets those to zero.

    Args:
        model: the beam model dict (nodes, supports).

    Returns:
        (v_expr, description) — the sympy expression and the case it was chosen
        for, e.g. "simply supported".
    """
    x0, xL, L = _span(model)
    if L <= 0:
        raise ValueError("model has zero span")
    s = (X - x0) / L
    held_v, held_slope = _held(model, 1), _held(model, 5)
    ends = [x0, xL]
    amp = _AMPLITUDE

    # Descriptions are in s = (x - x0)/L. A free end needs v'' and v''' to vanish
    # there, which is why the cantilever shapes carry a fourth power.
    if held_v == ends and not held_slope:
        return amp * sp.sin(sp.pi * s), "simply supported (v* = sin(pi s))"
    if held_v == ends and held_slope == ends:
        return amp * sp.sin(sp.pi * s) ** 2, "clamped-clamped (v* = sin^2(pi s))"
    if held_v == [x0] and held_slope == [x0]:
        return amp * s**2 * (1 - s) ** 4, "cantilever fixed at the left end (v* = s^2 (1-s)^4)"
    if held_v == [xL] and held_slope == [xL]:
        return amp * (1 - s) ** 2 * s**4, "cantilever fixed at the right end (v* = (1-s)^2 s^4)"
    if held_v == ends and held_slope == [x0]:
        return (
            amp * s**2 * (1 - s) ** 3,
            "propped cantilever, fixed at the left end (v* = s^2 (1-s)^3)",
        )
    if held_v == ends and held_slope == [xL]:
        return (
            amp * (1 - s) ** 2 * s**3,
            "propped cantilever, fixed at the right end (v* = (1-s)^2 s^3)",
        )

    raise ValueError(
        "no built-in manufactured solution for supports with deflection held at "
        f"{held_v} and slope held at {held_slope}; pass v_expr explicitly "
        "(it must vanish where the deflection is held, have zero slope where the "
        "slope is held, and leave the natural boundary terms zero at free ends)"
    )


def _sample(expr, x0, xL, count=201):
    fn = sp.lambdify(X, expr, "math")
    return [float(fn(x0 + (xL - x0) * k / (count - 1))) for k in range(count)]


def _peak(expr, x0, xL):
    return max(abs(value) for value in _sample(expr, x0, xL))


def _check_admissible(model, coeffs, v_expr):
    """Raise unless v* meets the model's essential and natural conditions."""
    x0, xL, _ = _span(model)
    slope = sp.diff(v_expr, X)
    moment = coeffs["v4"] * sp.diff(v_expr, X, 2)
    shear = sp.diff(moment, X) + coeffs["v2"] * slope

    for label, expr, held in (
        ("deflection", v_expr, _held(model, 1)),
        ("slope", slope, _held(model, 5)),
    ):
        scale = max(_peak(expr, x0, xL), 1e-300)
        for x_s in held:
            if abs(float(expr.subs(X, x_s))) > _ZERO * scale:
                raise ValueError(
                    f"manufactured v* has non-zero {label} at x = {x_s}, where the "
                    "model restrains it"
                )

    for label, expr, free_at in (
        ("moment", moment, [x for x in (x0, xL) if x not in _held(model, 5)]),
        ("shear", shear, [x for x in (x0, xL) if x not in _held(model, 1)]),
    ):
        scale = max(_peak(expr, x0, xL), 1e-300)
        for x_s in free_at:
            if abs(float(expr.subs(X, x_s))) > _ZERO * scale:
                raise ValueError(
                    f"manufactured v* leaves a non-zero {label} at the free end "
                    f"x = {x_s}; the weak form sets that natural boundary term to zero"
                )


def _refined_model(model, n_elements):
    """The same span and supports on a uniform mesh of n_elements, loads dropped.

    The manufactured load is the only forcing, so point and distributed loads
    from the original model are left out.
    """
    x0, _, L = _span(model)
    xs = [x0 + L * k / n_elements for k in range(n_elements + 1)]
    x_of = {node["id"]: float(node["x"]) for node in model["nodes"]}

    supports = {}
    for nid, flags in model["supports"].items():
        k = round((x_of[nid] - x0) / L * n_elements)
        if abs(xs[k] - x_of[nid]) > 1e-9 * L:
            raise ValueError(
                f"support at x = {x_of[nid]} does not land on a node of the "
                f"{n_elements}-element mesh; choose refinements that keep it on a node"
            )
        supports[f"N{k}"] = list(flags)

    refined = {
        key: value
        for key, value in model.items()
        if key not in ("nodes", "elements", "supports", "point_loads", "distributed_loads")
    }
    refined.update(
        nodes=[{"id": f"N{k}", "x": x} for k, x in enumerate(xs)],
        supports=supports,
        point_loads=[],
        distributed_loads=[],
    )
    return refined


def _solve(model, spec):
    """Solve one mesh with the equation solver (imported here, not at module load)."""
    from tools.fem import equation

    return equation.solve_equation_beam(model, spec)


def _observed_order(refinements, errors):
    """Least-squares slope of log(error) against log(mesh count), sign flipped."""
    points = [(n, e) for n, e in zip(refinements, errors) if e > 0 and math.isfinite(e)]
    if len(points) < 2:
        return 0.0  # nothing to fit; the exactness branch decides the verdict
    logs_n = [math.log(n) for n, _ in points]
    logs_e = [math.log(e) for _, e in points]
    return float(-np.polyfit(logs_n, logs_e, 1)[0])


def mms_check(model, spec, v_expr=None, refinements=(4, 8, 16)):
    """Verify the equation solver on this spec by the method of manufactured solutions.

    Picks a manufactured deflection v*(x) admissible for the model's supports
    (or takes the one given), derives the load that makes it exact, solves that
    load on each mesh, and measures the largest nodal deflection error relative
    to the peak of v*.

    Args:
        model: beam model dict; only its span and supports are used, since the
            manufactured load replaces every applied load.
        spec: equation spec dict (coeffs, params); its rhs is replaced.
        v_expr: optional manufactured v*(x); chosen from the supports if omitted.
        refinements: element counts to solve, increasing.

    Returns:
        dict with passed (bool), errors (relative nodal error per mesh), rate
        (observed order of convergence), refinements, v_star (the manufactured
        solution as a string) and detail (which v* was used and the pass rule).
    """
    coeffs = _coefficients(spec)
    if v_expr is None:
        v_star, how = choose_manufactured_solution(model)
    else:
        v_star = _parse(v_expr, spec.get("params") or {})
        how = "caller-supplied v*"
    _check_admissible(model, coeffs, v_star)

    load = manufactured_load(spec, v_star)
    mms_spec = {**spec, "rhs": sp.sstr(load, full_prec=True)}

    x0, xL, _ = _span(model)
    scale = _peak(v_star, x0, xL)
    v_exact = sp.lambdify(X, v_star, "math")

    errors = []
    refused = ""
    for n in refinements:
        refined = _refined_model(model, n)
        try:
            result = _solve(refined, mms_spec)
        except Exception as exc:
            # The solver refusing a mesh (no stable equilibrium, a degenerate
            # matrix) is a failed verification, not a crash: MMS owes its caller
            # a verdict, and "could not be solved" is one.
            errors.append(math.inf)
            refused = refused or f"the solver refused the {n}-element mesh: {exc}"
            continue
        errors.append(
            max(
                abs(result["displacements"][node["id"]]["v"] - v_exact(node["x"]))
                for node in refined["nodes"]
            )
            / scale
        )

    rate = _observed_order(refinements, errors)
    nodally_exact = errors[-1] <= _EXACT
    passed = bool(nodally_exact or (rate >= _RATE_MIN and errors[-1] <= _TOL))

    if refused:
        verdict = refused
    elif nodally_exact:
        verdict = (
            "error is at machine precision, so this scheme is nodally exact for "
            "this equation and the convergence rate carries no information"
        )
    else:
        verdict = f"observed order {rate:.2f}, finest relative error {errors[-1]:.3e}"
    detail = (
        f"{how}; meshes {tuple(refinements)} elements; {verdict}. "
        f"PASS rule: finest relative nodal error <= {_EXACT:g}, or observed order "
        f">= {_RATE_MIN} (Hermite cubics are 4th order in nodal deflection) with "
        f"finest relative error <= {_TOL:g}."
    )

    return {
        "passed": passed,
        "errors": errors,
        "rate": rate,
        "refinements": list(refinements),
        "v_star": str(v_star),
        "detail": detail,
    }

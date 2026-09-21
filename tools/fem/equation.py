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
"""

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

COEFF_KEYS = ("v4", "v2", "v1", "v0")

X = sp.Symbol("x")  # the beam coordinate, the one name a spec may not rebind

# Underscored so a spec may carry params of its own named L or x0.
_EL = sp.Symbol("_L", positive=True)  # element length
_X0 = sp.Symbol("_x0", real=True)  # global x at the element's left node
_XI = sp.Symbol("_xi", real=True)  # local coordinate, 0 <= _xi <= _L

# Parsing namespace. sympy's default globals bind E to Euler's number and I to
# the imaginary unit, which would silently turn the professor's "E*I" into
# e*sqrt(-1); only these names are predefined, everything else becomes a Symbol.
_PARSE_GLOBALS = {
    name: getattr(sp, name)
    for name in (
        "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
        "exp", "log", "sqrt", "Abs", "Min", "Max", "sign", "Piecewise",
    )
}
_PARSE_GLOBALS["pi"] = sp.pi
# constructors the tokenizer's own transformations emit
_PARSE_GLOBALS.update(
    Symbol=sp.Symbol, Integer=sp.Integer, Float=sp.Float, Rational=sp.Rational, Function=sp.Function
)

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
    """Parse one spec expression in the restricted namespace."""
    if isinstance(text, sp.Basic):
        return text
    if isinstance(text, (int, float, np.integer, np.floating)):
        return sp.Float(float(text)) if not isinstance(text, int) else sp.Integer(text)
    if not isinstance(text, str):
        raise EquationError(f"{what} must be a number or a sympy-parseable string, got {text!r}")
    if "__" in text:
        raise EquationError(f"{what} contains '__', which is not allowed: {text!r}")
    try:
        return parse_expr(
            text,
            local_dict={"x": X},
            global_dict=dict(_PARSE_GLOBALS),
            transformations=standard_transformations,
        )
    except Exception as exc:
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
    if sp.simplify(done["v4"]) == 0:
        raise EquationError(
            "coefficient 'v4' is identically zero; the Hermite C1 element needs a "
            "fourth-order term. Drop to a lower-order formulation instead."
        )

    return ParsedEquation(
        label=str(spec.get("label") or "custom equation"),
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


def element_matrices(spec: dict, L, x0=0):
    """Element stiffness and load vector for one element, by sympy integration.

    Args:
        spec: equation spec dict (or an already-parsed ParsedEquation).
        L: element length, a number or a sympy Symbol for symbolic work.
        x0: global x at the element's left node; only matters when a
            coefficient or the right-hand side depends on x.

    Returns:
        (k_e, f_e): a 4x4 and a 4x1 sympy Matrix over the DOFs
        [v_i, slope_i, v_j, slope_j].
    """
    eq = parse_spec(spec)
    k_e, f_e = _symbolic_element(eq)
    subs = {_EL: sp.sympify(L), _X0: sp.sympify(x0)}
    return k_e.subs(subs), f_e.subs(subs)


def _elements(model):
    """Element list, defaulting to consecutive-node connectivity (as solver.py)."""
    elems = model.get("elements")
    if elems:
        return elems
    nodes = model["nodes"]
    return [
        {"id": f"E{k + 1}", "i": nodes[k]["id"], "j": nodes[k + 1]["id"]}
        for k in range(len(nodes) - 1)
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
    evaluate = _numeric_element(eq)

    for e in _elements(model):
        L = x_of[e["j"]] - x_of[e["i"]]
        if L <= 0:
            raise EquationError(f"element {e['id']!r} has non-positive length {L}")
        k_e, f_e = evaluate(L, x_of[e["i"]])
        idx = [dof_map[(e["i"], "v")], dof_map[(e["i"], "slope")],
               dof_map[(e["j"], "v")], dof_map[(e["j"], "slope")]]
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


def _samples(model, eq, u, dof_map):
    """Sample v, slope, moment and shear along each element's own cubic."""
    x_of = {n["id"]: float(n["x"]) for n in model["nodes"]}
    elems = sorted(_elements(model), key=lambda e: x_of[e["i"]])
    N, N1, N2, N3 = _hermite_numeric()

    a4 = eq.coeffs["v4"]
    a4_fn = sp.lambdify(X, a4, "numpy")
    da4_fn = sp.lambdify(X, sp.diff(a4, X), "numpy")

    n_per = max(3, math.ceil(20 / len(elems)) + 1)  # >= 21 samples in total
    points = []
    for e in elems:
        L = x_of[e["j"]] - x_of[e["i"]]
        idx = [dof_map[(e["i"], "v")], dof_map[(e["i"], "slope")],
               dof_map[(e["j"], "v")], dof_map[(e["j"], "slope")]]
        d = u[idx]
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
        points {"x", "v", "slope", "moment", "shear"}, and "max_abs" of each.
    """
    eq = parse_spec(equation)
    K, F, dof_map = assemble_equation(model, eq)

    held = set(_restrained(model, dof_map))
    free = [g for g in range(K.shape[0]) if g not in held]

    u = np.zeros(K.shape[0])
    try:
        u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    except np.linalg.LinAlgError as exc:
        raise EquationError("structure is unstable: singular stiffness matrix") from exc

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
    max_abs = {k: max(abs(p[k]) for p in samples) for k in ("v", "slope", "moment", "shear")}

    return {
        "displacements": displacements,
        "reactions": reactions,
        "samples": samples,
        "max_abs": max_abs,
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
    evaluate = _numeric_element(eq)

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

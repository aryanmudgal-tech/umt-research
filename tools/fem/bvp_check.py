"""Independent check for the equation path: solve the strong form by collocation.

tools/fem/equation.py solves the weak form with Hermite-cubic Galerkin
elements. This module solves the same equation a different way, with scipy's
solve_bvp (collocation on the strong form written as four first-order ODEs),
so the two agree only if both are right. The one thing they share is the spec
parser, parse_spec.

State y = [v, theta, m, s] with theta = v', m = a4 v'' and s = (a4 v'')':

    v'     = theta
    theta' = m / a4
    m'     = s
    s'     = f - (a2' theta + a2 m / a4) - a1 theta - a0 v

which is (a4 v'')'' + (a2 v')' + a1 v' + a0 v = f, the self-adjoint reading
the FEM solves. moment is m and shear is s, as in the FEM's samples.

The boundary conditions come from the two end nodes, and they are the natural
boundary conditions of the FEM's own weak form, so the sign conventions match
it exactly. With P the FY load and M the MZ load applied at that node:

    deflection restrained -> v = 0,  otherwise  s + a2 theta = P  (x = 0)
                                                s + a2 theta = -P (x = L)
    slope restrained      -> theta = 0, otherwise  m = -M  (x = 0)
                                                   m = M   (x = L)

and a restrained end's reaction is what that balance leaves over.

A support or a point load at an interior node needs an interior condition,
which solve_bvp does not take, so the check reports itself not applicable
rather than solving a different problem.
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_bvp

from tools.fem.equation import X, parse_spec

METHOD = "scipy solve_bvp: collocation on the strong form, independent of the Galerkin FEM"
_GRID = 4001  # points the peaks are read from


def _vectorized(expr):
    """expr(x) as a function returning an array the shape of x."""
    if X not in expr.free_symbols:
        value = float(expr)
        return lambda x: np.full(np.shape(x), value)
    fn = sp.lambdify(X, expr, modules="numpy")
    return lambda x: np.asarray(fn(x), dtype=float) * np.ones(np.shape(x))


def _end_loads(model, node):
    fy = sum(p["value"] for p in model.get("point_loads") or [] if p["node"] == node and p["dof"] == "FY")
    mz = sum(p["value"] for p in model.get("point_loads") or [] if p["node"] == node and p["dof"] == "MZ")
    return float(fy), float(mz)


def _not_applicable(reason):
    return {"method": METHOD, "applicable": False, "reason": reason}


def solve_equation_bvp(model: dict, equation: dict) -> dict:
    """Solve model under equation by collocation, independently of the FEM.

    Args:
        model: the dict solve_with_equation takes: "nodes", "supports" and
            optional "point_loads".
        equation: the equation spec solve_with_equation takes.

    Returns:
        {"applicable": False, "reason"} when an interior support or point load
        puts the model out of reach, else "converged", "message",
        "max_rms_residual", "deflection_at_nodes" {node_id: v}, "reactions"
        {node_id: {"F", "M"}} at restrained DOFs, "midspan" {"x", "v"} and
        "max_abs" {"v", "moment", "shear": {"value", "x"}} read from a
        4001-point grid.
    """
    eq = parse_spec(equation)
    nodes = sorted(model["nodes"], key=lambda n: n["x"])
    left, right = nodes[0]["id"], nodes[-1]["id"]
    x0, x1 = float(nodes[0]["x"]), float(nodes[-1]["x"])
    span = x1 - x0
    ends = {left, right}

    supports = model.get("supports") or {}
    inner = sorted(n for n, flags in supports.items() if n not in ends and (flags[1] or flags[5]))
    if inner:
        return _not_applicable(f"interior support at {inner}; this check handles supports at the two ends only")
    loaded = sorted({p["node"] for p in model.get("point_loads") or [] if p["node"] not in ends})
    if loaded:
        return _not_applicable(f"point load at interior node {loaded}; this check handles end loads only")

    a4, a2, a1, a0 = (_vectorized(eq.coeffs[k]) for k in ("v4", "v2", "v1", "v0"))
    f = _vectorized(eq.rhs)
    da2 = _vectorized(sp.diff(eq.coeffs["v2"], X))

    # Scale to order one, or solve_bvp's tolerance means nothing: v is
    # millimetres while m and s are tens of kilonewtons.
    grid = np.linspace(x0, x1, _GRID)
    P0, M0 = _end_loads(model, left)
    P1, M1 = _end_loads(model, right)
    F = float(np.max(np.abs(f(grid)))) * span + abs(P0) + abs(P1) + (abs(M0) + abs(M1)) / span or 1.0
    a4_ref = float(np.max(np.abs(a4(grid))))
    Vs = F * span**3 / a4_ref  # deflection scale
    # y = [v / Vs, theta / (Vs / span), m / (F span), s / F], with xi = (x - x0) / span

    def rhs(xi, y):
        x = x0 + span * xi
        A4, A2 = a4(x), a2(x)
        theta = Vs / span * y[1]
        m = F * span * y[2]
        v = Vs * y[0]
        ds = f(x) - (da2(x) + a1(x)) * theta - A2 * m / A4 - a0(x) * v
        return np.vstack([y[1], a4_ref * y[2] / A4, y[3], span / F * ds])

    def bc(ya, yb):
        out = []
        for y, P, M, flags, sign in (
            (ya, P0, M0, supports.get(left), 1.0),
            (yb, P1, M1, supports.get(right), -1.0),
        ):
            x = x0 if sign > 0 else x1
            A2 = float(a2(np.array([x]))[0])
            fixed_v = bool(flags and flags[1])
            fixed_t = bool(flags and flags[5])
            out.append(y[0] if fixed_v else y[3] + A2 * Vs / (F * span) * y[1] - sign * P / F)
            out.append(y[1] if fixed_t else y[2] + sign * M / (F * span))
        return np.array(out)

    xi = np.linspace(0.0, 1.0, 201)
    try:
        sol = solve_bvp(rhs, bc, xi, np.zeros((4, xi.size)), tol=1e-9, bc_tol=1e-12, max_nodes=200000)
    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError, FloatingPointError) as exc:
        return {"method": METHOD, "applicable": True, "converged": False, "message": f"{type(exc).__name__}: {exc}"}
    if sol.status != 0 or not np.all(np.isfinite(sol.y)):
        return {"method": METHOD, "applicable": True, "converged": False, "message": sol.message}

    def state(x):
        y = sol.sol((np.asarray(x, dtype=float) - x0) / span)
        return Vs * y[0], Vs / span * y[1], F * span * y[2], F * y[3]

    v, _theta, m, s = state(grid)
    reactions = {}
    for node, x, P, M, sign in ((left, x0, P0, M0, 1.0), (right, x1, P1, M1, -1.0)):
        flags = supports.get(node)
        if not flags or not (flags[1] or flags[5]):
            continue
        _v, tt, mm, ss = (float(c[0]) for c in state(np.array([x])))
        A2 = float(a2(np.array([x]))[0])
        out = {}
        if flags[1]:
            out["F"] = sign * (ss + A2 * tt) - P
        if flags[5]:
            out["M"] = -sign * mm - M
        reactions[node] = out

    def peak(values):
        i = int(np.argmax(np.abs(values)))
        return {"value": float(abs(values[i])), "x": float(grid[i])}

    mid = x0 + span / 2
    return {
        "method": METHOD,
        "applicable": True,
        "converged": True,
        "message": sol.message,
        "max_rms_residual": float(np.max(sol.rms_residuals)),
        "deflection_at_nodes": {n["id"]: float(state(np.array([n["x"]]))[0][0]) for n in nodes},
        "reactions": reactions,
        "midspan": {"x": mid, "v": float(state(np.array([mid]))[0][0])},
        "max_abs": {"v": peak(v), "moment": peak(m), "shear": peak(s)},
    }

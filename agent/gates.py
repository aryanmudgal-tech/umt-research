"""Deterministic hard gate: invariants + closed form + PyNite. No LLM here.

Every check reports a status, not just a boolean: "pass", "fail", or
"skipped" for a check that did not apply to this run. A skipped check is not
a pass — it is counted and rendered separately everywhere — but it does not
fail the gate either.

Two kinds of run are gated. A standard beam solved by tools/fem/solver.py is
checked against physics invariants, the textbook closed form, and PyNite. A
run under a caller-supplied governing equation is checked against vertical
equilibrium in the form that equation implies, stiffness symmetry, its own
support conditions, and the method of manufactured solutions — MMS being the
general replacement for a closed form nobody has derived for the professor's
new equation.
"""

import numpy as np
import sympy as sp

from tools.fem import mms
from tools.fem.analytical import closed_form
from tools.fem.equation import X, assemble_equation, equilibrium_terms, parse_spec
from tools.fem.invariants import check_invariants
from tools.fem.pynite_check import solve_with_pynite
from tools.fem.solver import assemble

REL_TOL = 1e-6
EQUILIBRIUM_TOL = 1e-9  # dimensionless; equation.equilibrium_residual is normalized

PASS, FAIL, SKIPPED = "pass", "fail", "skipped"


def check_status(check) -> str:
    """The status of a check dict, tolerating records written before statuses.

    Anything the harness reads back — an old trace, a verifier verdict — may
    carry only "passed"; read that as pass/fail and never as skipped.
    """
    status = (check or {}).get("status")
    if status in (PASS, FAIL, SKIPPED):
        return status
    return PASS if (check or {}).get("passed") else FAIL


def tally(checks) -> dict:
    """{"passed", "skipped", "failed", "total"} over a list of check dicts.

    Skipped checks are excluded from "passed": a check that did not run has
    not verified anything, and reporting it as a pass is how a gate lies.
    """
    statuses = [check_status(c) for c in (checks or [])]
    return {
        "passed": sum(1 for s in statuses if s == PASS),
        "skipped": sum(1 for s in statuses if s == SKIPPED),
        "failed": sum(1 for s in statuses if s == FAIL),
        "total": len(statuses),
    }


def _check(name, status, detail) -> dict:
    # "passed" stays for backwards compatibility: a skipped check must not
    # fail the gate, so it is truthy there while the tally reads "status".
    return {"name": name, "status": status, "passed": status != FAIL, "detail": detail}


def _verdict(name, ok, detail) -> dict:
    return _check(name, PASS if ok else FAIL, detail)


def _emit(type, title, /, **data):
    """Publish a gate event if a tracer is around; never fail because of one.

    The import is lazy so gates.py stays importable (and testable) on its own.
    """
    try:
        from agent.trace import tracer

        tracer.emit("gate", type, title, **data)
    except Exception:
        pass


# ------------------------------------------------------------ standard beam


def detect_ss_udl(model: dict):
    """Params {"L","E","I","q"} when the model is a simply supported beam under
    a single uniform full-span y load, else None."""
    nodes = sorted(model["nodes"], key=lambda n: n["x"])
    if len(nodes) < 2:
        return None
    left, right = nodes[0]["id"], nodes[-1]["id"]
    supported = {n for n, flags in model.get("supports", {}).items() if any(flags)}
    if supported != {left, right}:
        return None
    for node_id in (left, right):
        flags = model["supports"][node_id]
        # pinned/roller: uy held, no rotational restraint about y or z
        if not flags[1] or flags[4] or flags[5]:
            return None
    if model.get("point_loads"):
        return None
    dist = model.get("distributed_loads") or []
    if len(dist) != 1:
        return None
    dl = dist[0]
    if (
        dl["element"] != "all"
        or dl["direction"] != "y"
        or dl["w1"] != dl["w2"]
        or dl["w1"] == 0
    ):
        return None
    return {
        "L": nodes[-1]["x"] - nodes[0]["x"],
        "E": model["material"]["E"],
        "I": model["section"]["Iz"],
        "q": abs(dl["w1"]),
    }


def _closed_form_checks(model, result):
    case = detect_ss_udl(model)
    if not case:
        return [
            _check(
                "closed_form_reference",
                SKIPPED,
                "model does not match a known textbook case, so no closed form "
                "was compared; nothing was verified by this check",
            )
        ]
    ref = closed_form("ss_udl", **case)
    rows = []
    for name, fem, exact in (
        ("closed_form_midspan_deflection", result["max_abs"]["uy"], ref["max_deflection"]),
        ("closed_form_max_moment", result["max_abs"]["Mz"], ref["max_moment"]),
        ("closed_form_end_shear", result["max_abs"]["Vy"], ref["end_shear"]),
    ):
        rel = abs(abs(fem) - exact) / exact
        rows.append(
            _verdict(
                name,
                rel < REL_TOL,
                f"FEM {abs(fem):.6e} vs closed form {exact:.6e}, rel err {rel:.3e}",
            )
        )
    return rows


def _pynite_check(model, result):
    try:
        py = solve_with_pynite(model)
    except Exception as exc:
        return _verdict("pynite_agreement", False, f"PyNite failed to solve the model: {exc}")
    # tolerances scaled by category magnitude so exact-zero components compare sanely
    d_scale = max(
        (abs(v) for n in result["displacements"].values() for v in n.values()),
        default=0.0,
    ) or 1.0
    r_scale = max(
        (abs(v) for n in result["reactions"].values() for v in n.values()),
        default=0.0,
    ) or 1.0
    worst = 0.0
    for node_id, ours in result["displacements"].items():
        theirs = py["displacements"][node_id]
        for dof, value in ours.items():
            worst = max(worst, abs(value - theirs[dof]) / d_scale)
    for node_id, ours in result["reactions"].items():
        theirs = py["reactions"].get(node_id, {})
        for dof, value in ours.items():
            worst = max(worst, abs(value - theirs.get(dof, 0.0)) / r_scale)
    return _verdict(
        "pynite_agreement",
        worst < REL_TOL,
        f"worst scaled mismatch vs PyNite = {worst:.3e}",
    )


# --------------------------------------------------------- custom equation


def _n_elements(model) -> int:
    elems = model.get("elements")
    return len(elems) if elems else len(model["nodes"]) - 1


def reduces_to_euler_bernoulli(equation):
    """{"EI", "q"} when the spec is a prismatic beam under a uniform load, else None.

    That is the one equation for which a textbook closed form exists here: the
    v2, v1 and v0 terms absent, and neither the bending stiffness nor the load
    depending on x.
    """
    try:
        eq = parse_spec(equation)
    except Exception:  # a detector answers "not this case", it never raises
        return None
    if eq.free_symbols():
        return None
    for key in ("v2", "v1", "v0"):
        if sp.simplify(eq.coeffs[key]) != 0:
            return None
    a4, f = sp.simplify(eq.coeffs["v4"]), sp.simplify(eq.rhs)
    if X in a4.free_symbols or X in f.free_symbols:
        return None
    return {"EI": float(a4), "q": float(f)}


def detect_ss_udl_equation(model: dict, equation: dict):
    """Params {"L","EI","q"} when an equation run is the simply supported UDL case.

    Requires both halves: the spec has to reduce to Euler-Bernoulli under a
    uniform load, and the model has to be a beam held vertically at its two
    ends, free to rotate, with no point loads.
    """
    reduced = reduces_to_euler_bernoulli(equation)
    if reduced is None or reduced["q"] == 0:
        return None
    nodes = sorted(model["nodes"], key=lambda n: n["x"])
    if len(nodes) < 2:
        return None
    ends = {nodes[0]["id"], nodes[-1]["id"]}
    supports = model.get("supports") or {}
    held = {node for node, flags in supports.items() if flags[1] or flags[5]}
    if held != ends:
        return None
    if any(supports[node][5] for node in ends):
        return None
    if any(pl.get("value") for pl in (model.get("point_loads") or [])):
        return None
    return {"L": nodes[-1]["x"] - nodes[0]["x"], **reduced}


def _equation_equilibrium_check(model, equation, result):
    """Vertical equilibrium in the form the equation implies.

    Reactions alone do not balance the applied load once a0 or a1 is nonzero —
    a foundation carries most of it — so the balance comes from the weak form
    with w = 1; see tools.fem.equation.equilibrium_terms.
    """
    try:
        terms = equilibrium_terms(model, equation, result)
    except Exception as exc:
        return _verdict("equation_equilibrium", False, f"could not be evaluated: {exc}")
    residual = terms["residual"] / terms["scale"]
    detail = (
        f"reactions {terms['reactions']:.6e} + point loads {terms['point_loads']:.6e} "
        f"+ distributed {terms['distributed']:.6e} - carried by the equation "
        f"{terms['carried']:.6e} N; normalized residual {residual:.3e} "
        f"(tolerance {EQUILIBRIUM_TOL:g})"
    )
    return _verdict("equation_equilibrium", abs(residual) < EQUILIBRIUM_TOL, detail)


def _equation_symmetry_check(model, equation):
    """K = K^T, which only holds when the equation has no first-derivative term."""
    try:
        eq = parse_spec(equation)
        if sp.simplify(eq.coeffs["v1"]) != 0:
            return _check(
                "equation_stiffness_symmetry",
                SKIPPED,
                f"a1 = {eq.coeffs['v1']} is non-zero, and the weak form's a1*N^T*N' "
                "term is genuinely unsymmetric, so K is not expected to equal its "
                "transpose; symmetry was not checked",
            )
        K, _F, _dof_map = assemble_equation(model, eq)
    except Exception as exc:
        return _verdict(
            "equation_stiffness_symmetry", False, f"could not be evaluated: {exc}"
        )
    worst = float(np.max(np.abs(K - K.T)))
    scale = float(np.max(np.abs(K))) or 1.0
    rel = worst / scale
    return _verdict(
        "equation_stiffness_symmetry",
        rel < REL_TOL,
        f"max abs(K - K^T) = {worst:.3e}, scaled by max abs(K) = {scale:.3e}, "
        f"rel {rel:.3e}",
    )


def _equation_support_check(model, result):
    """Every restrained DOF is actually zero in the solved displacements."""
    displacements = result["displacements"]
    scales = {
        name: max((abs(d[name]) for d in displacements.values()), default=0.0) or 1.0
        for name in ("v", "slope")
    }
    worst, where = 0.0, "no supports declared"
    held = 0
    try:
        for node, flags in (model.get("supports") or {}).items():
            for name, flag in (("v", 1), ("slope", 5)):
                if not flags[flag]:
                    continue
                held += 1
                rel = abs(displacements[node][name]) / scales[name]
                if rel >= worst:
                    worst, where = rel, f"{name} at node {node}"
    except (KeyError, IndexError, TypeError) as exc:
        # a support naming a node the result does not carry is a real defect,
        # and it must arrive as a failed check rather than as a traceback
        return _verdict(
            "equation_support_conditions", False, f"could not be evaluated: {exc!r}"
        )
    if not held:
        return _check(
            "equation_support_conditions",
            SKIPPED,
            "the model restrains no bending DOF, so there is no support "
            "condition to check",
        )
    return _verdict(
        "equation_support_conditions",
        worst < REL_TOL,
        f"{held} restrained DOF(s); worst scaled residual movement {worst:.3e} "
        f"({where})",
    )


def _equation_mms_check(model, equation):
    """Manufactured solutions on this spec and this mesh: the general verification.

    A closed form only exists for equations someone has already solved. MMS
    manufactures an exact solution for whatever the professor wrote, so the
    element matrices are verified for the equation actually being run.
    """
    n = _n_elements(model)
    refinements = (n, 2 * n, 4 * n)
    try:
        out = mms.mms_check(model, equation, refinements=refinements)
    except ValueError as exc:
        return _check(
            "equation_mms",
            SKIPPED,
            f"no manufactured solution applies to this model: {exc}",
        )
    except Exception as exc:
        return _verdict("equation_mms", False, f"could not be evaluated: {exc}")
    errors = ", ".join(f"{e:.3e}" for e in out["errors"])
    return _verdict(
        "equation_mms",
        out["passed"],
        f"relative nodal errors [{errors}]; {out['detail']}",
    )


def _equation_closed_form_checks(model, equation, result):
    """The textbook comparison, but only where a textbook answer exists.

    Nodal deflections are exact for this element family, so the deflection is
    compared tightly. Moment and shear are sampled from each element's own
    cubic, whose third derivative is constant per element: that costs O(h^2)
    on moment and O(h) on shear, so those two are compared against the bound
    the mesh itself sets rather than pretending to machine precision.
    """
    case = detect_ss_udl_equation(model, equation)
    if case is None:
        return [
            _check(
                "closed_form_reference",
                SKIPPED,
                "this equation and model are not the simply supported uniform-load "
                "Euler-Bernoulli case, so no closed form exists to compare; "
                "verification is by manufactured solutions instead",
            )
        ]
    n = _n_elements(model)
    ref = closed_form("ss_udl", L=case["L"], E=case["EI"], I=1.0, q=abs(case["q"]))
    rows = []
    for name, fem, exact, tol, why in (
        (
            "closed_form_midspan_deflection",
            result["max_abs"]["v"],
            ref["max_deflection"],
            max(REL_TOL, 1.0 / n**4),
            "nodal deflections are exact for Hermite cubics",
        ),
        (
            "closed_form_max_moment",
            result["max_abs"]["moment"],
            ref["max_moment"],
            1.0 / n**2,
            f"moment is O(h^2) from the element cubic on {n} elements",
        ),
        (
            "closed_form_end_shear",
            result["max_abs"]["shear"],
            ref["end_shear"],
            1.5 / n,
            f"shear is O(h) from the element cubic on {n} elements",
        ),
    ):
        rel = abs(abs(fem) - exact) / exact
        rows.append(
            _verdict(
                name,
                rel < tol,
                f"FEM {abs(fem):.6e} vs closed form {exact:.6e}, rel err {rel:.3e} "
                f"against tolerance {tol:.3e} ({why})",
            )
        )
    return rows


def equation_checks(model: dict, equation: dict, result: dict) -> list:
    """Every deterministic check for a run under a caller-supplied equation."""
    checks = [
        _equation_equilibrium_check(model, equation, result),
        _equation_symmetry_check(model, equation),
        _equation_support_check(model, result),
        _equation_mms_check(model, equation),
    ]
    checks.extend(_equation_closed_form_checks(model, equation, result))
    return checks


# ---------------------------------------------------------------- the gate


def deterministic_gate(model_dict: dict, result: dict, equation: dict = None) -> dict:
    """Run every deterministic check; the gate fails if any single check fails.

    Args:
        model_dict: the model the orchestrator built.
        result: the solver result recorded for it.
        equation: the equation spec, when the run went through
            solve_with_equation; None for a standard solve_beam_3d run.

    Returns:
        {"passed": bool, "checks": [check dicts], "tally": counts}. Each check
        carries "status" ("pass" / "fail" / "skipped"), "passed" (False only
        for a failure) and "detail".
    """
    checks = []

    def record(check):
        checks.append(check)
        _emit("gate_check", f"{check['name']}: {check['status'].upper()}", **check)

    if equation is None:
        for name, check in check_invariants(model_dict, result, assemble).items():
            record(_verdict(f"invariant_{name}", check["passed"], check["detail"]))
        for check in _closed_form_checks(model_dict, result):
            record(check)
        record(_pynite_check(model_dict, result))
    else:
        for check in equation_checks(model_dict, equation, result):
            record(check)

    counts = tally(checks)
    gate = {
        "passed": counts["failed"] == 0,
        "checks": checks,
        "tally": counts,
    }
    _emit(
        "gate_result",
        f"deterministic gate: {'PASS' if gate['passed'] else 'FAIL'} "
        f"({counts['passed']}/{counts['total']})",
        passed=gate["passed"],
        n_passed=counts["passed"],
        n_skipped=counts["skipped"],
        n_failed=counts["failed"],
        n_total=counts["total"],
    )
    return gate

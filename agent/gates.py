"""Deterministic hard gate: invariants + closed form + PyNite. No LLM here."""

from tools.fem.analytical import closed_form
from tools.fem.invariants import check_invariants
from tools.fem.pynite_check import solve_with_pynite
from tools.fem.solver import assemble

REL_TOL = 1e-6


def _emit(type, title, /, **data):
    """Publish a gate event if a tracer is around; never fail because of one.

    The import is lazy so gates.py stays importable (and testable) on its own.
    """
    try:
        from agent.trace import tracer

        tracer.emit("gate", type, title, **data)
    except Exception:
        pass


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
            {
                "name": "closed_form_reference",
                "passed": True,
                "detail": "model does not match a known textbook case; skipped",
            }
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
            {
                "name": name,
                "passed": rel < REL_TOL,
                "detail": f"FEM {abs(fem):.6e} vs closed form {exact:.6e}, rel err {rel:.3e}",
            }
        )
    return rows


def _pynite_check(model, result):
    try:
        py = solve_with_pynite(model)
    except Exception as exc:
        return {
            "name": "pynite_agreement",
            "passed": False,
            "detail": f"PyNite failed to solve the model: {exc}",
        }
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
    return {
        "name": "pynite_agreement",
        "passed": worst < REL_TOL,
        "detail": f"worst scaled mismatch vs PyNite = {worst:.3e}",
    }


def deterministic_gate(model_dict: dict, result: dict) -> dict:
    """Run every deterministic check; the gate passes only if all of them do."""
    checks = []

    def record(check):
        checks.append(check)
        status = "PASS" if check["passed"] else "FAIL"
        _emit("gate_check", f"{check['name']}: {status}", **check)

    for name, check in check_invariants(model_dict, result, assemble).items():
        record(
            {"name": f"invariant_{name}", "passed": check["passed"], "detail": check["detail"]}
        )
    for check in _closed_form_checks(model_dict, result):
        record(check)
    record(_pynite_check(model_dict, result))

    gate = {"passed": all(c["passed"] for c in checks), "checks": checks}
    n_passed = sum(1 for c in checks if c["passed"])
    _emit(
        "gate_result",
        f"deterministic gate: {'PASS' if gate['passed'] else 'FAIL'} "
        f"({n_passed}/{len(checks)})",
        passed=gate["passed"],
        n_passed=n_passed,
        n_total=len(checks),
    )
    return gate

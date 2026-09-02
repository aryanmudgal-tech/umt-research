"""Physics invariants for solve results — cheap checks that catch real bugs.

check_invariants(model, result, assemble_fn) runs named checks and returns
{name: {"passed": bool, "detail": str}}. assemble_fn is the stiffness
assembler from tools.fem.solver.
"""

import copy

import numpy as np

_TRANSLATIONS = ["ux", "uy", "uz"]
_FORCE_AXES = {"FX": 0, "FY": 1, "FZ": 2}


def _extract_matrix(assembled):
    """Pull the global stiffness matrix out of whatever assemble_fn returns."""
    candidates = assembled if isinstance(assembled, (tuple, list)) else [assembled]
    for item in candidates:
        arr = np.asarray(item, dtype=float) if not isinstance(item, np.ndarray) else item
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] > 1:
            return arr
    raise TypeError("assemble_fn did not return a square matrix")


def _check_equilibrium(model, result):
    applied = result["total_applied"]
    scale = max(1.0, *(abs(applied[a]) for a in _FORCE_AXES))
    residuals = {}
    for axis in _FORCE_AXES:
        rxn_sum = sum(r.get(axis, 0.0) for r in result["reactions"].values())
        residuals[axis] = rxn_sum + applied[axis]
    worst = max(abs(v) for v in residuals.values())
    passed = worst / scale < 1e-6
    detail = ", ".join(f"{a}: {v:.3e}" for a, v in residuals.items())
    return {"passed": passed, "detail": f"residual sum(R)+applied per axis: {detail}"}


def _check_symmetry(model, assemble_fn):
    K = _extract_matrix(assemble_fn(model))
    kmax = np.abs(K).max()
    asym = np.abs(K - K.T).max() / kmax if kmax > 0 else 0.0
    return {
        "passed": asym < 1e-10,
        "detail": f"max|K - K^T| / max|K| = {asym:.3e}",
    }


def _check_nullspace(model, assemble_fn):
    free = copy.deepcopy(model)
    free["supports"] = {}
    K = _extract_matrix(assemble_fn(free))
    n_nodes = len(model["nodes"])
    if K.shape[0] != 6 * n_nodes:
        return {
            "passed": False,
            "detail": f"unconstrained K is {K.shape[0]}x{K.shape[0]}, expected {6 * n_nodes}",
        }
    kmax = np.abs(K).max()
    worst = 0.0
    for t in range(3):  # rigid translation along x, y, z
        v = np.zeros(K.shape[0])
        v[t::6] = 1.0
        worst = max(worst, np.abs(K @ v).max() / kmax)
    return {
        "passed": worst < 1e-9,
        "detail": f"max ||K v|| / max|K| over rigid translations = {worst:.3e}",
    }


def _check_supports(model, result):
    umax = max(
        (abs(v) for d in result["displacements"].values() for v in d.values()),
        default=0.0,
    )
    tol = 1e-9 * max(1.0, umax)
    dof_names = ["ux", "uy", "uz", "rx", "ry", "rz"]
    violations = []
    for node_id, restrained in model.get("supports", {}).items():
        disp = result["displacements"][node_id]
        for name, flag in zip(dof_names, restrained):
            if flag and abs(disp[name]) > tol:
                violations.append(f"{node_id}.{name}={disp[name]:.3e}")
    return {
        "passed": not violations,
        "detail": "restrained DOFs move: " + ", ".join(violations) if violations
        else "all restrained DOFs at zero",
    }


def _check_deflection_sign(model, result):
    # Only meaningful when every applied load acts in -y.
    downward, other = 0, 0
    for dl in model.get("distributed_loads", []):
        if dl["direction"] == "y" and dl["w1"] <= 0 and dl["w2"] <= 0:
            downward += 1 if (dl["w1"] < 0 or dl["w2"] < 0) else 0
        else:
            other += 1
    for pl in model.get("point_loads", []):
        if pl["dof"] == "FY" and pl["value"] <= 0:
            downward += 1 if pl["value"] < 0 else 0
        else:
            other += 1
    if other or downward == 0:
        return {"passed": True, "detail": "not applicable (loads are not all -y)"}

    uys = [d["uy"] for d in result["displacements"].values()]
    umax = max(abs(u) for u in uys)
    tol = 1e-9 * max(1.0, umax)
    passed = min(uys) < 0 and max(uys) <= tol
    return {
        "passed": passed,
        "detail": f"uy range [{min(uys):.3e}, {max(uys):.3e}] under all-downward load",
    }


def check_invariants(model: dict, result: dict, assemble_fn) -> dict:
    return {
        "equilibrium_forces": _check_equilibrium(model, result),
        "stiffness_symmetric": _check_symmetry(model, assemble_fn),
        "rigid_body_nullspace": _check_nullspace(model, assemble_fn),
        "supports_respected": _check_supports(model, result),
        "deflection_negative_under_downward_load": _check_deflection_sign(model, result),
    }

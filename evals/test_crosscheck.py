"""Cross-check tools.fem.solver against PyNite, plus physics invariants."""

import math

import pytest

from tools.fem import solver
from tools.fem.invariants import check_invariants
from tools.fem.pynite_check import solve_with_pynite

E = 30e9
G = 12.5e9
SECTION = {"A": 0.5, "Iy": 0.004, "Iz": 0.005, "J": 0.001}


def _nodes(xs):
    return [{"id": f"N{k}", "x": float(x)} for k, x in enumerate(xs)]


def ss_udl_model():
    # 25 m simply supported, UDL 30 kN/m down, node at midspan.
    n = 10  # elements per half-span not needed; keep midspan nodal
    xs = [25.0 * k / n for k in range(n + 1)]
    return {
        "nodes": _nodes(xs),
        "material": {"E": E, "G": G},
        "section": dict(SECTION),
        "supports": {
            "N0": [True, True, True, True, False, False],
            f"N{n}": [False, True, True, False, False, False],
        },
        "distributed_loads": [
            {"element": "all", "direction": "y", "w1": -30e3, "w2": -30e3}
        ],
        "point_loads": [],
    }


def cantilever_model():
    xs = [25.0 * k / 5 for k in range(6)]
    return {
        "nodes": _nodes(xs),
        "material": {"E": E, "G": G},
        "section": dict(SECTION),
        "supports": {"N0": [True] * 6},
        "distributed_loads": [],
        "point_loads": [{"node": "N5", "dof": "FY", "value": -50e3}],
    }


def _assert_close(a, b, rel, scale, what):
    # rel is applied against the category magnitude so exact zeros compare sanely
    assert abs(a - b) <= rel * max(scale, abs(a), abs(b)), (
        f"{what}: ours={a!r} pynite={b!r}"
    )


def _crosscheck(model, rel=1e-9):
    ours = solver.solve_beam_3d(model)
    ref = solve_with_pynite(model)
    dscale = max(abs(v) for d in ref["displacements"].values() for v in d.values())
    for nid, disp in ref["displacements"].items():
        for dof in ("ux", "uy", "uz", "rx", "ry", "rz"):
            _assert_close(
                ours["displacements"][nid][dof], disp[dof], rel, dscale, f"{nid}.{dof}"
            )
    assert set(ours["reactions"]) == set(ref["reactions"])
    rscale = max(abs(v) for r in ref["reactions"].values() for v in r.values())
    for nid, rxns in ref["reactions"].items():
        assert set(ours["reactions"][nid]) == set(rxns)
        for key, val in rxns.items():
            _assert_close(
                ours["reactions"][nid][key], val, rel, rscale, f"rxn {nid}.{key}"
            )
    return ours


def test_ss_udl_matches_pynite():
    model = ss_udl_model()
    ours = _crosscheck(model)
    # Midspan nodal deflection also matches the closed form.
    mid = ours["displacements"]["N5"]["uy"]
    closed = -5 * 30e3 * 25.0**4 / (384 * E * SECTION["Iz"])
    assert math.isclose(mid, closed, rel_tol=1e-9)


def test_cantilever_tip_load_matches_pynite():
    model = cantilever_model()
    ours = _crosscheck(model)
    tip = ours["displacements"]["N5"]["uy"]
    closed = -50e3 * 25.0**3 / (3 * E * SECTION["Iz"])
    assert math.isclose(tip, closed, rel_tol=1e-9)


@pytest.mark.parametrize("make_model", [ss_udl_model, cantilever_model])
def test_invariants_pass(make_model):
    model = make_model()
    result = solver.solve_beam_3d(model)
    checks = check_invariants(model, result, solver.assemble)
    failed = {k: v["detail"] for k, v in checks.items() if not v["passed"]}
    assert not failed, failed


def test_broken_model_is_caught():
    # Remove the right support: the beam becomes a mechanism.
    model = ss_udl_model()
    del model["supports"]["N10"]
    try:
        result = solver.solve_beam_3d(model)
    except Exception:
        return  # solver refused the singular system — caught.
    checks = check_invariants(model, result, solver.assemble)
    assert any(not v["passed"] for v in checks.values()), (
        "mechanism went undetected: " + str(checks)
    )

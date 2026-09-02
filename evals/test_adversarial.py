"""Adversarial checks of the FEM solver: cases the basic suite does not cover.

Attacks: linearly varying loads (not just uniform), off-center point loads,
pure-moment loading, simultaneous biaxial bending, diagram values at exact
stations, and element-targeted loads cross-checked against PyNite.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.fem.pynite_check import solve_with_pynite  # noqa: E402
from tools.fem.solver import solve_beam_3d  # noqa: E402

E = 30e9
G = E / 2.4
IZ = 0.005
IY = 0.002  # Iy != Iz so any y/z plane mix-up changes the numbers
SECTION = {"A": 1.0, "Iy": IY, "Iz": IZ, "J": 0.001}

SS_ENDS = lambda last: {  # noqa: E731
    "N0": [True, True, True, True, False, False],
    last: [False, True, True, False, False, False],
}


def make_model(xs, supports, dist=None, points=None, elements=None):
    model = {
        "nodes": [{"id": f"N{k}", "x": float(x)} for k, x in enumerate(xs)],
        "material": {"E": E, "G": G},
        "section": dict(SECTION),
        "supports": supports,
        "distributed_loads": dist or [],
        "point_loads": points or [],
    }
    if elements is not None:
        model["elements"] = elements
    return model


def diagram_at(res, x, tol=1e-9):
    """All diagram samples at station x (element-boundary stations appear twice)."""
    pts = [p for p in res["diagrams"] if abs(p["x"] - x) <= tol]
    assert pts, f"no diagram sample at x={x}"
    return pts


def test_ss_triangular_load_reactions_and_midspan():
    # w = 0 at N0 growing to W (downward) at the far support.
    L, W = 25.0, 40e3
    model = make_model(
        [0, L / 2, L],
        SS_ENDS("N2"),
        dist=[{"element": "all", "direction": "y", "w1": 0.0, "w2": -W}],
    )
    res = solve_beam_3d(model)

    assert res["reactions"]["N0"]["FY"] == pytest.approx(W * L / 6, rel=1e-9)
    assert res["reactions"]["N2"]["FY"] == pytest.approx(W * L / 3, rel=1e-9)
    assert res["total_applied"]["FY"] == pytest.approx(-W * L / 2, rel=1e-9)

    # exact midspan values: M = WL^2/16 (sagging +), v = -5WL^4/(768 EI)
    assert res["displacements"]["N1"]["uy"] == pytest.approx(
        -5 * W * L**4 / (768 * E * IZ), rel=1e-9
    )
    for p in diagram_at(res, L / 2):
        assert p["Mz"] == pytest.approx(W * L**2 / 16, rel=1e-6)
        assert p["Vy"] == pytest.approx(W * L / 6 - W * (L / 2) ** 2 / (2 * L), rel=1e-6)


def test_ss_triangular_load_max_deflection_vs_roark():
    # Roark: max deflection ~ 0.00652 W L^4 / EI at x ~ 0.519 L; a fine mesh
    # must put a node close enough to the peak to land within 0.5%.
    L, W = 25.0, 40e3
    n = 20
    model = make_model(
        [L * k / n for k in range(n + 1)],
        SS_ENDS(f"N{n}"),
        dist=[{"element": "all", "direction": "y", "w1": 0.0, "w2": -W}],
    )
    res = solve_beam_3d(model)
    assert res["max_abs"]["uy"] == pytest.approx(0.00652 * W * L**4 / (E * IZ), rel=5e-3)


def test_ss_point_load_quarter_span():
    L, P = 20.0, 80e3
    a, b = L / 4, 3 * L / 4
    model = make_model(
        [0, a, L],
        SS_ENDS("N2"),
        points=[{"node": "N1", "dof": "FY", "value": -P}],
    )
    res = solve_beam_3d(model)

    # deflection under the load (exact, superposition-free): P a^2 b^2 / (3 E I L)
    assert res["displacements"]["N1"]["uy"] == pytest.approx(
        -P * a**2 * b**2 / (3 * E * IZ * L), rel=1e-9
    )
    assert res["reactions"]["N0"]["FY"] == pytest.approx(P * b / L, rel=1e-9)
    assert res["reactions"]["N2"]["FY"] == pytest.approx(P * a / L, rel=1e-9)
    for p in diagram_at(res, a):
        assert p["Mz"] == pytest.approx(P * a * b / L, rel=1e-6)

    # independent engine agrees on every displacement
    ref = solve_with_pynite(model)
    scale = max(abs(v) for d in ref["displacements"].values() for v in d.values())
    for nid, disp in ref["displacements"].items():
        for dof, val in disp.items():
            ours = res["displacements"][nid][dof]
            assert abs(ours - val) <= 1e-9 * max(scale, abs(val)), f"{nid}.{dof}"


def test_cantilever_tip_moment_y_plane():
    # Two elements, pure tip moment about z: constant curvature M/EI.
    L, M = 10.0, 2e5
    model = make_model(
        [0, L / 2, L],
        {"N0": [True] * 6},
        points=[{"node": "N2", "dof": "MZ", "value": M}],
    )
    res = solve_beam_3d(model)

    assert res["displacements"]["N2"]["uy"] == pytest.approx(M * L**2 / (2 * E * IZ), rel=1e-9)
    assert res["displacements"]["N2"]["rz"] == pytest.approx(M * L / (E * IZ), rel=1e-9)
    assert res["displacements"]["N1"]["uy"] == pytest.approx(M * L**2 / (8 * E * IZ), rel=1e-9)
    assert res["reactions"]["N0"]["MZ"] == pytest.approx(-M, rel=1e-9)
    assert res["reactions"]["N0"]["FY"] == pytest.approx(0.0, abs=1e-6 * M / L)
    for p in res["diagrams"]:
        assert p["Mz"] == pytest.approx(M, rel=1e-9)
        assert p["Vy"] == pytest.approx(0.0, abs=1e-9 * M / L)


def test_cantilever_tip_moment_z_plane():
    # Tip moment about y: with theta_y = -uz', uz bends NEGATIVE while ry
    # grows positive -- the case a wrong bending-plane sign gets backwards.
    L, M = 10.0, 2e5
    model = make_model(
        [0, L / 2, L],
        {"N0": [True] * 6},
        points=[{"node": "N2", "dof": "MY", "value": M}],
    )
    res = solve_beam_3d(model)

    assert res["displacements"]["N2"]["uz"] == pytest.approx(-M * L**2 / (2 * E * IY), rel=1e-9)
    assert res["displacements"]["N2"]["ry"] == pytest.approx(M * L / (E * IY), rel=1e-9)
    assert res["reactions"]["N0"]["MY"] == pytest.approx(-M, rel=1e-9)


def test_ss_biaxial_distributed_loads_independent():
    # UDLs in y AND z at once on 3 elements; each plane must match its own
    # closed form with its own inertia (v(L/3) = 11 q L^4 / (972 E I)).
    L, qy, qz = 24.0, 30e3, 18e3
    model = make_model(
        [0, L / 3, 2 * L / 3, L],
        SS_ENDS("N3"),
        dist=[
            {"element": "all", "direction": "y", "w1": -qy, "w2": -qy},
            {"element": "all", "direction": "z", "w1": -qz, "w2": -qz},
        ],
    )
    res = solve_beam_3d(model)

    for nid in ("N1", "N2"):  # symmetric stations
        assert res["displacements"][nid]["uy"] == pytest.approx(
            -11 * qy * L**4 / (972 * E * IZ), rel=1e-9
        )
        assert res["displacements"][nid]["uz"] == pytest.approx(
            -11 * qz * L**4 / (972 * E * IY), rel=1e-9
        )
    for nid in ("N0", "N3"):
        assert res["reactions"][nid]["FY"] == pytest.approx(qy * L / 2, rel=1e-9)
        assert res["reactions"][nid]["FZ"] == pytest.approx(qz * L / 2, rel=1e-9)
    assert res["total_applied"]["FY"] == pytest.approx(-qy * L, rel=1e-9)
    assert res["total_applied"]["FZ"] == pytest.approx(-qz * L, rel=1e-9)

    # y-only load must leave the z plane untouched (no cross-coupling)
    model_y = make_model(
        [0, L / 3, 2 * L / 3, L],
        SS_ENDS("N3"),
        dist=[{"element": "all", "direction": "y", "w1": -qy, "w2": -qy}],
    )
    res_y = solve_beam_3d(model_y)
    for disp in res_y["displacements"].values():
        assert disp["uz"] == 0.0 and disp["ry"] == 0.0


def test_ss_udl_diagram_midspan_moment_and_shear_zero():
    L, q = 25.0, 30e3
    model = make_model(
        [0, L / 2, L],
        SS_ENDS("N2"),
        dist=[{"element": "all", "direction": "y", "w1": -q, "w2": -q}],
    )
    res = solve_beam_3d(model)

    for p in diagram_at(res, L / 2):
        assert p["Mz"] == pytest.approx(q * L**2 / 8, rel=1e-6)
        assert p["Vy"] == pytest.approx(0.0, abs=1e-6 * q * L / 2)
    for p in diagram_at(res, 0.0):
        assert p["Vy"] == pytest.approx(q * L / 2, rel=1e-6)
    for p in diagram_at(res, L):
        assert p["Vy"] == pytest.approx(-q * L / 2, rel=1e-6)
    # shear changes sign across midspan
    left = [p["Vy"] for p in res["diagrams"] if 0 < p["x"] < L / 2]
    right = [p["Vy"] for p in res["diagrams"] if L / 2 < p["x"] < L]
    assert all(v > 0 for v in left) and all(v < 0 for v in right)


def test_ss_udl_diagram_z_plane_signs():
    # Right-hand-rule convention: dMy/dx = -Vz, so downward z load gives
    # My = -qL^2/8 at midspan while |Vz| still peaks at qL/2 at the ends.
    L, q = 25.0, 30e3
    model = make_model(
        [0, L / 2, L],
        SS_ENDS("N2"),
        dist=[{"element": "all", "direction": "z", "w1": -q, "w2": -q}],
    )
    res = solve_beam_3d(model)

    for p in diagram_at(res, L / 2):
        assert p["My"] == pytest.approx(-q * L**2 / 8, rel=1e-6)
        assert p["Vz"] == pytest.approx(0.0, abs=1e-6 * q * L / 2)
    for p in diagram_at(res, 0.0):
        assert p["Vz"] == pytest.approx(q * L / 2, rel=1e-6)


def test_element_targeted_load_matches_pynite():
    # UDL on ONE interior element only, elements left to the auto-id default:
    # both engines must resolve the same element for the same id.
    L, q = 20.0, 25e3
    n = 4
    model = make_model(
        [L * k / n for k in range(n + 1)],
        SS_ENDS(f"N{n}"),
        dist=[{"element": "E2", "direction": "y", "w1": -q, "w2": -q}],
    )
    res = solve_beam_3d(model)
    ref = solve_with_pynite(model)

    scale = max(abs(v) for d in ref["displacements"].values() for v in d.values())
    assert scale > 0  # the load must actually land somewhere
    for nid, disp in ref["displacements"].items():
        for dof, val in disp.items():
            ours = res["displacements"][nid][dof]
            assert abs(ours - val) <= 1e-9 * max(scale, abs(val)), f"{nid}.{dof}"
    assert res["total_applied"]["FY"] == pytest.approx(-q * L / n, rel=1e-9)

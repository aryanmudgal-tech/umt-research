"""Machine-precision checks of the FEM solver against closed-form beam theory."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.fem.solver import assemble, solve_beam_3d  # noqa: E402

E = 30e9
G = E / 2.4
IZ = 0.005
IY = 0.002
SECTION = {"A": 1.0, "Iy": IY, "Iz": IZ, "J": 0.001}


def make_model(xs, supports, dist=None, points=None):
    return {
        "nodes": [{"id": f"N{k}", "x": float(x)} for k, x in enumerate(xs)],
        "material": {"E": E, "G": G},
        "section": dict(SECTION),
        "supports": supports,
        "distributed_loads": dist or [],
        "point_loads": points or [],
    }


def fixed():
    return [True] * 6


def test_ss_udl_y():
    L, q = 25.0, 30e3
    model = make_model(
        [0, L / 2, L],
        {
            "N0": [True, True, True, True, False, False],
            "N2": [False, True, True, False, False, False],
        },
        dist=[{"element": "all", "direction": "y", "w1": -q, "w2": -q}],
    )
    res = solve_beam_3d(model)

    uy_mid = res["displacements"]["N1"]["uy"]
    assert uy_mid == pytest.approx(-5 * q * L**4 / (384 * E * IZ), rel=1e-9)
    assert res["reactions"]["N0"]["FY"] == pytest.approx(q * L / 2, rel=1e-9)
    assert res["reactions"]["N2"]["FY"] == pytest.approx(q * L / 2, rel=1e-9)
    assert res["max_abs"]["Mz"] == pytest.approx(q * L**2 / 8, rel=1e-6)
    assert res["max_abs"]["Vy"] == pytest.approx(q * L / 2, rel=1e-9)
    assert res["total_applied"]["FY"] == pytest.approx(-q * L, rel=1e-9)
    assert len(res["diagrams"]) >= 21


def test_cantilever_tip_point_load_y():
    L, P = 10.0, 50e3
    model = make_model(
        np.linspace(0, L, 5),
        {"N0": fixed()},
        points=[{"node": "N4", "dof": "FY", "value": -P}],
    )
    res = solve_beam_3d(model)
    assert res["displacements"]["N4"]["uy"] == pytest.approx(-P * L**3 / (3 * E * IZ), rel=1e-9)


def test_cantilever_tip_point_load_z():
    # same beam loaded in z: catches the theta_y = -d(uz)/dx sign convention
    L, P = 10.0, 50e3
    model = make_model(
        np.linspace(0, L, 5),
        {"N0": fixed()},
        points=[{"node": "N4", "dof": "FZ", "value": -P}],
    )
    res = solve_beam_3d(model)
    assert res["displacements"]["N4"]["uz"] == pytest.approx(-P * L**3 / (3 * E * IY), rel=1e-9)


def test_cantilever_udl_z():
    # z-direction UDL: catches stiffness/load-vector sign inconsistency per plane
    L, q = 10.0, 20e3
    model = make_model(
        np.linspace(0, L, 5),
        {"N0": fixed()},
        dist=[{"element": "all", "direction": "z", "w1": -q, "w2": -q}],
    )
    res = solve_beam_3d(model)
    assert res["displacements"]["N4"]["uz"] == pytest.approx(-q * L**4 / (8 * E * IY), rel=1e-9)


def test_fixed_fixed_udl():
    L, q = 25.0, 30e3
    model = make_model(
        [0, L / 2, L],
        {"N0": fixed(), "N2": fixed()},
        dist=[{"element": "all", "direction": "y", "w1": -q, "w2": -q}],
    )
    res = solve_beam_3d(model)
    assert res["displacements"]["N1"]["uy"] == pytest.approx(-q * L**4 / (384 * E * IZ), rel=1e-9)
    assert res["max_abs"]["Mz"] == pytest.approx(q * L**2 / 12, rel=1e-6)


def test_axial_bar():
    L, P = 10.0, 1e6
    model = make_model(
        [0, L],
        {"N0": fixed()},
        points=[{"node": "N1", "dof": "FX", "value": P}],
    )
    res = solve_beam_3d(model)
    assert res["displacements"]["N1"]["ux"] == pytest.approx(P * L / (E * SECTION["A"]), rel=1e-9)
    assert res["max_abs"]["N"] == pytest.approx(P, rel=1e-9)


def test_torsion_shaft():
    L, T = 10.0, 5e4
    model = make_model(
        [0, L],
        {"N0": fixed()},
        points=[{"node": "N1", "dof": "MX", "value": T}],
    )
    res = solve_beam_3d(model)
    assert res["displacements"]["N1"]["rx"] == pytest.approx(T * L / (G * SECTION["J"]), rel=1e-9)
    assert res["max_abs"]["T"] == pytest.approx(T, rel=1e-9)


def test_mesh_independence_ss_udl():
    L, q = 25.0, 30e3
    supports_of = lambda last: {  # noqa: E731
        "N0": [True, True, True, True, False, False],
        last: [False, True, True, False, False, False],
    }
    dist = [{"element": "all", "direction": "y", "w1": -q, "w2": -q}]

    coarse = solve_beam_3d(make_model([0, L / 2, L], supports_of("N2"), dist=dist))
    fine = solve_beam_3d(make_model(np.linspace(0, L, 11), supports_of("N10"), dist=dist))

    uy_coarse = coarse["displacements"]["N1"]["uy"]
    uy_fine = fine["displacements"]["N5"]["uy"]
    assert uy_fine == pytest.approx(uy_coarse, rel=1e-9)
    assert uy_coarse == pytest.approx(-5 * q * L**4 / (384 * E * IZ), rel=1e-9)


def test_assemble_invariants():
    L, q = 25.0, 30e3
    model = make_model(
        [0, L / 2, L],
        {"N0": fixed()},
        dist=[{"element": "all", "direction": "y", "w1": -q, "w2": -q}],
    )
    K, F, dof_map = assemble(model)
    assert K.shape == (18, 18) and len(dof_map) == 18
    assert np.allclose(K, K.T)  # stiffness must be symmetric
    fy_rows = [dof_map[(f"N{k}", 1)] for k in range(3)]
    assert F[fy_rows].sum() == pytest.approx(-q * L, rel=1e-9)  # load conservation


def test_derivation_is_deterministic_markdown():
    from tools.fem.derivation import galerkin_derivation_markdown

    md = galerkin_derivation_markdown()
    assert isinstance(md, str)
    assert "$$" in md and "12" in md and "6 L" in md
    assert md == galerkin_derivation_markdown()

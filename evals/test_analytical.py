"""Spot-checks of the closed-form beam library against hand-computed values."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.fem.analytical import closed_form, list_cases

# Bridge-beam reference numbers: L=25 m, E=30 GPa, I=0.005 m^4, q=30 kN/m.
L, E, I, Q = 25.0, 30e9, 0.005, 30e3


def test_ss_udl_hand_values():
    r = closed_form("ss_udl", L=L, E=E, I=I, q=Q)
    assert r["max_deflection"] == pytest.approx(1.0172526041666667, rel=1e-12)
    assert r["max_moment"] == pytest.approx(2_343_750.0)
    assert r["end_shear"] == pytest.approx(375_000.0)
    assert r["reactions"]["R_left"] == pytest.approx(375_000.0)
    assert r["reactions"]["R_right"] == pytest.approx(375_000.0)
    assert r["location_of_max"] == pytest.approx(12.5)


def test_ss_point_center_hand_values():
    r = closed_form("ss_point_center", L=10.0, E=200e9, I=1e-4, P=100e3)
    assert r["max_deflection"] == pytest.approx(0.10416666666666667, rel=1e-12)
    assert r["max_moment"] == pytest.approx(250_000.0)
    assert r["end_shear"] == pytest.approx(50_000.0)
    assert r["reactions"]["R_left"] == pytest.approx(50_000.0)
    assert r["location_of_max"] == pytest.approx(5.0)


def test_cantilever_udl_hand_values():
    r = closed_form("cantilever_udl", L=5.0, E=200e9, I=2e-5, q=10e3)
    assert r["max_deflection"] == pytest.approx(0.1953125, rel=1e-12)
    assert r["max_moment"] == pytest.approx(125_000.0)
    assert r["end_shear"] == pytest.approx(50_000.0)
    assert r["end_moment"] == pytest.approx(125_000.0)
    assert r["reactions"]["V_fixed"] == pytest.approx(50_000.0)
    assert r["reactions"]["M_fixed"] == pytest.approx(125_000.0)
    assert r["location_of_max"] == pytest.approx(5.0)


def test_cantilever_tip_point_hand_values():
    r = closed_form("cantilever_tip_point", L=5.0, E=200e9, I=2e-5, P=20e3)
    assert r["max_deflection"] == pytest.approx(0.20833333333333334, rel=1e-12)
    assert r["max_moment"] == pytest.approx(100_000.0)
    assert r["end_shear"] == pytest.approx(20_000.0)
    assert r["end_moment"] == pytest.approx(100_000.0)
    assert r["location_of_max"] == pytest.approx(5.0)


def test_fixed_fixed_udl_hand_values():
    r = closed_form("fixed_fixed_udl", L=L, E=E, I=I, q=Q)
    # 1/5 of the simply supported deflection for the same beam and load.
    assert r["max_deflection"] == pytest.approx(1.0172526041666667 / 5, rel=1e-12)
    assert r["max_moment"] == pytest.approx(1_562_500.0)
    assert r["end_shear"] == pytest.approx(375_000.0)
    assert r["end_moment"] == pytest.approx(1_562_500.0)
    assert r["reactions"]["M_left"] == pytest.approx(1_562_500.0)
    assert r["location_of_max"] == pytest.approx(12.5)


def test_ss_udl_cross_consistency():
    r = closed_form("ss_udl", L=L, E=E, I=I, q=Q)
    assert r["max_moment"] * 8 / L**2 == pytest.approx(Q, rel=1e-12)


def test_list_cases_matches_registry():
    cases = list_cases()
    names = {c["case"] for c in cases}
    assert names == {
        "ss_udl",
        "ss_point_center",
        "cantilever_udl",
        "cantilever_tip_point",
        "fixed_fixed_udl",
    }
    for c in cases:
        assert "L" in c["params"] and "E" in c["params"] and "I" in c["params"]
        assert "max_deflection" in c["formulas"]
        assert all(isinstance(f, str) for f in c["formulas"].values())


def test_unknown_case_and_missing_params_raise():
    with pytest.raises(ValueError):
        closed_form("nope", L=1, E=1, I=1, q=1)
    with pytest.raises(ValueError):
        closed_form("ss_udl", L=1, E=1, I=1)  # q missing

"""The closed-form gate must judge a mesh by what that mesh can know.

A mesh with no node at midspan interpolates the peak inside an element, so its
deflection carries the mesh's own O(h^4) error. That is an honest answer. Held
to machine precision it would be rejected, and a gate that fails correct work
erodes trust exactly as much as one that passes wrong work. The tolerance must
still be far tighter than any real mistake.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.gates import deterministic_gate
from tools.fem.solver import solve_beam_3d

L, E, I, Q = 25.0, 30e9, 0.005, 30e3
EXACT = 5 * Q * L**4 / (384 * E * I)


def ss_udl(n_elem):
    xs = [L * k / n_elem for k in range(n_elem + 1)]
    return {
        "nodes": [{"id": f"N{i}", "x": x} for i, x in enumerate(xs)],
        "material": {"E": E, "G": E / 2.4},
        "section": {"A": 0.5, "Iy": I, "Iz": I, "J": 0.001},
        "supports": {
            "N0": [True, True, True, True, False, False],
            f"N{n_elem}": [False, True, True, False, False, False],
        },
        "distributed_loads": [{"element": "all", "direction": "y", "w1": -Q, "w2": -Q}],
        "point_loads": [],
    }


def gate_for(n_elem, tweak=None):
    model = ss_udl(n_elem)
    result = solve_beam_3d(model)
    if tweak:
        result["max_abs"]["uy"] *= tweak
    return deterministic_gate(model, result)


def deflection_check(gate):
    return next(c for c in gate["checks"] if c["name"] == "closed_form_midspan_deflection")


@pytest.mark.parametrize("n_elem", [5, 7, 11])
def test_a_mesh_without_a_node_at_midspan_is_not_failed_for_being_coarse(n_elem):
    gate = gate_for(n_elem)
    check = deflection_check(gate)

    assert check["passed"], check["detail"]
    assert gate["passed"]
    assert "no node at midspan" in check["detail"]


@pytest.mark.parametrize("n_elem", [4, 8, 12])
def test_a_mesh_with_a_node_at_midspan_is_still_held_to_machine_precision(n_elem):
    gate = gate_for(n_elem)
    result = solve_beam_3d(ss_udl(n_elem))

    assert abs(result["max_abs"]["uy"] - EXACT) / EXACT < 1e-6
    assert deflection_check(gate)["passed"]
    assert gate["passed"]


@pytest.mark.parametrize("n_elem", [4, 5, 11])
def test_the_looser_tolerance_still_catches_a_real_mistake(n_elem):
    """Slack for discretisation, none for a wrong answer."""
    for tweak in (1.01, 0.99, 1.05):
        gate = gate_for(n_elem, tweak=tweak)
        assert not deflection_check(gate)["passed"], (n_elem, tweak)
        assert not gate["passed"]


@pytest.mark.parametrize("n_elem", [3, 4, 5, 7, 8, 9, 11, 12])
def test_the_peak_moment_and_shear_are_exact_on_any_mesh(n_elem):
    """Unlike deflection, these are recovered exactly inside an element.

    A moment peaks where its shear crosses zero, which an evenly spaced grid
    hits only by luck, so this used to be O(h^2) wrong on a mesh with no node
    at midspan and failed the gate at 7 and 9 elements.
    """
    result = solve_beam_3d(ss_udl(n_elem))

    assert result["max_abs"]["Mz"] == pytest.approx(Q * L**2 / 8, rel=1e-11)
    assert result["max_abs"]["Vy"] == pytest.approx(Q * L / 2, rel=1e-11)


@pytest.mark.parametrize("n_elem", [3, 5, 7, 9, 11])
def test_every_mesh_passes_the_whole_gate(n_elem):
    assert gate_for(n_elem)["passed"]


def test_the_tolerance_tightens_as_the_mesh_refines():
    tolerances = []
    for n_elem in (5, 7, 11):
        detail = deflection_check(gate_for(n_elem))["detail"]
        tolerances.append(float(detail.split("tolerance")[1].split()[0].rstrip(",.")))

    assert tolerances == sorted(tolerances, reverse=True), tolerances
    assert max(tolerances) < 1e-2

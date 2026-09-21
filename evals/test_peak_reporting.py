"""The reported maximum must be the maximum of the solution, not of the printout.

Both solvers used to report, as "the maximum", the largest value among the
points they happened to SAMPLE. A peak between two samples was never seen, and
refining the mesh did not fix it: the samples keep landing at the same relative
position inside each element, so the headline number stalls while the solution
underneath it goes on converging. On the 25 m beam with a square-root taper the
reported maximum sat at 4.7e-04 relative error from 8 elements to 16 while the
solver's own nodal error fell from 2.8e-06 to 1.8e-07 - a professor refining
his mesh would watch the headline stand still and conclude, reasonably, that
nothing was converging.

What is pinned here:

- the taper's reported peak against an independent scipy solve_bvp reference
  that shares no code with the FEM, and that it now IMPROVES with refinement;
- an off-centre point load, whose true peak is at no node, against the textbook
  maximum - the case the old nodal-only rule got 19.5 % wrong at any mesh;
- a cantilever loaded in z, which is the plane whose theta_y = -d(uz)/dx sign
  convention a wrong interpolation would silently corrupt;
- the professor's own beam, whose peak IS at a node: it must be unchanged to
  the last bit, by both the old rule and the closed form;
- that a coarse mesh does not report a LARGER peak than a fine one, which is
  what an interpolation that overshoots between nodes would do.

Offline and deterministic: numpy and scipy only, no LLM, no network.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_bvp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.fem.equation import solve_equation_beam  # noqa: E402
from agent.run_phase1 import _results_rows  # noqa: E402
from tools.fem.solver import _quadratic_roots, solve_beam_3d  # noqa: E402

SPAN, E_VAL, I_VAL, Q_VAL = 25.0, 30e9, 0.005, -30e3

TAPER = {
    "label": "square-root taper",
    "coeffs": {"v4": "E*I0*sqrt(1 + x/L)", "v2": "0", "v1": "0", "v0": "0"},
    "rhs": "q",
    "params": {"E": E_VAL, "I0": I_VAL, "L": SPAN, "q": Q_VAL},
}

PRISMATIC = {
    "label": "Euler-Bernoulli beam",
    "coeffs": {"v4": "E*I", "v2": "0", "v1": "0", "v0": "0"},
    "rhs": "q",
    "params": {"E": E_VAL, "I": I_VAL, "q": Q_VAL},
}


def equation_beam(n_elem):
    """A simply supported 25 m beam meshed into n_elem equal elements."""
    return {
        "nodes": [{"id": f"N{k}", "x": SPAN * k / n_elem} for k in range(n_elem + 1)],
        "supports": {
            "N0": [True, True, True, True, False, False],
            f"N{n_elem}": [False, True, True, False, False, False],
        },
        "point_loads": [],
    }


# --------------------------------------------------------- the independent reference


def solve_bvp_peak(nodes=2001, dense=200_001):
    """max |v| for the tapered beam, from scipy.integrate.solve_bvp.

    Shares no code with the FEM: the self-adjoint operator (EI(x) v'')'' = q is
    written as four first-order equations in y = [v, v', M, V] with M = EI v''
    and V = M', and the simply supported end conditions are v = M = 0 at both
    ends. The peak is then read off a dense evaluation of the returned spline
    and sharpened by fitting a parabola through its three highest points, which
    a smooth maximum makes exact to well past the accuracy in question.
    """

    def stiffness(x):
        return E_VAL * I_VAL * np.sqrt(1.0 + x / SPAN)

    def ode(x, y):
        return np.vstack([y[1], y[2] / stiffness(x), y[3], np.full_like(x, Q_VAL)])

    def boundary(left, right):
        return np.array([left[0], left[2], right[0], right[2]])

    mesh = np.linspace(0.0, SPAN, nodes)
    solution = solve_bvp(
        ode, boundary, mesh, np.zeros((4, mesh.size)), tol=1e-10, max_nodes=200_000
    )
    assert solution.status == 0, f"reference BVP did not converge: {solution.message}"

    x = np.linspace(0.0, SPAN, dense)
    v = np.abs(solution.sol(x)[0])
    k = int(np.argmax(v))
    left, middle, right = v[k - 1], v[k], v[k + 1]
    step = x[1] - x[0]
    shift = step * (left - right) / (2.0 * (left - 2.0 * middle + right))
    return float(np.abs(solution.sol(np.array([x[k] + shift]))[0][0]))


@pytest.fixture(scope="module")
def taper_peak():
    return solve_bvp_peak()


def test_the_reference_is_the_beam_we_think_it_is(taper_peak):
    """A reference nobody has checked is not a reference."""
    # the same beam without the taper has a closed form, and the taper, being
    # stiffer everywhere than EI0, must deflect less than it
    prismatic = 5 * abs(Q_VAL) * SPAN**4 / (384 * E_VAL * I_VAL)
    assert 0.5 * prismatic < taper_peak < prismatic
    assert taper_peak == pytest.approx(0.8352334111, rel=1e-9)


# ------------------------------------------------------------------ the taper itself


def test_the_tapered_peak_matches_the_reference_and_keeps_improving(taper_peak):
    """The reported maximum used to stall at 4.7e-04; now it converges."""
    meshes = (4, 8, 16, 32)
    reported = [solve_equation_beam(equation_beam(n), TAPER)["max_abs"]["v"] for n in meshes]
    errors = [abs(value - taper_peak) / taper_peak for value in reported]

    # 8 elements: 4.68e-04 before, 7.1e-06 now. The floor left is the element's
    # OWN error - an 8-element Hermite mesh puts its nodal value 2.8e-06 from
    # the truth here and its interpolation between nodes rather further - so
    # what is being asserted is that the reporting no longer costs anything
    # next to the discretisation, not that a coarse mesh became exact.
    assert errors[1] < 1e-5

    # once the mesh can resolve the peak, the reported figure follows it down
    assert errors[3] < 1e-6

    for coarse, fine, n_coarse, n_fine in zip(errors, errors[1:], meshes, meshes[1:]):
        assert fine < coarse / 2.0, (
            f"refining {n_coarse} -> {n_fine} elements moved the reported peak "
            f"from {coarse:.3e} to {fine:.3e} relative error; it used to stall"
        )


def test_the_peak_is_no_longer_read_off_the_sample_list(taper_peak):
    """The samples stay the readable series; the maximum is no longer one of them."""
    result = solve_equation_beam(equation_beam(8), TAPER)
    sampled = max(abs(p["v"]) for p in result["samples"])

    assert len(result["samples"]) >= 21  # the series contract is untouched
    assert sampled < result["max_abs"]["v"]  # the peak falls between two samples
    assert abs(sampled - taper_peak) / taper_peak > 50 * abs(
        result["max_abs"]["v"] - taper_peak
    ) / taper_peak


def test_a_coarse_mesh_does_not_out_report_a_fine_one():
    """An interpolation that overshoots between nodes would fail this."""
    coarse = solve_equation_beam(equation_beam(2), TAPER)["max_abs"]["v"]
    fine = solve_equation_beam(equation_beam(64), TAPER)["max_abs"]["v"]

    # the 2-element mesh's own discretisation error is 7.6e-04 relative; any
    # excess beyond that is the interpolation inventing deflection
    assert coarse <= fine * (1 + 1e-3)
    assert coarse == pytest.approx(fine, rel=2e-3)


# ---------------------------------------------- the professor's beam, bit for bit


def test_the_professors_beam_is_unchanged_to_the_last_bit():
    """Its peak is exactly at a node, so the old rule and the new must agree."""
    result = solve_equation_beam(equation_beam(4), PRISMATIC)
    by_the_old_rule = max(abs(p["v"]) for p in result["samples"])

    assert result["max_abs"]["v"] == by_the_old_rule
    for key in ("slope", "moment", "shear"):
        assert result["max_abs"][key] == max(abs(p[key]) for p in result["samples"])
    assert result["max_abs"]["v"] == pytest.approx(
        5 * abs(Q_VAL) * SPAN**4 / (384 * E_VAL * I_VAL), rel=1e-12
    )


# ------------------------------------------------------------- the 3D solver's path

E_3D, IZ, IY = 30e9, 0.005, 0.002
SECTION = {"A": 1.0, "Iy": IY, "Iz": IZ, "J": 0.001}
PINNED = [True, True, True, True, False, False]
ROLLER = [False, True, True, False, False, False]


def beam_3d(xs, supports, dist=None, points=None):
    return {
        "nodes": [{"id": f"N{k}", "x": float(x)} for k, x in enumerate(xs)],
        "material": {"E": E_3D, "G": E_3D / 2.4},
        "section": dict(SECTION),
        "supports": supports,
        "distributed_loads": dist or [],
        "point_loads": points or [],
    }


def point_load_maximum(P, L, near, EI):
    """Textbook maximum deflection for one off-centre point load, and where it is.

    `near` is the distance from the load to the NEARER support, so L - near is
    the longer segment and the peak lies in it, at sqrt((L^2 - near^2)/3) from
    the FARTHER support. Writing it this way removes the usual ambiguity in the
    a/b lettering: the formula only holds for the shorter of the two distances.
    """
    return (
        P * near * (L**2 - near**2) ** 1.5 / (9 * math.sqrt(3) * EI * L),
        L - math.sqrt((L**2 - near**2) / 3.0),
    )


@pytest.mark.parametrize("load_at", [5.0, 15.0], ids=["load-left", "load-right"])
def test_an_off_centre_point_load_peaks_between_the_nodes(load_at):
    """The true peak is at no node, and the mesh cannot be refined into it."""
    L, P = 20.0, 80e3
    model = beam_3d(
        [0.0, load_at, L],
        {"N0": PINNED, "N2": ROLLER},
        points=[{"node": "N1", "dof": "FY", "value": -P}],
    )
    result = solve_beam_3d(model)

    exact, where = point_load_maximum(P, L, min(load_at, L - load_at), E_3D * IZ)
    nodal = max(abs(d["uy"]) for d in result["displacements"].values())

    assert min(model["nodes"][1]["x"], L - model["nodes"][1]["x"]) > 0  # a real offset
    assert not any(abs(n["x"] - where) < 1e-9 for n in model["nodes"])  # no node on it
    assert result["max_abs"]["uy"] == pytest.approx(exact, rel=1e-6)
    assert nodal < 0.9 * exact  # what the nodal-only rule used to report


def test_the_off_centre_peak_does_not_move_when_the_mesh_is_refined():
    """The element cubic IS the exact shape here, so refinement changes nothing."""
    L, P = 20.0, 80e3
    peaks = []
    for n in (2, 8):
        xs = sorted({L * k / n for k in range(n + 1)} | {5.0})
        supports = {"N0": PINNED, f"N{len(xs) - 1}": ROLLER}
        model = beam_3d(
            xs, supports, points=[{"node": f"N{xs.index(5.0)}", "dof": "FY", "value": -P}]
        )
        peaks.append(solve_beam_3d(model)["max_abs"]["uy"])
    assert peaks[0] == pytest.approx(peaks[1], rel=1e-9)


def test_a_cantilever_under_udl_in_z_respects_the_sign_convention():
    """theta_y = -d(uz)/dx: get it wrong and this reports 3 % too much."""
    L, w = 10.0, 20e3
    model = beam_3d(
        [0.0, L / 2, L],
        {"N0": [True] * 6},
        dist=[{"element": "all", "direction": "z", "w1": -w, "w2": -w}],
    )
    result = solve_beam_3d(model)

    exact = w * L**4 / (8 * E_3D * IY)
    assert result["max_abs"]["uz"] == pytest.approx(exact, rel=1e-9)
    # a cantilever's deflection grows all the way to the free end, so the peak
    # is the tip node: an interpolation with the rotations entering as +ry
    # instead of -ry bulges between the nodes and reports about 3 % more
    assert result["max_abs"]["uz"] == pytest.approx(
        abs(result["displacements"]["N2"]["uz"]), rel=1e-12
    )
    assert result["max_abs"]["uy"] == 0.0


def test_an_off_centre_point_load_in_z_peaks_where_the_textbook_says():
    """The z plane carries the sign convention, so it gets the same check as y."""
    L, P, near = 20.0, 80e3, 5.0
    model = beam_3d(
        [0.0, near, L],
        {"N0": PINNED, "N2": ROLLER},
        points=[{"node": "N1", "dof": "FZ", "value": -P}],
    )
    result = solve_beam_3d(model)

    exact, _ = point_load_maximum(P, L, near, E_3D * IY)
    assert result["max_abs"]["uz"] == pytest.approx(exact, rel=1e-6)
    assert max(abs(d["uz"]) for d in result["displacements"].values()) < 0.9 * exact


def test_the_professors_beam_through_the_3d_solver_is_unchanged():
    """Four equal elements put a node on midspan, so nothing may move."""
    q = 30e3
    model = beam_3d(
        [SPAN * k / 4 for k in range(5)],
        {"N0": PINNED, "N4": ROLLER},
        dist=[{"element": "all", "direction": "y", "w1": -q, "w2": -q}],
    )
    result = solve_beam_3d(model)
    by_the_old_rule = max(abs(d["uy"]) for d in result["displacements"].values())

    assert result["max_abs"]["uy"] == by_the_old_rule
    assert result["max_abs"]["uy"] == pytest.approx(
        5 * q * SPAN**4 / (384 * E_3D * IZ), rel=1e-12
    )
    assert len(result["diagrams"]) >= 21  # the diagram contract is untouched


def test_a_coarse_3d_mesh_does_not_out_report_a_fine_one():
    """The overshoot check, in the solver that reports uy and uz."""
    L, w = 25.0, 30e3
    peaks = []
    for n in (2, 64):
        model = beam_3d(
            [L * k / n for k in range(n + 1)],
            {"N0": PINNED, f"N{n}": ROLLER},
            dist=[{"element": "all", "direction": "y", "w1": 0.0, "w2": -w}],
        )
        peaks.append(solve_beam_3d(model)["max_abs"]["uy"])

    coarse, fine = peaks
    assert coarse <= fine * (1 + 1e-3)
    assert coarse == pytest.approx(fine, rel=2e-3)


# --------------------------------------- the element with no shear in it at all


def four_point_midspan(P, L, a, EI):
    """Midspan deflection under two loads P, each a from its own support."""
    return P * a * (3 * L**2 - 4 * a**2) / (24 * EI)


@pytest.mark.parametrize(
    "direction, dof, second_moment, key",
    [("y", "FY", IZ, "uy"), ("z", "FZ", IY, "uz")],
    ids=["y-plane", "z-plane"],
)
def test_a_span_with_no_shear_in_it_still_reports_its_peak(direction, dof, second_moment, key):
    """Four-point bending: the middle element's cubic term is zero, or nearly.

    Between two equal loads the shear vanishes, so the element cubic degenerates
    to a parabola and the quadratic whose roots locate the peak loses its
    leading coefficient - not to exactly zero, which is easy, but to a few ulp
    of it, which is not. Solved by the textbook formula the interior root comes
    back as 0.0 and the peak at midspan is lost, leaving the deflection under
    the load: the old nodal-only answer, 1.5 % low, in the one arrangement a
    laboratory beam test is most likely to use. The loads sit at 7.9 m rather
    than 8 m so the arithmetic does NOT cancel to exactly zero.
    """
    L, P, a = 24.0, 100e3, 7.9
    model = beam_3d(
        [0.0, a, L - a, L],
        {"N0": PINNED, "N3": ROLLER},
        points=[
            {"node": "N1", "dof": dof, "value": -P},
            {"node": "N2", "dof": dof, "value": -P},
        ],
    )
    model["section"]["Iy"] = model["section"]["Iz"] = second_moment
    result = solve_beam_3d(model)

    exact = four_point_midspan(P, L, a, E_3D * second_moment)
    under_the_load = max(abs(d[key]) for d in result["displacements"].values())

    assert not any(abs(n["x"] - L / 2) < 1e-9 for n in model["nodes"])  # no node at midspan
    assert result["max_abs"][key] == pytest.approx(exact, rel=1e-12)
    assert under_the_load < 0.99 * exact  # what is lost when the root cancels away

    # the equation path meets the same beam with the same answer
    if direction == "y":
        equation = dict(PRISMATIC)
        equation["rhs"] = "0"
        beam = {
            "nodes": [{"id": f"N{k}", "x": x} for k, x in enumerate([0.0, a, L - a, L])],
            "supports": {"N0": PINNED, "N3": ROLLER},
            "point_loads": [
                {"node": "N1", "dof": "FY", "value": -P},
                {"node": "N2", "dof": "FY", "value": -P},
            ],
        }
        equation["params"] = dict(PRISMATIC["params"], I=second_moment)
        assert solve_equation_beam(beam, equation)["max_abs"]["v"] == pytest.approx(
            exact, rel=1e-12
        )


def test_the_quadratic_keeps_the_root_that_cancellation_would_eat():
    """A leading coefficient a few ulp off zero must not swallow the small root."""
    B, C = -1.2276, 0.62442
    # for a leading coefficient this small the root is -C/B to well past the
    # last bit, so any answer that is not that one is the cancellation
    expected = -C / B
    for A in (0.0, 1e-18, -1e-18, 1e-16, -1e-16, 1e-12, -1e-12):
        roots = _quadratic_roots(A, B, C)
        assert any(r == pytest.approx(expected, rel=1e-9) for r in roots), (
            f"A = {A!r} lost the root near {expected}: got {roots}"
        )
        assert len(roots) == (1 if A == 0.0 else 2)
        for r in roots:  # and every root returned is a root
            assert abs(A * r * r + B * r + C) <= 1e-12 * (abs(B * r) + abs(C))

    assert _quadratic_roots(1.0, 0.0, 1.0) == []  # no real root
    assert _quadratic_roots(0.0, 0.0, 3.0) == []  # a non-zero constant is never flat
    assert sorted(_quadratic_roots(1.0, 0.0, -0.25)) == pytest.approx([-0.5, 0.5])


# ------------------------------------------- what the professor's report says


def test_the_report_row_names_and_compares_the_same_quantity():
    """The row beside the headline number has to be the same question.

    max_abs["uy"] is the peak of the whole deflected shape, so the row may no
    longer call it the MIDSPAN deflection - on an odd mesh there is no node
    there - and the PyNite column may no longer be PyNite's largest NODAL
    deflection, which on this mesh is 2.5 % lower and would read as the two
    engines disagreeing when they are answering two different questions.
    """
    L, q, n = 25.0, 30e3, 3  # odd, so the peak falls between two nodes
    model = beam_3d(
        [L * k / n for k in range(n + 1)],
        {"N0": PINNED, f"N{n}": ROLLER},
        dist=[{"element": "all", "direction": "y", "w1": -q, "w2": -q}],
    )
    result = solve_beam_3d(model)
    rows = {name: (fem, exact, pynite) for name, fem, exact, pynite in _results_rows(model, result)}

    assert "Midspan deflection (m)" not in rows
    fem, exact, pynite = rows["Max deflection (m)"]
    nodal = max(abs(d["uy"]) for d in result["displacements"].values())

    assert fem == result["max_abs"]["uy"]
    assert exact == pytest.approx(5 * q * L**4 / (384 * E_3D * IZ), rel=1e-12)
    assert nodal < 0.9 * exact  # the nodes really are away from the peak here
    assert pynite == pytest.approx(fem, rel=5e-3)  # both engines, both peaks
    assert pynite > 0.99 * exact

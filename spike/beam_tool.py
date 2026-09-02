"""Deterministic beam-solving tool for the ADK spike.

Solves a simply supported prismatic beam under uniform load with PyNite,
with a node at midspan so the reported deflection is a NODAL value
(exact for Euler-Bernoulli, not a cubic interpolation of a quartic).
"""


def solve_simply_supported_beam(
    span_m: float, E_pa: float, I_m4: float, q_n_per_m: float
) -> dict:
    """Solve a simply supported beam under uniform downward load.

    Args:
        span_m: span length in meters.
        E_pa: elastic modulus in pascals.
        I_m4: second moment of area in m^4.
        q_n_per_m: uniform load magnitude in N/m (positive number, acts down).

    Returns:
        dict with midspan deflection (m), max moment (N*m), end shear (N),
        and the closed-form values for the same quantities.
    """
    from Pynite import FEModel3D

    m = FEModel3D()
    m.add_material("mat", E_pa, E_pa / 2.4, 0.2, 2400)
    m.add_section("sec", A=1.0, Iy=I_m4, Iz=I_m4, J=1.0)
    m.add_node("A", 0, 0, 0)
    m.add_node("MID", span_m / 2, 0, 0)
    m.add_node("B", span_m, 0, 0)
    m.add_member("M1", "A", "MID", "mat", "sec")
    m.add_member("M2", "MID", "B", "mat", "sec")
    m.def_support("A", True, True, True, True, False, False)
    m.def_support("B", False, True, True, False, False, False)
    m.add_member_dist_load("M1", "FY", -q_n_per_m, -q_n_per_m)
    m.add_member_dist_load("M2", "FY", -q_n_per_m, -q_n_per_m)
    m.analyze()

    fem_defl = m.nodes["MID"].DY["Combo 1"]
    fem_moment = m.members["M1"].moment("Mz", m.members["M1"].L())
    fem_shear = m.members["M1"].shear("Fy", 0.0)

    closed_defl = -5 * q_n_per_m * span_m**4 / (384 * E_pa * I_m4)
    closed_moment = q_n_per_m * span_m**2 / 8
    closed_shear = q_n_per_m * span_m / 2

    return {
        "fem_midspan_deflection_m": fem_defl,
        "fem_max_moment_nm": abs(fem_moment),
        "fem_end_shear_n": abs(fem_shear),
        "closed_form_deflection_m": closed_defl,
        "closed_form_moment_nm": closed_moment,
        "closed_form_shear_n": closed_shear,
        "deflection_rel_error": abs((fem_defl - closed_defl) / closed_defl),
    }


if __name__ == "__main__":
    from pprint import pprint

    pprint(solve_simply_supported_beam(25.0, 30e9, 0.005, 30e3))

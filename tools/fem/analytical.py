"""Closed-form reference solutions for standard beam cases.

Formulas are the classical Euler-Bernoulli results tabulated in
Roark's Formulas for Stress and Strain (8th ed., Table 8.1) and
Gere & Goodno, Mechanics of Materials (Appendix G).

Conventions:
- SI units (m, N, Pa). q is load intensity in N/m, P a point load in N.
- All returned values are positive magnitudes; direction is implied by
  the case (loads act transverse to the beam axis).
- "location_of_max" is the x-coordinate (m from the left/fixed end) of
  the maximum deflection.
- "reactions" is a dict of named support reactions ("R_left", "R_right",
  "V_fixed", "M_fixed" as applicable).
"""

# Registry of supported cases: params they take and LaTeX formulas for reports.
_CASES = {
    "ss_udl": {
        "params": ["L", "E", "I", "q"],
        "formulas": {
            "max_deflection": r"\delta_{max} = \frac{5 q L^4}{384 E I}",
            "max_moment": r"M_{max} = \frac{q L^2}{8}",
            "end_shear": r"V = \frac{q L}{2}",
            "reactions": r"R_A = R_B = \frac{q L}{2}",
            "location_of_max": r"x = \frac{L}{2}",
        },
    },
    "ss_point_center": {
        "params": ["L", "E", "I", "P"],
        "formulas": {
            "max_deflection": r"\delta_{max} = \frac{P L^3}{48 E I}",
            "max_moment": r"M_{max} = \frac{P L}{4}",
            "end_shear": r"V = \frac{P}{2}",
            "reactions": r"R_A = R_B = \frac{P}{2}",
            "location_of_max": r"x = \frac{L}{2}",
        },
    },
    "cantilever_udl": {
        "params": ["L", "E", "I", "q"],
        "formulas": {
            "max_deflection": r"\delta_{max} = \frac{q L^4}{8 E I}",
            "max_moment": r"M_{max} = \frac{q L^2}{2}",
            "end_shear": r"V = q L",
            "end_moment": r"M_{fixed} = \frac{q L^2}{2}",
            "reactions": r"V_{fixed} = q L,\ M_{fixed} = \frac{q L^2}{2}",
            "location_of_max": r"x = L",
        },
    },
    "cantilever_tip_point": {
        "params": ["L", "E", "I", "P"],
        "formulas": {
            "max_deflection": r"\delta_{max} = \frac{P L^3}{3 E I}",
            "max_moment": r"M_{max} = P L",
            "end_shear": r"V = P",
            "end_moment": r"M_{fixed} = P L",
            "reactions": r"V_{fixed} = P,\ M_{fixed} = P L",
            "location_of_max": r"x = L",
        },
    },
    "fixed_fixed_udl": {
        "params": ["L", "E", "I", "q"],
        "formulas": {
            "max_deflection": r"\delta_{max} = \frac{q L^4}{384 E I}",
            "max_moment": r"M_{max} = \frac{q L^2}{12}\ \text{(at supports)}",
            "end_shear": r"V = \frac{q L}{2}",
            "end_moment": r"M_{end} = \frac{q L^2}{12}",
            "reactions": r"R_A = R_B = \frac{q L}{2},\ M_A = M_B = \frac{q L^2}{12}",
            "location_of_max": r"x = \frac{L}{2}",
        },
    },
}


def closed_form(case: str, **params) -> dict:
    """Evaluate the closed-form solution for a standard beam case.

    Args:
        case: one of the keys returned by list_cases().
        **params: L (span, m), E (Pa), I (m^4), and q (N/m) or P (N)
            depending on the case.

    Returns:
        dict with keys among {"max_deflection", "max_moment", "end_shear",
        "reactions", "end_moment", "location_of_max"}. Values are positive
        magnitudes in SI units.
    """
    if case not in _CASES:
        raise ValueError(
            f"Unknown case '{case}'. Known: {sorted(_CASES)}"
        )
    missing = [p for p in _CASES[case]["params"] if p not in params]
    if missing:
        raise ValueError(f"Case '{case}' missing params: {missing}")

    L = float(params["L"])
    E = float(params["E"])
    I = float(params["I"])

    if case == "ss_udl":
        # Roark Table 8.1 case 2e / Gere App. G-1: SS beam, uniform load.
        q = float(params["q"])
        return {
            "max_deflection": 5 * q * L**4 / (384 * E * I),
            "max_moment": q * L**2 / 8,
            "end_shear": q * L / 2,
            "reactions": {"R_left": q * L / 2, "R_right": q * L / 2},
            "location_of_max": L / 2,
        }

    if case == "ss_point_center":
        # Roark Table 8.1 case 1e / Gere App. G-1: SS beam, midspan point load.
        P = float(params["P"])
        return {
            "max_deflection": P * L**3 / (48 * E * I),
            "max_moment": P * L / 4,
            "end_shear": P / 2,
            "reactions": {"R_left": P / 2, "R_right": P / 2},
            "location_of_max": L / 2,
        }

    if case == "cantilever_udl":
        # Roark Table 8.1 case 2a / Gere App. G-2: cantilever, uniform load.
        q = float(params["q"])
        return {
            "max_deflection": q * L**4 / (8 * E * I),
            "max_moment": q * L**2 / 2,
            "end_shear": q * L,
            "end_moment": q * L**2 / 2,
            "reactions": {"V_fixed": q * L, "M_fixed": q * L**2 / 2},
            "location_of_max": L,
        }

    if case == "cantilever_tip_point":
        # Roark Table 8.1 case 1a / Gere App. G-2: cantilever, tip point load.
        P = float(params["P"])
        return {
            "max_deflection": P * L**3 / (3 * E * I),
            "max_moment": P * L,
            "end_shear": P,
            "end_moment": P * L,
            "reactions": {"V_fixed": P, "M_fixed": P * L},
            "location_of_max": L,
        }

    # fixed_fixed_udl
    # Roark Table 8.1 case 2d / Gere App. G-2: fixed-fixed beam, uniform load.
    # Support moment qL^2/12 governs over the midspan moment qL^2/24.
    q = float(params["q"])
    return {
        "max_deflection": q * L**4 / (384 * E * I),
        "max_moment": q * L**2 / 12,
        "end_shear": q * L / 2,
        "end_moment": q * L**2 / 12,
        "reactions": {
            "R_left": q * L / 2,
            "R_right": q * L / 2,
            "M_left": q * L**2 / 12,
            "M_right": q * L**2 / 12,
        },
        "location_of_max": L / 2,
    }


def list_cases() -> list:
    """Return the supported cases with their params and LaTeX formulas."""
    return [
        {"case": name, "params": list(spec["params"]), "formulas": dict(spec["formulas"])}
        for name, spec in _CASES.items()
    ]

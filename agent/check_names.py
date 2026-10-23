"""Every verification check, named as a civil engineer would name it.

Shared by the website's narration and the chat report of the deployed agent,
so the professor sees one name for one check wherever he reads it.
"""

# Every verification check, as a civil engineer would name it: (name, what it tests).
CHECKS = {
    "equilibrium": (
        "Global vertical equilibrium",
        "Support reactions plus any foundation reaction must equal the applied load.",
    ),
    "residual": (
        "Residual of the stiffness equations",
        "K·u − F must vanish at the free degrees of freedom, and the reported reactions "
        "must equal K·u − F at the restrained ones.",
    ),
    "consistent": (
        "Reported results consistent with the solution",
        "Deflections, rotations, moments and shears in the report are recomputed from the "
        "nodal solution and compared.",
    ),
    "symmetric": ("Stiffness matrix symmetry", "K must equal Kᵀ, as it does for this self-adjoint problem."),
    "rigid": (
        "No mechanism",
        "With the boundary conditions applied the structure must be stable: no rigid-body motion.",
    ),
    "supports": ("Boundary conditions satisfied", "Restrained degrees of freedom show zero displacement."),
    "sign": ("Deflection sense", "A downward load must produce a downward deflection."),
    "mms": (
        "Convergence study (manufactured solution)",
        "An exact solution is manufactured for this equation and the load producing it derived; "
        "the finite element solution must converge to it at the theoretical rate, fourth order "
        "for Hermite cubic elements.",
    ),
    "textbook": (
        "Closed-form solution",
        "For a simply supported, uniformly loaded Euler–Bernoulli beam the results must match "
        "δ = 5qL⁴/384EI, M = qL²/8 and V = qL/2.",
    ),
    "pynite": (
        "Independent FEM program (PyNite)",
        "PyNite, an independent frame analysis program, solves the same model and must agree.",
    ),
}
_CHECK_KEY = {
    "equilibrium_forces": "equilibrium",
    "equation_equilibrium": "equilibrium",
    "equation_solution_residual": "residual",
    "equation_samples_consistent": "consistent",
    "stiffness_symmetric": "symmetric",
    "equation_stiffness_symmetry": "symmetric",
    "rigid_body_nullspace": "rigid",
    "supports_respected": "supports",
    "equation_support_conditions": "supports",
    "deflection_negative_under_downward_load": "sign",
    "equation_deflection_sign": "sign",
    "equation_mms": "mms",
    "pynite_agreement": "pynite",
}


def check_key(name: str):
    """The CHECKS key for a gate check's name, or None for one not listed."""
    name = name.removeprefix("invariant_")
    if name.startswith("closed_form"):
        return "textbook"
    return _CHECK_KEY.get(name)


def plain_check_name(name: str) -> str:
    """A gate check's name as a civil engineer would say it."""
    key = check_key(name)
    if key:
        return CHECKS[key][0]
    words = name.removeprefix("invariant_").replace("_", " ").strip()
    return words[:1].upper() + words[1:]

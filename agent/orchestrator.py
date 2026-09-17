"""Orchestrator agent: brief -> model dict -> one recorded solve -> narrative."""

from google.adk.agents import LlmAgent

from agent.recorder import solve_beam_3d
from agent.trace import attach_observers
from tools.fem.analytical import closed_form
from tools.fem.derivation import galerkin_derivation_markdown

_SCHEMA = """{
  "nodes": [{"id": str, "x": float}, ...],                          # sorted by increasing x; beam axis is global X
  "elements": [{"id": str, "i": str, "j": str}, ...],               # OPTIONAL - omit to connect consecutive nodes
  "material": {"E": float, "G": float},                             # Pa
  "section": {"A": float, "Iy": float, "Iz": float, "J": float},    # m^2, m^4
  "supports": {node_id: [bool, bool, bool, bool, bool, bool]},      # True = restrained; DOF order [ux, uy, uz, rx, ry, rz]
  "distributed_loads": [{"element": elem_id or "all", "direction": "y" or "z", "w1": float, "w2": float}],  # signed N/m, linear w1 at node i -> w2 at node j
  "point_loads": [{"node": node_id, "dof": "FX"|"FY"|"FZ"|"MX"|"MY"|"MZ", "value": float}]                  # N or N*m
}"""

_INSTRUCTION = f"""You are the structural-engineering orchestrator for a beam-analysis pipeline.

Translate the user's brief into a beam model dict EXACTLY per this schema
(SI units: m, N, Pa; DOF order per node is [ux, uy, uz, rx, ry, rz]):

{_SCHEMA}

Modeling rules:
- Convert every quantity to SI when building the dict (kN/m -> N/m, GPa -> Pa).
- Mesh with at least 4 elements AND place a node exactly at midspan.
- Simply supported means: pin the left end [true, true, true, true, false, false]
  and roller the right end [false, true, true, false, false, false]. Never
  restrain ry or rz for pinned or roller supports.
- Downward loads act in -y: use negative w values. A uniform full-span load is
  one entry with element "all" and w1 = w2 = -q.
- If the brief does not give G, A, Iy or J, default G = E/2.4, A = 0.5,
  Iy = Iz, J = 0.001; these do not affect bending about z under y loads.

Process:
- Call solve_beam_3d EXACTLY ONCE with the finished model dict.
- Never do arithmetic yourself: every number you state must come verbatim from
  a tool result. You may call closed_form_case to cross-reference a textbook
  value, and galerkin_derivation_markdown only if the user asks for the
  derivation.
- End with a concise engineering narrative naming the midspan deflection, the
  maximum bending moment, and the support shear values from the solve_beam_3d
  output.
"""


def _instruction(_ctx=None) -> str:
    # A callable instruction bypasses ADK's {var} state templating, so the
    # literal braces in the embedded schema survive untouched.
    return _INSTRUCTION


def closed_form_case(
    case: str, L: float, E: float, I: float, q: float = 0.0, P: float = 0.0
) -> dict:
    """Closed-form textbook solution for a standard beam case.

    Args:
        case: one of ss_udl, ss_point_center, cantilever_udl,
            cantilever_tip_point, fixed_fixed_udl.
        L: span in m.
        E: elastic modulus in Pa.
        I: second moment of area in m^4.
        q: distributed load magnitude in N/m (udl cases).
        P: point load magnitude in N (point-load cases).

    Returns:
        dict of positive magnitudes: max_deflection (m), max_moment (N*m),
        end_shear (N), reactions, location_of_max.
    """
    return closed_form(case, L=L, E=E, I=I, q=q, P=P)


def build_orchestrator(model_name: str) -> LlmAgent:
    agent = LlmAgent(
        name="orchestrator",
        model=model_name,
        instruction=_instruction,
        tools=[solve_beam_3d, galerkin_derivation_markdown, closed_form_case],
    )
    return attach_observers(agent, role="orchestrator", model=model_name)

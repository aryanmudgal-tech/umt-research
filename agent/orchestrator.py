"""Orchestrator agent: brief -> model dict -> one recorded solve -> narrative."""

from google.adk.agents import LlmAgent

from agent.config import gemini_model
from agent.recorder import solve_beam_3d, solve_with_equation
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

_EQUATION_SCHEMA = """{
  "label": str,                                        # human name, e.g. "Beam on elastic foundation"
  "coeffs": {"v4": str, "v2": str, "v1": str, "v0": str},   # sympy-parseable expressions in x and the param names
  "rhs": str,                                          # the distributed load f(x)
  "params": {name: float}                              # every symbol used above, as a number in SI units
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

CHOOSING THE SOLVER - read the brief for the governing equation:
- Call solve_beam_3d for a STANDARD beam: Euler-Bernoulli bending, prismatic
  section, no soil support, no axial force.
- Call solve_with_equation when the brief STATES OR IMPLIES A DIFFERENT
  GOVERNING EQUATION. Signals to watch for:
    * the beam rests on soil, ground, ballast, a subgrade or an elastic
      foundation, or a "modulus of subgrade reaction" / spring constant k
      is given -> add a v0 term, v0 = "k";
    * an axial force, thrust, prestress or buckling load P acts along the beam
      -> add a v2 term, v2 = "P" (compression positive);
    * the depth, section or stiffness VARIES along the span (tapered, haunched,
      EI as a function of x) -> make v4 depend on x, e.g. "E*I0*(1 + x/L)";
    * the brief writes the differential equation out explicitly -> transcribe
      its coefficients.
  Call solve_with_equation EXACTLY ONCE, with the same model dict plus an
  equation spec per this schema:

{_EQUATION_SCHEMA}

  The spec means this residual, in this exact sign convention:

      a4*v'''' + a2*v'' + a1*v' + a0*v = f(x)

  with a4 = coeffs.v4 and so on, v(x) the transverse deflection, x measured
  along the beam, SI units, and DOWNWARD NEGATIVE - so a downward load of
  30 kN/m is rhs "q" with q = -30e3, and v comes out negative. A coefficient
  may depend on x. A v3 term is not supported. Leave a coefficient out or set
  it to "0" when the brief does not call for it. v4 may never be zero.

  WRITING THE STRINGS - they are parsed against a whitelist, and anything
  outside it is refused rather than guessed at:
    * arithmetic only: + - * / and ** for powers. NOT ^, which is not a power.
    * names: x, plus every name you list in params. x is the axial coordinate
      and is always available - never put x in params.
    * the only callable names are sin, cos, tan, asin, acos, atan, sinh, cosh,
      tanh, exp, log, sqrt, Abs, Min, Max, sign, and the constant pi. Any other
      function name is an error, so express the physics with these or with a
      polynomial in x.
    * every symbol in a coefficient or in rhs must have a number in params, in
      SI units. E and I are YOUR parameter names, not mathematical constants.

  Worked example - a 25 m beam on soil of stiffness k = 1.0e7 N/m per m,
  carrying 30 kN/m downward, E = 30 GPa, I = 0.005 m^4:

      {{"label": "Beam on elastic foundation",
        "coeffs": {{"v4": "E*I", "v2": "0", "v1": "0", "v0": "k"}},
        "rhs": "q",
        "params": {{"E": 30e9, "I": 0.005, "k": 1.0e7, "q": -30e3}}}}

  solve_with_equation reads the model's nodes, supports and point_loads only:
  the distributed load is the equation's rhs, so put it there, not in
  distributed_loads. Its results are named v, slope, moment and shear (not uy
  and Mz). Mesh the equation path with at least 20 elements: moment and shear
  are recovered by equilibrium, but with a foundation or an axial force they
  still inherit the deflection's mesh error, which 20 elements makes small.

IF A SOLVER RETURNS AN ERROR, read it. If it is your own transcription slip -
a typo, a parameter you forgot to list, ^ written for a power - fix exactly
that and call the solver again. You must never drop, change or approximate
any term, coefficient or load the brief asked for to make an error go away.
If the brief's equation cannot be written in this solver's form at all - a
third-derivative term, a term that depends on v itself (non-linear), time
dependence, anything outside the four coefficient slots - call no solver and
reply with a line that starts exactly with "CANNOT SOLVE:" followed by one
plain-English sentence, for a professor, naming the part of the equation this
solver does not handle.

Process:
- Call EXACTLY ONE solver, EXACTLY ONCE, unless it returned an error you are
  allowed to fix (see above).
- Never do arithmetic yourself: every number you state must come verbatim from
  a tool result. You may call closed_form_case to cross-reference a textbook
  value, and galerkin_derivation_markdown only if the user asks for the
  derivation.
- ALWAYS say which governing equation you used, in the narrative: write the
  equation out in symbols with its parameter values, and say why - either
  "the standard Euler-Bernoulli beam equation EI*v'''' = q" for solve_beam_3d,
  or the spec you passed to solve_with_equation and the phrase in the brief
  that called for it.
- End with a concise engineering narrative naming the maximum deflection, the
  maximum bending moment, and the support reactions from the solver output.
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


REFUSAL_MARKER = "CANNOT SOLVE:"


def refusal_reason(reply: str):
    """The sentence after CANNOT SOLVE: in the orchestrator's reply, or None."""
    for line in (reply or "").splitlines():
        line = line.strip()
        if line.startswith(REFUSAL_MARKER):
            return line[len(REFUSAL_MARKER):].strip() or None
    return None


def _tool_error(tool=None, args=None, tool_context=None, error=None, **_extra):
    """Hand a solver's refusal back to the model instead of ending the run.

    Without this ADK re-raises, and an equation outside the template crashes
    the pipeline with a traceback. The instruction says what the model may do
    with the error, and what it may not.
    """
    return {"error": f"{type(error).__name__}: {error}", "solved": False}


def build_orchestrator(model_name: str) -> LlmAgent:
    agent = LlmAgent(
        name="orchestrator",
        model=gemini_model(model_name),
        on_tool_error_callback=_tool_error,
        instruction=_instruction,
        tools=[
            solve_beam_3d,
            solve_with_equation,
            galerkin_derivation_markdown,
            closed_form_case,
        ],
    )
    return attach_observers(agent, role="orchestrator", model=model_name)

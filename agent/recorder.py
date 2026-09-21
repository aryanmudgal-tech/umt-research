"""Module-level recorder around the FEM solvers.

The harness reads (model_dict, result) pairs from recorder.calls — never
from LLM prose — so every gate and every report row is grounded in an
actual tool call.

Two solvers can be called: solve_beam_3d, the fixed Euler-Bernoulli beam
element, and solve_with_equation, which takes the governing equation as
data. Each recorded call carries the tool name and, for the equation tool,
the spec that was solved, so the harness can tell which one produced the
numbers it is about to gate.
"""

import copy

import numpy as np

from tools.fem.equation import solve_equation_beam as _solve_equation_beam
from tools.fem.solver import solve_beam_3d as _solve_beam_3d

BEAM_TOOL = "solve_beam_3d"
EQUATION_TOOL = "solve_with_equation"


def to_plain(obj):
    """Recursively convert numpy scalars to Python floats/ints (JSON-safe)."""
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_plain(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):  # covers floating, integer, bool_
        return obj.item()
    return obj


class SolveCall:
    """One recorded solve: which tool, on what model, under what equation.

    Unpacks and indexes as the (model, result) pair the harness has always
    read, so `model_dict, result = recorder.last` keeps working; `tool` and
    `equation` are the new facts a caller needs to gate an equation run.
    """

    __slots__ = ("tool", "model", "result", "equation")

    def __init__(self, tool, model, result, equation=None):
        self.tool = tool
        self.model = model
        self.result = result
        self.equation = equation

    @property
    def is_equation(self) -> bool:
        """True when this call solved a caller-supplied governing equation."""
        return self.equation is not None

    def __iter__(self):
        return iter((self.model, self.result))

    def __getitem__(self, index):
        return (self.model, self.result)[index]

    def __len__(self):
        return 2

    def __eq__(self, other):
        if isinstance(other, SolveCall):
            return (self.tool, self.model, self.result, self.equation) == (
                other.tool,
                other.model,
                other.result,
                other.equation,
            )
        if isinstance(other, (tuple, list)):
            return [self.model, self.result] == list(other)
        return NotImplemented

    def __repr__(self):
        return f"SolveCall(tool={self.tool!r}, equation={self.equation!r})"


class ToolRecorder:
    def __init__(self):
        self.calls = []  # SolveCall per solve, in call order

    def reset(self):
        self.calls.clear()

    def record(self, model, result, tool=BEAM_TOOL, equation=None):
        self.calls.append(
            SolveCall(tool, copy.deepcopy(model), copy.deepcopy(result), copy.deepcopy(equation))
        )

    @property
    def last(self):
        return self.calls[-1] if self.calls else None


recorder = ToolRecorder()


def solve_beam_3d(model: dict) -> dict:
    """Solve a 3D beam model with the deterministic FEM solver.

    Use this for a standard beam: Euler-Bernoulli bending with no soil
    support, no axial force, and a stiffness that is constant along the span.

    Args:
        model: beam model dict per the schema in the agent instruction
            (nodes, material, section, supports, loads; SI units: m, N, Pa).

    Returns:
        dict with displacements per node, reactions at restrained DOFs,
        internal-force diagrams along the beam, max_abs values, and the
        total applied load — all in SI units.
    """
    result = to_plain(_solve_beam_3d(model))
    recorder.record(model, result, tool=BEAM_TOOL)
    return result


def solve_with_equation(model: dict, equation: dict) -> dict:
    """Solve a beam under a governing equation supplied as data.

    Use this when the brief states or implies an equation other than the
    standard Euler-Bernoulli beam: a soil or elastic foundation, an axial
    force, a bending stiffness that varies along the span, or an equation
    written out explicitly. The element matrices are integrated symbolically
    from the equation you pass, so no textbook matrix is assumed.

    Args:
        model: dict with "nodes" [{"id", "x"}], optional "elements"
            [{"id", "i", "j"}], "supports" {node_id: 6 flags, index 1 holding
            deflection and index 5 holding slope}, and "point_loads"
            [{"node", "dof": "FY" or "MZ", "value"}]. Any "distributed_loads"
            are ignored: the distributed load is the equation's rhs.
        equation: equation spec dict with "label", "coeffs" over v4/v2/v1/v0
            as sympy-parseable strings in x and the parameter names, "rhs" as
            f(x), and "params" mapping every name used to a number. SI units,
            deflection downward negative.

    Returns:
        dict with "displacements" {node_id: {"v", "slope"}}, "reactions" at
        restrained DOFs only, "samples" of at least 21 points {"x", "v",
        "slope", "moment", "shear"}, and "max_abs" of each.
    """
    result = to_plain(_solve_equation_beam(model, equation))
    recorder.record(model, result, tool=EQUATION_TOOL, equation=equation)
    return result

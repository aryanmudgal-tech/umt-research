"""Module-level recorder around the FEM solver.

The harness reads (model_dict, result) pairs from recorder.calls — never
from LLM prose — so every gate and every report row is grounded in an
actual tool call.
"""

import copy

import numpy as np

from tools.fem.solver import solve_beam_3d as _solve_beam_3d


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


class ToolRecorder:
    def __init__(self):
        self.calls = []  # (model_dict, result) per solve, in call order

    def reset(self):
        self.calls.clear()

    def record(self, model, result):
        self.calls.append((copy.deepcopy(model), copy.deepcopy(result)))

    @property
    def last(self):
        return self.calls[-1] if self.calls else None


recorder = ToolRecorder()


def solve_beam_3d(model: dict) -> dict:
    """Solve a 3D beam model with the deterministic FEM solver.

    Args:
        model: beam model dict per the schema in the agent instruction
            (nodes, material, section, supports, loads; SI units: m, N, Pa).

    Returns:
        dict with displacements per node, reactions at restrained DOFs,
        internal-force diagrams along the beam, max_abs values, and the
        total applied load — all in SI units.
    """
    result = to_plain(_solve_beam_3d(model))
    recorder.record(model, result)
    return result

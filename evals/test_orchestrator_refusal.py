"""Offline tests for what the orchestrator does with an equation it cannot solve.

A spec the solver refuses (a v''' term, a non-linear term) used to raise out of
ADK and crash the run with a traceback. The error now goes back to the model,
which may fix its own transcription slip but is told never to drop or
approximate a term to make the error go away, and to say CANNOT SOLVE instead.
No API calls.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.orchestrator import _INSTRUCTION, build_orchestrator, refusal_reason
from tools.fem.equation import EquationError


def test_a_solver_error_goes_back_to_the_model_instead_of_crashing():
    agent = build_orchestrator("gemini-probe")
    callbacks = agent.canonical_on_tool_error_callbacks
    assert callbacks

    error = EquationError("coefficient 'v3' is not supported: a third-derivative term")
    out = callbacks[0](tool=None, args={}, tool_context=None, error=error)

    assert out["solved"] is False
    assert "third-derivative" in out["error"]


def test_the_instruction_forbids_approximating_the_equation_away():
    assert "never drop, change or approximate" in _INSTRUCTION
    assert "CANNOT SOLVE:" in _INSTRUCTION


def test_the_refusal_reason_is_read_from_the_reply():
    reply = "I read the brief.\n\nCANNOT SOLVE: the equation has a third-derivative term, which this solver does not support."
    assert refusal_reason(reply) == "the equation has a third-derivative term, which this solver does not support."
    assert refusal_reason("All done, the midspan deflection is 3 mm.") is None
    assert refusal_reason("") is None

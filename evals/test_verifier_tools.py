"""Offline tests for the verifier's tools: what it is offered, and on what.

A verifier tool must check the beam that was actually solved, must not be
offered where it cannot apply, and must not take the whole run down when it
fails. No API calls.
"""

import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent import verifier as verifier_module
from agent.recorder import to_plain
from tools.fem.pynite_check import solve_with_pynite

L, E, G, I, Q = 25.0, 30e9, 12.5e9, 0.005, 30e3

WINKLER = {
    "label": "Beam on elastic foundation",
    "coeffs": {"v4": "E*I", "v2": "0", "v1": "0", "v0": "k"},
    "rhs": "q",
    "params": {"E": E, "I": I, "k": 1.0e7, "q": -Q},
}


def ss_udl_model(n_elem=4):
    return {
        "nodes": [{"id": f"N{k}", "x": L * k / n_elem} for k in range(n_elem + 1)],
        "material": {"E": E, "G": G},
        "section": {"A": 0.5, "Iy": 0.004, "Iz": I, "J": 0.001},
        "supports": {
            "N0": [True, True, True, True, False, False],
            f"N{n_elem}": [False, True, True, False, False, False],
        },
        "distributed_loads": [{"element": "all", "direction": "y", "w1": -Q, "w2": -Q}],
        "point_loads": [],
    }


def equation_model(n_elem=10):
    # what the orchestrator hands solve_with_equation: geometry and supports only
    return {
        "nodes": [{"id": f"N{k}", "x": L * k / n_elem} for k in range(n_elem + 1)],
        "supports": {
            "N0": [True, True, True, True, False, False],
            f"N{n_elem}": [False, True, True, False, False, False],
        },
    }


def names(tools):
    return {t.__name__ for t in tools}


def test_pynite_check_runs_on_the_solved_model_and_takes_no_arguments():
    model = ss_udl_model()
    pynite = next(
        t for t in verifier_module.verifier_tools(model, {}) if t.__name__ == "pynite_crosscheck"
    )

    # no model argument: the verifier cannot swap in a beam of its own
    assert inspect.signature(pynite).parameters == {}
    assert pynite() == to_plain(solve_with_pynite(model))


def test_standard_path_keeps_all_three_tools():
    assert names(verifier_module.verifier_tools(ss_udl_model(), {})) == {
        "pynite_crosscheck",
        "closed_form_case",
        "run_invariant_checks",
    }


def test_equation_path_is_not_offered_the_standard_beam_tools():
    offered = names(verifier_module.verifier_tools(equation_model(), {}, WINKLER))
    # both need a material, a section and 3D results; an equation model has none
    assert "pynite_crosscheck" not in offered
    assert "run_invariant_checks" not in offered


def test_a_failing_tool_is_handed_back_to_the_verifier_not_raised():
    agent = verifier_module.build_verifier("gemini-probe", [])
    callbacks = agent.canonical_on_tool_error_callbacks
    assert callbacks

    out = callbacks[0](tool=None, args={}, tool_context=None, error=KeyError("material"))

    assert out["verified"] is False
    assert "KeyError" in out["error"] and "material" in out["error"]


def test_equation_path_gets_the_independent_check_bound_to_the_solved_model():
    from tools.fem.bvp_check import solve_equation_bvp

    model = equation_model()
    check = next(
        t
        for t in verifier_module.verifier_tools(model, {}, WINKLER)
        if t.__name__ == "independent_equation_check"
    )

    assert inspect.signature(check).parameters == {}
    assert check() == to_plain(solve_equation_bvp(model, WINKLER))


def test_the_verifier_is_told_to_take_reference_numbers_from_its_tools(monkeypatch):
    prompts = []

    async def capture(agent, prompt):
        prompts.append((agent.instruction, prompt))
        return '{"refuted": false, "checks": [], "reasoning": ""}'

    monkeypatch.setattr(verifier_module, "_run_once", capture)
    verifier_module.run_verifier("brief", equation_model(), {}, ["m"], equation=WINKLER)

    instruction, prompt = prompts[0]
    assert "independent_equation_check" in prompt
    assert "do not compute reference values yourself" in instruction
    assert "never code, field or tool names" in instruction

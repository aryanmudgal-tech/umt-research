"""Isolated adversarial verifier: fresh agent, runner and session per run.

It receives ONLY the problem text and the raw numbers JSON — never
orchestrator prose — and tries to refute the results with its own tools:
PyNite, closed forms and physics invariants for a standard beam, and an
independent collocation solve of the same equation for a custom one.
"""

import asyncio
import json
import re

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from agent.config import VERIFIER_MODELS, retryable_error
from agent.orchestrator import closed_form_case
from agent.recorder import to_plain
from agent.trace import attach_observers, tracer
from tools.fem.bvp_check import solve_equation_bvp
from tools.fem.invariants import check_invariants
from tools.fem.pynite_check import solve_with_pynite
from tools.fem.solver import assemble

_INSTRUCTION = (
    "You are an adversarial verifier. You receive a problem statement and "
    "computed FEM results. Recompute independently with YOUR tools and try "
    "to REFUTE the results. Finding nothing wrong must be justified check "
    "by check. Every reference number you compare against must come from one "
    "of your tools; do not compute reference values yourself. If a tool "
    "returns an error, that check verified nothing: say so."
)


def _make_pynite_tool(model):
    # Bound, not an argument: given a model parameter, the verifier once passed
    # its own copy with both ends fully fixed and would have checked that beam.
    def pynite_crosscheck() -> dict:
        """Re-solve the beam under scrutiny with PyNiteFEA, an independent FEM engine.

        Takes no arguments — the model that was solved is already bound, so
        this always checks that beam and no other.

        Returns:
            dict with displacements and reactions per node (SI units).
        """
        return to_plain(solve_with_pynite(model))

    return pynite_crosscheck


def _make_invariants_tool(model, result):
    def run_invariant_checks() -> dict:
        """Run the deterministic physics invariants on the results under scrutiny.

        Takes no arguments — the model and result are already bound.

        Returns:
            dict of check name -> passed flag and detail string.
        """
        return to_plain(check_invariants(model, result, assemble))

    return run_invariant_checks


def _make_equation_check_tool(model, equation):
    def independent_equation_check() -> dict:
        """Re-solve this beam's own equation by a different method, as a reference.

        Takes no arguments — the solved model and its equation are already
        bound. Uses collocation on the strong form (scipy solve_bvp), not the
        Galerkin FEM that produced the results, so agreement means something.

        Returns:
            dict with "applicable" and, when it is, "converged",
            "deflection_at_nodes", "reactions" {node: {"F", "M"}}, "midspan"
            {"x", "v"} and "max_abs" {"v", "moment", "shear": {"value", "x"}}.
            SI units, deflection downward negative.
        """
        return to_plain(solve_equation_bvp(model, equation))

    return independent_equation_check


def verifier_tools(model_dict, result, equation=None) -> list:
    """The tools that can actually check this run.

    PyNite and the invariants re-solve the standard 3D beam, so they need a
    material, a section and 3D results. An equation run has none of those, and
    offering them there only invites a call that fails.
    """
    if equation is not None:
        return [_make_equation_check_tool(model_dict, equation), closed_form_case]
    return [
        _make_pynite_tool(model_dict),
        closed_form_case,
        _make_invariants_tool(model_dict, result),
    ]


def _tool_error(tool=None, args=None, tool_context=None, error=None, **_extra):
    """Hand a failing tool's error back to the verifier instead of ending the run.

    Without this ADK re-raises, and one bad tool call crashes the whole
    pipeline with no report and no trace.
    """
    return {"error": f"{type(error).__name__}: {error}", "verified": False}


def build_verifier(name: str, tools: list) -> LlmAgent:
    agent = LlmAgent(
        name="verifier",
        model=name,
        instruction=_INSTRUCTION,
        tools=tools,
        on_tool_error_callback=_tool_error,
    )
    return attach_observers(agent, role="verifier", model=name)


def extract_json(text: str):
    """First parseable JSON object in text (fenced or bare), else None."""
    if not text:
        return None
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL):
        try:
            return json.loads(m.group(1))
        except ValueError:
            pass
    # bare object embedded in prose: try each brace-balanced candidate
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : end + 1])
                    except ValueError:
                        break
    return None


def _emit_verdict(verdict: dict) -> dict:
    """Publish the verdict on the event bus, then hand it back unchanged."""
    banner = "REFUTED" if verdict["refuted"] else "not refuted"
    tracer.emit(
        "verifier",
        "verdict",
        f"verifier verdict: {banner}",
        refuted=verdict["refuted"],
        checks=verdict["checks"],
        reasoning=verdict["reasoning"],
        model=verdict["model"],
    )
    return verdict


async def _run_once(agent: LlmAgent, prompt: str) -> str:
    runner = InMemoryRunner(agent=agent, app_name="phase1_verifier")
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="verifier"
    )
    msg = types.Content(role="user", parts=[types.Part(text=prompt)])
    final = ""
    async for event in runner.run_async(
        user_id="verifier", session_id=session.id, new_message=msg
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final = "".join(p.text or "" for p in event.content.parts)
    return final


_EQUATION_NOTE = (
    "\n\nNOTE: these results were NOT produced by the standard Euler-Bernoulli "
    "beam element. The governing equation above was supplied with the problem "
    "and reads a4*v'''' + a2*v'' + a1*v' + a0*v = f(x), SI units, deflection "
    "downward negative. Call independent_equation_check: it solves this same "
    "equation by collocation instead of FEM. Compare against it every quantity "
    "the problem statement asks for, and every max_abs value in the results. "
    "closed_form_case assumes the standard equation, so it does NOT apply here "
    "unless every one of a2, a1 and a0 is zero: say so rather than refuting on "
    "that basis.\n"
)


def run_verifier(
    problem_text: str, model_dict: dict, result: dict, model_names=None, equation=None
) -> dict:
    """Verify (model_dict, result) against problem_text with an isolated agent.

    Pass `equation` when the run went through solve_with_equation, so the
    verifier judges the results against the equation actually solved rather
    than against the standard beam its own tools assume.

    Returns {"refuted": bool, "checks": [...], "reasoning": str, "model": str}.
    A verifier that cannot run or cannot be parsed counts as a refutation.
    """
    model_names = list(model_names or VERIFIER_MODELS)
    numbers = {"model": to_plain(model_dict), "result": to_plain(result)}
    if equation is not None:
        numbers["governing_equation"] = to_plain(equation)
    payload = json.dumps(numbers, indent=2)
    prompt = (
        "Problem statement:\n" + problem_text.strip() + "\n\n"
        "Computed FEM model and results (raw numbers, JSON):\n" + payload + "\n\n"
        + (_EQUATION_NOTE if equation is not None else "")
        + "Recompute independently with your tools and try to refute these "
        "results. Then finish with ONLY a JSON verdict object of the form:\n"
        '{"refuted": true|false, "checks": [{"name": "...", "passed": '
        'true|false, "detail": "..."}], "reasoning": "..."}'
    )
    tools = verifier_tools(model_dict, result, equation)

    last_error = None
    for name in model_names:
        tracer.emit(
            "verifier",
            "model_attempt",
            f"verifier trying {name}",
            role="verifier",
            model=name,
        )
        agent = build_verifier(name, tools)
        try:
            text = asyncio.run(_run_once(agent, prompt))
        except Exception as exc:
            if retryable_error(exc):
                print(f"[verifier] {name} unavailable ({exc}); trying next model")
                tracer.emit(
                    "verifier",
                    "model_fallback",
                    f"{name} unavailable; trying the next model",
                    role="verifier",
                    model=name,
                    error=str(exc),
                )
                last_error = exc
                continue
            raise
        verdict = extract_json(text)
        if not isinstance(verdict, dict) or "refuted" not in verdict:
            return _emit_verdict(
                {
                    "refuted": True,
                    "checks": [
                        {
                            "name": "verdict_parse",
                            "passed": False,
                            "detail": "unparseable verdict",
                        }
                    ],
                    "reasoning": text,
                    "model": name,
                }
            )
        return _emit_verdict(
            {
                "refuted": bool(verdict.get("refuted")),
                "checks": list(verdict.get("checks") or []),
                "reasoning": str(verdict.get("reasoning", "")),
                "model": name,
            }
        )

    return _emit_verdict(
        {
            "refuted": True,
            "checks": [
                {
                    "name": "verifier_run",
                    "passed": False,
                    "detail": f"all verifier models failed: {last_error}",
                }
            ],
            "reasoning": "verifier could not run",
            "model": None,
        }
    )

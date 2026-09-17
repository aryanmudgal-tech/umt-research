"""Isolated adversarial verifier: fresh agent, runner and session per run.

It receives ONLY the problem text and the raw numbers JSON — never
orchestrator prose — and tries to refute the results with its own tools
(PyNite, closed forms, physics invariants).
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
from tools.fem.invariants import check_invariants
from tools.fem.pynite_check import solve_with_pynite
from tools.fem.solver import assemble

_INSTRUCTION = (
    "You are an adversarial verifier. You receive a problem statement and "
    "computed FEM results. Recompute independently with YOUR tools and try "
    "to REFUTE the results. Finding nothing wrong must be justified check "
    "by check."
)


def pynite_crosscheck(model: dict) -> dict:
    """Re-solve the beam model dict with PyNiteFEA, an independent FEM engine.

    Args:
        model: the beam model dict under scrutiny (SI units).

    Returns:
        dict with displacements and reactions per node (SI units).
    """
    return to_plain(solve_with_pynite(model))


def _make_invariants_tool(model, result):
    def run_invariant_checks() -> dict:
        """Run the deterministic physics invariants on the results under scrutiny.

        Takes no arguments — the model and result are already bound.

        Returns:
            dict of check name -> passed flag and detail string.
        """
        return to_plain(check_invariants(model, result, assemble))

    return run_invariant_checks


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


def run_verifier(
    problem_text: str, model_dict: dict, result: dict, model_names=None
) -> dict:
    """Verify (model_dict, result) against problem_text with an isolated agent.

    Returns {"refuted": bool, "checks": [...], "reasoning": str, "model": str}.
    A verifier that cannot run or cannot be parsed counts as a refutation.
    """
    model_names = list(model_names or VERIFIER_MODELS)
    payload = json.dumps(
        {"model": to_plain(model_dict), "result": to_plain(result)}, indent=2
    )
    prompt = (
        "Problem statement:\n" + problem_text.strip() + "\n\n"
        "Computed FEM model and results (raw numbers, JSON):\n" + payload + "\n\n"
        "Recompute independently with your tools and try to refute these "
        "results. Then finish with ONLY a JSON verdict object of the form:\n"
        '{"refuted": true|false, "checks": [{"name": "...", "passed": '
        'true|false, "detail": "..."}], "reasoning": "..."}'
    )
    tools = [
        pynite_crosscheck,
        closed_form_case,
        _make_invariants_tool(model_dict, result),
    ]

    last_error = None
    for name in model_names:
        tracer.emit(
            "verifier",
            "model_attempt",
            f"verifier trying {name}",
            role="verifier",
            model=name,
        )
        agent = attach_observers(
            LlmAgent(
                name="verifier", model=name, instruction=_INSTRUCTION, tools=tools
            ),
            role="verifier",
            model=name,
        )
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

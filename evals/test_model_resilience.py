"""Offline tests for how the agents ride out a busy or silent Gemini.

Two failure shapes, two remedies. A 503 or 429 is usually a blip, so the same
model is retried a few times with backoff before anyone gives up on it. A model
that accepts the request and never answers is the other shape: one run sat in
the orchestrator stage for over three minutes. Each model attempt now has a
time budget, and running out of it counts as unavailable, so the roster moves
on. No API calls.
"""

import asyncio
import sys
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent import config, run_phase1
from agent import verifier as verifier_module
from agent.config import retryable_error
from agent.orchestrator import build_orchestrator
from agent.trace import tracer


@pytest.mark.parametrize("build", [build_orchestrator, lambda n: verifier_module.build_verifier(n, [])])
def test_every_agent_retries_a_busy_model_before_giving_up_on_it(build):
    retry = build("gemini-probe").model.retry_options
    assert retry.attempts >= 3
    assert {429, 503}.issubset(set(retry.http_status_codes))
    assert retry.initial_delay >= 1


@pytest.mark.parametrize(
    "exc, expected",
    [
        (RuntimeError("503 UNAVAILABLE"), True),
        (RuntimeError("429 RESOURCE_EXHAUSTED"), True),
        (TimeoutError(), True),
        (httpx.ReadTimeout("read timed out"), True),
        (ValueError("bad tool schema"), False),
    ],
)
def test_what_counts_as_the_model_being_unavailable(exc, expected):
    assert retryable_error(exc) is expected


class _FakeRunner:
    def __init__(self, agent, app_name):
        self.agent = agent


def test_an_orchestrator_model_that_never_answers_falls_through(monkeypatch):
    silent = config.ORCHESTRATOR_MODELS[0]

    async def fake_run_agent(runner, _brief):
        if runner.agent == silent:
            await asyncio.sleep(30)
        return f"narrative from {runner.agent}"

    monkeypatch.setattr(config, "ORCHESTRATOR_BUDGET_S", 0.2)
    monkeypatch.setattr(run_phase1, "build_orchestrator", lambda name: name)
    monkeypatch.setattr(run_phase1, "InMemoryRunner", _FakeRunner)
    monkeypatch.setattr(run_phase1, "_run_agent", fake_run_agent)
    tracer.reset()

    used, _ = run_phase1.run_orchestrator("brief")

    assert used == config.ORCHESTRATOR_MODELS[1]
    fallback = next(e for e in tracer.events if e["type"] == "model_fallback")
    assert fallback["data"]["model"] == silent
    assert "no answer within" in fallback["data"]["error"]


def test_a_verifier_model_that_never_answers_falls_through(monkeypatch):
    async def fake_run_once(agent, _prompt):
        if agent.name == "verifier" and agent.model.model == "silent-model":
            await asyncio.sleep(30)
        return '{"refuted": false, "checks": [], "reasoning": ""}'

    monkeypatch.setattr(config, "VERIFIER_BUDGET_S", 0.2)
    monkeypatch.setattr(verifier_module, "_run_once", fake_run_once)
    tracer.reset()

    verdict = verifier_module.run_verifier("brief", {}, {}, ["silent-model", "answering-model"], equation={})

    assert verdict["model"] == "answering-model"
    assert verdict["refuted"] is False

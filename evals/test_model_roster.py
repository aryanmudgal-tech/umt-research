"""Offline tests for the model rosters and the fallback down them.

The free tier returns 503 on whichever Flash model is busiest, so the
orchestrator walks down a roster of progressively lighter models. The verifier
must still never run on the model the orchestrator landed on. No API calls.
"""

import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent import run_phase1
from agent.config import (
    ORCHESTRATOR_MODELS,
    VERIFIER_MODELS,
    generation,
    verifier_roster,
)
from agent.recorder import recorder, solve_beam_3d
from agent.trace import tracer

L, E, G, I, Q = 25.0, 30e9, 12.5e9, 0.005, 30e3


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


# ------------------------------------------------------------------ rosters


def test_generation_reads_the_version_out_of_the_model_name():
    assert generation("gemini-3.8-flash") == "3.8"
    assert generation("gemini-3.5-flash-lite") == "3.5"
    assert generation("gemini-3-flash-preview") == "3"


def test_orchestrator_roster_falls_back_to_lighter_models():
    assert ORCHESTRATOR_MODELS[0] == "gemini-3.8-flash"
    assert len(ORCHESTRATOR_MODELS) >= 5
    assert len(set(ORCHESTRATOR_MODELS)) == len(ORCHESTRATOR_MODELS)
    versions = [float(generation(m)) for m in ORCHESTRATOR_MODELS]
    assert versions == sorted(versions, reverse=True)  # strongest first
    assert ORCHESTRATOR_MODELS[-1].endswith("-lite")


def test_no_roster_names_a_retired_model():
    # the 2.5 generation answers 404 on this key
    for name in ORCHESTRATOR_MODELS + VERIFIER_MODELS:
        assert not name.startswith("gemini-2")


@pytest.mark.parametrize("used", ORCHESTRATOR_MODELS)
def test_verifier_never_runs_on_the_orchestrators_model(used):
    roster = verifier_roster(used)
    assert roster
    assert used not in roster
    assert set(roster) <= set(VERIFIER_MODELS)


@pytest.mark.parametrize("used", ORCHESTRATOR_MODELS)
def test_verifier_tries_a_different_generation_first(used):
    assert generation(verifier_roster(used)[0]) != generation(used)


def test_verifier_roster_keeps_config_order_otherwise():
    # nothing to exclude and nothing to reorder: the configured order stands
    assert verifier_roster("gemini-3.8-flash") == VERIFIER_MODELS


# ------------------------------------------------------------------ fallback


class _FakeRunner:
    def __init__(self, agent, app_name):
        self.agent = agent


def _overloaded(monkeypatch, busy):
    """Make every model in `busy` answer 503; the rest answer normally."""

    async def fake_run_agent(runner, _brief):
        if runner.agent in busy:
            raise RuntimeError("503 UNAVAILABLE. This model is overloaded")
        return f"narrative from {runner.agent}"

    monkeypatch.setattr(run_phase1, "build_orchestrator", lambda name: name)
    monkeypatch.setattr(run_phase1, "InMemoryRunner", _FakeRunner)
    monkeypatch.setattr(run_phase1, "_run_agent", fake_run_agent)


def test_orchestrator_walks_down_the_roster_until_a_model_answers(monkeypatch):
    tracer.reset()
    _overloaded(monkeypatch, set(ORCHESTRATOR_MODELS[:4]))

    used, narrative = run_phase1.run_orchestrator("brief")

    assert used == ORCHESTRATOR_MODELS[4]
    assert narrative == f"narrative from {used}"
    fallbacks = [e["data"]["model"] for e in tracer.events if e["type"] == "model_fallback"]
    assert fallbacks == ORCHESTRATOR_MODELS[:4]


def test_orchestrator_gives_up_only_after_the_last_model(monkeypatch):
    tracer.reset()
    _overloaded(monkeypatch, set(ORCHESTRATOR_MODELS))

    with pytest.raises(RuntimeError, match="all orchestrator models failed"):
        run_phase1.run_orchestrator("brief")

    attempts = [e["data"]["model"] for e in tracer.events if e["type"] == "model_attempt"]
    assert attempts == ORCHESTRATOR_MODELS


def test_a_real_error_is_not_mistaken_for_overload(monkeypatch):
    async def broken(runner, _brief):
        raise ValueError("bad tool schema")

    monkeypatch.setattr(run_phase1, "build_orchestrator", lambda name: name)
    monkeypatch.setattr(run_phase1, "InMemoryRunner", _FakeRunner)
    monkeypatch.setattr(run_phase1, "_run_agent", broken)

    with pytest.raises(ValueError, match="bad tool schema"):
        run_phase1.run_orchestrator("brief")


def test_pipeline_hands_the_verifier_a_roster_without_the_orchestrators_model(
    monkeypatch, tmp_path
):
    used = "gemini-3.1-flash-lite"
    seen = {}

    def fake_orchestrator(_brief):
        recorder.reset()
        solve_beam_3d(ss_udl_model())
        return used, "narrative"

    def fake_verifier(_brief, _model, _result, model_names, equation=None):
        seen["roster"] = list(model_names)
        return {"refuted": False, "checks": [], "reasoning": "", "model": model_names[0]}

    monkeypatch.setattr(run_phase1, "run_orchestrator", fake_orchestrator)
    monkeypatch.setattr(run_phase1, "run_verifier", fake_verifier)
    monkeypatch.setattr(run_phase1, "write_report", lambda *a, **k: tmp_path / "r.md")

    tracer.reset()
    run_phase1._pipeline(
        Path("brief.md"), "brief", None, tmp_path / "t.json", time.monotonic()
    )

    assert seen["roster"] == verifier_roster(used)
    assert used not in seen["roster"]

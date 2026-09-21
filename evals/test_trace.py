"""Event-bus tests: schema, ordering, sanitization, ADK callbacks, gate events.

Nothing here needs an API key or the network — the ADK request/response
objects are constructed locally and handed straight to the callbacks.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from google.adk.agents import LlmAgent
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

import agent.verifier as verifier_module
from agent.gates import deterministic_gate
from agent.recorder import recorder, solve_beam_3d
from agent.trace import (
    USER_PREVIEW_CHARS,
    Tracer,
    attach_observers,
    make_model_callbacks,
    make_tool_callbacks,
    tracer,
)
from evals.test_harness_offline import ss_udl_model

EVENT_KEYS = {"seq", "t", "stage", "type", "title", "data"}


@pytest.fixture(autouse=True)
def clean_tracer():
    tracer.reset()
    yield
    tracer.reset()


def test_emit_fills_the_schema_with_monotonic_seq_and_time():
    bus = Tracer()
    bus.emit("brief", "run_start", "read the brief", brief_path="b.md", brief_text="x")
    bus.emit("orchestrator", "stage_start", "orchestrator", stage="orchestrator")

    events = bus.events
    assert [e["seq"] for e in events] == [0, 1]
    assert all(set(e) == EVENT_KEYS for e in events)
    assert events[0]["stage"] == "brief" and events[0]["type"] == "run_start"
    assert events[0]["data"] == {"brief_path": "b.md", "brief_text": "x"}
    # a data key named "stage" must not collide with the positional stage
    assert events[1]["stage"] == "orchestrator"
    assert events[1]["data"] == {"stage": "orchestrator"}

    times = [e["t"] for e in events]
    assert times == sorted(times)
    assert all(t >= 0.0 and round(t, 3) == t for t in times)


def test_emit_sanitizes_numpy_and_unencodable_values():
    bus = Tracer()
    bus.emit(
        "gate",
        "gate_check",
        "numpy payload",
        passed=np.bool_(True),
        value=np.float64(1.5),
        counts=np.array([1, 2, 3]),
        nested={"inner": [np.float32(0.5)]},
        odd=object(),
    )
    data = bus.events[0]["data"]

    assert data["passed"] is True and isinstance(data["passed"], bool)
    assert data["value"] == 1.5 and isinstance(data["value"], float)
    assert data["counts"] == [1, 2, 3]
    assert data["nested"]["inner"][0] == pytest.approx(0.5)
    assert isinstance(data["odd"], str)
    json.dumps(bus.events)
    assert json.loads(bus.to_json()) == bus.events


def test_subscribers_fire_synchronously_in_order():
    bus = Tracer()
    seen_a, seen_b = [], []
    bus.subscribe(seen_a.append)
    bus.subscribe(lambda e: seen_b.append(e["seq"]))

    bus.emit("brief", "run_start", "one")
    assert [e["title"] for e in seen_a] == ["one"]  # already delivered
    bus.emit("report", "run_end", "two")

    assert [e["title"] for e in seen_a] == ["one", "two"]
    assert seen_b == [0, 1]


def test_a_broken_subscriber_never_breaks_the_run():
    bus = Tracer()
    good = []
    bus.subscribe(lambda e: (_ for _ in ()).throw(RuntimeError("viewer died")))
    bus.subscribe(good.append)

    bus.emit("gate", "gate_result", "still fine", passed=True)
    assert len(bus.events) == 1
    assert len(good) == 1


def test_reset_clears_events_but_keeps_subscribers():
    bus = Tracer()
    seen = []
    bus.subscribe(seen.append)
    bus.emit("brief", "run_start", "before reset")

    bus.reset()
    assert bus.events == []

    bus.emit("brief", "run_start", "after reset")
    assert [e["seq"] for e in bus.events] == [0]
    assert [e["title"] for e in seen] == ["before reset", "after reset"]


def _llm_request():
    return LlmRequest(
        model="gemini-3.5-flash",
        contents=[
            types.Content(role="user", parts=[types.Part(text="first turn")]),
            types.Content(role="model", parts=[types.Part(text="thinking")]),
            types.Content(role="user", parts=[types.Part(text="B" * 900)]),
        ],
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(name="solve_beam_3d"),
                        types.FunctionDeclaration(name="closed_form_case"),
                    ]
                )
            ]
        ),
    )


def test_model_callbacks_log_request_and_response_without_altering_them():
    before, after = make_model_callbacks("orchestrator", "gemini-3.5-flash")

    assert before(callback_context=None, llm_request=_llm_request()) is None
    request_event = tracer.events[-1]
    assert request_event["stage"] == "orchestrator"
    assert request_event["type"] == "llm_request"
    assert request_event["data"]["tool_names"] == ["solve_beam_3d", "closed_form_case"]
    assert request_event["data"]["user_preview"] == "B" * USER_PREVIEW_CHARS

    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(text="calling the solver"),
                types.Part(
                    function_call=types.FunctionCall(
                        name="solve_beam_3d", args={"model": {"nodes": []}}
                    )
                ),
            ],
        )
    )
    assert after(callback_context=None, llm_response=response) is None
    response_event = tracer.events[-1]
    assert response_event["type"] == "llm_response"
    assert response_event["title"] == "gemini-3.5-flash called solve_beam_3d"
    assert response_event["data"]["text"] == "calling the solver"
    assert response_event["data"]["function_calls"] == [
        {"name": "solve_beam_3d", "args": {"model": {"nodes": []}}}
    ]


def test_model_callbacks_survive_unexpected_shapes():
    before, after = make_model_callbacks("verifier", "gemini-3-flash-preview")

    assert before(callback_context=None, llm_request=object()) is None
    assert after(callback_context=None, llm_response=object()) is None
    assert [e["type"] for e in tracer.events] == ["llm_request", "llm_response"]
    assert tracer.events[0]["data"]["tool_names"] == []
    json.dumps(tracer.events)


def test_tool_callbacks_log_full_args_and_a_one_line_summary():
    before, after = make_tool_callbacks("orchestrator")

    class _Tool:
        name = "solve_beam_3d"

    recorder.reset()
    model = ss_udl_model()
    result = solve_beam_3d(model)

    assert before(tool=_Tool(), args={"model": model}, tool_context=None) is None
    call_event = tracer.events[-1]
    assert call_event["type"] == "tool_call"
    assert call_event["data"]["name"] == "solve_beam_3d"
    assert call_event["data"]["args"]["model"]["material"]["E"] == model["material"]["E"]

    assert (
        after(
            tool=_Tool(), args={"model": model}, tool_context=None, tool_response=result
        )
        is None
    )
    result_event = tracer.events[-1]
    assert result_event["type"] == "tool_result"
    assert "max |uy|" in result_event["data"]["summary"]
    assert "max |Mz|" in result_event["data"]["summary"]
    assert result_event["data"]["result"]["max_abs"]["uy"] == pytest.approx(
        result["max_abs"]["uy"]
    )
    json.dumps(tracer.events)


def test_tool_summary_falls_back_to_a_truncated_string():
    _before, after = make_tool_callbacks("verifier")

    class _Tool:
        name = "pynite_crosscheck"

    after(tool=_Tool(), args={}, tool_context=None, tool_response={"x": "y" * 500})
    summary = tracer.events[-1]["data"]["summary"]
    assert len(summary) == 120


def test_attach_observers_sets_the_four_callbacks_and_returns_the_agent():
    agent = LlmAgent(name="probe", model="gemini-3.5-flash")
    assert attach_observers(agent, role="orchestrator", model="gemini-3.5-flash") is agent
    for slot in (
        "before_model_callback",
        "after_model_callback",
        "before_tool_callback",
        "after_tool_callback",
    ):
        assert callable(getattr(agent, slot)), slot


def test_verifier_emits_attempts_fallbacks_and_the_verdict(monkeypatch):
    attempts = []

    async def fake_run_once(agent, prompt):
        attempts.append(agent.model)
        if len(attempts) == 1:
            raise RuntimeError("503 UNAVAILABLE model overloaded")
        return (
            '{"refuted": false, "checks": [{"name": "shear", "passed": true, '
            '"detail": "matches"}], "reasoning": "nothing to refute"}'
        )

    monkeypatch.setattr(verifier_module, "_run_once", fake_run_once)
    verdict = verifier_module.run_verifier(
        "brief", ss_udl_model(), {"max_abs": {}}, ["model-a", "model-b"]
    )

    assert verdict["refuted"] is False
    assert verdict["model"] == "model-b"

    events = tracer.events
    assert all(e["stage"] == "verifier" for e in events)
    assert [e["type"] for e in events] == [
        "model_attempt",
        "model_fallback",
        "model_attempt",
        "verdict",
    ]
    assert [e["data"]["model"] for e in events] == [
        "model-a",
        "model-a",
        "model-b",
        "model-b",
    ]
    assert "503" in events[1]["data"]["error"]
    assert events[-1]["data"]["checks"] == verdict["checks"]
    assert events[-1]["data"]["reasoning"] == "nothing to refute"
    json.dumps(events)


def test_verifier_emits_a_verdict_even_when_no_model_runs(monkeypatch):
    async def always_overloaded(agent, prompt):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(verifier_module, "_run_once", always_overloaded)
    verdict = verifier_module.run_verifier(
        "brief", ss_udl_model(), {"max_abs": {}}, ["model-a"]
    )

    assert verdict["refuted"] is True
    assert verdict["model"] is None
    verdicts = [e for e in tracer.events if e["type"] == "verdict"]
    assert len(verdicts) == 1
    assert verdicts[0]["data"]["refuted"] is True


def test_gate_emits_one_event_per_check_plus_one_result():
    recorder.reset()
    model = ss_udl_model()
    result = solve_beam_3d(model)

    tracer.reset()
    gate = deterministic_gate(model, result)
    assert gate["passed"], [c for c in gate["checks"] if not c["passed"]]

    events = tracer.events
    assert all(e["stage"] == "gate" for e in events)
    checks = [e for e in events if e["type"] == "gate_check"]
    results = [e for e in events if e["type"] == "gate_result"]

    assert len(checks) == len(gate["checks"])
    assert len(results) == 1
    assert len(events) == len(checks) + 1
    assert events[-1]["type"] == "gate_result"  # the result comes last
    assert [e["data"]["name"] for e in checks] == [c["name"] for c in gate["checks"]]
    assert [e["data"]["passed"] for e in checks] == [c["passed"] for c in gate["checks"]]
    assert [e["data"]["status"] for e in checks] == [c["status"] for c in gate["checks"]]
    assert all(isinstance(e["data"]["detail"], str) for e in checks)

    summary = results[0]["data"]
    assert summary == {
        "passed": True,
        "n_passed": len(gate["checks"]),
        "n_skipped": 0,  # this model IS the textbook case, so nothing is skipped
        "n_failed": 0,
        "n_total": len(gate["checks"]),
    }
    json.dumps(events)


def test_gate_events_report_a_failing_check():
    recorder.reset()
    model = ss_udl_model()
    result = solve_beam_3d(model)
    result["max_abs"]["uy"] *= 1.1

    tracer.reset()
    gate = deterministic_gate(model, result)
    assert not gate["passed"]

    failed = [
        e["data"]["name"]
        for e in tracer.events
        if e["type"] == "gate_check" and not e["data"]["passed"]
    ]
    assert "closed_form_midspan_deflection" in failed
    summary = tracer.events[-1]["data"]
    assert summary["passed"] is False
    assert summary["n_passed"] == summary["n_total"] - len(failed)

"""Live-view tests: synthetic events only.

No API key, no network, no real Live -- the view is driven through a
non-terminal Console so it takes the plain-printing path.
"""

import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console

from agent.jsonfmt import compact_json
from agent.live_view import (
    DONE,
    FAILED,
    MAX_PANEL_LINES,
    PENDING,
    STAGES,
    LiveView,
    attach_live_view,
)

BRIEF = (
    "Design check for a simply supported bridge beam.\n"
    "Span 25 m, E = 30 GPa, Iz = 0.005 m^4.\n"
    "Uniform load 30 kN/m downward over the full span.\n"
)

MODEL = {
    "nodes": [{"id": f"N{k}", "x": 6.25 * k} for k in range(5)],
    "material": {"E": 30000000000.0, "G": 12500000000.0},
    "section": {"A": 0.5, "Iy": 0.005, "Iz": 0.005, "J": 0.001},
    "supports": {
        "N0": [True, True, True, True, False, False],
        "N4": [False, True, True, False, False, False],
    },
    "distributed_loads": [
        {"element": "all", "direction": "y", "w1": -30000.0, "w2": -30000.0}
    ],
    "point_loads": [],
}

GATE_CHECKS = [
    ("invariant_equilibrium_forces", True, "sum Fy residual 1.8e-10 N"),
    ("closed_form_midspan_deflection", True, "rel err 3.1e-15"),
    ("pynite_agreement", True, "worst scaled mismatch vs PyNite = 2.4e-13"),
]


def events(passed=True, refuted=False):
    """One event of every type in the schema, in run order."""
    gate_events = [
        {
            "seq": 8 + i,
            "t": 4.0 + i,
            "stage": "gate",
            "type": "gate_check",
            "title": f"check {name}",
            "data": {"name": name, "passed": ok, "detail": detail},
        }
        for i, (name, ok, detail) in enumerate(GATE_CHECKS)
    ]
    return [
        {
            "seq": 0,
            "t": 0.0,
            "stage": "brief",
            "type": "run_start",
            "title": "run started",
            "data": {"brief_path": "evals/brief_25m.md", "brief_text": BRIEF},
        },
        {
            "seq": 1,
            "t": 0.01,
            "stage": "orchestrator",
            "type": "stage_start",
            "title": "orchestrator",
            "data": {"stage": "orchestrator"},
        },
        {
            "seq": 2,
            "t": 0.02,
            "stage": "orchestrator",
            "type": "model_attempt",
            "title": "trying gemini-3.8-flash",
            "data": {"role": "orchestrator", "model": "gemini-3.8-flash"},
        },
        {
            "seq": 3,
            "t": 0.5,
            "stage": "orchestrator",
            "type": "model_fallback",
            "title": "gemini-3.8-flash overloaded",
            "data": {
                "role": "orchestrator",
                "model": "gemini-3.8-flash",
                "error": "503 UNAVAILABLE: model overloaded",
            },
        },
        {
            "seq": 4,
            "t": 0.6,
            "stage": "orchestrator",
            "type": "llm_request",
            "title": "prompting gemini-3.5-flash",
            "data": {
                "role": "orchestrator",
                "model": "gemini-3.5-flash",
                "tool_names": [
                    "solve_beam_3d",
                    "closed_form_case",
                    "galerkin_derivation_markdown",
                ],
                "user_preview": BRIEF[:400],
            },
        },
        {
            "seq": 5,
            "t": 1.9,
            "stage": "orchestrator",
            "type": "llm_response",
            "title": "gemini-3.5-flash decided to call solve_beam_3d",
            "data": {
                "role": "orchestrator",
                "model": "gemini-3.5-flash",
                "text": "Meshing the 25 m span into four elements with a node at midspan.",
                "function_calls": [{"name": "solve_beam_3d", "args": {"model": MODEL}}],
            },
        },
        {
            "seq": 6,
            "t": 1.95,
            "stage": "orchestrator",
            "type": "tool_call",
            "title": "Gemini called solve_beam_3d",
            "data": {"name": "solve_beam_3d", "args": {"model": MODEL}},
        },
        {
            "seq": 7,
            "t": 2.1,
            "stage": "orchestrator",
            "type": "tool_result",
            "title": "solve_beam_3d returned",
            "data": {
                "name": "solve_beam_3d",
                "summary": "solved 4 elements; midspan uy = -1.017e-01 m",
                "result": {
                    "max_abs": {
                        "uy": 0.10172526,
                        "Mz": 2343750.0,
                        "Vy": 375000.0,
                    }
                },
            },
        },
        *gate_events,
        {
            "seq": 11,
            "t": 7.0,
            "stage": "gate",
            "type": "gate_result",
            "title": "deterministic gate",
            "data": {
                "passed": passed,
                "n_passed": len(GATE_CHECKS) if passed else len(GATE_CHECKS) - 1,
                "n_total": len(GATE_CHECKS),
            },
        },
        {
            "seq": 12,
            "t": 7.1,
            "stage": "verifier",
            "type": "verdict",
            "title": "verifier verdict",
            "data": {
                "refuted": refuted,
                "checks": [
                    {
                        "name": "pynite_crosscheck",
                        "passed": not refuted,
                        "detail": "independent solve agrees to 1e-12",
                    }
                ],
                "reasoning": "Recomputed with PyNite and the textbook formula; both agree.",
                "model": "gemini-3-flash-preview",
            },
        },
        {
            "seq": 13,
            "t": 9.2,
            "stage": "report",
            "type": "run_end",
            "title": "run finished",
            "data": {
                "passed": passed and not refuted,
                "report_path": "results/phase1_report.md",
                "trace_path": "results/phase1_trace.jsonl",
                "duration_s": 9.237,
            },
        },
    ]


class FakeTracer:
    """Minimal stand-in for the run tracer: subscribe + emit + event history."""

    def __init__(self):
        self.events = []
        self._subscribers = []

    def subscribe(self, callback):
        self._subscribers.append(callback)
        return lambda: self._subscribers.remove(callback)

    def emit(self, event):
        self.events.append(event)
        for callback in list(self._subscribers):
            callback(event)


class ReceiptTracer:
    """Decorator-style tracer: subscribe hands the callback back (agent.trace)."""

    def __init__(self):
        self.events = []
        self._subscribers = []

    def subscribe(self, fn):
        self._subscribers.append(fn)
        return fn

    def emit(self, event):
        self.events.append(event)
        for callback in list(self._subscribers):
            callback(event)


class ListTracer:
    """A tracer that only exposes a listeners list (duck-typed subscription)."""

    def __init__(self):
        self.listeners = []

    def emit(self, event):
        for callback in list(self.listeners):
            callback(event)


def capture_console():
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def drive(all_events, tracer=None, console=None):
    """Feed events through the view, returning (output, view)."""
    tracer = tracer or FakeTracer()
    console = console or capture_console()
    with attach_live_view(tracer, console=console) as view:
        for event in all_events:
            tracer.emit(event)
    return console.file.getvalue(), view


def test_every_event_type_is_rendered():
    out, view = drive(events())

    # the input the user wants to see, verbatim
    assert "Brief (plain English)" in out
    assert "simply supported bridge beam" in out
    assert "Uniform load 30 kN/m downward" in out

    # the orchestrator's reasoning and model roster
    assert "gemini-3.8-flash" in out
    assert "503 UNAVAILABLE" in out
    assert "gemini-3.5-flash" in out
    assert "solve_beam_3d, closed_form_case, galerkin_derivation_markdown" in out
    assert "decided to call: solve_beam_3d" in out

    # the tool call and its result
    assert "solve_beam_3d" in out
    assert "solved 4 elements" in out
    assert "1.017253e-01" in out  # max_abs uy, engineering notation

    # the verifier and the final verdict
    assert "verifier found no refutation" in out
    assert "gemini-3-flash-preview" in out
    assert "GATE: PASS" in out
    assert view.event_count == len(events())


def panel(event):
    """Render one event on its own, returning the printed lines."""
    console = capture_console()
    LiveView(console=console).handle(event)
    return console.file.getvalue().rstrip("\n").split("\n")


def body(lines):
    """Panel lines with the box drawing stripped off."""
    return [line.strip("│ ").rstrip() for line in lines]


def tool_call_event(args):
    return {"seq": 6, "t": 1.95, "stage": "orchestrator", "type": "tool_call",
            "title": "Gemini called solve_beam_3d",
            "data": {"name": "solve_beam_3d", "args": args}}


def test_tool_call_shows_the_model_dict_compacted():
    out, _ = drive(events())

    for key in MODEL:
        assert f'"{key}"' in out, key
    assert "30000000000.0" in out  # E in Pa, not truncated or reformatted
    assert "12500000000.0" in out  # G
    assert "-30000.0" in out  # w1 = w2, downward N/m
    assert "6.25" in out
    assert "0.001" in out  # J
    # the six-flag support vector is one line now, not twelve
    assert '"N0": [true, true, true, true, false, false]' in out


def test_no_lone_boolean_lines_survive():
    lines = body(panel(tool_call_event({"model": MODEL})))

    assert [line for line in lines if line in ("true,", "false,", "true", "false")] == []


def test_long_node_array_folds_and_says_how_many():
    # A fine mesh, not the 5-node demo: folding only earns its marker line
    # once it hides more than a couple of entries.
    meshed = {**MODEL, "nodes": [{"id": f"N{k}", "x": 2.5 * k} for k in range(11)]}
    lines = body(panel(tool_call_event({"model": meshed})))
    joined = "\n".join(lines)

    assert '{"id": "N0", "x": 0.0}' in joined
    assert '{"id": "N3", "x": 7.5}' in joined
    assert "... 7 more items (11 total)" in joined


def test_the_real_model_dict_fits_the_terminal(tool_call_args):
    """The captured run's model dict, or a stand-in of the same shape."""
    lines = panel(tool_call_event(tool_call_args))

    assert len(lines) < 40, len(lines)
    joined = "\n".join(body(lines))
    for key in tool_call_args["model"]:
        assert f'"{key}"' in joined, key


def test_a_huge_payload_is_capped_with_a_pointer_to_the_html():
    huge = {f"node_{n}": {"ux": 0.0, "uy": -0.7247924804687, "uz": -0.0,
                          "rx": 0.0, "ry": 0.0, "rz": -0.0895182291666}
            for n in range(60)}
    lines = body(panel(tool_call_event(huge)))
    notes = [line for line in lines if "see the HTML trace" in line]

    assert len(notes) == 1, lines[-4:]
    hidden = int(notes[0].split()[1])
    json_lines = [line for line in lines if line.startswith(("{", '"', "}"))]
    assert len(json_lines) <= MAX_PANEL_LINES
    assert hidden > 0


def test_short_payloads_get_no_overflow_note():
    lines = body(panel(tool_call_event({"case": "ss_udl", "L": 25.0})))

    assert "see the HTML trace" not in "\n".join(lines)


def test_tool_result_panel_carries_the_solver_numbers(tool_result_payload):
    event = {"seq": 7, "t": 2.1, "stage": "orchestrator", "type": "tool_result",
             "title": "solve_beam_3d returned",
             "data": {"name": "solve_beam_3d", "summary": "midspan uy = -1.017e+00 m",
                      "result": tool_result_payload}}
    lines = body(panel(event))
    joined = "\n".join(lines)

    assert "midspan uy" in joined  # the summary still leads
    assert "1.017253e+00" in joined  # max_abs uy, from the headline table
    assert '"displacements"' in joined  # the payload itself now reaches the terminal
    json_lines = [line for line in lines if line.startswith(("{", '"', "}"))]
    assert len(json_lines) <= MAX_PANEL_LINES
    # a result this long is cut, and says so rather than scrolling the run away
    if len(compact_json(tool_result_payload).split("\n")) > MAX_PANEL_LINES:
        assert "see the HTML trace" in joined


def test_the_cap_applies_on_the_live_path_too():
    """Live and plain-print render through the same _json_block."""
    huge = {f"node_{n}": {"ux": 0.0, "uy": -0.7247924804687, "uz": -0.0,
                          "rx": 0.0, "ry": 0.0, "rz": -0.0895182291666}
            for n in range(60)}
    console = Console(file=io.StringIO(), force_terminal=True, width=120)
    view = LiveView(console=console)
    assert view.use_live is True
    view.start()
    view.handle(tool_call_event(huge))
    view.stop()

    assert "see the HTML trace" in console.file.getvalue()


def test_each_gate_check_appears_with_its_verdict():
    out, _ = drive(events())

    for name, _passed, detail in GATE_CHECKS:
        assert name in out
        assert detail in out
    assert "PASS " in out
    assert "deterministic gate: PASS" in out
    assert "3/3 checks passed" in out


def test_failing_run_renders_fail_banner_and_marks_stages():
    out, view = drive(events(passed=False, refuted=True))

    assert "deterministic gate: FAIL" in out
    assert "verifier REFUTED the results" in out
    assert "GATE: FAIL" in out
    assert view.stage_state["gate"] == FAILED
    assert view.stage_state["verifier"] == FAILED


def test_stage_tracker_advances_and_finishes_done():
    _out, view = drive(events())

    assert list(view.stage_state) == list(STAGES)
    assert view.stage_state["orchestrator"] == DONE
    assert view.stage_state["gate"] == DONE
    assert view.stage_state["report"] == DONE
    assert view.stage_state["brief"] in (DONE, PENDING)


def test_non_tty_never_starts_live_and_emits_no_ansi():
    console = capture_console()
    tracer = FakeTracer()
    with attach_live_view(tracer, console=console) as view:
        assert view.use_live is False
        assert view._live is None
        for event in events():
            tracer.emit(event)

    out = console.file.getvalue()
    assert "\x1b[" not in out  # no cursor control, safe in a pipe
    assert out.strip()


def test_malformed_events_do_not_raise():
    console = capture_console()
    tracer = FakeTracer()
    broken = [
        {},
        None,
        "not an event at all",
        {"type": "tool_call"},  # no data
        {"type": "tool_call", "data": None},
        {"type": "tool_call", "data": {"name": "solve_beam_3d", "args": object()}},
        {"type": "gate_check", "data": {}},
        {"type": "verdict", "data": {"checks": "not a list"}},
        {"type": "run_end", "data": {"duration_s": "soon"}},
        {"type": "no_such_type", "title": "mystery", "data": {"k": 1}},
        {"seq": 99, "stage": "nowhere", "type": "stage_start", "data": {"stage": "x"}},
    ]
    with attach_live_view(tracer, console=console) as view:
        for event in broken:
            tracer.emit(event)

    assert view.event_count == len(broken)
    assert console.file.getvalue().strip()


def test_view_unsubscribes_when_the_block_exits():
    console = capture_console()
    tracer = FakeTracer()
    with attach_live_view(tracer, console=console) as view:
        tracer.emit(events()[0])
    seen = view.event_count
    before = console.file.getvalue()

    tracer.emit(events()[-1])
    assert view.event_count == seen
    assert console.file.getvalue() == before


def test_events_recorded_before_attach_are_replayed():
    console = capture_console()
    tracer = FakeTracer()
    tracer.events.append(events()[0])  # emitted before anyone was listening
    with attach_live_view(tracer, console=console):
        pass

    assert "simply supported bridge beam" in console.file.getvalue()


def test_subscribes_to_a_listeners_list_tracer():
    console = capture_console()
    tracer = ListTracer()
    with attach_live_view(tracer, console=console):
        assert len(tracer.listeners) == 1
        tracer.emit(events()[0])
    assert tracer.listeners == []
    assert "Brief (plain English)" in console.file.getvalue()


def test_detaches_from_a_tracer_whose_subscribe_returns_the_callback():
    console = capture_console()
    tracer = ReceiptTracer()
    with attach_live_view(tracer, console=console) as view:
        tracer.emit(events()[0])
    assert tracer._subscribers == []

    tracer.emit(events()[-1])
    assert view.event_count == 1
    assert "GATE: PASS" not in console.file.getvalue()


def test_works_against_the_real_tracer():
    from agent.trace import Tracer

    console = capture_console()
    tracer = Tracer()
    with attach_live_view(tracer, console=console) as view:
        tracer.emit("brief", "run_start", "run started", brief_path="b.md", brief_text=BRIEF)
        tracer.emit("orchestrator", "tool_call", "calls solve_beam_3d",
                    name="solve_beam_3d", args={"model": MODEL})
        tracer.emit("report", "run_end", "run finished", passed=True,
                    report_path="results/phase1_report.md",
                    trace_path="results/phase1_trace.jsonl", duration_s=9.2)
    out = console.file.getvalue()
    assert view.event_count == 3
    assert "simply supported bridge beam" in out
    assert "solve_beam_3d" in out
    assert "30000000000.0" in out
    assert "GATE: PASS" in out

    tracer.emit("report", "run_end", "again", passed=False)
    assert view.event_count == 3  # detached at block exit


def test_tracer_without_a_subscribe_hook_is_a_clear_error():
    try:
        with attach_live_view(object(), console=capture_console()):
            pass
    except TypeError as exc:
        assert "live view" in str(exc)
    else:
        raise AssertionError("expected a TypeError for an unsubscribable tracer")


def test_handle_survives_a_broken_console():
    class ExplodingConsole(Console):
        def print(self, *args, **kwargs):
            raise RuntimeError("terminal went away")

    view = LiveView(console=ExplodingConsole(file=io.StringIO(), force_terminal=False))
    view.start()
    for event in events():
        view.handle(event)  # must not raise
    view.stop()
    assert view.event_count == len(events())

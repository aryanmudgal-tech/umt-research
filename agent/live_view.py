"""Live terminal view of a Phase-1 run.

Subscribes to the run tracer and renders every event as it arrives, so the
whole pipeline is visible while it happens: the English brief, the model dict
the orchestrator builds, each deterministic check, the verifier's verdict and
the final gate.

JSON payloads are rendered by agent.jsonfmt.compact_json, the same formatter
the HTML viewer uses, and a panel stops at MAX_PANEL_LINES lines with a note
pointing at the HTML trace, which carries the payload whole. A terminal is a
poor place to scroll 300 lines of one array past a reader.

Usage:

    from agent.live_view import attach_live_view

    with attach_live_view(tracer):
        ...run the pipeline...

Layout: a pinned status block (title, clock, stage tracker) sits at the bottom
of the terminal and the event log scrolls above it, which is how rich composes
Live with console output. The log therefore stays in the scrollback in full --
a long solve_beam_3d model dict is still there after the run.
"""

import json
import sys
import time
from contextlib import contextmanager

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from agent.gates import FAIL, PASS, SKIPPED, check_status
from agent.jsonfmt import compact_json

# How each gate-check status reads in the log. SKIPPED is deliberately not
# green: a check that did not run has verified nothing.
_STATUS_STYLE = {
    PASS: (" PASS ", "bold green"),
    FAIL: (" FAIL ", "bold red"),
    SKIPPED: (" SKIP ", "bold yellow"),
}

DEFAULT_TITLE = "The 25-Meter Agent"

# a JSON panel stops here; the rest is one line of prose and the HTML trace
MAX_PANEL_LINES = 40
OVERFLOW_NOTE = "... {hidden} more lines - see the HTML trace"

STAGES = ("brief", "orchestrator", "gate", "verifier", "report")

PENDING, RUNNING, DONE, FAILED = "pending", "running", "done", "failed"

_MARKERS = {
    PENDING: ("-", "dim"),
    RUNNING: (">", "bold cyan"),
    DONE: ("✓", "bold green"),
    FAILED: ("✗", "bold red"),
}

EVENT_TYPES = (
    "run_start",
    "stage_start",
    "model_attempt",
    "model_fallback",
    "llm_request",
    "llm_response",
    "tool_call",
    "tool_result",
    "gate_check",
    "gate_result",
    "verdict",
    "run_end",
)

_SUBSCRIBE_HOOKS = ("subscribe", "add_listener", "add_subscriber", "on_event", "listen")
_UNSUBSCRIBE_HOOKS = ("unsubscribe", "remove_listener", "off")
_LISTENER_LISTS = ("listeners", "subscribers", "_listeners", "_subscribers")


def _fmt(value) -> str:
    """Engineering-notation magnitude for a number, str() for anything else."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    return f"{value:.6e}"


def _json_block(data):
    """Compacted, syntax-highlighted JSON, capped at MAX_PANEL_LINES lines.

    The cap is the whole point: one 300-line array used to push the rest of
    the run out of the scrollback. Both the Live and the plain-print paths
    render through here, so both are capped.
    """
    try:
        text = compact_json(data)
    except (TypeError, ValueError, RecursionError):
        text = repr(data)
    lines = text.split("\n")
    hidden = len(lines) - MAX_PANEL_LINES
    body = Syntax(
        "\n".join(lines[:MAX_PANEL_LINES]) if hidden > 0 else text,
        "json",
        theme="ansi_dark",
        word_wrap=True,
        background_color="default",
    )
    if hidden <= 0:
        return body
    return Group(body, Text(OVERFLOW_NOTE.format(hidden=hidden), style="dim"))


def _indented(text: str, style: str):
    """Left-margin block, so wrapped and multi-line prose stays in its column."""
    return Padding(Text(text, style=style), (0, 0, 0, 5))


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _text(value) -> str:
    return "" if value is None else str(value)


class LiveView:
    """Renders tracer events. One instance per run; `handle` is the sink."""

    def __init__(self, console=None, title: str = DEFAULT_TITLE):
        if console is None:
            self.console = Console()
            is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        else:
            self.console = console
            is_tty = bool(console.is_terminal)
        # Live in a pipe writes cursor-control garbage, so decide once, here.
        self.use_live = is_tty
        self.title = title
        self.stage_state = {name: PENDING for name in STAGES}
        self.current_stage = None
        self.event_count = 0
        self._t0 = time.monotonic()
        self._live = None

    # ---------------------------------------------------------------- driving

    def start(self):
        if self.use_live:
            try:
                self._live = Live(
                    _Screen(self),
                    console=self.console,
                    refresh_per_second=8,
                    vertical_overflow="visible",
                )
                self._live.start()
                return self
            except Exception:  # no Live is better than no run
                self._live = None
                self.use_live = False
        self._safe_print(Rule(Text(self.title, style="bold"), style="dim"))
        return self

    def stop(self):
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            finally:
                self._live = None
        # leave the finished stage tracker in the scrollback
        self._safe_print(self._tracker())

    def handle(self, event):
        """Render one event. Never raises: the pipeline outranks the view."""
        try:
            self._handle(event)
        except Exception as exc:  # a view bug must not abort a live run
            self._safe_print(
                Text(f"  [live_view could not render {event!r}: {exc}]", style="dim red")
            )

    def _handle(self, event):
        self.event_count += 1
        if not isinstance(event, dict):
            self._emit(Text(f"  {event!r}", style="dim"))
            return
        self._note_stage(event)
        self._emit(self._render(event))

    def _emit(self, renderable):
        self._safe_print(renderable)
        if self._live is not None:
            try:
                self._live.refresh()
            except Exception:
                pass

    def _safe_print(self, renderable):
        # rich renders lazily, so print is where a bad renderable actually blows up
        try:
            self.console.print(renderable)
        except Exception as exc:
            try:
                self.console.print(f"[live_view print error: {exc}]")
            except Exception:
                pass

    # ------------------------------------------------------------- stage state

    def _note_stage(self, event):
        etype = _text(event.get("type"))
        data = _as_dict(event.get("data"))
        if etype == "run_start":
            self._t0 = time.monotonic()

        stage = data.get("stage") if etype == "stage_start" else event.get("stage")
        if stage in self.stage_state:
            for prior in STAGES[: STAGES.index(stage)]:
                if self.stage_state[prior] == RUNNING:
                    self.stage_state[prior] = DONE
            if self.stage_state[stage] == PENDING:
                self.stage_state[stage] = RUNNING
            self.current_stage = stage

        if etype == "gate_result" and not data.get("passed"):
            self.stage_state["gate"] = FAILED
        elif etype == "verdict" and data.get("refuted"):
            self.stage_state["verifier"] = FAILED
        elif etype == "run_end":
            final = DONE if data.get("passed") else FAILED
            for name in STAGES:
                if self.stage_state[name] == RUNNING:
                    self.stage_state[name] = final

    # ---------------------------------------------------------------- chrome

    def _elapsed(self) -> float:
        return time.monotonic() - self._t0

    def _header(self):
        grid = Table.grid(expand=True)
        grid.add_column(justify="left")
        grid.add_column(justify="right")
        grid.add_row(
            Text(self.title, style="bold"),
            Text(
                f"{self._elapsed():.1f}s   stage: {self.current_stage or '-'}",
                style="dim",
            ),
        )
        return Panel(grid, border_style="bright_black", padding=(0, 1))

    def _tracker(self):
        line = Text("  ")
        for index, name in enumerate(STAGES):
            if index:
                line.append("  ->  ", style="dim")
            marker, style = _MARKERS[self.stage_state[name]]
            line.append(f"{marker} {name}", style=style)
        return line

    def _screen(self):
        return Group(self._header(), self._tracker(), Text(""))

    # --------------------------------------------------------------- renderers

    def _render(self, event):
        etype = _text(event.get("type"))
        data = _as_dict(event.get("data"))
        if etype not in EVENT_TYPES:
            return self._render_unknown(event, etype, data)
        return getattr(self, f"_r_{etype}")(data)

    def _render_unknown(self, event, etype, data):
        title = _text(event.get("title")) or etype or "event"
        line = Text(f"  {title}", style="dim")
        if data:
            line.append(f"  {json.dumps(data, default=str)}", style="dim")
        return line

    def _r_run_start(self, data):
        brief = _text(data.get("brief_text")).strip() or "(empty brief)"
        path = _text(data.get("brief_path"))
        return Panel(
            Text(brief),
            title="Brief (plain English)",
            title_align="left",
            subtitle=path or None,
            subtitle_align="right",
            border_style="bright_white",
            padding=(1, 2),
        )

    def _r_stage_start(self, data):
        return Rule(Text(f" {_text(data.get('stage'))} ", style="bold"), style="dim")

    def _r_model_attempt(self, data):
        return Text(
            f"  {_text(data.get('role'))} model: {_text(data.get('model'))}", style="dim"
        )

    def _r_model_fallback(self, data):
        return Text(
            f"  {_text(data.get('role'))} falling back from "
            f"{_text(data.get('model'))}: {_text(data.get('error'))}",
            style="yellow",
        )

    def _r_llm_request(self, data):
        tools = data.get("tool_names") or []
        line = Text("  -> ", style="dim")
        line.append(_text(data.get("model")) or "?", style="dim cyan")
        if isinstance(tools, (list, tuple)) and tools:
            line.append(f"  (tools: {', '.join(_text(t) for t in tools)})", style="dim")
        preview = _text(data.get("user_preview")).strip()
        if not preview:
            return line
        return Group(line, _indented(preview, "dim italic"))

    def _r_llm_response(self, data):
        parts = []
        text = _text(data.get("text")).strip()
        if text:
            parts.append(_indented(text, "dim"))
        calls = data.get("function_calls")
        if isinstance(calls, (list, tuple)):
            for call in calls:
                name = _text(_as_dict(call).get("name")) or "?"
                parts.append(Text(f"  decided to call: {name}", style="cyan"))
        return Group(*parts) if parts else Text("     (no content)", style="dim")

    def _r_tool_call(self, data):
        name = _text(data.get("name")) or "?"
        return Panel(
            _json_block(data.get("args", {})),
            title=f"tool call  {name}",
            title_align="left",
            border_style="cyan",
            padding=(0, 1),
        )

    def _r_tool_result(self, data):
        name = _text(data.get("name")) or "?"
        body = [Text(_text(data.get("summary")) or "(no summary)")]
        result = data.get("result")
        max_abs = _as_dict(_as_dict(result).get("max_abs"))
        if max_abs:
            table = Table(box=box.SIMPLE, pad_edge=False, header_style="dim")
            table.add_column("max abs", style="dim")
            table.add_column("value", justify="right")
            for key, value in max_abs.items():
                table.add_row(_text(key), _fmt(value))
            body.append(table)
        if result is not None:
            body.append(_json_block(result))
        return Panel(
            Group(*body),
            title=f"tool result  {name}",
            title_align="left",
            border_style="dim cyan",
            padding=(0, 1),
        )

    def _r_gate_check(self, data):
        # A skipped check verified nothing: it reads SKIPPED, never PASS.
        status = check_status(data)
        label, style = _STATUS_STYLE[status]
        line = Text("  ")
        line.append(label, style=style)
        line.append(f" {_text(data.get('name'))}")
        detail = _text(data.get("detail"))
        if detail:
            line.append(f"  {detail}", style="dim")
        return line

    def _r_gate_result(self, data):
        passed = bool(data.get("passed"))
        style = "green" if passed else "red"
        body = Text()
        body.append(
            f"deterministic gate: {'PASS' if passed else 'FAIL'}", style=f"bold {style}"
        )
        tally = f"   {data.get('n_passed', '?')}/{data.get('n_total', '?')} checks passed"
        skipped = data.get("n_skipped")
        if skipped:
            tally += f", {skipped} skipped"
        body.append(tally, style="dim")
        return Panel(body, border_style=style, padding=(0, 1))

    def _r_verdict(self, data):
        refuted = bool(data.get("refuted"))
        style = "red" if refuted else "green"
        parts = [
            Text(
                "verifier REFUTED the results" if refuted else "verifier found no refutation",
                style=f"bold {style}",
            )
        ]
        checks = data.get("checks")
        if isinstance(checks, (list, tuple)):
            for check in checks:
                check = _as_dict(check)
                ok = bool(check.get("passed"))
                line = Text("  ")
                line.append("PASS " if ok else "FAIL ", style="green" if ok else "red")
                line.append(_text(check.get("name")))
                detail = _text(check.get("detail"))
                if detail:
                    line.append(f"  {detail}", style="dim")
                parts.append(line)
        reasoning = _text(data.get("reasoning")).strip()
        if reasoning:
            parts.append(Text(""))
            parts.append(Text(reasoning, style="dim"))
        return Panel(
            Group(*parts),
            title=f"verifier verdict  {_text(data.get('model')) or 'n/a'}",
            title_align="left",
            border_style=style,
            padding=(0, 1),
        )

    def _r_run_end(self, data):
        passed = bool(data.get("passed"))
        style = "green" if passed else "red"
        banner = Text(
            f"  GATE: {'PASS' if passed else 'FAIL'}  ",
            style=f"bold white on {style}",
            justify="center",
        )
        facts = Table.grid(padding=(0, 2))
        facts.add_column(style="dim")
        facts.add_column()
        duration = data.get("duration_s")
        shown = f"{duration:.3f} s" if isinstance(duration, (int, float)) else "n/a"
        facts.add_row("duration", shown)
        facts.add_row("report", _text(data.get("report_path")) or "n/a")
        facts.add_row("trace", _text(data.get("trace_path")) or "n/a")
        return Panel(
            Group(banner, Text(""), facts),
            title="run complete",
            title_align="left",
            border_style=style,
            padding=(1, 2),
        )


class _Screen:
    """Live's renderable proxy, so auto-refresh keeps the clock moving."""

    def __init__(self, view: LiveView):
        self._view = view

    def __rich__(self):
        return self._view._screen()


def _remover(tracer, callback):
    def remove():
        for name in _UNSUBSCRIBE_HOOKS:
            hook = getattr(tracer, name, None)
            if callable(hook):
                hook(callback)
                return
        for name in _LISTENER_LISTS:
            bucket = getattr(tracer, name, None)
            if isinstance(bucket, list) and callback in bucket:
                bucket.remove(callback)
                return

    return remove


def _subscribe(tracer, callback):
    """Attach `callback` to a tracer, returning an unsubscribe callable."""
    for name in _SUBSCRIBE_HOOKS:
        hook = getattr(tracer, name, None)
        if callable(hook):
            handle = hook(callback)
            # decorator-style hooks hand the callback straight back; that is a
            # registration receipt, not an unsubscribe.
            if callable(handle) and handle is not callback:
                return handle
            return _remover(tracer, callback)
    for name in _LISTENER_LISTS:
        bucket = getattr(tracer, name, None)
        if isinstance(bucket, list):
            bucket.append(callback)
            return _remover(tracer, callback)
    raise TypeError(
        f"{type(tracer).__name__} has no subscribe()/add_listener() hook and no "
        "listeners list; cannot attach the live view"
    )


@contextmanager
def attach_live_view(tracer, console=None, title: str = DEFAULT_TITLE):
    """Render tracer events in the terminal for the duration of the block."""
    view = LiveView(console=console, title=title)
    unsubscribe = _subscribe(tracer, view.handle)
    view.start()
    try:
        for event in list(getattr(tracer, "events", None) or []):
            view.handle(event)  # anything emitted before we attached
        yield view
    finally:
        try:
            unsubscribe()
        except Exception:
            pass
        view.stop()

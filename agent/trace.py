"""Live event bus for a Phase-1 run, plus ADK observer callbacks.

Every stage pushes plain dicts through the module-level `tracer`; subscribers
(a terminal UI, a file writer) are called synchronously as the run happens.

The ADK callbacks built here always return None, so they only watch the run:
ADK proceeds with its own request, response and tool result untouched.
"""

import json
import time

from agent.recorder import to_plain

USER_PREVIEW_CHARS = 400
RESULT_SUMMARY_CHARS = 120


def _json_safe(obj):
    """to_plain, then stringify whatever json still could not encode."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    plain = to_plain(obj)
    if isinstance(plain, (dict, list)):
        return _json_safe(plain)
    if plain is None or isinstance(plain, (bool, int, float, str)):
        return plain
    return str(plain)


class Tracer:
    """Append-only event log with synchronous fan-out to subscribers."""

    def __init__(self):
        self._events = []
        self._subscribers = []
        self._t0 = time.monotonic()

    def reset(self):
        """Drop every event and restart the clock. Subscribers stay attached."""
        self._events.clear()
        self._t0 = time.monotonic()

    def subscribe(self, fn):
        self._subscribers.append(fn)
        return fn

    @property
    def events(self):
        return list(self._events)

    def emit(self, stage, type, title, /, **data):
        # stage/type/title are positional-only so a payload key of the same
        # name (e.g. stage_start's {"stage": ...}) cannot collide with them.
        event = {
            "seq": len(self._events),
            "t": round(time.monotonic() - self._t0, 3),
            "stage": stage,
            "type": type,
            "title": title,
            "data": _json_safe(data),
        }
        self._events.append(event)
        for subscriber in list(self._subscribers):
            try:
                subscriber(event)
            except Exception:
                pass  # a broken viewer must never take the pipeline down
        return event

    def to_json(self, indent=2):
        return json.dumps(self._events, indent=indent)


tracer = Tracer()


def _tool_names(llm_request):
    names = []
    config = getattr(llm_request, "config", None)
    for tool in getattr(config, "tools", None) or []:
        for decl in getattr(tool, "function_declarations", None) or []:
            name = getattr(decl, "name", None)
            if name:
                names.append(name)
    return names or list(getattr(llm_request, "tools_dict", None) or [])


def _last_user_text(llm_request):
    for content in reversed(getattr(llm_request, "contents", None) or []):
        if getattr(content, "role", None) != "user":
            continue
        text = "".join(
            getattr(part, "text", None) or ""
            for part in getattr(content, "parts", None) or []
        )
        if text:
            return text[:USER_PREVIEW_CHARS]
    return ""


def _response_text(llm_response):
    content = getattr(llm_response, "content", None)
    return "".join(
        getattr(part, "text", None) or ""
        for part in getattr(content, "parts", None) or []
    )


def _function_calls(llm_response):
    getter = getattr(llm_response, "get_function_calls", None)
    raw = getter() if callable(getter) else None
    if raw is None:
        content = getattr(llm_response, "content", None)
        raw = [
            getattr(part, "function_call", None)
            for part in getattr(content, "parts", None) or []
        ]
    calls = []
    for call in raw or []:
        if call is None:
            continue
        calls.append(
            {
                "name": getattr(call, "name", "") or "",
                "args": dict(getattr(call, "args", None) or {}),
            }
        )
    return calls


def _tool_name(tool):
    try:
        return getattr(tool, "name", None) or type(tool).__name__
    except Exception:
        return "unknown_tool"


def _summarize(name, result):
    """One human-readable line about a tool result."""
    if name == "solve_beam_3d":
        try:
            peak = result["max_abs"]
            return (
                f"max |uy| = {abs(float(peak['uy'])):.6e} m, "
                f"max |Mz| = {abs(float(peak['Mz'])):.6e} N*m"
            )
        except Exception:
            pass
    if name == "solve_with_equation":
        # The raw dict truncates inside the displacements, which hides how the
        # element matrices were integrated - and a numerically integrated
        # element must not be able to read as an exact one anywhere.
        try:
            peak, how = result["max_abs"], result["integration"]
            order = f" ({how['points']} points)" if how["method"] == "quadrature" else ""
            return (
                f"max |v| = {abs(float(peak['v'])):.6e} m, "
                f"max |M| = {abs(float(peak['moment'])):.6e} N*m, "
                f"element integrals {how['method']}{order}"
            )
        except Exception:
            pass
    return str(result)[:RESULT_SUMMARY_CHARS]


def make_model_callbacks(role, model):
    """(before, after) LLM callbacks logging the request and the reply.

    ADK 2.8 invokes these by keyword; **_extra absorbs any argument a later
    version adds, and every extraction is guarded so a shape change degrades
    to a partial event instead of a crash.
    """

    def before(callback_context=None, llm_request=None, **_extra):
        data = {"role": role, "model": model, "tool_names": [], "user_preview": ""}
        try:
            data["tool_names"] = _tool_names(llm_request)
            data["user_preview"] = _last_user_text(llm_request)
        except Exception:
            pass
        title = f"{role} -> {model} ({len(data['tool_names'])} tools)"
        tracer.emit(role, "llm_request", title, **data)
        return None

    def after(callback_context=None, llm_response=None, **_extra):
        # Streaming chunks would flood the log; only assembled replies count.
        if getattr(llm_response, "partial", False):
            return None
        data = {"role": role, "model": model, "text": "", "function_calls": []}
        try:
            data["text"] = _response_text(llm_response)
            data["function_calls"] = _function_calls(llm_response)
        except Exception:
            pass
        called = [c["name"] for c in data["function_calls"] if c["name"]]
        title = f"{model} called {', '.join(called)}" if called else f"{model} answered"
        tracer.emit(role, "llm_response", title, **data)
        return None

    return before, after


def make_tool_callbacks(role):
    """(before, after) tool callbacks logging full args and the full result."""

    def before(tool=None, args=None, tool_context=None, **_extra):
        name = _tool_name(tool)
        payload = {}
        try:
            payload = dict(args or {})
        except Exception:
            payload = {"args": args}
        tracer.emit(role, "tool_call", f"{role} calls {name}", name=name, args=payload)
        return None

    def after(tool=None, args=None, tool_context=None, tool_response=None, **_extra):
        name = _tool_name(tool)
        tracer.emit(
            role,
            "tool_result",
            f"{name} returned",
            name=name,
            summary=_summarize(name, tool_response),
            result=tool_response,
        )
        return None

    return before, after


def attach_observers(agent, role, model):
    """Wire the four observer callbacks onto an LlmAgent and return it."""
    agent.before_model_callback, agent.after_model_callback = make_model_callbacks(
        role, model
    )
    agent.before_tool_callback, agent.after_tool_callback = make_tool_callbacks(role)
    return agent

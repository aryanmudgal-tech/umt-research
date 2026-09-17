"""Self-contained HTML viewer for a recorded Phase-1 trace.

Renders the harness event list as one offline file: inline CSS and JS only,
no CDN and no network, so it opens by double-click and can be handed to
anyone. Every string goes through html.escape — the events carry user text,
model prose and JSON braces.

JSON goes through agent.jsonfmt.compact_json, the same formatter the terminal
view uses, and each block is capped so no one payload can own the page; when
compaction folds anything the raw dump is a click away underneath.
"""

import datetime
import html
import json
from pathlib import Path

from agent.jsonfmt import compact_json, was_folded

STAGE_ORDER = ["brief", "orchestrator", "gate", "verifier", "report"]

# a JSON block scrolls past this instead of growing the page
PRE_MAX_PX = 420

STAGE_LABEL = {
    "brief": "Brief",
    "orchestrator": "Orchestrator",
    "gate": "Deterministic gate",
    "verifier": "Independent verifier",
    "report": "Report",
}

STAGE_BLURB = {
    "brief": "the plain-English input, exactly as written",
    "orchestrator": "Gemini reads the brief, builds the model dict, calls the tools",
    "gate": "arithmetic only - closed form, PyNite, physics invariants; no LLM",
    "verifier": "a separate agent on a different model, trying to refute",
    "report": "final verdict and artifacts",
}

_CSS = """
:root {
  color-scheme: light dark;
  --paper:#f2efe7; --panel:#fbf9f4; --panel-2:#eeebe2;
  --ink:#1b1a16; --ink-soft:#5d574c; --ink-faint:#8a8376;
  --rule:#ddd7c9; --rule-strong:#b8b0a0;
  --accent:#34618E; --accent-ink:#ffffff; --accent-soft:#e2eaf3;
  --ok:#256b45; --ok-bg:#e0efe5;
  --bad:#97302a; --bad-bg:#f7e2df;
  --warn:#845400; --warn-bg:#f6ecd6;
  --tok-punct:#7a7367; --tok-key:#34618E; --tok-str:#2c6a44;
  --tok-num:#9d4f18; --tok-bool:#75399e; --tok-null:#8a8376;
  --font-sans: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-cond: "Avenir Next Condensed", "Roboto Condensed", "Arial Narrow", system-ui, sans-serif;
  --font-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper:#121519; --panel:#181c21; --panel-2:#1e232a;
    --ink:#e8e4da; --ink-soft:#a6adb7; --ink-faint:#7b838d;
    --rule:#282f37; --rule-strong:#3a434e;
    --accent:#7fb0e6; --accent-ink:#0e1116; --accent-soft:#1b2836;
    --ok:#7ec89b; --ok-bg:#15291e;
    --bad:#e8918a; --bad-bg:#2d1a18;
    --warn:#ddac5b; --warn-bg:#2a2114;
    --tok-punct:#8b939d; --tok-key:#7fb0e6; --tok-str:#8fd0a6;
    --tok-num:#e0a06a; --tok-bool:#c39be0; --tok-null:#7b838d;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 20px 16px 64px;
  font: 15px/1.55 var(--font-sans);
  color: var(--ink);
  background-color: var(--paper);
  background-image:
    repeating-linear-gradient(0deg, transparent 0 27px, var(--rule) 27px 28px),
    repeating-linear-gradient(90deg, transparent 0 27px, var(--rule) 27px 28px);
}
.wrap { max-width: 1040px; margin: 0 auto; }
.mono { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }

/* title block, drawn like a drawing sheet's */
.sheet { background: var(--panel); border: 1px solid var(--rule-strong); }
.sheet .bar { height: 5px; background: var(--accent); }
.sheet .top { padding: 16px 18px 12px; border-bottom: 1px solid var(--rule); }
h1 { margin: 0; font: 600 30px/1.1 var(--font-cond); letter-spacing: .03em; text-transform: uppercase; }
.sub { margin: 6px 0 0; color: var(--ink-soft); font-size: 13px; }
.specs { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); margin: 0; }
.spec { padding: 9px 18px; border-right: 1px solid var(--rule); border-bottom: 1px solid var(--rule); }
.spec .k { display: block; font: 600 10px/1.6 var(--font-cond); letter-spacing: .14em;
  text-transform: uppercase; color: var(--ink-faint); }
.spec .v { font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-size: 13px; word-break: break-word; }
.banner { margin: 0; padding: 14px 18px; font: 600 24px/1 var(--font-cond);
  letter-spacing: .12em; text-transform: uppercase; text-align: center; }
.banner.pass { background: var(--ok-bg); color: var(--ok); }
.banner.fail { background: var(--bad-bg); color: var(--bad); }
.banner.unknown { background: var(--panel-2); color: var(--ink-soft); }

.toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 16px 0 4px; }
.toolbar button { font: 600 11px/1 var(--font-cond); letter-spacing: .1em; text-transform: uppercase;
  padding: 7px 12px; color: var(--accent); background: var(--panel);
  border: 1px solid var(--rule-strong); cursor: pointer; }
.toolbar button:hover { background: var(--accent-soft); }
.toolbar .note { color: var(--ink-faint); font-size: 12px; }

/* stage sections */
.section { margin: 18px 0; background: var(--panel); border: 1px solid var(--rule-strong); }
.section > summary { padding: 11px 14px; cursor: pointer; list-style: none;
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px;
  border-bottom: 1px solid var(--rule); border-left: 5px solid var(--accent); }
.section > summary::-webkit-details-marker { display: none; }
.section > summary .name { font: 600 15px/1.2 var(--font-cond); letter-spacing: .1em; text-transform: uppercase; }
.section > summary .why { color: var(--ink-soft); font-size: 12px; }
.section > summary .count { margin-left: auto; color: var(--ink-faint); font-size: 12px;
  font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
.section .body { padding: 4px 0 10px; }

/* timeline rows */
.row { display: grid; grid-template-columns: 6.5rem 1fr; gap: 12px; padding: 7px 14px; }
.gutter { text-align: right; padding-top: 3px; color: var(--ink-faint);
  font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-size: 12px; white-space: nowrap; }
.card { min-width: 0; border-left: 1px solid var(--rule-strong); padding-left: 14px; position: relative; }
.card::before { content: ""; position: absolute; left: -4px; top: 8px; width: 7px; height: 7px;
  background: var(--accent); border-radius: 50%; }
.row.slim .card::before { background: var(--rule-strong); }
.head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px; }
.head .badge { font: 11px/1.7 var(--font-mono); color: var(--ink-faint);
  border: 1px solid var(--rule); padding: 0 6px; white-space: nowrap; }
.head .ttl { font-weight: 600; }
.head .seq { margin-left: auto; color: var(--ink-faint); font: 11px/1.7 var(--font-mono); }
summary.head { cursor: pointer; list-style: none; }
summary.head::-webkit-details-marker { display: none; }
.caret { display: inline-block; width: 1em; color: var(--accent); font-family: var(--font-mono); }
.caret::after { content: "+"; }
details[open] > .head .caret::after { content: "\\2212"; }
.row.accent .card::before { box-shadow: 0 0 0 3px var(--accent-soft); }
.row.accent .head .badge { color: var(--accent); border-color: var(--accent); }

.label { margin: 10px 0 4px; font: 600 10px/1.6 var(--font-cond); letter-spacing: .14em;
  text-transform: uppercase; color: var(--ink-faint); }
.kv { font-size: 13px; color: var(--ink-soft); margin-top: 3px; }
.kv .k { display: inline-block; min-width: 88px; color: var(--ink-faint);
  font: 600 10px/1.6 var(--font-cond); letter-spacing: .12em; text-transform: uppercase; }
.kv .v { font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-size: 12.5px;
  color: var(--ink); word-break: break-word; }
.gist { margin-top: 6px; font-size: 14px; }

pre { margin: 6px 0 0; padding: 10px 12px; overflow: auto; max-height: __PRE_MAX__px;
  tab-size: 2; overscroll-behavior: contain;
  background: var(--panel-2); border: 1px solid var(--rule);
  font: 12.5px/1.5 var(--font-mono); font-variant-numeric: tabular-nums; }
pre.json { color: var(--tok-punct); white-space: pre; }
.fold { margin: 4px 0 0; }
.fold > summary { cursor: pointer; list-style: none; display: inline-block;
  padding: 2px 8px; color: var(--ink-faint); background: var(--panel-2);
  border: 1px dashed var(--rule-strong);
  font: 11px/1.6 var(--font-mono); }
.fold > summary::-webkit-details-marker { display: none; }
.fold > summary:hover { color: var(--accent); border-color: var(--accent); }
pre.text { white-space: pre-wrap; word-break: break-word; color: var(--ink); }
.brief { margin-top: 6px; padding: 14px 16px; white-space: pre-wrap; word-break: break-word;
  background: var(--accent-soft); border: 1px solid var(--rule-strong); border-left: 5px solid var(--accent);
  font-size: 14.5px; }
blockquote { margin: 6px 0 0; padding: 10px 14px; border-left: 3px solid var(--accent);
  background: var(--panel-2); color: var(--ink); font-size: 14px; white-space: pre-wrap; word-break: break-word; }
.j-key { color: var(--tok-key); }
.j-fold { color: var(--ink-faint); font-style: italic; }
.j-str { color: var(--tok-str); }
.j-num { color: var(--tok-num); }
.j-bool { color: var(--tok-bool); }
.j-null { color: var(--tok-null); }

table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }
th { text-align: left; font: 600 10px/1.8 var(--font-cond); letter-spacing: .12em;
  text-transform: uppercase; color: var(--ink-faint); border-bottom: 1px solid var(--rule-strong); }
td, th { padding: 5px 8px 5px 0; vertical-align: top; }
td { border-bottom: 1px solid var(--rule); }
td.t, td.det { font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-size: 12px; }
td.t { color: var(--ink-faint); white-space: nowrap; }
td.name { font-family: var(--font-mono); font-size: 12.5px; white-space: nowrap; }

.pill { display: inline-block; padding: 1px 9px; border: 1px solid currentColor; border-radius: 999px;
  font: 600 11px/1.7 var(--font-mono); letter-spacing: .06em; white-space: nowrap; }
.pill.ok { color: var(--ok); background: var(--ok-bg); }
.pill.bad { color: var(--bad); background: var(--bad-bg); }
.pill.warn { color: var(--warn); background: var(--warn-bg); }

.tag-indep { display: inline-block; margin-top: 6px; padding: 4px 10px;
  background: var(--accent-soft); border: 1px dashed var(--accent); color: var(--accent);
  font: 600 11px/1.5 var(--font-mono); }
.final { margin-top: 8px; padding: 12px 14px; text-align: center;
  font: 600 20px/1 var(--font-cond); letter-spacing: .12em; text-transform: uppercase; }
.final.pass { background: var(--ok-bg); color: var(--ok); }
.final.fail { background: var(--bad-bg); color: var(--bad); }
.foot { margin-top: 26px; padding-top: 10px; border-top: 1px solid var(--rule-strong);
  color: var(--ink-faint); font-size: 12px; }

@media (max-width: 640px) {
  body { padding: 12px 10px 48px; }
  .row { grid-template-columns: 1fr; gap: 2px; padding: 8px 10px; }
  .gutter { text-align: left; padding-left: 14px; }
  .head .seq { margin-left: 0; }
  h1 { font-size: 24px; }
  .banner { font-size: 18px; }
}
"""

_JS = """
(function () {
  function esc(s) {
    return s.replace(/[&<>]/g, function (c) {
      return c === '&' ? '&amp;' : c === '<' ? '&lt;' : '&gt;';
    });
  }
  var TOKEN = /"(?:\\\\.|[^"\\\\])*"(\\s*:)?|\\b(?:true|false|null)\\b|-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?/g;
  var FOLD = /^\\s*\\.\\.\\. \\d+ more items \\(\\d+ total\\)$/;
  function highlightLine(raw) {
    if (FOLD.test(raw)) {
      return '<span class="j-fold">' + esc(raw) + '</span>';
    }
    var out = '', last = 0, m;
    TOKEN.lastIndex = 0;
    while ((m = TOKEN.exec(raw)) !== null) {
      out += esc(raw.slice(last, m.index));
      var tok = m[0];
      if (tok.charAt(0) === '"') {
        if (m[1] !== undefined) {
          var quoted = tok.slice(0, tok.length - m[1].length);
          out += '<span class="j-key">' + esc(quoted) + '</span>' + esc(m[1]);
        } else {
          out += '<span class="j-str">' + esc(tok) + '</span>';
        }
      } else if (tok === 'true' || tok === 'false') {
        out += '<span class="j-bool">' + tok + '</span>';
      } else if (tok === 'null') {
        out += '<span class="j-null">null</span>';
      } else {
        out += '<span class="j-num">' + tok + '</span>';
      }
      last = TOKEN.lastIndex;
    }
    return out + esc(raw.slice(last));
  }
  function highlight(raw) {
    return raw.split('\\n').map(highlightLine).join('\\n');
  }
  var blocks = document.querySelectorAll('pre.json');
  for (var i = 0; i < blocks.length; i++) {
    blocks[i].innerHTML = highlight(blocks[i].textContent);
  }
  var buttons = document.querySelectorAll('button[data-all]');
  for (var b = 0; b < buttons.length; b++) {
    buttons[b].addEventListener('click', function (ev) {
      var open = ev.currentTarget.getAttribute('data-all') === 'open';
      var all = document.querySelectorAll('details');
      for (var k = 0; k < all.length; k++) { all[k].open = open; }
    });
  }
})();
"""


def _esc(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _json(obj) -> str:
    """Compacted JSON in a <pre> the inline highlighter colours.

    When compaction folded elements away the untruncated dump follows in a
    collapsed <details>, so nothing in the trace is unreachable.
    """
    text = compact_json(obj)
    block = '<pre class="json">' + _esc(text) + "</pre>"
    if not was_folded(text):
        return block
    full = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    lines = full.count("\n") + 1
    return (
        block
        + '<details class="fold"><summary>'
        + _esc(f"full JSON ({lines} lines)")
        + '</summary><pre class="json">'
        + _esc(full)
        + "</pre></details>"
    )


def _text_block(value) -> str:
    return '<pre class="text">' + _esc(value) + "</pre>"


def _kv(key, value) -> str:
    return f'<div class="kv"><span class="k">{_esc(key)}</span> <span class="v">{_esc(value)}</span></div>'


def _label(text) -> str:
    return f'<div class="label">{_esc(text)}</div>'


def _pill(passed, ok_text="PASS", bad_text="FAIL") -> str:
    cls, text = ("ok", ok_text) if passed else ("bad", bad_text)
    return f'<span class="pill {cls}">{_esc(text)}</span>'


def _fmt_t(value) -> str:
    try:
        return f"{float(value):.3f} s"
    except (TypeError, ValueError):
        return "-"


def _fmt_duration(value) -> str:
    try:
        return f"{float(value):.2f} s"
    except (TypeError, ValueError):
        return "n/a"


def _details(open_, summary_html, inner_html) -> str:
    flag = " open" if open_ else ""
    return f"<details{flag}><summary class=\"head\">{summary_html}</summary>{inner_html}</details>"


def _head_inner(event, caret=False) -> str:
    caret_html = '<span class="caret"></span>' if caret else ""
    seq = event.get("seq")
    seq_html = f'<span class="seq">#{_esc(seq)}</span>' if seq is not None else ""
    return (
        caret_html
        + f'<span class="badge">{_esc(event.get("type", ""))}</span>'
        + f'<span class="ttl">{_esc(event.get("title", ""))}</span>'
        + seq_html
    )


def _row(event, card_html, extra_class="") -> str:
    classes = ("row " + extra_class).strip()
    return (
        f'<div class="{classes}">'
        f'<div class="gutter">{_esc(_fmt_t(event.get("t")))}</div>'
        f'<div class="card">{card_html}</div>'
        "</div>"
    )


def _checks_table(checks, times=None) -> str:
    head = "<tr>"
    if times:
        head += "<th>t</th>"
    head += "<th>Check</th><th>Result</th><th>Detail</th></tr>"
    rows = []
    for index, check in enumerate(checks or []):
        if not isinstance(check, dict):
            check = {"name": str(check)}
        cells = ""
        if times:
            cells += f'<td class="t">{_esc(times[index])}</td>'
        cells += f'<td class="name">{_esc(check.get("name", ""))}</td>'
        cells += f"<td>{_pill(bool(check.get('passed')))}</td>"
        cells += f'<td class="det">{_esc(check.get("detail", ""))}</td>'
        rows.append(f"<tr>{cells}</tr>")
    if not rows:
        span = 4 if times else 3
        rows.append(f'<tr><td colspan="{span}">no checks recorded</td></tr>')
    return "<table>" + head + "".join(rows) + "</table>"


def _render_run_start(event, data) -> str:
    body = _kv("brief file", data.get("brief_path", "n/a"))
    body += _label("plain-English input - the whole specification the agent got")
    body += f'<div class="brief">{_esc(data.get("brief_text", ""))}</div>'
    return f'<div class="head">{_head_inner(event)}</div>' + body


def _render_llm_request(event, data) -> str:
    tools = data.get("tool_names") or []
    inner = _kv("role", data.get("role", "")) + _kv("model", data.get("model", ""))
    inner += _kv("tools offered", ", ".join(str(t) for t in tools) if tools else "none")
    inner += _label("prompt preview")
    inner += _text_block(data.get("user_preview", ""))
    return _details(False, _head_inner(event, caret=True), inner)


def _render_llm_response(event, data) -> str:
    inner = _kv("role", data.get("role", "")) + _kv("model", data.get("model", ""))
    if data.get("text"):
        inner += _label("model text")
        inner += _text_block(data.get("text"))
    calls = data.get("function_calls") or []
    if calls:
        inner += _label(f"function calls requested ({len(calls)})")
        for call in calls:
            if not isinstance(call, dict):
                call = {"name": str(call)}
            inner += _kv("call", call.get("name", ""))
            inner += _json(call.get("args", {}))
    return _details(False, _head_inner(event, caret=True), inner)


def _render_tool_call(event, data) -> str:
    inner = _kv("tool", data.get("name", ""))
    inner += _label("arguments, verbatim - exactly the dict the model built")
    inner += _json(data.get("args", {}))
    return _details(True, _head_inner(event, caret=True), inner)


def _render_tool_result(event, data) -> str:
    head = f'<div class="head">{_head_inner(event)}</div>'
    gist = f'<div class="gist">{_esc(data.get("summary", ""))}</div>'
    inner = _json(data.get("result", {}))
    fold = _details(
        False,
        '<span class="caret"></span><span class="badge">full result</span>'
        f'<span class="ttl">{_esc(data.get("name", ""))} returned</span>',
        inner,
    )
    return head + gist + fold


def _render_gate_checks(events) -> str:
    checks = [(e.get("data") or {}) for e in events]
    times = [_fmt_t(e.get("t")) for e in events]
    head = (
        '<div class="head"><span class="badge">gate_check</span>'
        f'<span class="ttl">Deterministic checks ({len(checks)})</span></div>'
    )
    body = _label("arithmetic only - no model is asked whether the numbers are right")
    body += _checks_table(checks, times=times)
    return _row(events[0], head + body)


def _render_gate_result(event, data) -> str:
    head = f'<div class="head">{_head_inner(event)}</div>'
    body = f'<div class="gist">{_pill(bool(data.get("passed")))} '
    body += _esc(f"{data.get('n_passed', 0)} of {data.get('n_total', 0)} checks passed")
    body += "</div>"
    return head + body


def _render_verdict(event, data) -> str:
    head = f'<div class="head">{_head_inner(event)}</div>'
    refuted = bool(data.get("refuted"))
    body = (
        '<div class="tag-indep">independent agent, separate session, different model: '
        + _esc(data.get("model") or "n/a")
        + "</div>"
    )
    body += '<div class="gist">' + _pill(not refuted, "NOT REFUTED", "REFUTED") + "</div>"
    body += _checks_table(data.get("checks"))
    body += _label("verifier reasoning")
    body += f"<blockquote>{_esc(data.get('reasoning', ''))}</blockquote>"
    return head + body


def _render_run_end(event, data) -> str:
    passed = bool(data.get("passed"))
    head = f'<div class="head">{_head_inner(event)}</div>'
    body = _kv("duration", _fmt_duration(data.get("duration_s")))
    body += _kv("report", data.get("report_path", "n/a"))
    body += _kv("trace", data.get("trace_path", "n/a"))
    body += f'<div class="final {"pass" if passed else "fail"}">GATE {"PASS" if passed else "FAIL"}</div>'
    return head + body


def _render_event(event) -> str:
    etype = str(event.get("type") or "")
    data = event.get("data")
    if not isinstance(data, dict):
        data = {"value": data} if data is not None else {}

    if etype == "run_start":
        return _row(event, _render_run_start(event, data), "accent")
    if etype == "stage_start":
        head = f'<div class="head">{_head_inner(event)}</div>'
        return _row(event, head + _kv("stage", data.get("stage", "")), "slim")
    if etype == "model_attempt":
        head = f'<div class="head">{_head_inner(event)}</div>'
        body = _kv("role", data.get("role", "")) + _kv("model", data.get("model", ""))
        return _row(event, head + body, "slim")
    if etype == "model_fallback":
        head = f'<div class="head">{_head_inner(event)}</div>'
        body = '<div class="gist"><span class="pill warn">FALLBACK</span></div>'
        body += _kv("role", data.get("role", "")) + _kv("model", data.get("model", ""))
        body += _label("error returned")
        body += _text_block(data.get("error", ""))
        return _row(event, head + body)
    if etype == "llm_request":
        return _row(event, _render_llm_request(event, data))
    if etype == "llm_response":
        return _row(event, _render_llm_response(event, data))
    if etype == "tool_call":
        return _row(event, _render_tool_call(event, data), "accent")
    if etype == "tool_result":
        return _row(event, _render_tool_result(event, data))
    if etype == "gate_result":
        return _row(event, _render_gate_result(event, data), "accent")
    if etype == "verdict":
        return _row(event, _render_verdict(event, data), "accent")
    if etype == "run_end":
        return _row(event, _render_run_end(event, data), "accent")

    head = f'<div class="head">{_head_inner(event)}</div>'
    return _row(event, head + _json(data))


def _render_body(events) -> str:
    """Timeline rows, grouped into runs of consecutive same-stage events."""
    parts = []
    index = 0
    while index < len(events):
        event = events[index]
        if event.get("type") == "gate_check":
            end = index
            while end < len(events) and events[end].get("type") == "gate_check":
                end += 1
            parts.append(_render_gate_checks(events[index:end]))
            index = end
            continue
        parts.append(_render_event(event))
        index += 1
    return "".join(parts)


def _sections(events):
    groups = []
    for event in events:
        stage = str(event.get("stage") or "other")
        if not groups or groups[-1][0] != stage:
            groups.append((stage, []))
        groups[-1][1].append(event)
    return groups


def _render_section(stage, events) -> str:
    label = STAGE_LABEL.get(stage, stage.replace("_", " ").title())
    blurb = STAGE_BLURB.get(stage, "")
    times = [e.get("t") for e in events if isinstance(e.get("t"), (int, float))]
    span = f"{min(times):.3f} - {max(times):.3f} s" if times else ""
    summary = (
        f'<span class="name">{_esc(label)}</span>'
        f'<span class="why">{_esc(blurb)}</span>'
        f'<span class="count">{len(events)} events &middot; {_esc(span)}</span>'
    )
    return (
        '<details class="section" open><summary>'
        + summary
        + '</summary><div class="body">'
        + _render_body(events)
        + "</div></details>"
    )


def _last_data(events, etypes, predicate=None):
    found = None
    for event in events:
        if event.get("type") not in etypes:
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        if predicate and not predicate(data):
            continue
        found = data
    return found


def _resolve(events, meta):
    """Header facts: meta wins, otherwise read them back off the events."""
    orchestrator = meta.get("orchestrator_model")
    if not orchestrator:
        data = _last_data(
            events,
            {"model_attempt", "llm_request", "llm_response"},
            lambda d: d.get("role") == "orchestrator",
        )
        orchestrator = (data or {}).get("model")

    verifier = meta.get("verifier_model")
    if not verifier:
        data = _last_data(
            events,
            {"model_attempt", "llm_request", "llm_response"},
            lambda d: d.get("role") == "verifier",
        )
        verifier = (data or {}).get("model") or (_last_data(events, {"verdict"}) or {}).get("model")

    end = _last_data(events, {"run_end"}) or {}
    duration = meta.get("duration_s", end.get("duration_s"))
    if duration is None:
        times = [e.get("t") for e in events if isinstance(e.get("t"), (int, float))]
        duration = max(times) if times else None

    passed = meta.get("passed", end.get("passed"))
    if passed is None:
        gate = _last_data(events, {"gate_result"})
        verdict = _last_data(events, {"verdict"})
        if gate is not None and verdict is not None:
            passed = bool(gate.get("passed")) and not bool(verdict.get("refuted"))

    return {
        "title": meta.get("title") or "Phase 1 Trace",
        "subtitle": meta.get("subtitle")
        or "The 25-Meter Agent - every step from the English brief to the verdict",
        "date": meta.get("date") or meta.get("run_date") or datetime.date.today().isoformat(),
        "orchestrator_model": orchestrator or "n/a",
        "verifier_model": verifier or "n/a",
        "duration": duration,
        "passed": passed,
    }


def _render_header(events, facts) -> str:
    n_tools = sum(1 for e in events if e.get("type") == "tool_call")
    if facts["passed"] is None:
        banner_cls, banner_text = "unknown", "gate incomplete"
    elif facts["passed"]:
        banner_cls, banner_text = "pass", "gate pass"
    else:
        banner_cls, banner_text = "fail", "gate fail"

    specs = [
        ("run date", facts["date"]),
        ("orchestrator", facts["orchestrator_model"]),
        ("verifier", facts["verifier_model"]),
        ("duration", _fmt_duration(facts["duration"])),
        ("events", len(events)),
        ("tool calls", n_tools),
    ]
    spec_html = "".join(
        f'<div class="spec"><span class="k">{_esc(k)}</span><span class="v">{_esc(v)}</span></div>'
        for k, v in specs
    )
    return (
        '<header class="sheet"><div class="bar"></div>'
        f'<div class="top"><h1>{_esc(facts["title"])}</h1>'
        f'<p class="sub">{_esc(facts["subtitle"])}</p></div>'
        f'<div class="specs">{spec_html}</div>'
        f'<div class="banner {banner_cls}">{_esc(banner_text)}</div>'
        "</header>"
    )


def write_trace_html(events: list, out_path, meta: dict = None) -> Path:
    """Write the event list to a standalone HTML trace viewer.

    Args:
        events: recorded events, each {seq, t, stage, type, title, data}.
        out_path: file to write; parent directories are created.
        meta: optional header overrides - title, subtitle, date,
            orchestrator_model, verifier_model, duration_s, passed. Anything
            missing is read back off the events.

    Returns:
        The path written.
    """
    events = list(events or [])
    facts = _resolve(events, dict(meta or {}))

    sections = "".join(_render_section(stage, group) for stage, group in _sections(events))
    if not sections:
        sections = '<p class="foot">No events recorded.</p>'

    doc = (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(facts['title'])}</title>\n"
        "<style>" + _CSS.replace("__PRE_MAX__", str(PRE_MAX_PX)) + "</style>\n</head>\n<body>\n"
        '<div class="wrap">'
        + _render_header(events, facts)
        + '<div class="toolbar">'
        '<button type="button" data-all="open">Expand all</button>'
        '<button type="button" data-all="close">Collapse all</button>'
        '<span class="note">Times are seconds from the start of the run.</span>'
        "</div>"
        + sections
        + '<p class="foot">Preliminary engineering for research and teaching; '
        "not a sealed design; not for construction.</p>"
        "</div>\n<script>" + _JS + "</script>\n</body>\n</html>\n"
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out

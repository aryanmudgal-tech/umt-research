"""Phase-1 CLI: brief -> orchestrator -> deterministic gate -> verifier -> report.

Usage:  .venv/bin/python agent/run_phase1.py [path/to/brief.md] [options]
Exit code 0 only when the final gate passes (deterministic gate AND verifier).

Every stage publishes events on agent.trace.tracer, so the run can be watched
live in the terminal and read back afterwards from results/phase1_trace.html.
"""

import argparse
import asyncio
import contextlib
import datetime
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import sympy as sp  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

from agent.config import ORCHESTRATOR_MODELS, VERIFIER_MODELS, load_api_key, retryable_error  # noqa: E402
from agent.gates import (  # noqa: E402
    check_status,
    detect_ss_udl,
    detect_ss_udl_equation,
    deterministic_gate,
    tally,
)
from agent.jsonfmt import compact_json  # noqa: E402
from agent.live_view import attach_live_view  # noqa: E402
from agent.orchestrator import build_orchestrator  # noqa: E402
from agent.recorder import recorder, to_plain  # noqa: E402
from agent.trace import tracer  # noqa: E402
from agent.trace_html import write_trace_html  # noqa: E402
from agent.verifier import run_verifier  # noqa: E402
from tools.fem.analytical import closed_form  # noqa: E402
from tools.fem.derivation import (  # noqa: E402
    _residual_latex,  # reused so the report and the derivation print one equation
    equation_derivation_markdown,
    galerkin_derivation_markdown,
    validate_equation_spec,
)
from tools.fem.pynite_check import solve_with_pynite  # noqa: E402

DEFAULT_BRIEF = REPO_ROOT / "evals" / "brief_25m.md"
REPORT_PATH = REPO_ROOT / "results" / "phase1_report.md"
DEFAULT_TRACE_HTML = REPO_ROOT / "results" / "phase1_trace.html"

# The run's five stages, in order. agent.live_view.STAGES and
# agent.trace_html.STAGE_ORDER must stay identical to this.
STAGES = ("brief", "orchestrator", "gate", "verifier", "report")

DISCLAIMER = (
    "Preliminary engineering for research and teaching; "
    "not a sealed design; not for construction."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_phase1.py",
        description="Run the Phase-1 pipeline and watch every step of it.",
    )
    parser.add_argument(
        "brief",
        nargs="?",
        default=str(DEFAULT_BRIEF),
        help="English brief to analyze (default: evals/brief_25m.md)",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="no live terminal view; print only the final summary",
    )
    parser.add_argument(
        "--trace-html",
        default=str(DEFAULT_TRACE_HTML),
        metavar="PATH",
        help="where to write the standalone HTML trace "
        "(default: results/phase1_trace.html)",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="skip the HTML trace; the raw event dump is still written",
    )
    return parser


def _stage(name: str):
    """Open a stage. Every stage's events begin with one of these."""
    return tracer.emit(name, "stage_start", f"stage: {name}", stage=name)


async def _run_agent(runner: InMemoryRunner, prompt: str) -> str:
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="phase1"
    )
    msg = types.Content(role="user", parts=[types.Part(text=prompt)])
    final = ""
    async for event in runner.run_async(
        user_id="phase1", session_id=session.id, new_message=msg
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final = "".join(p.text or "" for p in event.content.parts)
    return final


def run_orchestrator(brief_text: str):
    """Run the orchestrator, falling through the model roster on overload."""
    last_error = None
    for name in ORCHESTRATOR_MODELS:
        tracer.emit(
            "orchestrator",
            "model_attempt",
            f"orchestrator trying {name}",
            role="orchestrator",
            model=name,
        )
        recorder.reset()
        runner = InMemoryRunner(agent=build_orchestrator(name), app_name="phase1")
        try:
            narrative = asyncio.run(_run_agent(runner, brief_text))
        except Exception as exc:
            if retryable_error(exc):
                print(f"[orchestrator] {name} unavailable ({exc}); trying next model")
                tracer.emit(
                    "orchestrator",
                    "model_fallback",
                    f"{name} unavailable; trying the next model",
                    role="orchestrator",
                    model=name,
                    error=str(exc),
                )
                last_error = exc
                continue
            raise
        return name, narrative
    raise RuntimeError(f"all orchestrator models failed: {last_error}")


def _fmt(value, unit=""):
    return f"{value:.6e}{unit}" if value is not None else "n/a"


def _cell(text) -> str:
    """Table-cell safe text: a bare '|' would start a new column."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _nest(md: str, levels: int = 2) -> str:
    """Push an embedded document's headings down so they sit under a report section."""
    out, fenced = [], False
    for line in md.splitlines():
        if line.startswith("```") or line.strip() == "$$":
            fenced = not fenced
        elif not fenced:
            line = re.sub(
                r"^(#{1,6})(?=\s)", lambda m: "#" * min(6, len(m.group(1)) + levels), line
            )
        out.append(line)
    return "\n".join(out)


def _equation_rows(model_dict, result, equation):
    """(quantity, FEM, closed form, PyNite) rows for a custom-equation run.

    PyNite models the standard beam element, so it has nothing to say about an
    equation with a foundation, an axial force or a varying stiffness: that
    column is empty by honesty, not by omission. The closed form appears only
    when the spec reduces to the one textbook case.
    """
    case = detect_ss_udl_equation(model_dict, equation)
    ref = (
        closed_form("ss_udl", L=case["L"], E=case["EI"], I=1.0, q=abs(case["q"]))
        if case
        else {}
    )
    return [
        ("Max deflection (m)", result["max_abs"]["v"], ref.get("max_deflection"), None),
        ("Max bending moment (N*m)", result["max_abs"]["moment"], ref.get("max_moment"), None),
        ("Max shear (N)", result["max_abs"]["shear"], ref.get("end_shear"), None),
    ]


def _results_rows(model_dict, result, equation=None):
    """(quantity, FEM, closed form, PyNite) rows; magnitudes in SI units."""
    if equation is not None:
        return _equation_rows(model_dict, result, equation)
    case = detect_ss_udl(model_dict)
    ref = closed_form("ss_udl", **case) if case else {}
    try:
        py = to_plain(solve_with_pynite(model_dict))
    except Exception:
        py = None
    py_defl = py_shear = None
    if py:
        py_defl = max(abs(d["uy"]) for d in py["displacements"].values())
        left = sorted(model_dict["nodes"], key=lambda n: n["x"])[0]["id"]
        py_shear = abs(py["reactions"].get(left, {}).get("FY", 0.0))
    return [
        ("Midspan deflection (m)", result["max_abs"]["uy"], ref.get("max_deflection"), py_defl),
        ("Max bending moment (N*m)", result["max_abs"]["Mz"], ref.get("max_moment"), None),
        ("End shear (N)", result["max_abs"]["Vy"], ref.get("end_shear"), py_shear),
    ]


def _equation_section(equation):
    """The '## Governing equation' section: symbols, parameters, derivation.

    Shown instead of the canned Galerkin derivation whenever the professor's
    own equation was solved, so the report documents the equation that
    produced its numbers rather than the one the pipeline ships with.
    """
    parsed = validate_equation_spec(equation)
    residual = f"{_residual_latex(parsed['coeffs'])} = {sp.latex(parsed['rhs'])}"
    lines = [
        "## Governing equation",
        "",
        "This run did NOT use the pipeline's default beam equation. The "
        "equation below came with the brief and was solved by "
        "`solve_with_equation`, which integrates its element matrices "
        "symbolically at run time.",
        "",
        f"**{_cell(parsed['label'])}**",
        "",
        "$$",
        residual,
        "$$",
        "",
        "Sign convention: $v(x)$ is the transverse deflection, $x$ runs along "
        "the beam, SI units, **downward negative**.",
        "",
        "| Parameter | Value (SI) |",
        "|---|---|",
    ]
    for name, value in parsed["params"].items():
        lines.append(f"| `{_cell(name)}` | {_fmt(value)} |")
    lines += [
        "",
        "### Derivation for this equation",
        "",
        _nest(equation_derivation_markdown(equation).strip(), levels=3),
        "",
    ]
    return lines


def write_report(
    brief_text, model_used, model_dict, result, det, verdict, passed, equation=None
):
    banner = "PASS" if passed else "FAIL"
    lines = [
        "# Phase 1 Report - The 25-Meter Agent",
        "",
        f"- Date: {datetime.date.today().isoformat()}",
        f"- Orchestrator model: {model_used}",
        f"- Verifier model: {verdict.get('model') or 'n/a'}",
        f"- Solver: {'solve_with_equation' if equation is not None else 'solve_beam_3d'}",
        f"- Final gate: {banner}",
        "",
        "## Brief",
        "",
        _nest(brief_text.strip()),
        "",
    ]
    if equation is not None:
        lines += _equation_section(equation)
    else:
        lines += [
            "## Galerkin derivation",
            "",
            _nest(galerkin_derivation_markdown().strip()),
            "",
        ]
    lines += [
        "## Model (as built by the orchestrator)",
        "",
        "```json",
        compact_json(to_plain(model_dict), fold_after=10**9),
        "```",
        "",
        "## Results",
        "",
        "| Quantity | FEM (solver) | Closed form | PyNite |",
        "|---|---|---|---|",
    ]
    for name, fem, cf, py in _results_rows(model_dict, result, equation):
        lines.append(f"| {_cell(name)} | {_fmt(fem)} | {_fmt(cf)} | {_fmt(py)} |")
    counts = det.get("tally") or tally(det["checks"])
    lines += [
        "",
        "## Deterministic gate",
        "",
        f"{counts['passed']} of {counts['total']} checks passed"
        + (f", {counts['skipped']} skipped (a skipped check verified nothing)" if counts["skipped"] else "")
        + ".",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for check in det["checks"]:
        status = check_status(check).upper()
        lines.append(f"| {_cell(check['name'])} | {status} | {_cell(check['detail'])} |")
    lines += [
        "",
        "## Verifier verdict (verbatim)",
        "",
        "```json",
        json.dumps(verdict, indent=2, ensure_ascii=False),
        "```",
        "",
        f"## GATE: {banner}",
        "",
        f"**GATE: {banner}**",
        "",
        f"> {DISCLAIMER}",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    return REPORT_PATH


def write_trace_json(path: Path) -> Path:
    """Dump the raw event list; the HTML viewer's machine-readable twin."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tracer.to_json())
    return path


def _finish(passed, report_path, html_path, json_path, started) -> float:
    """Close the run: emit run_end, then write the trace artifacts.

    run_end goes out first so both artifacts carry the final card.
    """
    elapsed = round(time.monotonic() - started, 3)
    tracer.emit(
        "report",
        "run_end",
        f"GATE {'PASS' if passed else 'FAIL'}",
        passed=passed,
        report_path=str(report_path or ""),
        trace_path=str(html_path or json_path),
        duration_s=elapsed,
    )
    if html_path is not None:
        write_trace_html(
            tracer.events, html_path, {"passed": passed, "duration_s": elapsed}
        )
    write_trace_json(json_path)
    return elapsed


def _summary_lines(model_used, model_dict, result, det, verdict, passed, report,
                   html_path, json_path, equation=None):
    lines = [
        "Phase 1 summary",
        f"  orchestrator model : {model_used}",
        f"  verifier model     : {verdict.get('model') or 'n/a'}",
        f"  solver             : "
        f"{'solve_with_equation' if equation is not None else 'solve_beam_3d'}",
    ]
    if equation is not None:
        lines.append(f"  equation           : {equation.get('label') or 'custom equation'}")
    for name, fem, _cf, _py in _results_rows(model_dict, result, equation):
        lines.append(f"  {name:<26}: {_fmt(fem)}")
    counts = det.get("tally") or tally(det["checks"])
    lines += [
        f"  gate checks        : {counts['passed']}/{counts['total']} passed"
        + (f", {counts['skipped']} skipped" if counts["skipped"] else ""),
        f"  deterministic gate : {'PASS' if det['passed'] else 'FAIL'}",
        f"  verifier refuted   : {verdict['refuted']}",
        f"  GATE               : {'PASS' if passed else 'FAIL'}",
        f"  report             : {report}",
        f"  trace (html)       : {html_path or 'skipped'}",
        f"  trace (json)       : {json_path}",
    ]
    return lines


def _pipeline(brief_path, brief_text, html_path, json_path, started):
    """Run all five stages. Returns (exit_code, summary_lines)."""
    _stage("brief")
    tracer.emit(
        "brief",
        "run_start",
        f"brief: {brief_path.name}",
        brief_path=str(brief_path),
        brief_text=brief_text,
    )

    _stage("orchestrator")
    model_used, _narrative = run_orchestrator(brief_text)
    if not recorder.calls:
        _stage("report")
        _finish(False, None, html_path, json_path, started)
        return 2, [
            "GATE FAIL: the orchestrator never called a solver; nothing to verify.",
            f"  trace (html)       : {html_path or 'skipped'}",
            f"  trace (json)       : {json_path}",
        ]
    if len(recorder.calls) > 1:
        print(f"note: a solver was called {len(recorder.calls)} times; using the last call")
    call = recorder.last
    model_dict, result, equation = call.model, call.result, call.equation

    _stage("gate")
    det = deterministic_gate(model_dict, result, equation)

    _stage("verifier")
    verdict = run_verifier(brief_text, model_dict, result, VERIFIER_MODELS, equation=equation)
    passed = det["passed"] and not verdict["refuted"]

    _stage("report")
    report = write_report(
        brief_text, model_used, model_dict, result, det, verdict, passed, equation
    )
    _finish(passed, report, html_path, json_path, started)

    lines = _summary_lines(
        model_used, model_dict, result, det, verdict, passed, report, html_path,
        json_path, equation,
    )
    return (0 if passed else 1), lines


def main(argv) -> int:
    args = build_parser().parse_args(list(argv)[1:])
    brief_path = Path(args.brief)
    brief_text = brief_path.read_text()
    load_api_key()

    html_path = None if args.no_html else Path(args.trace_html)
    # the raw dump is the HTML trace's sibling, so redirecting one moves both
    json_path = Path(args.trace_html).with_suffix(".json")

    tracer.reset()
    started = time.monotonic()
    view = contextlib.nullcontext() if args.plain else attach_live_view(tracer)
    with view:
        code, lines = _pipeline(brief_path, brief_text, html_path, json_path, started)
    print("\n".join(lines))
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))

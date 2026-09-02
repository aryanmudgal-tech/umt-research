"""Phase-1 CLI: orchestrator -> recorder -> deterministic gate -> verifier -> report.

Usage:  .venv/bin/python agent/run_phase1.py [path/to/brief.md]
Exit code 0 only when the final gate passes (deterministic gate AND verifier).
"""

import asyncio
import datetime
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

from agent.config import ORCHESTRATOR_MODELS, VERIFIER_MODELS, load_api_key, retryable_error  # noqa: E402
from agent.gates import deterministic_gate, detect_ss_udl  # noqa: E402
from agent.orchestrator import build_orchestrator  # noqa: E402
from agent.recorder import recorder, to_plain  # noqa: E402
from agent.verifier import run_verifier  # noqa: E402
from tools.fem.analytical import closed_form  # noqa: E402
from tools.fem.derivation import galerkin_derivation_markdown  # noqa: E402
from tools.fem.pynite_check import solve_with_pynite  # noqa: E402

DEFAULT_BRIEF = REPO_ROOT / "evals" / "brief_25m.md"
REPORT_PATH = REPO_ROOT / "results" / "phase1_report.md"
DISCLAIMER = (
    "Preliminary engineering for research and teaching; "
    "not a sealed design; not for construction."
)


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
        recorder.reset()
        runner = InMemoryRunner(agent=build_orchestrator(name), app_name="phase1")
        try:
            narrative = asyncio.run(_run_agent(runner, brief_text))
        except Exception as exc:
            if retryable_error(exc):
                print(f"[orchestrator] {name} unavailable ({exc}); trying next model")
                last_error = exc
                continue
            raise
        return name, narrative
    raise RuntimeError(f"all orchestrator models failed: {last_error}")


def _fmt(value, unit=""):
    return f"{value:.6e}{unit}" if value is not None else "n/a"


def _results_rows(model_dict, result):
    """(quantity, FEM, closed form, PyNite) rows; magnitudes in SI units."""
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


def write_report(brief_text, model_used, model_dict, result, det, verdict, passed):
    banner = "PASS" if passed else "FAIL"
    lines = [
        "# Phase 1 Report - The 25-Meter Agent",
        "",
        f"- Date: {datetime.date.today().isoformat()}",
        f"- Orchestrator model: {model_used}",
        f"- Verifier model: {verdict.get('model') or 'n/a'}",
        f"- Final gate: {banner}",
        "",
        "## Brief",
        "",
        brief_text.strip(),
        "",
        "## Galerkin derivation",
        "",
        galerkin_derivation_markdown().strip(),
        "",
        "## Model (as built by the orchestrator)",
        "",
        "```json",
        json.dumps(to_plain(model_dict), indent=2),
        "```",
        "",
        "## Results",
        "",
        "| Quantity | FEM (solver) | Closed form | PyNite |",
        "|---|---|---|---|",
    ]
    for name, fem, cf, py in _results_rows(model_dict, result):
        lines.append(f"| {name} | {_fmt(fem)} | {_fmt(cf)} | {_fmt(py)} |")
    lines += [
        "",
        "## Deterministic gate",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for check in det["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(f"| {check['name']} | {status} | {check['detail']} |")
    lines += [
        "",
        "## Verifier verdict (verbatim)",
        "",
        "```json",
        json.dumps(verdict, indent=2),
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


def main(argv) -> int:
    brief_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_BRIEF
    brief_text = brief_path.read_text()
    load_api_key()

    model_used, _narrative = run_orchestrator(brief_text)
    if not recorder.calls:
        print("GATE FAIL: the orchestrator never called solve_beam_3d; nothing to verify.")
        return 2
    if len(recorder.calls) > 1:
        print(f"note: solve_beam_3d called {len(recorder.calls)} times; using the last call")
    model_dict, result = recorder.last

    det = deterministic_gate(model_dict, result)
    verdict = run_verifier(brief_text, model_dict, result, VERIFIER_MODELS)
    passed = det["passed"] and not verdict["refuted"]
    report = write_report(brief_text, model_used, model_dict, result, det, verdict, passed)

    print("Phase 1 summary")
    print(f"  orchestrator model : {model_used}")
    print(f"  verifier model     : {verdict.get('model') or 'n/a'}")
    for name, fem, _cf, _py in _results_rows(model_dict, result):
        print(f"  {name:<26}: {_fmt(fem)}")
    print(f"  deterministic gate : {'PASS' if det['passed'] else 'FAIL'}")
    print(f"  verifier refuted   : {verdict['refuted']}")
    print(f"  GATE               : {'PASS' if passed else 'FAIL'}")
    print(f"  report             : {report}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

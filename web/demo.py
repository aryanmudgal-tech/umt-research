"""Demo mode: the web UI with a replayed run instead of a live one.

    .venv/bin/python -m web.demo --port 8765 --speed 0.5

The pipeline here replays a recorded run of the soil brief with its real
timing (scaled by --speed), but re-solves the beam, re-runs the deterministic
gate and writes a fresh report, so the page shows current output. Only the
two Gemini turns are replayed. For developing the page and for browser tests;
it never calls an LLM.
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

from agent.gates import deterministic_gate
from agent.recorder import to_plain
from agent.run_phase1 import RunOutcome, write_report
from agent.trace import tracer
from tools.fem.equation import solve_equation_beam

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "evals" / "fixtures" / "trace_soil_pass.json"
SOLVERS = {"solve_beam_3d", "solve_with_equation"}


def demo_pipeline(speed: float = 1.0, events_path: Path = FIXTURE):
    events = json.loads(Path(events_path).read_text())
    call = next(e["data"] for e in events if e["type"] == "tool_call" and e["data"].get("name") in SOLVERS)
    model, equation = call["args"]["model"], call["args"]["equation"]
    verdict = next(e["data"] for e in events if e["type"] == "verdict")

    def pipeline(brief_text, out_dir, brief_name="brief.md"):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tracer.reset()
        result = to_plain(solve_equation_beam(model, equation))
        det = None
        last_t = 0.0
        for event in events:
            time.sleep(max(0.0, event["t"] - last_t) * speed)
            last_t = event["t"]
            if event["stage"] == "gate":
                if event["type"] == "stage_start":
                    tracer.emit("gate", "stage_start", event["title"], stage="gate")
                    det = deterministic_gate(model, result, equation)  # emits the real checks
                continue
            data = dict(event["data"])
            if event["type"] == "tool_result" and data.get("name") in SOLVERS:
                data["result"] = result
            if event["type"] == "run_start":
                data["brief_text"] = brief_text
            tracer.emit(event["stage"], event["type"], event["title"], **data)

        passed = det["passed"] and not verdict["refuted"]
        report = write_report(
            brief_text, "gemini (replayed)", model, result, det, verdict, passed, equation,
            path=out_dir / "report.md",
        )
        (out_dir / "trace.json").write_text(tracer.to_json())
        return RunOutcome(
            code=0 if passed else 1, passed=passed, lines=[], report_path=report,
            narrative="replayed", model_used="gemini (replayed)", verdict=verdict, gate=det,
        )

    return pipeline


def main():
    import uvicorn

    from web.app import create_app
    from web.storage import LocalStore

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--speed", type=float, default=0.5, help="replay speed factor (1 = recorded timing)")
    parser.add_argument("--runs-dir", default=None, help="where demo runs are kept (default: a temp folder)")
    args = parser.parse_args()

    try:
        from web.pdf import render_pdf
    except ImportError:
        def render_pdf(report_md, title):
            return b"%PDF-1.4\n% demo placeholder\n"

    runs_dir = Path(args.runs_dir or tempfile.mkdtemp(prefix="web_demo_runs_"))
    app = create_app(
        store=LocalStore(runs_dir), pipeline=demo_pipeline(args.speed), render_pdf=render_pdf, load_key=False,
    )
    print(f"demo runs kept in {runs_dir}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

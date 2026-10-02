"""Offline tests for run_pipeline: the pipeline as a function, one folder per run.

The web server calls this instead of the CLI, so every run has to land in its
own folder and hand back what the page needs. No API calls: the orchestrator
and the verifier are replaced by fakes that record what a real one would.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent import run_phase1
from agent.recorder import recorder, solve_beam_3d

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


def fake_agents(monkeypatch, solve=True, refuted=False):
    def orchestrator(_brief):
        recorder.reset()
        if solve:
            solve_beam_3d(ss_udl_model())
            return "gemini-fake", "narrative: solved"
        return "gemini-fake", "No supports were given, so I could not build a model."

    def verifier(_brief, _model, _result, model_names, equation=None):
        return {"refuted": refuted, "checks": [], "reasoning": "", "model": "verifier-fake"}

    monkeypatch.setattr(run_phase1, "run_orchestrator", orchestrator)
    monkeypatch.setattr(run_phase1, "run_verifier", verifier)


def test_a_run_lands_in_its_own_folder(monkeypatch, tmp_path):
    fake_agents(monkeypatch)
    out = run_phase1.run_pipeline("# Brief\n\nA beam.", tmp_path / "run-1")

    assert out.passed and out.code == 0
    assert out.report_path == tmp_path / "run-1" / "report.md"
    for name in ("report.md", "trace.json", "trace.html"):
        assert (tmp_path / "run-1" / name).is_file(), name
    assert "GATE: PASS" in out.report_path.read_text()
    assert out.model_used == "gemini-fake"
    assert out.verdict["model"] == "verifier-fake"


def test_two_runs_do_not_overwrite_each_other(monkeypatch, tmp_path):
    fake_agents(monkeypatch)
    first = run_phase1.run_pipeline("# Brief one", tmp_path / "a")
    fake_agents(monkeypatch, refuted=True)
    second = run_phase1.run_pipeline("# Brief two", tmp_path / "b")

    assert first.passed and not second.passed
    assert "Brief one" in first.report_path.read_text()
    assert "Brief two" in second.report_path.read_text()


def test_a_run_with_no_solver_call_returns_the_agents_reason(monkeypatch, tmp_path):
    fake_agents(monkeypatch, solve=False)
    out = run_phase1.run_pipeline("# Brief", tmp_path / "run")

    assert out.code == 2 and not out.passed
    assert out.report_path is None
    assert out.narrative == "No supports were given, so I could not build a model."
    assert (tmp_path / "run" / "trace.json").is_file()


def test_the_cli_still_writes_to_the_results_report(monkeypatch, tmp_path):
    fake_agents(monkeypatch)
    report = tmp_path / "phase1_report.md"
    monkeypatch.setattr(run_phase1, "REPORT_PATH", report)  # where results/ would be
    monkeypatch.setattr(run_phase1, "load_api_key", lambda: "key")
    code = run_phase1.main(
        ["run_phase1.py", str(REPO_ROOT / "evals" / "brief_25m.md"), "--plain", "--trace-html", str(tmp_path / "t.html")]
    )

    assert code == 0
    assert "GATE: PASS" in report.read_text()

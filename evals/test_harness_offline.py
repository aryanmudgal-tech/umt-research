"""Offline harness tests: recorder, deterministic gate, brief, JSON extraction.

Nothing here needs an API key or the network.
"""

import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.gates import deterministic_gate, detect_ss_udl
from agent.recorder import recorder, solve_beam_3d
from agent.verifier import extract_json

# Bridge-beam reference numbers: L=25 m, E=30 GPa, I=0.005 m^4, q=30 kN/m.
L, E, G, I, Q = 25.0, 30e9, 12.5e9, 0.005, 30e3


def ss_udl_model(n_elem=4):
    nodes = [{"id": f"N{k}", "x": L * k / n_elem} for k in range(n_elem + 1)]
    return {
        "nodes": nodes,
        "material": {"E": E, "G": G},
        "section": {"A": 0.5, "Iy": 0.004, "Iz": I, "J": 0.001},
        "supports": {
            "N0": [True, True, True, True, False, False],
            f"N{n_elem}": [False, True, True, False, False, False],
        },
        "distributed_loads": [
            {"element": "all", "direction": "y", "w1": -Q, "w2": -Q}
        ],
        "point_loads": [],
    }


def test_recorder_records_model_and_result():
    recorder.reset()
    model = ss_udl_model()
    result = solve_beam_3d(model)

    assert len(recorder.calls) == 1
    rec_model, rec_result = recorder.last
    assert rec_model == model
    assert rec_result == result

    # recorded copies are decoupled from the caller's dict
    model["material"]["E"] = 1.0
    assert rec_model["material"]["E"] == E

    # recorded result is JSON-plain (no numpy scalars survive)
    json.dumps(rec_result)

    recorder.reset()
    assert recorder.calls == []
    assert recorder.last is None


def test_gate_passes_on_good_ss_udl_solve():
    recorder.reset()
    model = ss_udl_model()
    result = solve_beam_3d(model)

    assert detect_ss_udl(model) == {"L": L, "E": E, "I": I, "q": Q}
    gate = deterministic_gate(model, result)
    failed = [c for c in gate["checks"] if not c["passed"]]
    assert gate["passed"], failed
    names = {c["name"] for c in gate["checks"]}
    assert "closed_form_midspan_deflection" in names
    assert "pynite_agreement" in names
    assert "invariant_equilibrium_forces" in names


def test_gate_fails_on_tampered_deflection():
    model = ss_udl_model()
    result = solve_beam_3d(model)

    tampered = copy.deepcopy(result)
    for disp in tampered["displacements"].values():
        disp["uy"] *= 1.1
    tampered["max_abs"]["uy"] *= 1.1

    gate = deterministic_gate(model, tampered)
    assert not gate["passed"]
    failed = {c["name"] for c in gate["checks"] if not c["passed"]}
    assert "closed_form_midspan_deflection" in failed


def test_brief_file_exists_and_names_the_beam():
    brief = REPO_ROOT / "evals" / "brief_25m.md"
    text = brief.read_text()
    assert "25" in text
    assert "30 GPa" in text
    assert "0.005" in text
    assert "30 kN/m" in text


def test_extract_json_fenced():
    text = (
        "Verdict below.\n```json\n"
        '{"refuted": false, "checks": [], "reasoning": "ok"}\n'
        "```\nDone."
    )
    assert extract_json(text) == {"refuted": False, "checks": [], "reasoning": "ok"}


def test_extract_json_unfenced_with_prose():
    text = (
        'After checking everything: {"refuted": true, "checks": '
        '[{"name": "x", "passed": false, "detail": "d"}], "reasoning": "bad"} end'
    )
    verdict = extract_json(text)
    assert verdict["refuted"] is True
    assert verdict["checks"][0]["name"] == "x"


def test_extract_json_garbage_is_none():
    assert extract_json("") is None
    assert extract_json("no json here { unbalanced") is None

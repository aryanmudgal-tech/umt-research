"""Shared fixtures for the eval suite.

The captured run in results/ is gitignored, so anything that wants the real
payloads has to cope with it being absent. These fixtures hand back the real
solve_beam_3d call and result when the trace is on disk and a stand-in of the
same shape when it is not, so the rendering tests run either way.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

REAL_TRACE = REPO_ROOT / "results" / "phase1_trace.json"

FALLBACK_ARGS = {
    "model": {
        "supports": {
            "N1": [True, True, True, True, False, False],
            "N5": [False, True, True, False, False, False],
        },
        "nodes": [{"id": f"N{k + 1}", "x": 6.25 * k} for k in range(5)],
        "distributed_loads": [
            {"w2": -30000, "w1": -30000, "element": "all", "direction": "y"}
        ],
        "section": {"Iz": 0.005, "J": 0.001, "A": 0.5, "Iy": 0.005},
        "material": {"G": 12500000000, "E": 30000000000},
    }
}

FALLBACK_RESULT = {
    "displacements": {
        f"N{k + 1}": {"ux": 0.0, "uy": -0.72479248046875, "uz": -0.0, "rx": 0.0,
                      "ry": 0.0, "rz": -0.08951822916666667}
        for k in range(5)
    },
    "reactions": {
        "N1": {"FX": 0.0, "FY": 374999.9999999999, "FZ": 0.0, "MX": 0.0},
        "N5": {"FY": 375000.0, "FZ": 0.0},
    },
    "diagrams": [
        {"x": 25.0 * n / 23, "N": -0.0, "Vy": 375000.0, "Vz": 0.0, "T": -0.0,
         "My": -0.0, "Mz": 445312.5}
        for n in range(24)
    ],
    "max_abs": {"uy": 1.0172526041666667, "uz": 0.0, "Mz": 2343750.000000002,
                "My": 0.0, "Vy": 375000.0, "Vz": 0.0, "N": 0.0, "T": 0.0},
    "total_applied": {"FX": 0.0, "FY": -750000.0, "FZ": 0.0},
}


def _from_real_trace(etype, key, tool="solve_beam_3d"):
    """The first `tool` payload in the captured run, or None if there is none."""
    if not REAL_TRACE.exists():
        return None
    try:
        events = json.loads(REAL_TRACE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for event in events if isinstance(events, list) else []:
        data = event.get("data") if isinstance(event, dict) else None
        if not isinstance(data, dict):
            continue
        if event.get("type") == etype and data.get("name") == tool:
            payload = data.get(key)
            if isinstance(payload, dict) and payload:
                return payload
    return None


@pytest.fixture(scope="session")
def tool_call_args():
    """The model dict Gemini built: the real one when the trace is on disk."""
    return _from_real_trace("tool_call", "args") or FALLBACK_ARGS


@pytest.fixture(scope="session")
def tool_result_payload():
    """The solver's return: displacements, reactions, diagrams, max_abs."""
    return _from_real_trace("tool_result", "result") or FALLBACK_RESULT


@pytest.fixture(scope="session")
def using_real_trace():
    return _from_real_trace("tool_call", "args") is not None

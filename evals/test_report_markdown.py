"""The Phase-1 report must render as intended in Markdown viewers.

A report that reads fine as raw text can still collapse in a viewer: a line
starting with '+' inside an equation becomes a list item, and a '|' inside a
table cell becomes an extra column. These tests pin down the structure every
viewer relies on. No API, no network.
"""

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent import run_phase1
from agent.gates import deterministic_gate
from tools.fem.derivation import galerkin_derivation_markdown
from tools.fem.solver import solve_beam_3d

L, E, G, I, Q = 25.0, 30e9, 12.5e9, 0.005, 30e3

BRIEF = """# Brief: 25 m bridge beam

- Simply supported, span 25 m, q = 30 kN/m downward.
"""


def ss_udl_model():
    return {
        "nodes": [{"id": f"N{k}", "x": L * k / 4} for k in range(5)],
        "material": {"E": E, "G": G},
        "section": {"A": 0.5, "Iy": 0.004, "Iz": I, "J": 0.001},
        "supports": {
            "N0": [True, True, True, True, False, False],
            "N4": [False, True, True, False, False, False],
        },
        "distributed_loads": [{"element": "all", "direction": "y", "w1": -Q, "w2": -Q}],
        "point_loads": [],
    }


@pytest.fixture
def report(tmp_path, monkeypatch):
    monkeypatch.setattr(run_phase1, "REPORT_PATH", tmp_path / "report.md")
    model = ss_udl_model()
    result = solve_beam_3d(model)
    det = deterministic_gate(model, result)
    verdict = {
        "refuted": False,
        "checks": [{"name": "moment", "passed": True, "detail": "M = qL²/8, δ = 5qL⁴/384EI"}],
        "reasoning": "recomputed; nothing to refute",
        "model": "verifier-model",
    }
    path = run_phase1.write_report(BRIEF, "orchestrator-model", model, result, det, verdict, True)
    return path.read_text()


def math_block_lines(md):
    return [i for i, line in enumerate(md.splitlines()) if "$$" in line]


# ------------------------------------------------------------ equations


def test_every_display_equation_is_a_standalone_block():
    lines = galerkin_derivation_markdown().splitlines()
    marks = math_block_lines("\n".join(lines))

    assert marks and len(marks) % 2 == 0
    assert all(lines[i].strip() == "$$" for i in marks), "a $$ shares its line with text"
    for open_, close in zip(marks[::2], marks[1::2]):
        body = lines[open_ + 1 : close]
        assert len(body) == 1, f"equation at line {open_} spans {len(body)} lines"
        assert not re.match(r"\s*([-+*>]|\d+[.)])\s", body[0]), body[0][:60]


def test_equations_are_separated_by_blank_lines():
    lines = galerkin_derivation_markdown().splitlines()
    marks = math_block_lines("\n".join(lines))
    for open_, close in zip(marks[::2], marks[1::2]):
        assert open_ == 0 or lines[open_ - 1] == ""
        assert close == len(lines) - 1 or lines[close + 1] == ""


def test_report_math_stays_balanced(report):
    lines = report.splitlines()
    marks = math_block_lines(report)
    assert len(marks) % 2 == 0
    assert all(lines[i].strip() == "$$" for i in marks)


# --------------------------------------------------------------- tables


def unescaped_pipes(line):
    return len(re.findall(r"(?<!\\)\|", line))


def test_every_table_row_keeps_its_column_count(report):
    rows = [line for line in report.splitlines() if line.startswith("|")]
    tables, current = [], []
    for line in report.splitlines():
        if line.startswith("|"):
            current.append(line)
        elif current:
            tables.append(current)
            current = []
    assert tables
    for table in tables:
        header = unescaped_pipes(table[0])
        for row in table:
            assert unescaped_pipes(row) == header, row
    assert rows


def test_pipes_inside_gate_details_are_escaped(report):
    assert r"max\|K - K^T\|" in report
    assert "max|K - K^T|" not in report.replace(r"\|", "")


# ------------------------------------------------------------- headings


def test_the_report_has_exactly_one_title(report):
    titles = [line for line in report.splitlines() if re.match(r"#\s", line)]
    assert titles == ["# Phase 1 Report - The 25-Meter Agent"]


def test_embedded_headings_sit_under_their_report_section(report):
    headings = [line for line in report.splitlines() if re.match(r"#{1,6}\s", line)]
    assert "### Brief: 25 m bridge beam" in headings
    assert "### Galerkin derivation of the 3D Euler-Bernoulli beam element" in headings
    assert "#### 1. Strong and weak form" in headings
    for section in ("## Brief", "## Galerkin derivation", "## Results", "## Deterministic gate"):
        assert any(h.startswith(section) for h in headings), section


def test_nesting_leaves_fenced_content_alone():
    md = "# Title\n\n```\n# not a heading\n```\n\n$$\n# x\n$$\n"
    nested = run_phase1._nest(md)
    assert nested.splitlines()[0] == "### Title"
    assert "# not a heading" in nested.splitlines()
    assert "# x" in nested.splitlines()


# ---------------------------------------------------------- readability


def test_symbols_are_written_as_characters_not_escapes(report):
    assert "qL²/8" in report and "δ" in report
    assert "\\u00b2" not in report and "\\u03b4" not in report


def test_model_block_is_compact_and_complete_json(report):
    block = report.split("```json\n", 1)[1].split("\n```", 1)[0]
    parsed = json.loads(block)
    assert parsed == json.loads(json.dumps(ss_udl_model()))
    assert len(block.splitlines()) < 25
    assert "more items" not in block

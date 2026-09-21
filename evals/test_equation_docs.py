"""The equation is an input, so its documentation has to hold up for any input.

These tests cover the two things the professor actually touches: the derivation
document generated for an arbitrary equation spec, and the ready-made specs in
equations/. No API, no network, no LLM.
"""

import importlib
import json
import os
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

import pytest
import sympy as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from test_report_markdown import math_block_lines  # the rules the report already obeys

from tools.fem.derivation import (
    _COEFF_KEYS,
    _element_forms,
    equation_derivation_markdown,
    galerkin_derivation_markdown,
    validate_equation_spec,
)

EQUATIONS_DIR = REPO_ROOT / "equations"
LIBRARY_FILES = sorted(EQUATIONS_DIR.glob("*.json"))

EULER_BERNOULLI = {
    "label": "Euler-Bernoulli beam",
    "coeffs": {"v4": "E*I", "v2": "0", "v1": "0", "v0": "0"},
    "rhs": "q",
    "params": {"E": 30e9, "I": 0.005, "q": -30e3},
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def assert_markdown_block_rules(md):
    """Every $$ block: bare delimiters, one line of content, blank lines around.

    Same three rules test_report_markdown pins down for the fixed derivation —
    a generated document has to keep them for every equation, not just one.
    """
    lines = md.splitlines()
    marks = math_block_lines(md)
    assert marks and len(marks) % 2 == 0, "unbalanced $$ delimiters"
    assert all(lines[i].strip() == "$$" for i in marks), "a $$ shares its line with text"
    for open_, close in zip(marks[::2], marks[1::2]):
        body = lines[open_ + 1 : close]
        assert len(body) == 1, f"equation at line {open_} spans {len(body)} lines"
        assert not re.match(r"\s*([-+*>]|\d+[.)])\s", body[0]), body[0][:60]
        assert open_ == 0 or lines[open_ - 1] == "", f"no blank line before line {open_}"
        assert close == len(lines) - 1 or lines[close + 1] == "", f"line {close}"


def spec_ids(paths):
    return [p.name for p in paths]


# ------------------------------------------------------- the derivation document


@pytest.mark.parametrize("path", LIBRARY_FILES, ids=spec_ids(LIBRARY_FILES))
def test_generated_derivation_obeys_the_markdown_block_rules(path):
    assert_markdown_block_rules(equation_derivation_markdown(load(path)))


def test_derivation_is_deterministic():
    spec = load(EQUATIONS_DIR / "tapered_beam.json")
    assert equation_derivation_markdown(spec) == equation_derivation_markdown(spec)


def test_derivation_survives_a_different_hash_seed():
    """Sympy iterates sets internally; a string hash seed must not reorder output."""
    code = (
        "import json, sys; sys.path.insert(0, %r);"
        "from tools.fem.derivation import equation_derivation_markdown as d;"
        "print(d(json.load(open(%r))), end='')"
        % (str(REPO_ROOT), str(EQUATIONS_DIR / "beam_column.json"))
    )
    outs = [
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout
        for seed in ("0", "12345")
    ]
    assert outs[0] == outs[1]


def test_the_fixed_derivation_still_works_unchanged():
    md = galerkin_derivation_markdown()
    assert md.startswith("# Galerkin derivation of the 3D Euler-Bernoulli beam element")
    assert "## 1. Strong and weak form" in md
    assert_markdown_block_rules(md)


@pytest.mark.parametrize("path", LIBRARY_FILES, ids=spec_ids(LIBRARY_FILES))
def test_derivation_shows_this_specs_own_coefficients_and_params(path):
    spec = load(path)
    md = equation_derivation_markdown(spec)
    assert spec["label"] in md.splitlines()[0]
    for key, text in spec["coeffs"].items():
        if sp.sympify(text.replace("^", "**")).is_zero:
            continue
        assert f"`{text}`" in md, f"{path.name}: coefficient {key} = {text} is not shown"
        assert f"$a_{key[1]}$" in md, f"{path.name}: the {key} term is never named"
    for name in spec["params"]:
        assert f"`{name}`" in md, f"{path.name}: parameter {name} is not listed"


def test_foundation_spec_documents_its_a0_term():
    md = equation_derivation_markdown(load(EQUATIONS_DIR / "elastic_foundation.json"))
    assert "$a_0$" in md and "Winkler" in md
    assert r"\mathbf{k}_e^{(0)}" in md
    assert "156" in md, "the consistent foundation matrix never appears"


def test_a_spec_without_a_term_does_not_document_it():
    md = equation_derivation_markdown(EULER_BERNOULLI)
    assert r"\mathbf{k}_e^{(4)}" in md
    for absent in ("(0)", "(2)", "(1)"):
        assert rf"\mathbf{{k}}_e^{{{absent}}}" not in md
    assert "Winkler" not in md and "buckling" not in md


def test_x_dependent_coefficient_reaches_the_element_matrix():
    md = equation_derivation_markdown(load(EQUATIONS_DIR / "tapered_beam.json"))
    assert "x_{i}" in md, "the element's left-node coordinate never appears"
    assert "self-adjoint" in md, "the varying-EI caveat is missing"


# ------------------------------------------------------- the symbolic matrices


def element_terms(spec):
    s, xi = sp.symbols("s x_i", real=True)
    Le = sp.Symbol("L_e", positive=True)
    _N, _a, terms, f_e = _element_forms(validate_equation_spec(spec), s, Le, xi)
    return terms, f_e, Le


def test_each_term_reduces_to_its_textbook_matrix():
    """The three sanity targets from the equation contract, straight from sympy."""
    spec = {
        "coeffs": {"v4": "EI", "v2": "P", "v0": "k"},
        "rhs": "0",
        "params": {"EI": 1.0, "P": 1.0, "k": 1.0},
    }
    terms, _f_e, L = element_terms(spec)
    EI, P, k = (sp.Symbol(n, real=True) for n in ("EI", "P", "k"))

    targets = {
        "v4": EI / L**3 * sp.Matrix([
            [12, 6 * L, -12, 6 * L],
            [6 * L, 4 * L**2, -6 * L, 2 * L**2],
            [-12, -6 * L, 12, -6 * L],
            [6 * L, 2 * L**2, -6 * L, 4 * L**2],
        ]),
        "v0": k * L / 420 * sp.Matrix([
            [156, 22 * L, 54, -13 * L],
            [22 * L, 4 * L**2, 13 * L, -3 * L**2],
            [54, 13 * L, 156, -22 * L],
            [-13 * L, -3 * L**2, -22 * L, 4 * L**2],
        ]),
        "v2": -P / (30 * L) * sp.Matrix([
            [36, 3 * L, -36, 3 * L],
            [3 * L, 4 * L**2, -3 * L, -(L**2)],
            [-36, -3 * L, 36, -3 * L],
            [3 * L, -(L**2), -3 * L, 4 * L**2],
        ]),
    }
    for key, target in targets.items():
        assert sp.simplify(terms[key] - target) == sp.zeros(4, 4), key


def test_the_geometric_term_softens_the_beam_under_compression():
    """The a2 sign decides whether compression buckles the beam or braces it."""
    terms, _f_e, L = element_terms(
        {"coeffs": {"v4": "EI", "v2": "P"}, "rhs": "0", "params": {"EI": 1.0, "P": 1.0}}
    )
    numeric = {L: 1.0, sp.Symbol("EI", real=True): 1.0, sp.Symbol("P", real=True): 1.0}
    bending = sp.matrix2numpy(terms["v4"].subs(numeric), dtype=float)
    geometric = sp.matrix2numpy(terms["v2"].subs(numeric), dtype=float)

    def energy(m):
        d = [0.0, 0.0, 1.0, 0.0]  # push node j sideways, hold everything else
        return sum(d[r] * m[r][c] * d[c] for r in range(4) for c in range(4))

    assert energy(geometric) < 0 < energy(bending) + energy(geometric)


def test_uniform_load_gives_the_consistent_nodal_loads():
    _terms, f_e, L = element_terms(
        {"coeffs": {"v4": "EI"}, "rhs": "q", "params": {"EI": 1.0, "q": -30e3}}
    )
    q = sp.Symbol("q", real=True)
    expected = sp.Matrix([q * L / 2, q * L**2 / 12, q * L / 2, -q * L**2 / 12])
    assert sp.simplify(f_e - expected) == sp.zeros(4, 1)


# ------------------------------------------------------------ the spec contract


def assert_obeys_spec_contract(spec, where):
    """The contract as written in equations/README.md, checked field by field."""
    assert isinstance(spec, dict), where
    assert isinstance(spec["coeffs"], dict) and spec["coeffs"], f"{where}: no coeffs"
    assert set(spec["coeffs"]) <= set(_COEFF_KEYS), f"{where}: {set(spec['coeffs'])}"
    assert "v3" not in spec["coeffs"], f"{where}: v3 is not supported"
    assert all(isinstance(v, str) for v in spec["coeffs"].values()), where
    assert "v4" in spec["coeffs"] and spec["coeffs"]["v4"] != "0", f"{where}: no v4"
    assert isinstance(spec["rhs"], str), f"{where}: rhs must be a string"

    params = spec["params"]
    assert isinstance(params, dict) and params, f"{where}: no params"
    for name, value in params.items():
        assert name.isidentifier() and name != "x", f"{where}: bad param {name!r}"
        assert isinstance(value, (int, float)) and not isinstance(value, bool), where

    declared = set(params) | {"x"}
    for key, text in list(spec["coeffs"].items()) + [("rhs", spec["rhs"])]:
        used = {str(s) for s in sp.sympify(text, locals={n: sp.Symbol(n) for n in declared}).free_symbols}
        assert used <= declared, f"{where}: {key} uses undeclared {sorted(used - declared)}"

    assert isinstance(spec.get("label", ""), str), where
    validate_equation_spec(spec)


def test_the_library_holds_the_four_promised_equations():
    assert {p.name for p in LIBRARY_FILES} == {
        "euler_bernoulli.json",
        "elastic_foundation.json",
        "beam_column.json",
        "tapered_beam.json",
    }


@pytest.mark.parametrize("path", LIBRARY_FILES, ids=spec_ids(LIBRARY_FILES))
def test_library_file_parses_and_obeys_the_contract(path):
    spec = load(path)
    assert_obeys_spec_contract(spec, path.name)
    assert isinstance(spec.get("_comment"), str) and spec["_comment"].strip(), (
        f"{path.name}: needs a _comment saying what the physics is"
    )


@pytest.mark.parametrize("path", LIBRARY_FILES, ids=spec_ids(LIBRARY_FILES))
def test_library_file_keeps_the_downward_negative_convention(path):
    """Transverse load down means q < 0; a positive q would lift the beam."""
    spec = load(path)
    for name, value in spec["params"].items():
        if name.startswith("q"):
            assert value < 0, f"{path.name}: {name} = {value} is upward"


def test_a_v3_term_is_rejected_with_a_clear_error():
    spec = dict(EULER_BERNOULLI, coeffs={**EULER_BERNOULLI["coeffs"], "v3": "c"})
    with pytest.raises(ValueError, match="v3"):
        equation_derivation_markdown(spec)


@pytest.mark.parametrize(
    "spec, message",
    [
        ({"coeffs": {"v4": "E*I", "v9": "0"}, "rhs": "q", "params": {"E": 1, "I": 1, "q": 1}}, "v9"),
        ({"coeffs": {"v4": "E*Iz"}, "rhs": "q", "params": {"E": 1, "q": 1}}, "Iz"),
        ({"coeffs": {"v4": "0"}, "rhs": "q", "params": {"q": 1}}, "non-zero"),
        ({"coeffs": {"v4": "E*("}, "rhs": "q", "params": {"E": 1, "q": 1}}, "valid expression"),
        ({"rhs": "q", "params": {}}, "coeffs"),
        ({"coeffs": {"v4": "E*I"}, "rhs": "q", "params": {"E": 1, "I": "wide"}}, "must be a number"),
    ],
)
def test_broken_specs_fail_loudly(spec, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        validate_equation_spec(spec)


# --------------------------------------------------------- the README examples


def readme_json_blocks():
    text = (EQUATIONS_DIR / "README.md").read_text(encoding="utf-8")
    return re.findall(r"```json\n(.*?)\n```", text, re.DOTALL)


def test_readme_shows_at_least_two_worked_examples():
    assert len(readme_json_blocks()) >= 3, "the template plus two worked examples"


@pytest.mark.parametrize("index", range(len(readme_json_blocks())))
def test_readme_json_block_is_a_real_spec(index):
    block = readme_json_blocks()[index]
    spec = json.loads(block)
    assert_obeys_spec_contract(spec, f"README block {index}")
    assert_markdown_block_rules(equation_derivation_markdown(spec))


def test_readme_worked_examples_show_the_two_promised_changes():
    blocks = [json.loads(b) for b in readme_json_blocks()]
    assert any(b["coeffs"].get("v0", "0") != "0" for b in blocks), "no soil-spring example"
    assert any("x" in b["coeffs"]["v4"] for b in blocks), "no varying-EI example"


def test_readme_states_what_is_not_supported():
    text = (EQUATIONS_DIR / "README.md").read_text(encoding="utf-8").lower()
    for topic in ("nonlinear", "coupled", "time", "plate"):
        assert topic in text, f"the README never says {topic} problems are out of scope"


# ------------------------------------------- hand-off to the equation solver


def find_in_fem(name):
    """A function the equation solver exposes, wherever in tools.fem it lives."""
    import tools.fem

    for info in pkgutil.iter_modules(tools.fem.__path__):
        if info.name == "derivation":
            continue
        try:
            module = importlib.import_module(f"tools.fem.{info.name}")
        except Exception:  # noqa: BLE001 - an unrelated import failure is not our test
            continue
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    return None


PARSE_SPEC = find_in_fem("parse_spec")
ELEMENT_MATRICES = find_in_fem("element_matrices")
needs_solver = pytest.mark.skipif(
    PARSE_SPEC is None or ELEMENT_MATRICES is None,
    reason="the equation solver is not in tools/fem yet",
)


@needs_solver
@pytest.mark.parametrize("path", LIBRARY_FILES, ids=spec_ids(LIBRARY_FILES))
def test_the_document_shows_the_matrices_the_solver_assembles(path):
    """Otherwise the derivation is a description of some other element.

    Both sides integrate symbolically, but independently: this pins the
    document's k_e and f_e to the ones that actually carry the numbers.
    """
    length, x_left = 6.25, 12.5  # off-origin, so an x-dependent coefficient shows
    spec = load(path)
    terms, f_doc, Le = element_terms(spec)
    k_doc = sum(terms.values(), sp.zeros(4, 4))

    parsed = validate_equation_spec(spec)
    subs = {sp.Symbol(n, real=True): v for n, v in parsed["params"].items()}
    subs.update({Le: length, sp.Symbol("x_i", real=True): x_left})

    k_solver, f_solver = ELEMENT_MATRICES(spec, length, x_left)
    for doc, solver, what in ((k_doc, k_solver, "k_e"), (f_doc, f_solver, "f_e")):
        got = sp.matrix2numpy(doc.subs(subs), dtype=float)
        want = sp.matrix2numpy(sp.Matrix(solver), dtype=float)
        assert got.shape == want.shape, f"{path.name}: {what} shape"
        scale = max(abs(want).max(), 1.0)
        assert abs(got - want).max() <= 1e-9 * scale, f"{path.name}: {what} differs"


@pytest.mark.skipif(PARSE_SPEC is None, reason="the equation solver's parse_spec is not in tools/fem yet")
@pytest.mark.parametrize("path", LIBRARY_FILES, ids=spec_ids(LIBRARY_FILES))
def test_library_file_round_trips_through_the_solver(path):
    assert PARSE_SPEC(load(path)) is not None


@pytest.mark.skipif(PARSE_SPEC is None, reason="the equation solver's parse_spec is not in tools/fem yet")
@pytest.mark.parametrize("index", range(len(readme_json_blocks())))
def test_readme_example_round_trips_through_the_solver(index):
    assert PARSE_SPEC(json.loads(readme_json_blocks()[index])) is not None


@needs_solver
def test_both_modules_agree_on_the_coefficient_keys():
    """Two modules name the four slots; if one grows a fifth, say so here."""
    from tools.fem import equation

    assert set(_COEFF_KEYS) == set(equation.COEFF_KEYS)


@needs_solver
@pytest.mark.parametrize(
    "spec",
    [
        {"coeffs": {"v4": "E*I", "v3": "c"}, "rhs": "q", "params": {"E": 1, "I": 1, "c": 1, "q": 1}},
        {"coeffs": {"v4": "E*I", "v9": "c"}, "rhs": "q", "params": {"E": 1, "I": 1, "c": 1, "q": 1}},
        {"coeffs": {"v4": "0"}, "rhs": "q", "params": {"q": 1}},
        {"coeffs": {"v4": "E*("}, "rhs": "q", "params": {"E": 1, "q": 1}},
        {"coeffs": {"v4": "E*Iz"}, "rhs": "q", "params": {"E": 1, "q": 1}},
    ],
)
def test_the_document_and_the_solver_reject_the_same_specs(spec):
    """A spec the solver refuses must not quietly produce a derivation, or the
    professor would read a document for an equation that never runs."""
    with pytest.raises(ValueError):
        PARSE_SPEC(spec)
    with pytest.raises(ValueError):
        validate_equation_spec(spec)

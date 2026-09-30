"""What an equation string is allowed to be.

The coefficient and rhs strings in a spec are the trust boundary of the whole
equation feature: they are written by the professor, or by the orchestrator
model reading his brief, and sympy's parse_expr eval()s what it tokenizes.
Three modules read the same strings - the solver, the manufactured-solution
check that verifies it, and the derivation the report prints - so every case
below is asserted against ALL THREE. A rejection in one and an acceptance in
another is the hole: it would mean the gate verified an expression the solver
never ran, or the report documented one it did.

Offline and deterministic; nothing here needs a network or an LLM.
"""

import sys
from pathlib import Path

import pytest
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.fem.mms as mms  # noqa: E402
from tools.fem.derivation import validate_equation_spec  # noqa: E402
from tools.fem.equation import EquationError, parse_spec, solve_equation_beam  # noqa: E402
from tools.fem.safe_expr import (  # noqa: E402
    ALLOWED_FUNCTIONS,
    MAX_CHARS,
    MAX_NODES,
    PARSE_GLOBALS,
    ExpressionError,
    parse_expression,
)

PARAMS = {"E": 30e9, "I": 0.005, "q": -30e3, "L": 25.0}


def spec_with(v4="E*I", rhs="q", params=None):
    return {
        "label": "probe",
        "coeffs": {"v4": v4, "v2": "0", "v1": "0", "v0": "0"},
        "rhs": rhs,
        "params": dict(PARAMS if params is None else params),
    }


def beam(n_elements=8, span=25.0):
    xs = [span * k / n_elements for k in range(n_elements + 1)]
    return {
        "nodes": [{"id": f"N{k}", "x": x} for k, x in enumerate(xs)],
        "supports": {
            "N0": [True, True, True, True, False, False],
            f"N{n_elements}": [False, True, True, False, False, False],
        },
        "point_loads": [],
    }


def all_three(spec, field="coeffs"):
    """Run one spec through every parser that reads it, returning the errors.

    mms reads the coefficients and the right-hand side through different
    entry points, so the field being probed picks the one that sees it.
    """
    read_by_mms = (
        (lambda: mms._coefficients(spec))
        if field == "coeffs"
        else (lambda: mms.residual(spec, "x**2"))
    )
    outcomes = {}
    for name, call in (
        ("solver", lambda: parse_spec(spec)),
        ("mms", read_by_mms),
        ("derivation", lambda: validate_equation_spec(spec)),
    ):
        try:
            call()
            outcomes[name] = None
        except ValueError as exc:  # EquationError and ExpressionError are ValueErrors
            outcomes[name] = exc
    return outcomes


# --------------------------------------------------------------- code injection

# Expressions that must never be evaluated. Each is data for the parser, and
# the assertion is that it is refused before sympy is ever handed it.
HOSTILE = [
    pytest.param("__import__('os').system('echo pwned')", id="import-call"),
    pytest.param("E*I + __import__('os').getcwd()", id="import-inside-arithmetic"),
    pytest.param("pi.__class__.__base__", id="dunder-attribute-chain"),
    pytest.param("sqrt(4).is_integer", id="plain-attribute"),
    pytest.param("sqrt(2).evalf(3)", id="method-call"),
    pytest.param("open('/etc/passwd')", id="builtin-by-name"),
    pytest.param("eval('1+1')", id="eval-by-name"),
    pytest.param("lambda: 1", id="lambda"),
    pytest.param("[x for x in (1, 2)]", id="comprehension"),
    pytest.param("(1, 2)[0]", id="subscript"),
    pytest.param("'E*I'", id="string-literal"),
    pytest.param("f'{x}'", id="f-string"),
    pytest.param("x if x else 1", id="conditional"),
    pytest.param("E*I; q", id="two-statements"),
    pytest.param("E and I", id="boolean-operator"),
    pytest.param("~E", id="bitwise-invert"),
    pytest.param("E << I", id="shift"),
]


@pytest.mark.parametrize("text", HOSTILE)
def test_a_hostile_expression_is_refused_by_the_parser(text):
    """It must fail as an expression, not run as a program."""
    with pytest.raises(ExpressionError):
        parse_expression(text, local_dict={"x": sp.Symbol("x")})


@pytest.mark.parametrize("text", HOSTILE)
def test_every_parser_refuses_the_same_hostile_expression(text):
    outcomes = all_three(spec_with(v4=text))
    accepted = [name for name, err in outcomes.items() if err is None]
    assert not accepted, f"{accepted} accepted {text!r}"


@pytest.mark.parametrize("text", HOSTILE)
def test_a_hostile_rhs_is_refused_too(text):
    """The load side goes through the same parser as the coefficients."""
    outcomes = all_three(spec_with(rhs=text), field="rhs")
    accepted = [name for name, err in outcomes.items() if err is None]
    assert not accepted, f"{accepted} accepted rhs {text!r}"


@pytest.mark.parametrize("text", HOSTILE)
def test_a_hostile_parameter_value_is_refused(text):
    """params values are parsed too, so they are the same trust boundary."""
    with pytest.raises(EquationError):
        parse_spec(spec_with(params={**PARAMS, "I": text}))


def test_the_refusal_says_what_was_wrong_in_the_professors_terms():
    with pytest.raises(EquationError) as exc:
        parse_spec(spec_with(v4="E*I.conjugate()"))
    assert "attribute access" in str(exc.value)
    assert "coeffs['v4']" in str(exc.value)


def test_the_parse_namespace_hands_eval_no_builtins():
    """eval() supplies the real builtins to any globals dict lacking the key."""
    assert PARSE_GLOBALS["__builtins__"] == {}


def test_no_undeclared_name_can_reach_a_python_object():
    """Whatever survives parsing is sympy, never a callable or a module."""
    expr = parse_expression("E*I*sin(pi*x/L)", local_dict={"x": sp.Symbol("x")})
    assert isinstance(expr, sp.Expr)
    assert {s.name for s in expr.free_symbols} == {"E", "I", "L", "x"}


# ------------------------------------------------------------- silent nonsense


def test_an_unknown_function_is_named_not_silently_made_symbolic():
    """sympy's default reading of "heaviside(x)" is an undefined function of x,
    which carries no free symbol for the missing-parameter check to catch: the
    typo would become an opaque term the solver integrates."""
    with pytest.raises(EquationError) as exc:
        parse_spec(spec_with(v4="E*I*heaviside(x)"))
    assert "heaviside" in str(exc.value)
    assert "sin" in str(exc.value)  # the message lists what may be called


def test_a_parameter_used_as_a_function_is_refused():
    with pytest.raises(EquationError, match="unknown function 'q'"):
        parse_spec(spec_with(v4="E*I", rhs="q(x)"))


def test_a_caret_is_refused_with_the_python_power_spelled_out():
    with pytest.raises(EquationError) as exc:
        parse_spec(spec_with(v4="E*I*(1 + x/L)^2"))
    assert "**" in str(exc.value)


# ---------------------------------------------------------------- resource use


def test_an_absurdly_long_expression_is_refused_by_length():
    with pytest.raises(EquationError, match="characters long"):
        parse_spec(spec_with(v4="E*I" + "+0" * MAX_CHARS))


def test_an_expression_with_too_many_terms_is_refused_by_size():
    text = "+".join(["x"] * MAX_NODES)
    assert len(text) < MAX_CHARS  # short enough to pass the length check
    with pytest.raises(EquationError, match="terms"):
        parse_spec(spec_with(v4="E*I*(" + text + ")"))


def test_a_tower_of_exponents_does_not_hang_the_parser():
    """10**10**10 is a ten-billion-digit integer if anything evaluates it."""
    with pytest.raises(EquationError):
        parse_spec(spec_with(v4="E*I*10**10**10"))


# ------------------------------------------------------- the equations that work


LEGITIMATE = [
    pytest.param("E*I", id="prismatic"),
    pytest.param("E*I*(1 + x/L)", id="tapered-linear"),
    pytest.param("E*I*(1 + 2*(1 - 2*x/L)**2)", id="haunched-quadratic"),
    pytest.param("E*I*(2 - sin(pi*x/L))", id="trigonometric"),
    pytest.param("E*I*exp(-x/L)", id="exponential"),
    pytest.param("E*I*sqrt(2)", id="square-root-of-a-constant"),
    pytest.param("2.5e8", id="bare-number"),
    pytest.param("-E*I*-1", id="unary-signs"),
]


@pytest.mark.parametrize("text", LEGITIMATE)
def test_every_parser_accepts_the_same_legitimate_coefficient(text):
    outcomes = all_three(spec_with(v4=text))
    refused = {name: str(err) for name, err in outcomes.items() if err is not None}
    assert not refused, refused


@pytest.mark.parametrize("text", LEGITIMATE)
def test_a_legitimate_coefficient_still_solves(text):
    result = solve_equation_beam(beam(), spec_with(v4=text))
    assert result["max_abs"]["v"] > 0


def test_a_corner_in_v4_on_a_node_solves_with_the_right_moment():
    """Shear used to be (a4 v'')', which a corner in v4 cannot be differentiated
    for, so this was refused. Moment and shear are now recovered by equilibrium
    and nothing differentiates a4. The beam is statically determinate, so the
    moment is q(x^2 - Lx)/2 whatever the stiffness does."""
    result = solve_equation_beam(beam(8), spec_with(v4="E*I*(2 + Abs(x - 12.5)/L)"))
    q, span = PARAMS["q"], 25.0
    for p in result["samples"]:
        exact = q * (p["x"] ** 2 - span * p["x"]) / 2
        assert p["moment"] == pytest.approx(exact, abs=1e-9 * abs(q) * span**2 / 8)


def test_a_corner_in_v4_inside_an_element_is_refused_in_the_professors_words():
    """No quadrature rule integrates across a corner reliably; the self-check
    says so, naming the coefficient and the element, instead of guessing."""
    with pytest.raises(EquationError) as exc:
        solve_equation_beam(beam(7), spec_with(v4="E*I*(2 + Abs(x - 12.5)/L)"))
    assert "v4" in str(exc.value)
    assert "10.71" in str(exc.value)  # the element the corner sits in


def test_the_same_corner_is_fine_on_the_load_side():
    """rhs is only ever integrated, so Abs there is legitimate."""
    result = solve_equation_beam(beam(), spec_with(rhs="q*Abs(x - 12.5)/L"))
    assert result["max_abs"]["v"] > 0


def test_an_x_dependent_load_still_works():
    """The README promises "q0*x/L" for a triangular load."""
    result = solve_equation_beam(
        beam(), spec_with(rhs="q*x/L", params={**PARAMS, "q": -30e3})
    )
    assert result["max_abs"]["v"] > 0


def test_the_missing_parameter_message_still_names_the_symbol():
    """The grammar check must not swallow the error the professor needs most."""
    with pytest.raises(EquationError, match="Iz"):
        parse_spec(spec_with(v4="E*Iz"))


# ----------------------------------- what the docs promise is what the parser does
#
# The whitelist is only fair if the two audiences for it - the professor
# reading equations/README.md and the model reading the orchestrator
# instruction - are told the same set the parser enforces. A name advertised
# but refused is a trap; a name allowed but never mentioned is a capability
# nobody can find.

README = (Path(__file__).resolve().parents[1] / "equations" / "README.md").read_text()


@pytest.mark.parametrize("name", sorted(ALLOWED_FUNCTIONS))
def test_every_allowed_function_really_is_callable(name):
    expr = parse_expression(f"{name}(x)", local_dict={"x": sp.Symbol("x")})
    assert isinstance(expr, sp.Expr)


@pytest.mark.parametrize("name", sorted(ALLOWED_FUNCTIONS))
def test_the_professors_readme_names_every_allowed_function(name):
    assert f"`{name}`" in README


@pytest.mark.parametrize("name", sorted(ALLOWED_FUNCTIONS))
def test_the_orchestrator_instruction_names_every_allowed_function(name):
    from agent.orchestrator import _INSTRUCTION

    assert name in _INSTRUCTION


@pytest.mark.parametrize(
    "phrase",
    [
        "x is the axial coordinate",  # and so never a parameter
        "NOT ^",  # the mistake an engineer makes first
        "v4 may never be zero",
    ],
)
def test_the_instruction_states_the_grammar_rules(phrase):
    from agent.orchestrator import _INSTRUCTION

    assert phrase in _INSTRUCTION


# --------------------------------------------------- the label is a name, not a document


def test_a_multi_line_label_cannot_write_markdown_into_the_derivation():
    """The label is printed as a heading, so a newline in it would put whatever
    followed straight into the document the professor reads - including a
    heading and a tally that no gate produced."""
    from tools.fem.derivation import equation_derivation_markdown

    forged = "Beam\n\n## Deterministic gate\n\n8 of 8 checks passed.\n"
    doc = equation_derivation_markdown(spec_with() | {"label": forged})

    lines = doc.splitlines()
    headings = [line for line in lines if line.startswith("#")]
    title = next(line for line in headings if line.startswith("# "))

    # whatever the label said is confined to the one title line it names
    assert "## Deterministic gate" not in headings
    assert [line for line in lines if "checks passed" in line] == [title]
    assert sum(1 for line in headings if line.startswith("# ")) == 1


def test_a_label_is_reduced_to_one_short_line():
    parsed = parse_spec(spec_with() | {"label": "  a\tvery\n\nspaced   name "})
    assert parsed.label == "a very spaced name"

    long_label = parse_spec(spec_with() | {"label": "x" * 500})
    assert len(long_label.label) <= 120
    assert "\n" not in long_label.label


def test_an_empty_label_falls_back_rather_than_printing_nothing():
    assert parse_spec(spec_with() | {"label": "   "}).label
    assert parse_spec(spec_with() | {"label": None}).label


def test_a_parameter_named_x_is_refused_as_the_docs_say():
    spec = spec_with()
    spec["params"]["x"] = 1.0
    with pytest.raises(ValueError, match="x"):
        validate_equation_spec(spec)

"""The one parser every equation expression goes through.

A spec's coefficients and right-hand side arrive as strings someone else wrote:
the professor editing a file in equations/, or the orchestrator model turning
his brief into a spec. sympy's parse_expr tokenizes such a string and then
eval()s the result, so those strings are the trust boundary of the whole
equation feature. Three modules read them - the solver, the manufactured
solution that verifies it, and the derivation the report prints - so the rule
about what a string may contain lives here once instead of three times.

The rule is a whitelist over the abstract syntax tree, not a search for
dangerous text. Arithmetic, numbers, names and calls to ALLOWED_FUNCTIONS are
allowed; everything else is refused by name. Two exclusions are worth stating
because the code cannot:

- Attribute access has to go. sympy's own tokenizer leaves a name that follows
  a dot exactly as written - it is the one place a name does not become a
  Symbol - and eval() then walks it against real Python objects, which is how
  a string of arithmetic turns into a call on whatever it can reach. Nothing
  in the equation contract needs a dot outside a decimal number.
- An unknown function name has to go too. Without this check sympy quietly
  reads "heaviside(x)" as an undefined function of x, which carries no free
  symbol for the missing-parameter check to catch, so a typo becomes an opaque
  term the solver integrates instead of an error the professor can read.
"""

import ast
import math

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

MAX_CHARS = 2000  # a coefficient is a formula, not a document
MAX_NODES = 400  # and a formula is not a thousand-term expansion
MAX_POWER = 64  # no beam coefficient needs x**65
MAX_DIGITS = 1000  # nor a constant with more digits than that

# Every function an equation expression may call. "E" and "I" are deliberately
# absent, here and in the parse namespace: they are the professor's Young's
# modulus and second moment of area, not Euler's number and the imaginary unit.
ALLOWED_FUNCTIONS = {
    name: getattr(sp, name)
    for name in (
        "sin", "cos", "tan", "asin", "acos", "atan",
        "sinh", "cosh", "tanh", "exp", "log", "sqrt",
        "Abs", "Min", "Max", "sign",
    )
}

ALLOWED_CONSTANTS = {"pi": sp.pi}

# Constructors sympy's own tokenizer emits (auto_symbol writes Symbol(...),
# auto_number writes Integer(...)). The professor cannot name them: they are
# not in ALLOWED_FUNCTIONS, so the syntax check refuses them before eval.
_EMITTED = {
    "Symbol": sp.Symbol,
    "Function": sp.Function,
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
}

# parse_expr eval()s its output, and eval() helpfully supplies the real
# builtins to any globals dict that lacks the key. Supply an empty one.
PARSE_GLOBALS = {**ALLOWED_FUNCTIONS, **ALLOWED_CONSTANTS, **_EMITTED, "__builtins__": {}}

_BINARY_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
_UNARY_OPS = (ast.UAdd, ast.USub)


class ExpressionError(ValueError):
    """An expression string is unparseable or outside the allowed grammar."""


MAX_LABEL_CHARS = 120


def clean_label(value, default="custom equation") -> str:
    """A spec's label reduced to what a label is: one short line of text.

    The label is printed as a Markdown heading in the generated derivation and
    embedded in the report, so a label carrying newlines writes whatever it
    likes into that document - a forged "## Deterministic gate" and a tally to
    go under it. A name has no line breaks in it, so the fix is to say so here
    rather than to escape Markdown in three renderers.

    Args:
        value: the spec's "label", of any type.
        default: what an empty or missing label becomes.

    Returns:
        a single-line string of at most MAX_LABEL_CHARS characters.
    """
    text = " ".join(str(value if value is not None else "").split())
    if len(text) > MAX_LABEL_CHARS:
        text = text[: MAX_LABEL_CHARS - 1].rstrip() + "…"
    return text or default


def function_names() -> str:
    """The allowed function names, comma separated, for an error message."""
    return ", ".join(sorted(ALLOWED_FUNCTIONS) + sorted(ALLOWED_CONSTANTS))


def _refuse(message):
    raise ExpressionError(message)


_APPLY = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Mod: lambda a, b: a % b,
}


def _check_power(base, exponent):
    """Refuse a power whose value nothing should try to write down.

    10**10**10 is a ten-billion-digit integer. Nothing rejects it downstream:
    sympy would simply set about building it, and the run would look like a
    hang rather than a bad equation.
    """
    if abs(exponent) > MAX_POWER:
        _refuse(
            f"the exponent {exponent:g} is above the limit of {MAX_POWER}; "
            "a beam coefficient is a low-order polynomial in x"
        )
    if base is not None and abs(base) > 1:
        digits = abs(exponent) * math.log10(abs(base))
        if digits > MAX_DIGITS:
            _refuse(
                f"{base:g}**{exponent:g} is a number with about {digits:.0f} "
                f"digits; the limit is {MAX_DIGITS}"
            )


def _literal(node):
    """The value of a constant-only subtree, or None when a name appears in it.

    Powers are checked on the way, which is the point: the check has to happen
    before anything evaluates them, and it has to reach the innermost one.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            return None
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _UNARY_OPS):
        value = _literal(node.operand)
        if value is None:
            return None
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and isinstance(node.op, _BINARY_OPS):
        if isinstance(node.op, ast.Pow):
            exponent = _literal(node.right)
            base = _literal(node.left)
            if exponent is not None:
                _check_power(base, exponent)  # even when the base is symbolic
            if base is None or exponent is None:
                return None
            return base**exponent
        left, right = _literal(node.left), _literal(node.right)
        if left is None or right is None:
            return None
        try:
            return _APPLY[type(node.op)](left, right)
        except (ArithmeticError, ValueError):
            return None
    return None


def _check_node(node):
    """Refuse any syntax outside arithmetic, numbers, names and allowed calls."""
    if isinstance(node, ast.Attribute):
        _refuse(
            f"attribute access '.{node.attr}' is not allowed; an equation is "
            "arithmetic over x and the names in params"
        )
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            _check_node(node.func)  # says "attribute access", the real objection
        if not isinstance(node.func, ast.Name):
            _refuse("only a plain function call is allowed, e.g. sin(2*pi*x/L)")
        if node.func.id not in ALLOWED_FUNCTIONS:
            _refuse(
                f"unknown function {node.func.id!r}; an equation may call only: "
                + ", ".join(sorted(ALLOWED_FUNCTIONS))
            )
        if node.keywords:
            _refuse(f"{node.func.id}() takes no keyword arguments here")
        return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            _refuse(f"{node.value!r} is not a real number")
        return
    if isinstance(node, ast.Name):
        if not isinstance(node.ctx, ast.Load):
            _refuse(f"{node.id!r} cannot be assigned to in an equation")
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _BINARY_OPS):
            hint = (
                ". Note '^' is not a power in Python: write x**2"
                if isinstance(node.op, ast.BitXor)
                else ""
            )
            _refuse(f"operator {type(node.op).__name__} is not allowed{hint}")
        if isinstance(node.op, ast.Pow):
            _literal(node)  # refuses a power too large to evaluate
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _UNARY_OPS):
            _refuse(f"operator {type(node.op).__name__} is not allowed")
        return
    if isinstance(node, (ast.Expression, ast.Load, *_BINARY_OPS, *_UNARY_OPS)):
        return
    _refuse(
        f"{type(node).__name__} is not allowed in an equation expression; "
        "write arithmetic over x, the names in params, and " + function_names()
    )


def check_syntax(text: str) -> None:
    """Raise ExpressionError unless text is arithmetic this parser will accept.

    Args:
        text: the expression string exactly as the spec carries it.

    Raises:
        ExpressionError: the string is too long, is not a single Python
            expression, or uses syntax outside the allowed grammar, with a
            message naming what was refused.
    """
    if len(text) > MAX_CHARS:
        _refuse(f"expression is {len(text)} characters long; the limit is {MAX_CHARS}")
    if "__" in text:
        # Redundant once attribute access is gone, and kept anyway: a name with
        # a dunder in it has no place in an equation, and saying so is clearer
        # than whatever the grammar check would say about it.
        _refuse("'__' is not allowed in an equation expression")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"not a valid expression ({exc.msg})") from exc
    except (ValueError, MemoryError, RecursionError) as exc:
        raise ExpressionError(f"could not be read ({type(exc).__name__})") from exc

    nodes = 0
    for node in ast.walk(tree):
        nodes += 1
        if nodes > MAX_NODES:
            _refuse(f"expression has more than {MAX_NODES} terms")
        _check_node(node)


def parse_expression(text, local_dict=None) -> sp.Expr:
    """Parse one expression string into a sympy expression, safely.

    Args:
        text: the expression, as a string, or an already-sympy object or a
            number, both of which are returned as-is.
        local_dict: names bound to symbols before parsing, typically
            {"x": X} plus the spec's parameters.

    Returns:
        a sympy Expr.

    Raises:
        ExpressionError: the string is outside the allowed grammar, sympy
            cannot parse it, or what it parses to is not a scalar expression.
    """
    if isinstance(text, sp.Expr):
        return text
    if isinstance(text, bool):
        raise ExpressionError(f"{text!r} is not a real number")
    if isinstance(text, int):
        return sp.Integer(text)
    if isinstance(text, float):
        return sp.Float(text)
    if not isinstance(text, str):
        raise ExpressionError(
            f"must be a number or a sympy-parseable string, got {type(text).__name__}"
        )

    check_syntax(text)
    try:
        expr = parse_expr(
            text,
            local_dict=dict(local_dict or {}),
            global_dict=dict(PARSE_GLOBALS),
            transformations=standard_transformations,
        )
    except ExpressionError:
        raise
    except Exception as exc:  # sympy raises a zoo of parser errors
        raise ExpressionError(f"sympy could not parse it ({exc})") from exc

    # The grammar check should make this unreachable; it is the backstop that
    # keeps a non-expression from reaching code that assumes .free_symbols.
    if not isinstance(expr, sp.Expr):
        raise ExpressionError(
            f"is not a scalar expression (it parsed to {type(expr).__name__})"
        )
    return expr

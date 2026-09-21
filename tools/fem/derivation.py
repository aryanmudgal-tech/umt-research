"""Symbolic Galerkin derivations, rendered as Markdown teaching documents.

Two of them: galerkin_derivation_markdown() for the fixed 12-DOF
Euler-Bernoulli beam element the solver uses, and
equation_derivation_markdown(spec) for whatever equation the professor writes
into an equations/*.json file.

Deterministic: SymPy does the calculus, the text explains it. No LLM involved —
nothing in the output is a typed-in textbook matrix.

An equation whose coefficients are not polynomials in x has no element matrix
in closed form, and the solver integrates it numerically instead. The document
then says so, states the quadrature rule and its order, and prints no matrix,
rather than showing a closed form that was never computed. integration_method
mirrors the solver's own choice on this module's own parse of the spec, and
_claimed_method then holds that mirror to what the solver actually decided, so
that the worst a disagreement can cost is a closed form the reader does not
get - never an exactness claim over numbers that were integrated.
"""

import re

import sympy as sp

# The quadrature ORDER is tools/fem/equation.py's to set, so it is read off
# that module at call time rather than restated or bound at import: a document
# quoting a different order than the one that ran would be worse than one
# quoting none. The DECISION of which method applies is re-derived below
# (see integration_method) and then checked against the solver's own
# (see _claimed_method).
from tools.fem import equation as equation_solver
from tools.fem.safe_expr import ExpressionError, clean_label, parse_expression

_DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)


def _display_math_blocks(md: str) -> str:
    """Give every $$...$$ equation its own bare $$ lines, content on one line.

    Markdown reads a line starting with '+' or '-' as a list item, which splits
    a multi-line equation and throws every later $$ out of step. This form
    renders the same in GitHub, VS Code and KaTeX-based viewers.
    """

    def block(match):
        return "\n\n$$\n" + " ".join(match.group(1).split()) + "\n$$\n\n"

    return re.sub(r"\n{3,}", "\n\n", _DISPLAY_MATH.sub(block, md)).strip() + "\n"


def _hermite_shape_functions(x, L):
    """Derive N1..N4 by imposing the four nodal conditions on a general cubic."""
    v1, t1, v2, t2 = sp.symbols("v_1 theta_1 v_2 theta_2")
    a = sp.symbols("a_0:4")
    v = sum(a[k] * x**k for k in range(4))
    sol = sp.solve(
        [
            v.subs(x, 0) - v1,
            v.diff(x).subs(x, 0) - t1,
            v.subs(x, L) - v2,
            v.diff(x).subs(x, L) - t2,
        ],
        a,
    )
    vx = sp.expand(v.subs(sol))
    return [sp.factor(vx.coeff(d)) for d in (v1, t1, v2, t2)]


def _shape_function_lines(N, var):
    """One display equation per Hermite shape function, in the variable `var`."""
    return "\n".join(f"$$N_{k + 1}({var}) = {sp.latex(Nk)}$$" for k, Nk in enumerate(N))


def _beam_12x12(E, G, A, Iy, Iz, J, L, kb):
    """Assemble the symbolic 12x12 from the bending core kb (= L^3/EI * k)."""
    k = sp.zeros(12, 12)
    for a, b, c in ((0, 6, E * A / L), (3, 9, G * J / L)):
        k[a, a] = k[b, b] = c
        k[a, b] = k[b, a] = -c
    y_plane, z_plane = [1, 5, 7, 11], [2, 4, 8, 10]
    S = sp.diag(1, -1, 1, -1)  # theta_y = -d(uz)/dx sign map
    kz = E * Iz / L**3 * kb
    ky = E * Iy / L**3 * (S * kb * S)
    for r in range(4):
        for c in range(4):
            k[y_plane[r], y_plane[c]] += kz[r, c]
            k[z_plane[r], z_plane[c]] += ky[r, c]
    return k


def galerkin_derivation_markdown() -> str:
    """Render the full Galerkin/Hermite derivation as Markdown with LaTeX."""
    x, L = sp.symbols("x L", positive=True)
    E, I, q = sp.symbols("E I q", positive=True)
    A, G, Iy, Iz, J = sp.symbols("A G I_y I_z J", positive=True)

    N = _hermite_shape_functions(x, L)
    Nvec = sp.Matrix(N)
    B = Nvec.diff(x, 2)

    k_bend = sp.Matrix(4, 4, lambda r, c: sp.factor(sp.integrate(E * I * B[r] * B[c], (x, 0, L))))
    k_core = sp.expand(k_bend * L**3 / (E * I))  # the classic [[12, 6L, ...]] numbers
    f_udl = sp.simplify(sp.integrate(Nvec * q, (x, 0, L)))

    Na = sp.Matrix([1 - x / L, x / L])
    k_axial = sp.integrate(E * A * Na.diff(x) * Na.diff(x).T, (x, 0, L))
    k_torsion = k_axial.subs({E: G, A: J})

    k12 = _beam_12x12(E, G, A, Iy, Iz, J, L, k_core)

    ltx = sp.latex
    n_list = _shape_function_lines(N, "x")

    return _display_math_blocks(f"""# Galerkin derivation of the 3D Euler-Bernoulli beam element

## 1. Strong and weak form

Bending of an Euler-Bernoulli beam in the $x$-$y$ plane is governed by

$$EI \\, v''''(x) = q(x), \\qquad 0 \\le x \\le L .$$

Galerkin's method multiplies the residual by a test function $w(x)$ and
integrates over the element. Integrating by parts **twice** moves two
derivatives onto $w$, which is what lets us use $C^1$ (Hermite) interpolation
instead of $C^3$ functions:

$$\\int_0^L EI \\, v'' \\, w'' \\, dx
  = \\int_0^L q \\, w \\, dx
  + \\Big[ V w \\Big]_0^L + \\Big[ M w' \\Big]_0^L .$$

The boundary terms are the end shears and moments — exactly the nodal forces
the element exchanges with its neighbours.

## 2. Hermite cubic shape functions

The weak form contains $v''$, so the trial space needs continuous deflection
*and* slope at the nodes. Interpolate with a cubic in terms of the four nodal
DOFs $(v_1, \\theta_1, v_2, \\theta_2)$, where $\\theta = v'$:

$$v(x) = N_1 v_1 + N_2 \\theta_1 + N_3 v_2 + N_4 \\theta_2$$

Imposing $v(0)=v_1$, $v'(0)=\\theta_1$, $v(L)=v_2$, $v'(L)=\\theta_2$ on a
general cubic and solving for its coefficients (sympy did this) gives

{n_list}

## 3. Bending stiffness by Galerkin integration

With $v = \\mathbf{{N}} \\mathbf{{d}}$ and $w$ drawn from the same space
(Bubnov-Galerkin), the weak form becomes $\\mathbf{{k}} \\mathbf{{d}} =
\\mathbf{{f}}$ with $\\mathbf{{B}} = \\mathbf{{N}}''$ and

$$\\mathbf{{k}} = \\int_0^L EI \\, \\mathbf{{B}}^T \\mathbf{{B}} \\, dx
  = \\frac{{EI}}{{L^3}} {ltx(k_core)}$$

which is the classic Euler-Bernoulli bending matrix.

## 4. Consistent (work-equivalent) nodal loads

The load side of the weak form for a uniform $q$ gives

$$\\mathbf{{f}} = \\int_0^L \\mathbf{{N}}^T q \\, dx = {ltx(f_udl)}$$

i.e. $[qL/2, \\; qL^2/12, \\; qL/2, \\; -qL^2/12]$ — each node carries half the
load plus the fixed-end moments. The solver integrates the same expression for
linearly varying loads.

## 5. Axial and torsion blocks

Axial displacement and twist only need $C^0$ continuity, so linear shape
functions $[1 - x/L, \\; x/L]$ suffice:

$$\\mathbf{{k}}_{{axial}} = \\int_0^L EA \\, \\mathbf{{N}}_a'^T \\mathbf{{N}}_a' \\, dx
  = {ltx(k_axial)} , \\qquad
  \\mathbf{{k}}_{{torsion}} = {ltx(k_torsion)}$$

## 6. Assembling the 12x12

Per node the DOFs are $[u_x, u_y, u_z, r_x, r_y, r_z]$. Four independent
blocks fill the 12x12:

- axial $EA/L$ on $(u_{{x1}}, u_{{x2}})$,
- torsion $GJ/L$ on $(r_{{x1}}, r_{{x2}})$,
- bending about $z$ (the matrix of section 3 with $I = I_z$) on
  $(u_{{y1}}, r_{{z1}}, u_{{y2}}, r_{{z2}})$, since $r_z = +v'$,
- bending about $y$ with $I = I_y$ on $(u_{{z1}}, r_{{y1}}, u_{{z2}}, r_{{y2}})$
  — but here the right-hand-rule rotation is $\\theta_y = -\\,d u_z/dx$, so the
  section-3 matrix is transformed by $S = \\mathrm{{diag}}(1, -1, 1, -1)$:
  $\\mathbf{{k}}_y = S \\, \\mathbf{{k}} \\, S$, which flips the sign of every
  $6L$ term.

The result:

$$\\mathbf{{k}}_{{12 \\times 12}} = {ltx(k12)}$$
""")


# --------------------------------------------------------------------------
# Arbitrary equations: a4 v'''' + a2 v'' + a1 v' + a0 v = f(x)
# --------------------------------------------------------------------------

_COEFF_KEYS = ("v4", "v2", "v1", "v0")

_TERM_NOTES = {
    "v4": (
        "$v''''$", "$a_4$", "bending stiffness $EI$",
        lambda a, Le: a / Le**3,
        lambda tex: rf"\frac{{{tex}}}{{L_e^3}}",
        "This is the classic Euler-Bernoulli bending stiffness matrix "
        "$\\frac{EI}{L_e^3}[[12, 6L_e, -12, 6L_e], \\ldots]$.",
    ),
    "v2": (
        "$v''$", "$a_2$", "axial force $P$, compression positive",
        lambda a, Le: -a / (30 * Le),
        lambda tex: rf"-\frac{{{tex}}}{{30 L_e}}",
        "This is the standard geometric stiffness matrix "
        "$-\\frac{P}{30 L_e}[[36, 3L_e, -36, 3L_e], \\ldots]$. The minus sign is "
        "the whole point: a compressive $a_2$ subtracts from the bending "
        "stiffness, and the load at which the sum goes singular is the Euler "
        "buckling load.",
    ),
    "v1": (
        "$v'$", "$a_1$", "first-order (slope-proportional) term",
        lambda a, Le: a / 60,
        lambda tex: rf"\frac{{{tex}}}{{60}}",
        "This is a non-symmetric first-order matrix. It has no classical beam "
        "analogue - a beam whose equation needs it is not a self-adjoint "
        "energy problem, so expect a non-symmetric global stiffness.",
    ),
    "v0": (
        "$v$", "$a_0$", "foundation modulus $k$ (N/m per m of span)",
        lambda a, Le: a * Le / 420,
        lambda tex: rf"\frac{{{tex} L_e}}{{420}}",
        "This is the consistent Winkler-foundation matrix "
        "$\\frac{k L_e}{420}[[156, 22L_e, 54, -13L_e], \\ldots]$ - the same "
        "156/22/54/13 pattern as the consistent mass matrix, with the soil "
        "modulus $k$ in place of $\\rho A$.",
    ),
}


def _parse_scalar(text, field, local_dict):
    """Parse one spec string into a sympy expression, or raise ValueError.

    Parsed by tools.fem.safe_expr, the whitelist grammar the solver and the
    manufactured-solution check both use, so the document cannot be generated
    for an expression the solver would refuse.
    """
    if isinstance(text, bool) or not isinstance(text, (str, int, float)):
        raise ValueError(f"{field} must be a sympy-parseable string, got {text!r}")
    try:
        expr = parse_expression(text, local_dict=dict(local_dict))
    except ExpressionError as exc:
        raise ValueError(f"{field} is not a valid expression: {text!r} ({exc})") from exc
    if not isinstance(expr, sp.Expr):
        raise ValueError(f"{field} must be a scalar expression, got {text!r}")
    return expr


def validate_equation_spec(spec: dict) -> dict:
    """Check an equation spec against the contract and parse it symbolically.

    The spec describes the residual a4*v'''' + a2*v'' + a1*v' + a0*v = f(x),
    SI units, transverse deflection v(x), downward negative. See
    equations/README.md for the full contract and worked examples.

    The solver's parse_spec applies the same contract but substitutes the
    parameter values; this one keeps the symbols, so the document can print
    "E*I" rather than 1.5e8. evals/test_equation_docs.py pins the two to
    accept and reject the same specs.

    Args:
        spec: dict with "coeffs" (any of "v4", "v2", "v1", "v0" as
            sympy-parseable strings in x and the declared params), "rhs",
            "params" (name -> number) and an optional "label".

    Returns:
        dict with "label" (str), "coeffs" ({key: sympy expression}, absent
        coefficients as zero), "rhs" (sympy expression), "params"
        ({name: float}) and "x" (the axial-coordinate Symbol).

    Raises:
        ValueError: the spec breaks the contract, with a message naming the
            offending field.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"equation spec must be a dict, got {type(spec).__name__}")

    label = spec.get("label", "equation")
    if not isinstance(label, str):
        raise ValueError(f"'label' must be a string, got {label!r}")
    # printed as a heading in the generated document, so it stays one line
    label = clean_label(label, default="equation")

    params = spec.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("'params' must be a dict of name -> number")
    clean_params = {}
    for name, value in params.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"parameter name {name!r} is not a valid identifier")
        if name == "x":
            raise ValueError("'x' is the axial coordinate and cannot be a parameter")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"parameter {name!r} must be a number, got {value!r}")
        clean_params[name] = float(value)

    coeffs = spec.get("coeffs")
    if not isinstance(coeffs, dict) or not coeffs:
        raise ValueError("equation spec needs a non-empty 'coeffs' dict")
    if "v3" in coeffs:
        raise ValueError(
            "coefficient 'v3' is not supported: the Hermite cubic element has no "
            "v''' term. Allowed keys are v4, v2, v1, v0."
        )
    unknown = sorted(set(coeffs) - set(_COEFF_KEYS))
    if unknown:
        raise ValueError(
            f"unknown coefficient key(s) {unknown}; allowed keys are {list(_COEFF_KEYS)}"
        )

    x = sp.Symbol("x", real=True)
    local = {"x": x, **{name: sp.Symbol(name, real=True) for name in clean_params}}

    parsed = {
        key: _parse_scalar(coeffs[key], f"coeffs[{key!r}]", local)
        if key in coeffs
        else sp.Integer(0)
        for key in _COEFF_KEYS
    }
    rhs = _parse_scalar(spec.get("rhs", "0"), "'rhs'", local)

    for field, expr in list(parsed.items()) + [("rhs", rhs)]:
        missing = sorted(str(s) for s in expr.free_symbols if str(s) not in local)
        if missing:
            raise ValueError(
                f"{field} references undeclared parameter(s) {missing}; "
                "add them to 'params'"
            )

    if parsed["v4"].is_zero:
        raise ValueError(
            "coefficient 'v4' must be non-zero: with no v'''' term the Hermite "
            "element has no bending stiffness and the system is singular"
        )

    return {"label": label, "coeffs": parsed, "rhs": rhs, "params": clean_params, "x": x}


def _element_pieces(parsed, s, Le, xi):
    """The Hermite shape functions and this spec's coefficients in the local coord.

    Nothing is integrated here: both halves of equation_derivation_markdown
    need these, and only the symbolic half goes on to integrate them.
    """
    N = sp.Matrix(_hermite_shape_functions(s, Le))
    sub = {parsed["x"]: xi + s}
    a = {key: sp.expand(parsed["coeffs"][key].subs(sub)) for key in _COEFF_KEYS}
    return N, a


def _is_polynomial_in(expr, x) -> bool:
    """True when expr is a polynomial in x; a constant counts, at degree zero."""
    if x not in expr.free_symbols:
        return True
    try:
        sp.Poly(expr, x)
    except Exception:  # noqa: BLE001 - every sympy "not a polynomial" means the same here
        return False
    return True


def integration_method(parsed: dict) -> tuple:
    """("symbolic" | "quadrature", why) for a validated spec — see validate_equation_spec.

    Mirrors the rule in tools/fem/equation.py, on the same expressions with the
    same parameters substituted, so the document cannot claim a closed form the
    solver never computed. The two are pinned to each other by
    evals/test_quadrature.py rather than sharing code, for the same reason
    validate_equation_spec does not share code with parse_spec.

    Args:
        parsed: the dict validate_equation_spec returns.

    Returns:
        (method, reason) — reason names the first field that forced quadrature.
    """
    subs = {sp.Symbol(name, real=True): value for name, value in parsed["params"].items()}
    fields = [(f"coefficient {key!r}", parsed["coeffs"][key]) for key in _COEFF_KEYS]
    fields.append(("the right-hand side", parsed["rhs"]))
    for name, expr in fields:
        if not _is_polynomial_in(expr.subs(subs), parsed["x"]):
            return "quadrature", f"{name} is not polynomial in $x$"
    return "symbolic", "every coefficient and the right-hand side is polynomial in $x$"


def _claimed_method(spec: dict, parsed: dict) -> tuple:
    """(method, why) this document is allowed to claim: never above the solver's.

    integration_method mirrors the solver's rule, but it runs on THIS module's
    parse of the spec, where x carries real=True. That is not cosmetic: sympy
    folds Abs(x)**2 to x**2 for a real x while it is still being parsed, so the
    mirror calls "symbolic" an equation the solver integrates numerically. The
    solver also has a route to quadrature the mirror cannot see at all — a
    polynomial spec whose symbolic integration raises. Either one would leave
    this document printing a closed form nobody computed, next to a report
    paragraph that says the run was numerical.

    So the solver is asked, and only a DOWNGRADE to quadrature is taken from
    the answer. The mirror stays: it is what makes a solver that wrongly went
    symbolic show up as a document with no closed form in it, and
    evals/test_quadrature.py holds the two to each other.

    Args:
        spec: the raw equation spec dict, as the solver parses it.
        parsed: the dict validate_equation_spec returns for that spec.

    Returns:
        (method, reason) - "quadrature" whenever either side says so.
    """
    method, why = integration_method(parsed)
    if method != "symbolic":
        return method, why
    try:
        plan = equation_solver.integration_plan(spec)
    except Exception:  # noqa: BLE001 - the solver refuses a spec this document took
        return method, why
    if plan["method"] == "symbolic":
        return method, why
    # the solver's reason already ends in "so ...", which this document supplies
    return "quadrature", f"the solver reports that {plan['reason'].split(', so ')[0]}"


def _element_forms(parsed, s, Le, xi):
    """Integrate k_e term by term and f_e over one element, in the local coord s.

    The spec is written in the global x, so x is replaced by x_i + s (x_i = the
    element's left node) before integrating; for x-independent coefficients
    that substitution is a no-op and the textbook matrices drop straight out.

    Only safe for a spec integration_method calls "symbolic": sympy does not
    return in any useful time on, say, a square-root taper.
    """
    N, a = _element_pieces(parsed, s, Le, xi)
    d1, d2 = N.diff(s), N.diff(s, 2)

    integrands = {
        "v4": lambda: a["v4"] * d2 * d2.T,
        "v2": lambda: -a["v2"] * d1 * d1.T,
        "v1": lambda: a["v1"] * N * d1.T,
        "v0": lambda: a["v0"] * N * N.T,
    }
    terms = {}
    for key, build in integrands.items():
        if a[key].is_zero:
            terms[key] = sp.zeros(4, 4)
            continue
        m = build()
        terms[key] = sp.Matrix(4, 4, lambda r, c: sp.factor(sp.integrate(m[r, c], (s, 0, Le))))

    f = sp.expand(parsed["rhs"].subs({parsed["x"]: xi + s}))
    f_e = sp.Matrix(4, 1, lambda r, _c: sp.factor(sp.integrate(f * N[r], (s, 0, Le))))
    return N, a, terms, f_e


def _residual_latex(a):
    """LaTeX for the residual with this spec's coefficients written in."""
    pieces = []
    for key, deriv in zip(_COEFF_KEYS, ("v''''", "v''", "v'", "v")):
        if a[key].is_zero:
            continue
        tex = sp.latex(a[key])
        if a[key].is_Add or a[key].could_extract_minus_sign():
            tex = rf"\left({tex}\right)"
        pieces.append(rf"{tex} \, {deriv}")
    return " + ".join(pieces)


def _scalar_tex(expr):
    """LaTeX for a scalar factored out in front of a matrix, parenthesised if needed."""
    tex = sp.latex(expr)
    return rf"\left({tex}\right)" if expr.could_extract_minus_sign() else tex


def equation_derivation_markdown(spec: dict) -> str:
    """Render the Galerkin derivation for one equation spec as Markdown.

    Use this when the governing equation is not the plain Euler-Bernoulli beam:
    it derives the element matrices for whatever equation the spec describes.
    Every matrix is integrated by sympy at call time - nothing is typed in.

    Args:
        spec: equation spec dict - "coeffs" (any of "v4", "v2", "v1", "v0" as
            sympy-parseable strings), "rhs", "params" and an optional "label".
            See equations/README.md and the ready-made files in equations/.

    Returns:
        Markdown with LaTeX: the residual for this spec, the weak form with the
        boundary terms each integration by parts produced, the four Hermite
        shape functions, the element matrix k_e and load vector f_e for this
        equation, and which textbook matrix each coefficient reduces to.

    Raises:
        ValueError: the spec breaks the contract (a 'v3' term, an unknown
            coefficient key, an unparseable expression, an undeclared param).
    """
    parsed = validate_equation_spec(spec)
    s, xi = sp.symbols("s x_i", real=True)
    Le = sp.Symbol("L_e", positive=True)

    method, why = _claimed_method(spec, parsed)
    terms = None
    if method == "symbolic":
        try:
            N, a, terms, f_e = _element_forms(parsed, s, Le, xi)
        except Exception as exc:  # noqa: BLE001 - mirrors the solver's own fallback
            method, terms = "quadrature", None
            why = (
                "every coefficient is polynomial in $x$, but symbolic integration "
                f"failed ({type(exc).__name__}: {str(exc)[:160]})"
            )
    if terms is None:
        N, a = _element_pieces(parsed, s, Le, xi)
    else:
        k_e = sp.expand(sum(terms.values(), sp.zeros(4, 4)))

    ltx = sp.latex
    active = [key for key in _COEFF_KEYS if not a[key].is_zero]
    f_expr = sp.expand(parsed["rhs"].subs({parsed["x"]: xi + s}))
    quadrature = method == "quadrature"
    intro = (
        "Every matrix below was integrated by sympy when this document was "
        "generated - none of them is a typed-in textbook matrix."
        if not quadrature
        else "This equation is integrated numerically, so the element entries below "
        "are described by the quadrature rule that produces them rather than "
        "printed as closed forms - see section 4."
    )

    coeff_rows = "\n".join(
        f"| {_TERM_NOTES[key][0]} | {_TERM_NOTES[key][1]} | "
        f"`{spec['coeffs'].get(key, 0)}` | {_TERM_NOTES[key][2]} |"
        for key in active
    )
    zeroed = [_TERM_NOTES[key][1] for key in _COEFF_KEYS if a[key].is_zero]
    zero_note = (
        f"\nCoefficients you left at zero: {', '.join(zeroed)} - "
        "those terms drop out of every integral below.\n"
        if zeroed
        else ""
    )

    param_rows = "\n".join(
        f"| `{name}` | {value:g} |" for name, value in sorted(parsed["params"].items())
    ) or "| _(none)_ | |"

    if quadrature:
        n = equation_solver.QUADRATURE_POINTS
        evaluation = (
            f"**How these are evaluated: numerically, by {n}-point Gauss-Legendre "
            "quadrature.** The method is chosen by structure, not by a timer: "
            f"{why}, so these integrals have no closed form sympy can be relied on "
            "to finish. No closed-form $\\mathbf{k}_e$ is printed below because "
            "there is none to print. The solver evaluates every entry as a "
            f"weighted sum over {n} points inside each element:\n"
            "\n"
            "$$\\mathbf{k}_e = \\frac{L_e}{2} \\sum_{g=1}^{" + str(n) + "} w_g "
            "\\left( a_4 \\mathbf{N}''^T \\mathbf{N}'' - a_2 \\mathbf{N}'^T "
            "\\mathbf{N}' + a_1 \\mathbf{N}^T \\mathbf{N}' + a_0 \\mathbf{N}^T "
            "\\mathbf{N} \\right)_{s = s_g}$$\n"
            "\n"
            "$$\\mathbf{f}_e^T = \\frac{L_e}{2} \\sum_{g=1}^{" + str(n) + "} w_g "
            "\\, f(x_i + s_g) \\, \\mathbf{N}^T(s_g), \\qquad "
            "s_g = \\frac{L_e}{2} \\left( t_g + 1 \\right)$$\n"
            "\n"
            "where $t_g$ and $w_g$ are the Gauss-Legendre nodes and weights on "
            f"$[-1, 1]$, and $f = {ltx(f_expr)}$ over the element.\n"
            "\n"
            "**The entries are therefore numerical, not closed-form.** An "
            "$n$-point rule integrates a polynomial of degree $2n - 1$ exactly, "
            "and the richest element integrand here is degree 6 in $s$, so "
            f"$n = {n}$ reproduces any polynomial spec exactly with a wide margin "
            "left over. For a spec that is not polynomial there is no exactness "
            "to claim, so the solver checks each element instead: it repeats "
            f"every integration at {2 * n} points and refuses the equation, "
            "naming the coefficient, unless each entry agrees to 1 part in "
            "$10^{10}$.\n"
        )
    else:
        evaluation = (
            f"**How these are evaluated: symbolically.** {why[0].upper() + why[1:]}, "
            "so sympy integrates each entry in closed form when this document is "
            "generated and the matrices below are exact:\n"
            "\n"
            f"$$\\mathbf{{k}}_e = {ltx(k_e)}$$\n"
            "\n"
            f"$$\\mathbf{{f}}_e^T = {ltx(f_e.T)}$$\n"
            "\n"
            f"with $f = {ltx(f_expr)}$ over the element.\n"
        )

    term_blocks = []
    for key in active:
        _, sym, _, factor, factor_tex, textbook = _TERM_NOTES[key]
        block = f"**{sym} term.** {textbook}\n"
        if quadrature:
            block += (
                f"\nYour $a_{key[1]} = {ltx(a[key])}$ is integrated numerically, so "
                "this term has no closed-form matrix of its own to show; it reaches "
                "$\\mathbf{k}_e$ through the quadrature sum of section 4.\n"
            )
        elif parsed["x"] not in parsed["coeffs"][key].free_symbols:
            scale = factor(a[key], Le)
            core = sp.expand(terms[key] / scale)
            block += (
                f"\nFor your $a_{key[1]} = {ltx(a[key])}$ it integrates to\n\n"
                f"$$\\mathbf{{k}}_e^{{({key[1]})}} = "
                f"{factor_tex(_scalar_tex(a[key]))} {ltx(core)}$$\n"
            )
        else:
            block += (
                f"\nYour $a_{key[1]}$ varies along $x$, so no constant factor comes "
                "out in front; sympy integrates it as it stands:\n\n"
                f"$$\\mathbf{{k}}_e^{{({key[1]})}} = {ltx(terms[key])}$$\n"
            )
        term_blocks.append(block)

    return _display_math_blocks(f"""# Galerkin derivation: {parsed["label"]}

Derived from your equation spec. {intro}

## 1. The equation you gave

Written as a residual, in the pipeline's sign convention (SI units, $v(x)$ the
transverse deflection, **downward negative**):

$$\\mathcal{{R}}(v) = a_4 \\, v'''' + a_2 \\, v'' + a_1 \\, v' + a_0 \\, v - f(x) = 0$$

With your coefficients substituted:

$${_residual_latex(parsed["coeffs"])} = {ltx(parsed["rhs"])}$$

| term | coefficient | your value | what it models |
| --- | --- | --- | --- |
{coeff_rows}
{zero_note}
Parameter values used for the numbers:

| parameter | value (SI) |
| --- | --- |
{param_rows}

## 2. Weak form

Multiply the residual by a test function $w(x)$ and integrate over one element
of length $L_e$, in the local coordinate $s = x - x_i$:

$$\\int_0^{{L_e}} w \\left( a_4 v'''' + a_2 v'' + a_1 v' + a_0 v \\right) ds = \\int_0^{{L_e}} w \\, f \\, ds$$

Two terms get integrated by parts. The $v''''$ term goes **twice**, which is
what drops the continuity requirement from $C^3$ to $C^1$ and lets Hermite
cubics be used at all. The $v''$ term goes **once**. The $v'$ and $v$ terms are
left alone - they already sit at the lowest derivative order.

| term | integrations by parts | boundary term it leaves |
| --- | --- | --- |
| $a_4 v''''$ | 2 | $\\big[ w \\, (a_4 v'')' \\big]_0^{{L_e}} - \\big[ w' \\, a_4 v'' \\big]_0^{{L_e}}$ |
| $a_2 v''$ | 1 | $\\big[ w \\, a_2 v' \\big]_0^{{L_e}}$ |
| $a_1 v'$ | 0 | none |
| $a_0 v$ | 0 | none |

Those boundary terms are the end shear $(a_4 v'')'$ and the end moment
$a_4 v''$ - exactly the nodal forces the element hands to its neighbours, which
is why they cancel internally and only survive at the beam's two ends.

What is left is the weak form the element matrices come from:

$$\\int_0^{{L_e}} \\left( a_4 w'' v'' - a_2 w' v' + a_1 w \\, v' + a_0 w \\, v \\right) ds = \\int_0^{{L_e}} w \\, f \\, ds + \\text{{boundary terms}}$$

Note the sign flip on $a_2$: integrating by parts once moves one derivative
onto $w$ and takes a minus sign with it.

**One honesty note about $x$-dependent coefficients.** Moving both derivatives
onto $w$ turns $a_4 v''''$ into the self-adjoint form $(a_4 v'')''$, and
$a_2 v''$ into $(a_2 v')'$. When $a_4$ and $a_2$ are constant the two readings
are identical. When they vary along $x$ - a tapered beam, say - the
self-adjoint form is the physically correct one: $(EI v'')'' = q$ is the real
tapered-beam equation, and it is the one this element solves.

## 3. Hermite shape functions

The weak form contains $v''$, so the trial space needs continuous deflection
*and* slope at the nodes. Interpolating a cubic on the four nodal DOFs
$(v_1, \\theta_1, v_2, \\theta_2)$ with $\\theta = v'$ gives

{_shape_function_lines(N, "s")}

These do not depend on the equation - they are fixed by the element's four
DOFs, so every spec shares them.

## 4. Element matrix and load vector for this equation

Substituting $v = \\mathbf{{N}} \\mathbf{{d}}$ and drawing $w$ from the same
space (Bubnov-Galerkin):

$$\\mathbf{{k}}_e = \\int_0^{{L_e}} \\left( a_4 \\mathbf{{N}}''^T \\mathbf{{N}}'' - a_2 \\mathbf{{N}}'^T \\mathbf{{N}}' + a_1 \\mathbf{{N}}^T \\mathbf{{N}}' + a_0 \\mathbf{{N}}^T \\mathbf{{N}} \\right) ds$$

$$\\mathbf{{f}}_e = \\int_0^{{L_e}} f \\, \\mathbf{{N}}^T ds$$

{evaluation}
## 5. What each term reduces to

{chr(10).join(term_blocks)}
""")


if __name__ == "__main__":
    print(galerkin_derivation_markdown())

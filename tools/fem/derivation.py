"""Symbolic Galerkin derivation of the 12-DOF Euler-Bernoulli beam element.

Deterministic teaching document: SymPy does the calculus, the text explains it.
No LLM involved — every matrix in the output is computed by sympy at call time.
"""

import re

import sympy as sp

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
    n_list = "\n".join(
        f"$$N_{k + 1}(x) = {ltx(Nk)}$$" for k, Nk in enumerate(N)
    )

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


if __name__ == "__main__":
    print(galerkin_derivation_markdown())

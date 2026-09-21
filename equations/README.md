# The equation library

The governing equation is **data, not code**. Nothing in this pipeline has the
Euler-Bernoulli stiffness matrix typed into it: the element matrices are
integrated symbolically from whatever equation one of these files describes.
To change the physics, edit a JSON file here — or copy one and write your own.

| file | equation it describes |
| --- | --- |
| `euler_bernoulli.json` | the default prismatic beam in pure bending |
| `elastic_foundation.json` | a beam bedded on soil that pushes back |
| `beam_column.json` | a beam that also carries an axial force |
| `tapered_beam.json` | a beam whose `EI` grows along the span |

## The contract

Every file describes one ordinary differential equation for the transverse
deflection `v(x)`, written as a **residual**:

```
a4 * v'''' + a2 * v'' + a1 * v' + a0 * v = f(x)
```

- `x` runs along the beam axis, in metres from the left-hand end.
- `v(x)` is the transverse deflection, and **downward is negative**. This is
  why the professor's own beam carries `q = -30e3`: 30 kN/m pressing down. Get
  this sign wrong and the beam will deflect upwards.
- Everything is SI: metres, newtons, pascals. Convert `kN/m` to `N/m` and `GPa`
  to `Pa` before you write the number down.
- Both sides are whole terms, not magnitudes. `a4*v'''' + a0*v = q` means the
  soil term sits on the *same* side as the bending term, with a plus sign.

### The four coefficient slots

`coeffs` may contain these keys and no others:

| key | term it multiplies | what it usually is |
| --- | --- | --- |
| `v4` | `v''''` | bending stiffness, `E*I`. **Required and non-zero** |
| `v2` | `v''` | axial force `P`, compression positive |
| `v1` | `v'` | a slope-proportional term (rare in statics) |
| `v0` | `v` | foundation modulus `k`, N/m per metre of span |

A slot you do not need is `"0"`, or simply left out. `v4` may not be zero: with
no fourth-order term the Hermite element has no bending stiffness at all and
the global matrix is singular.

There is **no `v3` slot**. A file containing one is rejected with an error
rather than silently ignored — see "What is not supported" below.

### Referencing parameters

Coefficients and `rhs` are strings that sympy parses. They may use:

- `x`, the axial coordinate — always available, and never declared in `params`
  (a parameter called `x` is an error, not a redefinition);
- any name you list in `params`, which is a flat map of name to number;
- ordinary arithmetic — `+ - * / %`, brackets, and `**` for powers. `^` is
  **not** a power here, and writing one is an error rather than a wrong answer;
- these functions, and no others: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`,
  `sinh`, `cosh`, `tanh`, `exp`, `log`, `sqrt`, `Abs`, `Min`, `Max`, `sign`,
  plus the constant `pi`.

That list is a whitelist, and anything outside it is refused by name — a call
to `heaviside(x)` fails with a message listing what you may call, rather than
quietly becoming an unknown function the solver then integrates. An expression
is arithmetic and nothing else: no attribute access, no indexing, no strings.

`Abs`, `Min`, `Max` and `sign` have corners in them, and shear is
`(a4*v'')'`, so a `v4` built from any of them cannot be differentiated along
the span and is refused with a message saying so. They are fine in `rhs` and in
the other three slots, which are only ever integrated.

Every symbol in a coefficient must appear in `params`. A typo like `E*Iz` when
you declared `I` is an error naming the missing parameter, not a silent zero.
Note that `E` and `I` mean *your* parameters, not Euler's number and the
imaginary unit.

One practical limit: the element matrices are integrated symbolically, and
sympy is fast on polynomials in `x` and slow on some other forms. A polynomial
taper such as `"E*I0*(1 + 2*(1 - 2*x/L)**2)"` integrates in well under a
second; a coefficient under a square root, such as `"E*I0*sqrt(1 + x/L)"`, can
take minutes or longer with no sign of progress. Prefer a polynomial fit of the
stiffness you want — for a second moment of area it is an approximation either
way.

A complete file looks like this — this is `euler_bernoulli.json`:

```json
{
  "_comment": "The pipeline's default: a prismatic Euler-Bernoulli beam in pure bending, carrying a uniform downward load, with no axial force and no support from the ground.",
  "label": "Euler-Bernoulli beam",
  "coeffs": {
    "v4": "E*I",
    "v2": "0",
    "v1": "0",
    "v0": "0"
  },
  "rhs": "q",
  "params": {
    "E": 30e9,
    "I": 0.005,
    "q": -30e3
  }
}
```

`_comment` is yours: one sentence on the physics, so the next reader knows what
the file is for. `label` is the human name that appears at the top of the
generated derivation. Both are optional; `coeffs` is not.

## Worked example 1: adding a soil spring

Suppose the 25 m beam no longer sits on two bearings but on compacted ground
that pushes back as it sinks. The classic Winkler model says the upward
pressure is proportional to the local deflection, `p = -k*v`, so the equation
gains one term:

```
E*I*v'''' + k*v = q
```

`k` is the modulus of subgrade reaction times the bearing width — for a 0.5 m
wide beam on a subgrade of 20 MN/m³ that is `0.5 * 20e6 = 1.0e7` N/m per metre
of span.

Start from `euler_bernoulli.json`, fill the `v0` slot with `k`, and declare `k`
in `params`. Nothing else changes:

```json
{
  "_comment": "The 25 m beam bedded on compacted fill: the ground pushes back in proportion to local settlement.",
  "label": "25 m beam on compacted fill",
  "coeffs": {
    "v4": "E*I",
    "v2": "0",
    "v1": "0",
    "v0": "k"
  },
  "rhs": "q",
  "params": {
    "E": 30e9,
    "I": 0.005,
    "k": 1.0e7,
    "q": -30e3
  }
}
```

The derivation document will now show a second matrix under the `v0` term: the
consistent foundation matrix `k*L/420 * [[156, 22L, 54, -13L], ...]`, added to
the usual bending matrix. Stiffer soil means a larger `k` means less
deflection, which is the check to run first on the result.

## Worked example 2: making EI vary along the span

A haunched bridge girder is deeper over its supports than at midspan, so `I` is
a function of `x`. Coefficients may depend on `x` directly — write the
variation into the string and declare the span `L` as a parameter:

```json
{
  "_comment": "A haunched girder, three times stiffer over the supports than at midspan, with EI varying quadratically along the 25 m span.",
  "label": "Haunched girder",
  "coeffs": {
    "v4": "E*I0*(1 + 2*(1 - 2*x/L)**2)",
    "v2": "0",
    "v1": "0",
    "v0": "0"
  },
  "rhs": "q",
  "params": {
    "E": 30e9,
    "I0": 0.005,
    "L": 25.0,
    "q": -30e3
  }
}
```

That expression is `3*E*I0` at `x = 0` and `x = L`, and `E*I0` at midspan. Two
things are worth knowing:

- `I0` is a *second moment of area*, and for a rectangular section it goes as
  the cube of the depth. A stiffness that varies quadratically is not a depth
  that varies quadratically — work out `I(x)` from your real section first.
- With `a4` varying, the weak form uses the self-adjoint reading
  `(E*I*v'')'' = q`, which is the physically correct tapered-beam equation.
  When `a4` is constant that is identical to `E*I*v''''`; when it varies it is
  not, and the self-adjoint form is the one you want. The generated derivation
  says so in section 2.

The same trick covers a load that varies along the span: `rhs` takes `x` too,
so a triangular load is `"q0*x/L"` and a patch is a `sin` or a polynomial fit.

## Checking your equation before you run it

The derivation document is generated from the file, so it is the fastest way to
see whether you wrote down the equation you meant:

```python
import json
from tools.fem.derivation import equation_derivation_markdown

spec = json.loads(open("equations/elastic_foundation.json").read())
print(equation_derivation_markdown(spec))
```

It prints the residual with your coefficients substituted, the weak form and
the boundary terms each integration by parts left behind, the Hermite shape
functions, and the element matrix and load vector sympy integrated for *your*
equation — plus a note on which textbook matrix each term reduces to. If the
residual at the top is not the equation you intended, stop there.

A malformed file raises a `ValueError` naming the field, so a typo fails
immediately and loudly rather than producing plausible wrong numbers.

## What is not supported

This element solves one **linear, static, one-dimensional, one-field** ODE.
The following are outside it, and the tooling will not pretend otherwise:

- **Nonlinear terms.** Anything where a coefficient depends on `v` or its
  derivatives — large deflections, `P` that grows with the deflection,
  yielding, cracked-section stiffness, contact that lifts off the soil.
  Coefficients may depend on `x` only.
- **Coupled fields.** One unknown, `v(x)`. Bending coupled to torsion or to
  axial stretch, composite layers with their own displacements, or
  thermo-mechanical coupling all need more than one field.
- **Time dependence.** No `v_tt` slot, so no dynamics, no modal analysis, no
  creep, no moving loads. Note that `v0` produces exactly the matrix a mass
  term would, so the pieces are there — the time integration is not.
- **Two-dimensional plates and shells.** `x` is a line along a beam. Plate
  bending needs a two-dimensional element, not a Hermite cubic on a line.
- **A `v'''` term.** The four-DOF Hermite element has no slot for it; a spec
  containing `v3` is rejected rather than quietly dropped.

Any of these is a real extension, not a config change. If your problem needs
one, say so and it can be built — describe the equation you want and what it
models, and the element and the verifier can be extended to match.

## "No stable equilibrium" — what that error means

Some equations parse, assemble, and have exactly one solution, and that
solution is still nonsense: it is an *unstable* equilibrium, and the beam comes
back deflecting **upwards** under a downward load. The solver refuses those
instead of reporting them, with an error beginning `no stable equilibrium`.

The test is that the symmetric part of the assembled stiffness matrix is
positive definite. That is the energy minimum for an equation with no `v1`
term, and the standard well-posedness condition when there is one. Four ways to
trip it, in rough order of how often they happen:

- **The supports leave a mechanism.** A single pin, or two nodes that hold
  slope but never hold deflection, lets the beam move without straining. Hold
  deflection at two points (or fully fix one end).
- **`v2` is compressive and at or past the buckling load.** A simply supported
  beam-column goes critical at `P = pi^2*E*I/L^2` — for the 25 m beam that is
  2.37 MN. Just below it the deflection is real but enormous (it grows like
  `1/(1 - P/P_cr)`, so 0.9 P_cr already multiplies it by ten); at it, the beam
  buckles and no linear static answer exists. If you want the buckling load
  itself, that is an eigenvalue problem, not this solve.
- **`v4` is zero or negative somewhere on the span.** `E*I*(1 - 2*x/L)` looks
  like a taper but goes negative past midspan, which is a beam with negative
  bending stiffness. Plot your `a4` over `0 <= x <= L` before running it and
  check it stays positive. A taper that only touches zero exactly at an end
  point is fine.
- **`v1` is large enough to swamp the bending term.** The `v1` term is not an
  energy term at all, so a big one has nothing holding it stable.

The error reports the smallest and largest eigenvalues, so the margin is
visible: a smallest eigenvalue barely below zero means you are just past a
critical point, and a large negative one means the equation is far from being a
beam.

Two related refusals in the same family: a parameter that is `inf` or `nan` is
rejected by name rather than solved into a result full of `nan`, and a `v4`
that is identically zero is rejected because the Hermite element then has no
bending stiffness at all.

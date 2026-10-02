# Phase 1 Report - The 25-Meter Agent

- Date: 2026-10-02
- Orchestrator model: gemini (replayed)
- Verifier model: gemini-3-flash-preview
- Solver: solve_with_equation
- Final gate: PASS

## Key results

The answers to the brief, read from the solver's output. Positions are measured from the left end of the beam.

| Quantity | Value | Where |
|---|---|---|
| Midspan deflection | 3.015 mm downward | x = 12.5 m |
| Maximum deflection | 3.197 mm downward | x = 6.53 m |
| Maximum bending moment | 37.5 kN·m | x = 2.17 m |
| Reaction at N0 (pin, x = 0 m) | 41.77 kN upward | |
| Reaction at N20 (roller, x = 25 m) | 41.77 kN upward | |

## Brief

### Brief: 25 m beam on soil

Analyze a bridge beam that rests on the ground along its whole length:

- Simply supported: pinned at one end, roller at the other.
- Span: 25 m.
- Elastic modulus: E = 30 GPa.
- Second moment of area about the bending axis: I = 0.005 m^4.
- Loading: uniform distributed load q = 30 kN/m acting downward over the full span.
- The beam rests on soil with a modulus of subgrade reaction k = 1.0e7 N/m per metre of span.

Report the midspan deflection, the maximum bending moment, and the support
shear forces.

Use this governing equation (EI v'''' + k v = q, with downward negative):

```json
{
  "label": "Beam on elastic foundation",
  "coeffs": {"v4": "E*I", "v2": "0", "v1": "0", "v0": "k"},
  "rhs": "q",
  "params": {"E": 30e9, "I": 0.005, "k": 1.0e7, "q": -30e3}
}
```

To analyze a different beam, change the numbers above. To change the physics,
describe it in words (an axial force, a depth that varies along the span) or
edit the equation: v4 is the bending stiffness, v2 an axial force
(compression positive), v0 the soil stiffness, and rhs the load.

## Governing equation

This run did NOT use the pipeline's default beam equation. The equation below came with the brief and was solved by `solve_with_equation`, which integrates its element matrices from the equation itself at run time.

**Beam on elastic foundation**

$$
E I \, v'''' + k \, v = q
$$

Sign convention: $v(x)$ is the transverse deflection, $x$ runs along the beam, SI units, **downward negative**.

| Parameter | Value (SI) |
|---|---|
| `k` | 1.000000e+07 |
| `E` | 3.000000e+10 |
| `I` | 5.000000e-03 |
| `q` | -3.000000e+04 |

**Element integrals: symbolic (exact).** Every coefficient and the right-hand side is a polynomial in x, so sympy integrates each element entry exactly.


### Derivation for this equation

#### Galerkin derivation: Beam on elastic foundation

Derived from your equation spec. Every matrix below was integrated by sympy when this document was generated - none of them is a typed-in textbook matrix.

##### 1. The equation you gave

Written as a residual, in the pipeline's sign convention (SI units, $v(x)$ the
transverse deflection, **downward negative**):

$$
\mathcal{R}(v) = a_4 \, v'''' + a_2 \, v'' + a_1 \, v' + a_0 \, v - f(x) = 0
$$

With your coefficients substituted:

$$
E I \, v'''' + k \, v = q
$$

| term | coefficient | your value | what it models |
| --- | --- | --- | --- |
| $v''''$ | $a_4$ | `E*I` | bending stiffness $EI$ |
| $v$ | $a_0$ | `k` | foundation modulus $k$ (N/m per m of span) |

Coefficients you left at zero: $a_2$, $a_1$ - those terms drop out of every integral below.

Parameter values used for the numbers:

| parameter | value (SI) |
| --- | --- |
| `E` | 3e+10 |
| `I` | 0.005 |
| `k` | 1e+07 |
| `q` | -30000 |

##### 2. Weak form

Multiply the residual by a test function $w(x)$ and integrate over one element
of length $L_e$, in the local coordinate $s = x - x_i$:

$$
\int_0^{L_e} w \left( a_4 v'''' + a_2 v'' + a_1 v' + a_0 v \right) ds = \int_0^{L_e} w \, f \, ds
$$

Two terms get integrated by parts. The $v''''$ term goes **twice**, which is
what drops the continuity requirement from $C^3$ to $C^1$ and lets Hermite
cubics be used at all. The $v''$ term goes **once**. The $v'$ and $v$ terms are
left alone - they already sit at the lowest derivative order.

| term | integrations by parts | boundary term it leaves |
| --- | --- | --- |
| $a_4 v''''$ | 2 | $\big[ w \, (a_4 v'')' \big]_0^{L_e} - \big[ w' \, a_4 v'' \big]_0^{L_e}$ |
| $a_2 v''$ | 1 | $\big[ w \, a_2 v' \big]_0^{L_e}$ |
| $a_1 v'$ | 0 | none |
| $a_0 v$ | 0 | none |

Those boundary terms are the end shear $(a_4 v'')'$ and the end moment
$a_4 v''$ - exactly the nodal forces the element hands to its neighbours, which
is why they cancel internally and only survive at the beam's two ends.

What is left is the weak form the element matrices come from:

$$
\int_0^{L_e} \left( a_4 w'' v'' - a_2 w' v' + a_1 w \, v' + a_0 w \, v \right) ds = \int_0^{L_e} w \, f \, ds + \text{boundary terms}
$$

Note the sign flip on $a_2$: integrating by parts once moves one derivative
onto $w$ and takes a minus sign with it.

**One honesty note about $x$-dependent coefficients.** Moving both derivatives
onto $w$ turns $a_4 v''''$ into the self-adjoint form $(a_4 v'')''$, and
$a_2 v''$ into $(a_2 v')'$. When $a_4$ and $a_2$ are constant the two readings
are identical. When they vary along $x$ - a tapered beam, say - the
self-adjoint form is the physically correct one: $(EI v'')'' = q$ is the real
tapered-beam equation, and it is the one this element solves.

##### 3. Hermite shape functions

The weak form contains $v''$, so the trial space needs continuous deflection
*and* slope at the nodes. Interpolating a cubic on the four nodal DOFs
$(v_1, \theta_1, v_2, \theta_2)$ with $\theta = v'$ gives

$$
N_1(s) = \frac{\left(- L_{e} + s\right)^{2} \left(L_{e} + 2 s\right)}{L_{e}^{3}}
$$

$$
N_2(s) = \frac{s \left(- L_{e} + s\right)^{2}}{L_{e}^{2}}
$$

$$
N_3(s) = - \frac{s^{2} \left(- 3 L_{e} + 2 s\right)}{L_{e}^{3}}
$$

$$
N_4(s) = \frac{s^{2} \left(- L_{e} + s\right)}{L_{e}^{2}}
$$

These do not depend on the equation - they are fixed by the element's four
DOFs, so every spec shares them.

##### 4. Element matrix and load vector for this equation

Substituting $v = \mathbf{N} \mathbf{d}$ and drawing $w$ from the same
space (Bubnov-Galerkin):

$$
\mathbf{k}_e = \int_0^{L_e} \left( a_4 \mathbf{N}''^T \mathbf{N}'' - a_2 \mathbf{N}'^T \mathbf{N}' + a_1 \mathbf{N}^T \mathbf{N}' + a_0 \mathbf{N}^T \mathbf{N} \right) ds
$$

$$
\mathbf{f}_e = \int_0^{L_e} f \, \mathbf{N}^T ds
$$

**How these are evaluated: symbolically.** Every coefficient and the right-hand side is polynomial in $x$, so sympy integrates each entry in closed form when this document is generated and the matrices below are exact:

$$
\mathbf{k}_e = \left[\begin{matrix}\frac{12 E I}{L_{e}^{3}} + \frac{13 L_{e} k}{35} & \frac{6 E I}{L_{e}^{2}} + \frac{11 L_{e}^{2} k}{210} & - \frac{12 E I}{L_{e}^{3}} + \frac{9 L_{e} k}{70} & \frac{6 E I}{L_{e}^{2}} - \frac{13 L_{e}^{2} k}{420}\\\frac{6 E I}{L_{e}^{2}} + \frac{11 L_{e}^{2} k}{210} & \frac{4 E I}{L_{e}} + \frac{L_{e}^{3} k}{105} & - \frac{6 E I}{L_{e}^{2}} + \frac{13 L_{e}^{2} k}{420} & \frac{2 E I}{L_{e}} - \frac{L_{e}^{3} k}{140}\\- \frac{12 E I}{L_{e}^{3}} + \frac{9 L_{e} k}{70} & - \frac{6 E I}{L_{e}^{2}} + \frac{13 L_{e}^{2} k}{420} & \frac{12 E I}{L_{e}^{3}} + \frac{13 L_{e} k}{35} & - \frac{6 E I}{L_{e}^{2}} - \frac{11 L_{e}^{2} k}{210}\\\frac{6 E I}{L_{e}^{2}} - \frac{13 L_{e}^{2} k}{420} & \frac{2 E I}{L_{e}} - \frac{L_{e}^{3} k}{140} & - \frac{6 E I}{L_{e}^{2}} - \frac{11 L_{e}^{2} k}{210} & \frac{4 E I}{L_{e}} + \frac{L_{e}^{3} k}{105}\end{matrix}\right]
$$

$$
\mathbf{f}_e^T = \left[\begin{matrix}\frac{L_{e} q}{2} & \frac{L_{e}^{2} q}{12} & \frac{L_{e} q}{2} & - \frac{L_{e}^{2} q}{12}\end{matrix}\right]
$$

with $f = q$ over the element.

##### 5. What each term reduces to

**$a_4$ term.** This is the classic Euler-Bernoulli bending stiffness matrix $\frac{EI}{L_e^3}[[12, 6L_e, -12, 6L_e], \ldots]$.

For your $a_4 = E I$ it integrates to

$$
\mathbf{k}_e^{(4)} = \frac{E I}{L_e^3} \left[\begin{matrix}12 & 6 L_{e} & -12 & 6 L_{e}\\6 L_{e} & 4 L_{e}^{2} & - 6 L_{e} & 2 L_{e}^{2}\\-12 & - 6 L_{e} & 12 & - 6 L_{e}\\6 L_{e} & 2 L_{e}^{2} & - 6 L_{e} & 4 L_{e}^{2}\end{matrix}\right]
$$

**$a_0$ term.** This is the consistent Winkler-foundation matrix $\frac{k L_e}{420}[[156, 22L_e, 54, -13L_e], \ldots]$ - the same 156/22/54/13 pattern as the consistent mass matrix, with the soil modulus $k$ in place of $\rho A$.

For your $a_0 = k$ it integrates to

$$
\mathbf{k}_e^{(0)} = \frac{k L_e}{420} \left[\begin{matrix}156 & 22 L_{e} & 54 & - 13 L_{e}\\22 L_{e} & 4 L_{e}^{2} & 13 L_{e} & - 3 L_{e}^{2}\\54 & 13 L_{e} & 156 & - 22 L_{e}\\- 13 L_{e} & - 3 L_{e}^{2} & - 22 L_{e} & 4 L_{e}^{2}\end{matrix}\right]
$$

## Model (as built by the orchestrator)

```json
{
  "material": {"E": 30000000000, "G": 12500000000},
  "supports": {
    "N0": [true, true, true, true, false, false],
    "N20": [false, true, true, false, false, false]
  },
  "nodes": [
    {"id": "N0", "x": 0},
    {"id": "N1", "x": 1.25},
    {"id": "N2", "x": 2.5},
    {"id": "N3", "x": 3.75},
    {"id": "N4", "x": 5},
    {"id": "N5", "x": 6.25},
    {"id": "N6", "x": 7.5},
    {"x": 8.75, "id": "N7"},
    {"x": 10, "id": "N8"},
    {"x": 11.25, "id": "N9"},
    {"x": 12.5, "id": "N10"},
    {"id": "N11", "x": 13.75},
    {"x": 15, "id": "N12"},
    {"x": 16.25, "id": "N13"},
    {"x": 17.5, "id": "N14"},
    {"x": 18.75, "id": "N15"},
    {"x": 20, "id": "N16"},
    {"id": "N17", "x": 21.25},
    {"id": "N18", "x": 22.5},
    {"id": "N19", "x": 23.75},
    {"id": "N20", "x": 25}
  ],
  "section": {"Iy": 0.005, "Iz": 0.005, "J": 0.001, "A": 0.5},
  "point_loads": []
}
```

## Results

| Quantity | FEM (solver) | Closed form | PyNite |
|---|---|---|---|
| Max deflection (m) | 3.197402e-03 | n/a | n/a |
| Max bending moment (N*m) | 3.749517e+04 | n/a | n/a |
| Max shear (N) | 4.176814e+04 | n/a | n/a |

## Deterministic gate

6 of 8 checks passed, 2 skipped (a skipped check verified nothing).

| Check | Result | Detail |
|---|---|---|
| equation_equilibrium | PASS | reactions 8.353628e+04 + point loads 0.000000e+00 + distributed -7.500000e+05 - carried by the equation -6.664637e+05 N; normalized residual 2.173e-15 (tolerance 1e-09) |
| equation_solution_residual | PASS | scaled \|K u - F\| at the free DOFs is 2.484e-14; the reported reactions differ from K u - F by at most 0.000e+00 (F at node N20); tolerance 1e-06 |
| equation_samples_consistent | PASS | worst scaled disagreement between what was reported and what the reported displacements imply is 0.000e+00 (max_abs['shear']) |
| equation_stiffness_symmetry | PASS | max abs(K - K^T) = 0.000e+00, scaled by max abs(K) = 1.852e+09, rel 0.000e+00 |
| equation_support_conditions | PASS | 2 restrained DOF(s); worst scaled residual movement 0.000e+00 (v at node N20) |
| equation_deflection_sign | SKIPPED | the sign of the deflection was not checked: the equation is not pure bending (v0 non-zero), and a foundation, an axial force or a first-derivative term can lift part of a beam away from its own load |
| equation_mms | PASS | relative nodal errors [8.418e-07, 5.264e-08, 3.291e-09]; simply supported (v* = sin(pi s)); meshes (20, 40, 80) elements; observed order 4.00, finest relative error 3.291e-09. PASS rule: finest relative nodal error <= 1e-10, or observed order >= 2.5 (Hermite cubics are 4th order in nodal deflection) with finest relative error <= 0.001. |
| closed_form_reference | SKIPPED | this equation and model are not the simply supported uniform-load Euler-Bernoulli case, so no closed form exists to compare; verification is by manufactured solutions instead |

## Verifier verdict (verbatim)

```json
{
  "refuted": false,
  "checks": [
    {
      "name": "Midspan deflection check",
      "passed": true,
      "detail": "FEM value -0.00301473 m vs independent check -0.00301475 m (diff < 0.001%)."
    },
    {
      "name": "Maximum bending moment check",
      "passed": true,
      "detail": "FEM value 37495.2 N*m vs independent check 37491.5 N*m (diff ~0.01%)."
    },
    {
      "name": "Support shear/reactions check",
      "passed": true,
      "detail": "FEM reaction 41768.1 N vs independent check 41761.3 N (diff ~0.02%)."
    },
    {
      "name": "Maximum deflection location check",
      "passed": true,
      "detail": "Both methods confirm max deflection (~0.003197 m) occurs near x=6.5 m, not at midspan, consistent with Winkler foundation theory."
    }
  ],
  "reasoning": "All FEM results are consistent with the independent numerical solution of the governing ODE. Discrepancies are minimal and attributable to the different numerical methods (Galerkin FEM vs. strong-form collocation).",
  "model": "gemini-3-flash-preview"
}
```

## GATE: PASS

**GATE: PASS**

> Preliminary engineering for research and teaching; not a sealed design; not for construction.

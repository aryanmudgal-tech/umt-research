# Agentic-AI — The 25-Meter Agent

Agentic AI for Prof. Basaran's bridge-beam pipeline: Galerkin FEM → ACI 318
concrete design → cost analysis → DXF drawings, with an adversarial verifier
and hard verification gates. Full plan and evidence: [PROPOSAL.md](PROPOSAL.md).

Stack: Google ADK + Gemini (free tier, AI Studio key) + deterministic Python
tools (NumPy, PyNite, ezdxf, concreteproperties).

## Setup

```bash
/usr/local/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then put your GEMINI_API_KEY in .env (gitignored)
```

## Phase 0 checks

```bash
.venv/bin/python spike/beam_tool.py        # tool layer, no AI needed
.venv/bin/python scripts/probe_models.py   # which models this key can use
.venv/bin/python spike/run_spike.py        # ADK round-trip, both variants
```

## Watching it run

```bash
.venv/bin/python agent/run_phase1.py                    # default brief, live view
.venv/bin/python agent/run_phase1.py evals/brief_25m.md # any other brief
.venv/bin/python agent/run_phase1.py --plain            # summary only, no live view
.venv/bin/python agent/run_phase1.py --no-html          # skip the HTML trace
.venv/bin/python agent/run_phase1.py --trace-html /tmp/run.html
```

Exit code is 0 only on GATE PASS (deterministic gate passed **and** verifier did
not refute), 1 on a failed gate, 2 if the orchestrator never called the solver.

The live view shows the run as it happens, in five stages:

| Stage | What you see |
|---|---|
| brief | the plain-English brief, verbatim, as the agent received it |
| orchestrator | which Gemini model answered (and any 503 fallback), the prompt it got, the tools it was offered, and the **full model dict** it built for `solve_beam_3d` |
| gate | each deterministic check as it is decided — invariants, closed form, PyNite for a standard beam; equilibrium, symmetry, supports and manufactured solutions for a custom equation — then the tally, counting PASS, FAIL and SKIPPED apart. No LLM in this stage |
| verifier | the independent agent on a different model, its tool calls, and its verdict check by check |
| report | GATE PASS/FAIL, duration, and where the artifacts landed |

Each run writes three things to `results/` (gitignored):

- `phase1_report.md` — the report
- `phase1_trace.html` — standalone, offline trace of the whole run; open it by
  double-click, expand any step, read the model dict and the verdict
- `phase1_trace.json` — the raw event list behind that page

## Changing the equation

The governing equation is an input, not code. Nothing in `tools/fem/equation.py`
knows which beam equation you mean: it reads the equation you give it,
integrates the element matrices for that equation with sympy, and solves. Three
ways to change it, easiest first.

**1. Say it in the brief, in plain English.** The orchestrator picks
`solve_with_equation` instead of `solve_beam_3d` whenever the brief states or
implies a different equation, and writes the spec itself. Phrases it acts on:
the beam *rests on soil / ground / an elastic foundation*, or a *modulus of
subgrade reaction k* is given; an *axial force / thrust / prestress P* acts
along the beam; the *depth or stiffness varies along the span* (tapered,
haunched, EI as a function of x); or the differential equation written out.

```
Simply supported, span 25 m, q = 30 kN/m downward, resting on ground with a
modulus of subgrade reaction k = 1.0e7 N/m per metre of span.
```

**2. Name a file from `equations/`.** Four ready-made equations live there —
`euler_bernoulli.json`, `elastic_foundation.json`, `beam_column.json`,
`tapered_beam.json` — each with a comment saying what it models. Copy one,
change the numbers in `params`, and refer to it in the brief.

**3. Write your own JSON.** The full contract, with worked examples, is in
[equations/README.md](equations/README.md). The short version:

```json
{
  "label": "Beam on elastic foundation",
  "coeffs": {"v4": "E*I", "v2": "0", "v1": "0", "v0": "k"},
  "rhs": "q",
  "params": {"E": 30e9, "I": 0.005, "k": 1.0e7, "q": -30e3}
}
```

which means `a4*v'''' + a2*v'' + a1*v' + a0*v = f(x)` with `a4 = coeffs.v4` and
so on — SI units, `v(x)` the transverse deflection, **downward negative**, so a
30 kN/m downward load is `q = -30e3` and `v` comes out negative. A coefficient
may depend on `x` (`"E*I0*(1 + x/L)"` tapers the beam). Every symbol used must
have a number in `params`.

Whatever route you take, the report's **Governing equation** section prints the
equation that produced its numbers, its parameters, and the derivation sympy
performed for it — so the equation in the document is always the equation that
was solved.

### How a new equation is still verified

There is no textbook answer for an equation nobody has solved, so the gate
verifies it another way: it manufactures an exact solution for *your* equation,
derives the load that produces it, and checks that the FEM converges to it at
the fourth-order rate Hermite cubics promise (`tools/fem/mms.py`). It also
checks vertical equilibrium in the form your equation implies — with a
foundation, the reactions carry only part of the load and the soil carries the
rest — plus stiffness symmetry and the support conditions. It also recomputes
`K u - F` from the displacements that were reported, which is what catches a
result whose numbers were edited after the solve: for a beam in pure bending
the equilibrium balance cannot see a corrupted deflection at all, because rows
0 and 2 of every element matrix sum to zero. The classic
closed-form comparison is added only when your equation really does reduce to
the simply supported uniform-load Euler-Bernoulli case; otherwise that check
reports **SKIPPED**, and a skipped check is never counted as a pass.

### Out of scope

- A `v3` (third-derivative) term: rejected with an error. The Hermite C1
  element has no symmetric Galerkin form for it.
- A `v4` coefficient of zero: rejected. Without a fourth-order term this
  element has no bending stiffness.
- Non-linear equations, coefficients that depend on `v` itself, time
  dependence, and 2D/3D behaviour (torsion, axial extension, out-of-plane
  bending). `solve_with_equation` solves one linear ODE in the transverse
  deflection of a single beam axis; anything else still needs `solve_beam_3d`.
- Mesh advice: moment and shear are sampled from each element's own cubic, so
  they carry O(h²) and O(h) error. Use at least 8 elements on the equation
  path, more if the shear diagram matters.

## Layout

- `agent/` — orchestrator, verifier, gates, trace/live view (Phase 1+)
- `tools/` — deterministic engineering tools: `fem/`, `rc_design/`, `cost/`, `drawings/`
- `spike/` — Phase-0 harness spike
- `scripts/` — utilities (model probe)
- `evals/` — benchmark problems with known answers
- `sandbox/` — scratch area for agent-written code

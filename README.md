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
| gate | each deterministic check as it is decided — invariants, closed form, PyNite — then the tally. No LLM in this stage |
| verifier | the independent agent on a different model, its tool calls, and its verdict check by check |
| report | GATE PASS/FAIL, duration, and where the artifacts landed |

Each run writes three things to `results/` (gitignored):

- `phase1_report.md` — the report
- `phase1_trace.html` — standalone, offline trace of the whole run; open it by
  double-click, expand any step, read the model dict and the verdict
- `phase1_trace.json` — the raw event list behind that page

## Layout

- `agent/` — orchestrator, verifier, gates, trace/live view (Phase 1+)
- `tools/` — deterministic engineering tools: `fem/`, `rc_design/`, `cost/`, `drawings/`
- `spike/` — Phase-0 harness spike
- `scripts/` — utilities (model probe)
- `evals/` — benchmark problems with known answers
- `sandbox/` — scratch area for agent-written code

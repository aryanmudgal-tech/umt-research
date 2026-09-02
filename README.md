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

## Layout

- `agent/` — orchestrator, verifier, gates (Phase 1+)
- `tools/` — deterministic engineering tools: `fem/`, `rc_design/`, `cost/`, `drawings/`
- `spike/` — Phase-0 harness spike
- `scripts/` — utilities (model probe)
- `evals/` — benchmark problems with known answers
- `sandbox/` — scratch area for agent-written code

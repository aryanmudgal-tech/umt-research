# The 25-Meter Agent — working agreements

Agentic AI for Prof. Cemal Basaran's bridge-beam pipeline: Galerkin FEM → ACI 318
concrete design → cost analysis → DXF drawings, with an adversarial verifier and
hard verification gates. Plan and evidence: [PROPOSAL.md](PROPOSAL.md).

## Git: commit very frequently, always as Aryan

**Repository:** https://github.com/aryanmudgal-tech/umt-research (remote `origin`,
branch `main`). This directory is the repository root.

- **Commit very frequently.** Commit at every natural checkpoint — a file finished,
  a stage working, tests going green, a doc updated — without waiting to be asked.
  Push after committing. Small, frequent commits, not one big one at the end.
- **The author is always Aryan Mudgal `<aryanmudgal4493@gmail.com>`.** Never Claude.
- **No Claude attribution of any kind in commits:** no `Co-Authored-By: Claude`
  trailers, no `Claude-Session:` trailers, no "Generated with Claude Code" footers.
  Commits must look exactly like Aryan made them from his own terminal.
- Never change `user.name` / `user.email`, locally or globally.
- Don't commit `.env` (it holds the Gemini API key) or anything under `results/`
  or `sandbox/` — `.gitignore` already covers these.

## Running things

```bash
.venv/bin/python -m pytest evals/ -q     # full offline suite, no API calls
.venv/bin/python agent/run_phase1.py     # live run: brief -> verified report
```

The venv is Python 3.12 (`/usr/local/bin/python3.12`); a 3.14 venv fails to build.
Gemini runs on the free tier, which is Flash-only and returns 503 under load, so
the model roster falls back. Tests must never call an LLM or the network.

## How the system is built

- The LLM never does arithmetic. It reads the brief, builds the model, and explains.
  Deterministic Python computes; the harness reads numbers from recorded tool calls,
  never from LLM prose.
- Every result passes a plain-Python gate plus an adversarial verifier that runs on a
  different model in its own session. A skipped check is reported as SKIPPED, never
  counted as a pass.
- Reports are preliminary engineering for research and teaching — not a sealed design.

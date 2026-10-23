# The 25-Meter Agent — working agreements

Agentic AI for Prof. Cemal Basaran's bridge-beam pipeline: Galerkin FEM → ACI 318
concrete design → cost analysis → DXF drawings, with an adversarial verifier and
hard verification gates. Plan and evidence: [PROPOSAL.md](PROPOSAL.md).

## Git: commit very frequently, always as Aryan

**Repository:** https://github.com/aryanmudgal-tech/umt-research (remote `origin`,
branch `main`). This directory is the repository root.

- **Commit very frequently.** Commit at every natural checkpoint — a file finished,
  a stage working, tests going green, a doc updated — without waiting to be asked.
  Push immediately after each commit; work that sits uncommitted on the machine
  does not exist as far as this repo is concerned.
- **One logical change per commit.** As a rule of thumb a commit touches a handful
  of files, not a dozen, and hundreds of lines, not thousands. A commit of 20 files
  and 3,000 lines means several commits were missed. Concretely:
  - a new module and its tests go in together, on their own, the moment they pass;
  - a bug fix and its regression test are one commit, separate from any other fix;
  - documentation and comment changes are their own commit;
  - a refactor is never mixed with a behaviour change.
- **Never batch a whole feature or a whole background task into one commit.** When
  work is delegated to subagents, tell them in their prompt to commit and push each
  piece as it goes green rather than once at the end, and commit each phase's output
  as it lands instead of waiting for the run to finish. A long run should show up as
  a steady series of commits, not silence followed by one enormous one.
- **The author is always Aryan Mudgal `<aryanmudgal4493@gmail.com>`.** Never Claude.
- **No Claude attribution of any kind in commits:** no `Co-Authored-By: Claude`
  trailers, no `Claude-Session:` trailers, no "Generated with Claude Code" footers.
  Commits must look exactly like Aryan made them from his own terminal.
- Never change `user.name` / `user.email`, locally or globally.
- Don't commit `.env` (it holds the Gemini API key) or anything under `results/`
  or `sandbox/` — `.gitignore` already covers these.

## Running things

```bash
.venv/bin/python -m pytest evals/ -q                        # full offline suite, no API calls
.venv/bin/python agent/run_phase1.py                        # live run: brief -> verified report
.venv/bin/python -m web.demo --port 8765                    # the website with a replayed run
.venv/bin/uvicorn web.app:create_app --factory --port 8080  # the website, live
```

The venv is Python 3.12 (`/usr/local/bin/python3.12`); a 3.14 venv fails to build.
The browser and PDF tests need Playwright's Chromium (`python -m playwright
install chromium`) and skip without it. Since 2026-10-02 the Gemini key is on
the paid tier; Flash models still return 503 under load, so calls retry with
backoff, each model turn has a time budget, and the roster falls back. Tests
must never call an LLM or the network.

Since 2026-10-23 the professor uses the agent on Google's Agent Platform
(formerly Vertex AI): he chats with `beam_agent/` on its Playground page in the
Cloud console. It reaches Gemini with the project's own account, no API key,
and calls the models on the `global` endpoint (none answer in us-central1).
Deploy changes to the same agent, from the repository root:

```bash
.venv-deploy/bin/python deploy/agent_platform.py --project gen-lang-client-0841593868 --update 3441449679584231424
```

`.venv-deploy` holds the deploy library (setup in the script's docstring). The
cost settings live in `beam_agent/.agent_engine_config.json` (no instance kept
running, at most one, 1 CPU) and the script refuses anything costlier. The
project's billing is real money: ask Aryan before any deploy or cloud change.

The older website (`web/`, no login by Aryan's decision; `deploy/cloudrun.sh`)
is no longer the professor's interface and is due to be removed.

## How the system is built

- The LLM never does arithmetic. It reads the brief, builds the model, and explains.
  Deterministic Python computes; the harness reads numbers from recorded tool calls,
  never from LLM prose.
- Every result passes a plain-Python gate plus an adversarial verifier that runs on a
  different model in its own session. A skipped check is reported as SKIPPED, never
  counted as a pass.
- Reports are preliminary engineering for research and teaching — not a sealed design.

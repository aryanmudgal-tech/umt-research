# The 25-Meter Agent — build proposal (rev. 2, Gemini stack)

Agentic AI for Prof. Cemal Basaran's four-stage bridge-beam pipeline.
Full research-backed version (with sources): https://claude.ai/code/artifact/972a9e56-4cb6-42bd-9015-79e104533776

**Rev. 2 changes (2026-09-02):** (1) UMT book server dropped for now — may be added later, design preserved
in the appendix; (2) build moves to the **Google Gemini ecosystem** — harness layer re-researched and rewritten
(104-agent sweep, 24 claims verified 3-0, 1 refuted). The deterministic tool layer is model-agnostic and unchanged.

## Reading the professor's note

The handwritten equation is **EI ∂⁴v/∂x⁴ = q_y** (q sub y = distributed load), i.e. the standard
Euler–Bernoulli beam equation — a boundary value problem, not `d⁴y/dx⁴ = 7y`.
Simply supported, L = 25 m. Analytical checks: midspan deflection 5qL⁴/384EI, M_mid = qL²/8, V_end = qL/2.
"3-D beam" ⇒ 12-DOF two-node element: axial AE/L, torsion GJ/L, biaxial bending (Hermite cubics).

## The recommendation

**One orchestrator agent on Google ADK + local deterministic Python tools + one context-isolated
adversarial verifier + a hard verification gate written as plain Python.** Not a multi-agent crew.

Architecture evidence (stack-agnostic, verified in sweep 1):
- Fixed-sequence pipelines = coded workflow, not autonomous crew; multi-agent costs 3–10× tokens.
- arXiv:2408.13406 (1,120 FEA trials): only Coder–Executor–Critic triad reliable; redundant reviewers *degraded*
  results; naive critics show 85–92% affirmation bias; executable-but-wrong code passes undetected.
- MASSE (arXiv:2510.11004) + Automation-in-Construction RC paper (97% acc vs SAP2000): deterministic tools +
  structured handoffs + auditable step-by-step calcs are what works.
- LLM never does arithmetic. It orchestrates, derives, explains.

## The Gemini harness (sweep 2, verified 2026-09-02)

**Framework: ADK (google/adk-python).** Apache-2.0, official Google, code-first Python, ~21.4k stars,
roughly bi-weekly releases. Current: v2.8.0 (2.x line) with maintained 1.x branch (v1.39.1). Model-agnostic
via LiteLLM (exit path from Gemini preserved). Decision needed via hands-on spike: **2.x graph Workflow API**
(current; templated SequentialAgent/ParallelAgent/LoopAgent deprecated in 2.0) vs **1.x SequentialAgent**
(battle-tested; verified verbatim: "not controlled by an AI model, and is deterministic in how it executes
its sub-agents").

**Three verified gotchas that shape the build:**
1. **Tools run locally, never in Gemini's hosted code sandbox.** ADK Python enforces that built-in code
   execution excludes ALL other tools in the same agent (400 INVALID_ARGUMENT) and built-in tools can't be
   used in sub-agents at all. So: all four stage tools are local `FunctionTool`s in our own pinned venv —
   which is where PyNiteFEA/ezdxf/concreteproperties live anyway. (Gemini 3's raw-API "tool combination"
   preview may lift this someday; build on what ADK enforces today.)
2. **A verifier inside the same SequentialAgent is NOT isolated** — all sub-agents share the same
   InvocationContext and session state (verified verbatim from ADK docs). Fresh-context isolation requires an
   **AgentTool-wrapped agent or a separate Runner with its own session**. That's how we implement the skeptic.
3. **The hard gate is plain Python** in custom orchestration code (BaseAgent subclass / code between stages) —
   ADK supports this pattern natively; the gate is an if-statement, not a prompt.

**Inter-stage data:** small results via session state `output_key` / `tool_context.state`; large artifacts
(FEM arrays, DXF) via files — output_key carries final response text, not binaries.

**Models & pricing (verified live 2026-09-02, all "Preview"-suffixed names still in flux):**

| Role | Model | Price in/out per 1M tok (≤200k ctx) |
|------|-------|--------------------------------------|
| Orchestrator | Gemini 3.1 Pro Preview | $2.00 / $12.00 |
| Verifier (different model for independence) | Gemini 2.5 Pro | $1.25 / $10.00 |
| Cheap tasks (rendering, formatting) | Gemini 3 Flash / 3.1 Flash-Lite | $0.50/$3.00 · $0.25/$1.50 |

A full pipeline run is a few hundred k tokens ⇒ **dollars, not tens of dollars, per run** on paid Tier 1.

**Free tier warning (verified):** AI Studio free tier needs no billing account BUT (a) free-tier content is
used by Google to improve products, with possible human review — wrong for unpublished research; (b) Pro
models reportedly no longer on the free tier post-Dec-2025 (secondary source; medium confidence); (c) live
RPM/RPD limits only visible per-project in the AI Studio dashboard. **Use paid Tier 1** (link billing account);
cost at our volumes is trivial and paid-tier content is not used for training.

**Access route (RESOLVED 2026-09-02, verified 3-0):** Aryan's Google AI Ultra subscription does NOT cover
Gemini API usage — Google states plan benefits "apply only within the Google AI Studio web interface"; API-key
use "is billed and managed separately." BUT since Jan 27 2026, Ultra includes Google Developer Program premium
benefits at no extra cost = **$100/month in Google Cloud credits usable toward the Gemini API**. Setup:
(1) AI Studio API key + link billing, prepay min $10 (Tier 1); (2) activate GDP premium benefits at
developers.google.com (requires GCP project + Cloud Billing enabled; prepay balance > $0 to activate
promotional credits). Net effect: the $100/mo credit should cover the pipeline's entire token spend, on
paid-tier privacy terms. Sources: ai.google.dev/gemini-api/docs/google-ai-plans, /docs/billing,
blog.google GDP-premium announcement.

**Fallback harnesses (documented, not chosen):** Gemini CLI headless mode (`--output-format stream-json`,
JSONL tool events; known issue #9281: exits on non-fatal tool errors) — subprocess-per-stage pattern; NOTE:
the consumer-subscription OAuth login for Gemini CLI/Code Assist was shut off June 18 2026 (migrated to
Antigravity), so the CLI runs on API keys too — Ultra quota can't drive it. Raw google-genai SDK automatic
function calling (local execution by default, 10-call cap via `AutomaticFunctionCallingConfig`) — hand-rolled
loop, no state/multi-agent machinery.

## Stages (unchanged — model-agnostic)

| # | Stage | Build | Cross-checks |
|---|-------|-------|--------------|
| 1 | Galerkin FEM | Agent-written NumPy Hermite-cubic 12-DOF 3D beam element (show the weak-form derivation) | PyNite (verified: true 12-DOF Member3D, CI-tested vs textbooks); OpenSeesPy backup; closed form 5qL⁴/384EI to machine precision |
| 2 | RC design | Agent-written ACI 318-19 checks (Whitney block, C=T, tension-controlled φ, ρ limits, Vn=Vc+Vs, stirrups, serviceability) as handcalcs-style audit sheets. **Verified gap: no open-source ACI 318 library** (concreteproperties = AS/NZS; structuralcodes = Eurocode) | concreteproperties moment-curvature on final section |
| 3 | Cost | Takeoff × WSDOT unit bid prices (rebar $1.50–2.00/lb, concrete $900–1,500/CY, +10% mobilization, +20% contingency) | Parametric: FHWA 2025 avg $393/ft² deck; TxDOT PS I-girder 51–100 ft bin $102.3/ft². One 25 m span ≈ $1–3M ⇒ agent must surface the $50M gap, not hide it. RSMeans = paid, no API — skip |
| 4 | Drawings | ezdxf → AutoCAD-compatible DXF (R12–R2018). Trial etacad (v0.0.14) for beam sections + rebar detailing; raw ezdxf fallback. GA/elevation = custom ezdxf | Re-parse DXF, geometry must match design values; agent renders PNG preview and inspects it |

## Verification model (unchanged)

Three layers; green light comes from the bottom:
1. **Deterministic checks** (code, no AI): closed-form matches, independent-library recompute, physics
   invariants (equilibrium, K symmetric/PSD, rigid-body zero energy, convergence, units), ACI inequalities,
   cost-band reconciliation, DXF read-back.
2. **Skeptic subagent** (AgentTool / separate Runner, fresh session): receives problem + answers only, never
   the solver's reasoning; prompted to refute; checks the specification (right problem modeled?). Run it on a
   *different* model than the orchestrator for extra independence.
3. **Hard gate** (plain Python between stages): machine-readable verdict required; orchestrator cannot argue
   past an if-statement. Checks are a library; the check *plan* is composed per problem (invariants always;
   reference solutions derived per case; manufactured solutions when no closed form exists).

## Professional posture (unchanged)

NSPE BER Case 24-2: AI use fine; failing *Responsible Charge* over output is not. Agent = supervised intern:
auditable calc sheets, every number carries its cross-check + verdict, reports labeled "preliminary /
research & teaching / not a sealed design." Data privacy: paid API tier only (free tier content feeds
Google's training + possible human review).

## Interface

- **Mode 1 (professor):** web chat page over the ADK backend — plain-language brief (or photo of a handwritten
  note), stage-progress display with per-stage verifier verdicts, deliverables as downloads (report, calc
  sheets, estimate, DXF).
- **Mode 2 (research):** `run_pipeline brief.md` → results folder (report.pdf, drawings/*.dxf, estimate,
  verification.json). Reproducible runs, batch evals, no babysitting.
- Phase 1 has no UI: driven from terminal by Aryan; demos run live.

## Build order

- **Phase 0 — DONE (2026-09-02).** Repo + venv (Python 3.12, ADK 2.8.0, PyNiteFEA 3.0). Empirical results:
  free tier is Flash-only (Pro → 429 quota; 2.5-gen retired with 404). Working models on this key:
  gemini-3.5-flash (reliable), 3.5/3.1-flash-lite, 3-flash-preview; 3.8/3.7-flash exist but 503 under load —
  use with retry+fallback. Beam tool matches closed form at rel. error 0.0 (midspan node trick: nodal values
  exact, interior interpolation isn't). ADK spike PASSED on gemini-3.5-flash in both patterns: single
  LlmAgent+FunctionTool and SequentialAgent with output_key state handoff. Model plan: 3.5-flash orchestrator
  (3.8-flash opportunistic), different-generation Flash for verifier, flash-lite for cheap tasks.
- **Phase 1** — FEM tool + isolated verifier + hard gate (credibility milestone; demo first)
- **Phase 2** — ACI 318 calc sheets + concreteproperties cross-check
- **Phase 3** — cost takeoff + parametric reconciliation
- **Phase 4** — etacad spot-check; DXF sheets; AutoCAD acceptance test
- **Phase 5** — end-to-end + eval suite (beam cases with known answers); targeted test of Gemini function-calling
  reliability over 20+ turn loops (no published data survived verification — measure it ourselves)

## Open questions

1. ~~Billing~~ **DECIDED (2026-09-02): free tier.** Aryan has an AI Studio API key; team accepts
   free-tier terms (content may be used for training) and skips paid billing for now. Consequence:
   Flash-class models only (Pro likely unavailable on free tier post-Dec-2025) and lower rate limits —
   run all roles on Gemini 3 Flash / 3.1 Flash-Lite; check live RPM/RPD in the AI Studio dashboard.
   Upgrade path if Pro is ever needed: Tier 1 ($10 min prepay) + GDP premium activation from the Ultra
   sub → $100/mo API credits.
2. **Spike (Phase 0):** ADK 2.x graph Workflow vs 1.x SequentialAgent.
3. **For the prof:** design code ACI 318-19 vs Eurocode; what the $50M covers.
4. **Later (book phase):** whether Gemini File Search/context caching gives citation-granular grounding
   comparable to Anthropic's citations API — no claims survived verification; needs its own research pass
   when the book returns to scope.

## Appendix: deferred UMT book design (v1, kept for later)

When the book returns to scope: PDF → page images → vision-model transcription to LaTeX (rule-based PDF
extractors verifiably garble equations) → semantic-unit chunks (one derivation/example each) → local index →
retrieval tool the orchestrator can query mid-task. One-time automated ingestion (~hours, ~$10–30); ask the
professor for the LaTeX manuscript to skip transcription entirely; confirm Springer electronic-reproduction
rights; corpus stays local.

## Research provenance

Two deep-research sweeps: 2026-08-31 (110 agents, 27 sources, 25 claims verified 3-0, 0 refuted — architecture,
FEM tools, RC gap, cost data, drawings, book RAG) and 2026-09-02 (104 agents, 22 sources, 25 claims panelled:
24 confirmed, 1 refuted — Gemini/ADK harness; the refuted claim was "the Gemini 3 lineup is complete and all
models are preview" — don't state lineup completeness). Quote-verified-but-not-panelled claims are labeled
in the artifact.

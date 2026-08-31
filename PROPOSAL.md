# The 25-Meter Agent — build proposal

Agentic AI for Prof. Cemal Basaran's four-stage bridge-beam pipeline.
Full research-backed version (with sources): https://claude.ai/code/artifact/972a9e56-4cb6-42bd-9015-79e104533776

## Reading the professor's note

The handwritten equation is **EI ∂⁴v/∂x⁴ = q_y** (q sub y = distributed load), i.e. the standard
Euler–Bernoulli beam equation — a boundary value problem, not `d⁴y/dx⁴ = 7y`.
Simply supported, L = 25 m. Analytical checks: midspan deflection 5qL⁴/384EI, M_mid = qL²/8, V_end = qL/2.
"3-D beam" ⇒ 12-DOF two-node element: axial AE/L, torsion GJ/L, biaxial bending (Hermite cubics).

## The recommendation

**One orchestrator agent (Claude Agent SDK) + deterministic Python tools + one adversarial
verifier subagent + one MCP server for the UMT book.** Not a multi-agent crew.

- Anthropic guidance (verified): fixed-sequence pipelines = prompt-chained workflow; multi-agent costs 3–10× tokens.
- arXiv:2408.13406 (1,120 FEA trials): only Coder–Executor–Critic triad was reliable; redundant reviewers
  *degraded* results; naive critics show 85–92% affirmation bias; executable-but-wrong code passes undetected.
- MASSE (arXiv:2510.11004) + Automation-in-Construction RC-design paper (97% acc vs SAP2000): what works is
  deterministic tools + structured handoffs + human-auditable step-by-step calcs.
- The SDK gives the triad in one process: agent = coder, sandbox = executor, refutation-prompted subagent = critic.
- LLM never does arithmetic. It orchestrates, derives, explains, cites.

## Stages

| # | Stage | Build | Cross-checks |
|---|-------|-------|--------------|
| 1 | Galerkin FEM | Agent-written NumPy Hermite-cubic 12-DOF 3D beam element (show the weak-form derivation — it's the point) | PyNite (verified: true 12-DOF Member3D, CI-tested vs textbooks, `pip install PyNiteFEA`); OpenSeesPy `elasticBeamColumn` backup; closed form 5qL⁴/384EI to machine precision |
| 2 | RC design | Agent-written ACI 318-19 checks (Whitney block, C=T, tension-controlled φ, ρ limits, Vn=Vc+Vs, stirrups, serviceability) rendered as handcalcs/efficalc-style audit sheets. **Verified gap: no open-source ACI 318 library exists** (concreteproperties = AS/NZS only; structuralcodes = Eurocode only) | concreteproperties moment-curvature / ultimate capacity on the final section |
| 3 | Cost | Quantity takeoff × WSDOT unit bid prices (rebar $1.50–2.00/lb, concrete $900–1,500/CY, +10% mobilization, +20% contingency) | Parametric: FHWA NBI 2025 avg $393/ft² deck (state range $128–1,306); TxDOT PS I-girder 51–100 ft bin $102.3/ft². **$50M reconciliation is a feature**: one 25 m span ≈ $1–3M ⇒ agent must surface the gap and ask what $50M covers. RSMeans = paid, no API — skip |
| 4 | Drawings | ezdxf → AutoCAD-compatible DXF (R12–R2018). Trial etacad (v0.0.14) for beam long./transverse sections + rebar detailing + bar schedule; raw ezdxf fallback. GA/elevation sheets = custom ezdxf | Acceptance test: professor opens the DXF in AutoCAD |

## The UMT book (RAG)

- **Never rule-based PDF extraction** — benchmarks (arXiv:2410.09871) show all rule-based parsers garble equations.
- Pipeline: PDF → page images → **vision-model transcription to LaTeX** (LemmaHead pattern, arXiv:2501.15797)
  → **semantic-unit chunks** (one derivation / worked example each, never fixed-size) → index → MCP retrieval tool.
- Serve via Claude **citations API** (page-level cites; scanned pages uncitable — transcribe first).
- Warning (verified): naive RAG *degraded* a model 9.4%→2.3% in the closest precedent. Retrieval quality decides everything.
  Fallback hardening: Confident RAG (N embedders, keep highest-confidence answer, ~+5–10%).
- ⚠️ Book PDF is NOT yet on disk. ⚠️ Confirm Springer electronic-reproduction rights with the professor; keep corpus local.

## Professional posture

NSPE BER Case 24-2: AI use is fine; failing *Responsible Charge* over its output is not. Operate the agent as a
supervised intern: auditable calc sheets, every number carries its cross-check + verifier verdict, reports labeled
"preliminary / research & teaching / not a sealed design." Book corpus and any client data stay local / no-training API.

## Build order

- **Phase 0** — repo, Agent SDK harness, sandbox; ingest book (blocked on PDF); MCP book tool + citation smoke test
- **Phase 1** — FEM tool + verifier subagent (credibility milestone; demo first)
- **Phase 2** — ACI 318 calc sheets + concreteproperties cross-check
- **Phase 3** — cost takeoff + parametric reconciliation
- **Phase 4** — etacad spot-check; DXF beam sheets; custom GA/elevation
- **Phase 5** — end-to-end run + eval set (beam cases with known answers, book Q&A with page-cite ground truth)

## Questions for the professor

1. Springer rights for local book ingestion + get the PDF.
2. How much Galerkin derivation to exhibit in output (assumed: full, pedagogical).
3. Design code: ACI 318-19 assumed — confirm (Eurocode would let structuralcodes carry weight).
4. What the $50M covers (single structure / crossing / program) for Stage 3 framing.

## Research provenance

Deep-research workflow, 2026-08-31: 110 agents, 6 angles, 27 sources fetched, 129 claims extracted,
25 adversarially verified (3-vote refutation panels), 0 refuted. Claims for RC-libraries / cost-data / LLM+FEM
angles are quote-verified from primary sources but were not tri-vote panelled (budget-dropped).

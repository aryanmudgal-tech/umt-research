# Professor web UI — design

Agreed with Aryan on 2026-10-02. Prof. Basaran must be able to run the agent
without a terminal: upload a brief, watch it run, read the report as a
formatted document.

## Decisions

- **Hosting:** Google Cloud Run (free tier) with runs stored in a Cloud Storage
  bucket in a US region. One instance at most.
- **Users:** the professor and Aryan. **No login of any kind.** Safeguards that
  are not logins: an unguessable service URL, one run at a time, a daily cap
  (default 30 runs).
- **Gemini:** the API key is on the paid tier.
- **Input:** a brief as a `.md` file holding the questions and the equation.
- **Output:** the report rendered as a document (headings, tables, typeset
  math), never raw markdown. Downloads: PDF and `.md`. A shared past-runs list.

## Screens

1. **Upload.** Drop zone or file picker, `.md` only. The chosen brief is shown
   rendered so he can confirm it, then **Run analysis**. A link to a sample
   brief. Past runs in a sidebar (title, date, verdict).
2. **Watching it run.** Six stage cards: Reading the brief, Building the model,
   Solving, Checking the numbers, Independent check, Writing the report. Done
   stages show a check, waiting ones are greyed, the active one shows a
   spinner. Above the cards, a stage-specific animation plays for the active
   stage only (brief being scanned with facts popping out; beam drawn and split
   into pieces; beam sagging under load, with soil springs only for a soil
   term, end thrust only for an axial term; checks ticking off; two solutions
   overlaid; report writing itself). Below it, an always-visible plain-English
   explanation of the current stage, filled with the run's real values. A
   timer and "keep this page open". Stages advance on real pipeline events,
   not on a timer. The technical trace is saved with the run but never shown
   on this page.
3. **Report.** Verdict banner (green "Verified — all checks passed", or red
   "Not verified" naming the failed check in plain words), PDF and `.md`
   download buttons, then the rendered report. The report table answers the
   brief: midspan deflection, each support's reaction, and where the peaks are.

## Architecture

```
browser ──upload──► FastAPI on Cloud Run ──► pipeline (existing agent) in a worker thread
   ▲                     │ tracer events ──► narrator ──► stage updates
   └──── streamed NDJSON over the same POST response ◄──────┘
end of run ──► store: brief.md, report.md, report.pdf, trace.json, meta.json
```

New code lives in `web/`:

| Unit | Job |
|---|---|
| `web/app.py` | Routes: page, `POST /api/runs` (upload + streamed progress), `GET /api/runs`, `GET /api/runs/{id}`, `GET /api/runs/{id}/report.md`, `GET /api/runs/{id}/report.pdf`, `GET /sample-brief.md`. Run lock and daily cap. |
| `web/narrator.py` | Pure function from tracer events to stage updates (stage, status, headline, explanation, facts). No I/O. |
| `web/storage.py` | `RunStore` with a local-folder implementation and a Cloud Storage implementation behind one interface. |
| `web/pdf.py` | Prints the rendered report page to PDF with headless Chromium (Playwright), so the PDF matches the screen. |
| `web/static/` | The single page: HTML, CSS, animations, markdown-it + KaTeX rendering. |

Changes to the existing agent:

1. The pipeline becomes a library function writing into a per-run folder; the
   CLI keeps working unchanged.
2. Gemini calls get per-call retries with backoff and a per-call timeout; a
   timeout counts as "busy" and falls through to the next model.
3. Orchestrator tool errors (an equation outside the template) return to the
   model instead of crashing the run, and the instruction says to stop and say
   why rather than approximate. The run then ends with that reason.
4. The report gains midspan deflection, per-support reactions and peak
   locations.

## Errors (each ends in a plain sentence, never a crash)

Wrong file (not `.md`, empty, over 200 KB) is refused before any AI call. A
brief with no solvable beam ends with the agent's reason. An equation outside
the template ends with which term is unsupported. Gemini unavailable after
retries and fallbacks: "The AI service is busy right now. Try again in a few
minutes." A failed check or a refuting verifier still shows the report under
a red banner. A second concurrent run is refused with a message. The daily cap
is refused with a message. A run whose stream drops is listed as interrupted.
An unexpected exception is saved as an error run with its trace.

## Limitation

Cloud Run allocates CPU only while a request is open, so a run lives inside
the professor's open page. Closing the tab may stop it; it then appears as
interrupted, never as a result.

## Testing

No test calls Gemini. Narrator tests replay recorded event streams. Server
tests use a fake pipeline that replays a scripted run. A Playwright test
drives the page against the fake: upload, stages in order, report rendered
with no raw `$$` or `#`, PDF download. All existing tests keep passing. After
deploy, one live run of the soil brief, checked against the exact answers
(3.01 mm midspan, 37.5 kN·m, 41.8 kN per support).

## Deployment

One container (Python 3.12, the engineering stack, Playwright Chromium),
Cloud Run in `us-central1`, max instances 1, 2 GiB, request timeout 15 min.
`GEMINI_API_KEY` from Secret Manager. Runs in a `us-central1` bucket. A $1
budget alert. Built remotely with `gcloud run deploy --source`, so no local
Docker is needed.

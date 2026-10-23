"""The beam agent as it runs on Agent Platform: a chat around the verified pipeline.

The professor types or pastes a brief into the agent's Playground page. One
tool runs the whole pipeline: a model builds the beam model, Python solves it
and runs the engineering checks, and a second model reviews the result. The
reply he reads is the pipeline's own report, handed over word for word. A
callback returns it in place of the next model call, so no model retypes a
number on the way to him.

The conversation model only decides when to run an analysis, edits a brief
for a what-if he asks for, and answers questions about a finished run.
"""

import asyncio
import tempfile
import threading
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import ToolContext
from google.genai import types

from agent.chat_report import chat_report
from agent.config import CONVERSATION_MODEL, connect_gemini, gemini_model
from agent.run_phase1 import run_pipeline

# On Agent Platform this finds the platform's settings; on a laptop, the key in
# .env. strict=False so that importing the agent never needs either.
connect_gemini(strict=False)

SUMMARY_KEY = "beam_summary"
REPORT_KEY = "beam_report"

# The pipeline's tracer and recorder are module-level, so two analyses must
# never run at once in one process. A second request waits for the first.
_one_analysis_at_a_time = threading.Lock()


def _run(brief: str):
    with _one_analysis_at_a_time, tempfile.TemporaryDirectory() as out:
        outcome = run_pipeline(brief, Path(out))
        report = outcome.report_path.read_text() if outcome.report_path else ""
    return outcome, report


async def analyze_brief(brief: str, tool_context: ToolContext) -> dict:
    """Run the full verified beam analysis on a brief.

    The pipeline builds the beam model from the brief, solves it with the
    finite element solver, runs the engineering checks and has a second model
    review the result. It takes one to four minutes.

    Args:
        brief: the brief exactly as the user wrote it, or, for a what-if, the
            earlier brief with only the requested change made.

    Returns:
        dict with "verified" (bool) and "summary" (the answer shown to the user).
    """
    try:
        # in a worker thread: the pipeline starts event loops of its own
        outcome, report = await asyncio.to_thread(_run, brief)
    except Exception as exc:  # the user gets a sentence, not a stack trace
        summary = (
            "**The analysis stopped before it finished.** "
            f"{type(exc).__name__}: {exc}\n\n"
            "This is usually Google's model service being busy. Send the brief again in a minute."
        )
        tool_context.state[SUMMARY_KEY] = summary
        return {"verified": False, "summary": summary}
    summary = chat_report(outcome, report)
    tool_context.state[SUMMARY_KEY] = summary
    tool_context.state[REPORT_KEY] = report or summary
    return {"verified": outcome.passed, "summary": summary}


def show_full_report(tool_context: ToolContext) -> dict:
    """Show the complete report of the most recent analysis in this conversation.

    It holds the beam model, the derivation of the element equations, the full
    results and the detail of every check.

    Returns:
        dict with "available" (bool).
    """
    return {"available": bool(tool_context.state.get(REPORT_KEY))}


_HANDED_OVER = {"analyze_brief": SUMMARY_KEY, "show_full_report": REPORT_KEY}


def hand_over_report(callback_context, llm_request):
    """Answer with the pipeline's own text, without a model call, after a report tool.

    Returns None, so that the model answers, for every other turn, and when
    there is no report yet to show.
    """
    last = llm_request.contents[-1] if llm_request.contents else None
    for part in (last.parts or []) if last else []:
        response = part.function_response
        key = _HANDED_OVER.get(response.name) if response else None
        text = callback_context.state.get(key) if key else None
        if text:
            return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))
    return None


INSTRUCTION = """You are the structural engineering assistant of Prof. Cemal Basaran's research group at the University at Buffalo. You run beam analyses through a verified pipeline: a Galerkin finite element solver, deterministic engineering checks, and an independent review by a second model.

- When the user gives a beam problem (span, supports, section and material, loads, and possibly the governing equation), call analyze_brief once with the brief EXACTLY as the user wrote it. Do not rewrite, shorten, correct or complete it: the pipeline reads it as given, and judging what is missing is part of its job. Tell the user in one short sentence that the analysis takes a few minutes.
- For a what-if on an earlier brief, copy that brief, change only what the user asked, say in one sentence which change you made, then call analyze_brief with the edited brief.
- When the user asks for the full report, the beam model, the derivation or the details of the checks, call show_full_report.
- Never state a number that did not come from a result in this conversation. Do not compute, estimate, convert or round anything yourself. If a question needs a number the results do not contain, say so and offer to run an analysis.
- For questions about theory (unified mechanics theory, the governing equation, the finite element method), answer briefly, and say when something is your general understanding rather than a result of the analysis.
- Write for a civil engineering professor: engineering terms are welcome; never mention code, functions, tools, JSON, files or model names."""


root_agent = LlmAgent(
    name="umt_beam_engineer",
    model=gemini_model(CONVERSATION_MODEL),
    description="Runs verified finite element beam analyses from a brief and explains the results.",
    instruction=INSTRUCTION,
    tools=[analyze_brief, show_full_report],
    before_model_callback=hand_over_report,
)

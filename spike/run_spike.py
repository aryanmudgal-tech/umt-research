"""Phase-0 ADK spike: prove the harness round-trip on the free tier.

Two variants, run back to back:
  A. Single LlmAgent with the beam FunctionTool (the orchestrator pattern).
  B. SequentialAgent: solver agent -> output_key state -> reporter agent
     (the templated pipeline pattern; tests inter-stage state passing).

Run:  .venv/bin/python spike/run_spike.py [model-name]
Model defaults to the first Flash model the probe found; pass one explicitly
to override, e.g.:  .venv/bin/python spike/run_spike.py gemini-3-flash-preview
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GEMINI_API_KEY")
if not key or key.startswith("paste-"):
    sys.exit("No key: copy .env.example to .env and set GEMINI_API_KEY first.")
# ADK reads the Gemini key from the environment; set both common names.
os.environ.setdefault("GOOGLE_API_KEY", key)
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")

from google.adk.agents import LlmAgent, SequentialAgent  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

from beam_tool import solve_simply_supported_beam  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemini-flash-latest"

PROBLEM = (
    "Solve the professor's beam: simply supported, span 25 m, E = 30 GPa, "
    "I = 0.005 m^4, uniform load q = 30 kN/m. Use the tool, then report "
    "midspan deflection, max moment, and end shear, and state the relative "
    "error of the FEM deflection against the closed form."
)


async def run_agent(runner: InMemoryRunner, prompt: str) -> str:
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="spike"
    )
    msg = types.Content(role="user", parts=[types.Part(text=prompt)])
    final = ""
    async for event in runner.run_async(
        user_id="spike", session_id=session.id, new_message=msg
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final = "".join(p.text or "" for p in event.content.parts)
    return final


async def variant_a() -> None:
    print(f"\n=== Variant A: single LlmAgent + FunctionTool ({MODEL}) ===")
    agent = LlmAgent(
        name="beam_solver",
        model=MODEL,
        instruction=(
            "You are a structural analysis assistant. Always compute numbers "
            "with the provided tool - never do arithmetic yourself."
        ),
        tools=[solve_simply_supported_beam],
    )
    out = await run_agent(InMemoryRunner(agent=agent, app_name="spike_a"), PROBLEM)
    print(out or "(no final response)")


async def variant_b() -> None:
    print(f"\n=== Variant B: SequentialAgent, state via output_key ({MODEL}) ===")
    solver = LlmAgent(
        name="solver",
        model=MODEL,
        instruction=(
            "Call the tool for the beam described by the user and output ONLY "
            "the tool's raw result as compact JSON. No prose."
        ),
        tools=[solve_simply_supported_beam],
        output_key="fem_results",
    )
    reporter = LlmAgent(
        name="reporter",
        model=MODEL,
        instruction=(
            "FEM results from the previous stage: {fem_results}\n"
            "Write a 3-sentence engineering summary: deflection, moment, "
            "shear, and whether FEM matches the closed form."
        ),
    )
    pipeline = SequentialAgent(name="pipeline", sub_agents=[solver, reporter])
    out = await run_agent(InMemoryRunner(agent=pipeline, app_name="spike_b"), PROBLEM)
    print(out or "(no final response)")


if __name__ == "__main__":
    asyncio.run(variant_a())
    asyncio.run(variant_b())
    print("\nSpike complete.")

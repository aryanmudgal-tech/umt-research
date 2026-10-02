"""Model rosters and environment loading for the Phase-1 harness."""

import asyncio
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google.adk.models.google_llm import Gemini
from google.genai import types

REPO_ROOT = Path(__file__).resolve().parents[1]

# Try in order, strongest first: the free tier throws 503 on whichever Flash
# model is busiest, and the lighter ones are usually still answering. Every
# result still has to pass the deterministic gate, so a weaker model that
# builds a wrong model dict fails the run rather than passing it.
ORCHESTRATOR_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

# Led by a DIFFERENT generation from the orchestrator models, so the verifier
# does not share the orchestrator's failure modes. verifier_roster() takes out
# whichever model the orchestrator actually used.
VERIFIER_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
]

_RETRYABLE_MARKERS = (
    "503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded", "timed out",
    "no answer within",
)

# A 503 or 429 is usually a blip: retry the same call with backoff (about 2 s,
# 4 s, 8 s) before the model is given up on for this run. Without this a single
# blip on the second of two calls threw away a model that was answering.
RETRY_OPTIONS = types.HttpRetryOptions(
    attempts=4,
    initial_delay=2.0,
    max_delay=16.0,
    http_status_codes=[429, 500, 502, 503, 504],
)

# Time budget for one model's whole turn, all its calls and retries included.
# A model that accepts a request and never answers raises nothing, so without
# a budget the fallback never starts: one run sat in the orchestrator stage for
# over three minutes. A healthy orchestrator turn takes about 15 s and a
# verifier turn, with its tool calls, under a minute.
ORCHESTRATOR_BUDGET_S = 150
VERIFIER_BUDGET_S = 240


def gemini_model(name: str) -> Gemini:
    """A Gemini model handle that retries transient overload on its own."""
    return Gemini(model=name, retry_options=RETRY_OPTIONS)


async def within_budget(coroutine, budget_s: float, name: str):
    """Await coroutine, or raise TimeoutError naming the model past budget_s."""
    try:
        return await asyncio.wait_for(coroutine, budget_s)
    except TimeoutError as exc:
        raise TimeoutError(f"{name}: no answer within {budget_s:g} s") from exc


def load_api_key() -> str:
    """Load the Gemini key from .env and export it the way ADK expects.

    Uses load_dotenv with an explicit path — find_dotenv breaks under some
    invocations (different cwd, pytest, launchd).
    """
    load_dotenv(REPO_ROOT / ".env")
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key or key.startswith("paste-"):
        raise RuntimeError(
            "No API key: copy .env.example to .env and set GEMINI_API_KEY."
        )
    os.environ["GOOGLE_API_KEY"] = key
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    return key


def generation(model: str) -> str:
    """The version in a Gemini model name: "gemini-3.5-flash-lite" -> "3.5"."""
    match = re.match(r"gemini-(\d+(?:\.\d+)?)", model)
    return match.group(1) if match else model


def verifier_roster(orchestrator_model: str) -> list:
    """VERIFIER_MODELS for a run whose orchestrator answered on orchestrator_model.

    That model is never used, whatever fallback put it there, and models of a
    different generation are tried before any of the same generation.
    """
    pool = [m for m in VERIFIER_MODELS if m != orchestrator_model]
    same = generation(orchestrator_model)
    return [m for m in pool if generation(m) != same] + [
        m for m in pool if generation(m) == same
    ]


def retryable_error(exc: BaseException) -> bool:
    """True for overload, quota and timeout errors worth trying the next model for."""
    seen = exc
    while seen is not None:
        if isinstance(seen, (TimeoutError, httpx.TimeoutException)):
            return True
        if any(marker in str(seen) for marker in _RETRYABLE_MARKERS):
            return True
        seen = seen.__cause__
    return False

"""Model rosters and environment loading for the Phase-1 harness."""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]

# Try in order: the free tier throws 503 on the newest model under load.
ORCHESTRATOR_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash"]

# Deliberately a DIFFERENT generation from the orchestrator models, so the
# verifier does not share the orchestrator's failure modes.
VERIFIER_MODELS = ["gemini-3-flash-preview", "gemini-3.5-flash-lite"]

_RETRYABLE_MARKERS = ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded")


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


def retryable_error(exc: BaseException) -> bool:
    """True for overload/quota errors worth retrying on the next model."""
    text = str(exc)
    return any(marker in text for marker in _RETRYABLE_MARKERS)

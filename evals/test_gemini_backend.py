"""Offline tests for choosing how Gemini is reached: Agent Platform or an API key.

On Agent Platform the container sets the flag, the project and the region, and
the agent's service account signs each call; anywhere else the key in .env is
used. No API calls: the real .env is never read here.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import config  # noqa: E402

ENV = (
    "GOOGLE_GENAI_USE_ENTERPRISE",
    "GOOGLE_GENAI_USE_VERTEXAI",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)


def platform(monkeypatch, flag="GOOGLE_GENAI_USE_ENTERPRISE", value="1"):
    monkeypatch.setenv(flag, value)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "gen-lang-client-0841593868")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")


@pytest.mark.parametrize(
    "flag, value",
    [
        ("GOOGLE_GENAI_USE_ENTERPRISE", "1"),
        ("GOOGLE_GENAI_USE_ENTERPRISE", "true"),
        ("GOOGLE_GENAI_USE_VERTEXAI", "True"),
    ],
)
def test_the_platform_container_needs_no_key(monkeypatch, flag, value):
    platform(monkeypatch, flag, value)
    assert config.connect_gemini() == "agent-platform"
    assert "GOOGLE_API_KEY" not in config.os.environ


def test_the_platform_settings_are_left_as_the_container_set_them(monkeypatch):
    platform(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "a-key-that-must-not-be-used")
    config.connect_gemini()
    # load_api_key would switch the client to the key by setting this to FALSE
    assert "GOOGLE_GENAI_USE_VERTEXAI" not in config.os.environ
    assert "GOOGLE_API_KEY" not in config.os.environ


def test_the_key_is_used_everywhere_else(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "abc123")
    assert config.connect_gemini() == "api-key"
    assert config.os.environ["GOOGLE_API_KEY"] == "abc123"
    assert config.os.environ["GOOGLE_GENAI_USE_VERTEXAI"] == "FALSE"


def test_a_flag_without_a_project_is_not_the_platform(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "abc123")
    assert not config.on_agent_platform()
    assert config.connect_gemini() == "api-key"


@pytest.mark.parametrize("value", ["0", "false", "", "no"])
def test_a_flag_set_off_is_off(monkeypatch, value):
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", value)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    assert not config.on_agent_platform()


def test_with_neither_the_error_names_both_ways(monkeypatch):
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY.*GOOGLE_GENAI_USE_ENTERPRISE"):
        config.connect_gemini()


def test_importing_the_deployed_agent_never_needs_a_key(monkeypatch):
    assert config.connect_gemini(strict=False) == "none"

"""Probe which Gemini models this API key can actually use.

Answers empirically what the free tier gives us (docs are vague, secondary
sources say Pro is off the free tier). Run:

    .venv/bin/python scripts/probe_models.py
"""

import os
import sys

from dotenv import load_dotenv
from google import genai

load_dotenv()
key = os.getenv("GEMINI_API_KEY")
if not key or key.startswith("paste-"):
    sys.exit("No key: copy .env.example to .env and set GEMINI_API_KEY first.")

client = genai.Client(api_key=key)

print("=== Models visible to this key (generateContent-capable) ===")
candidates = []
for m in client.models.list():
    actions = getattr(m, "supported_actions", None) or []
    if "generateContent" in actions:
        name = m.name.removeprefix("models/")
        candidates.append(name)
        print(f"  {name}")

print("\n=== Live generate test (1 short call per family) ===")
# Test one representative per family, newest first; skip exotic variants.
families = ["gemini-3.1-pro", "gemini-3-pro", "gemini-2.5-pro",
            "gemini-3.1-flash", "gemini-3-flash", "gemini-2.5-flash",
            "flash-lite"]
tested = set()
for fam in families:
    match = next((c for c in candidates if fam in c and c not in tested), None)
    if not match:
        print(f"  {fam:<18} -> not in model list")
        continue
    tested.add(match)
    try:
        r = client.models.generate_content(
            model=match, contents="Reply with the single word: ok")
        print(f"  {match:<40} -> OK ({r.text.strip()[:20]!r})")
    except Exception as e:
        msg = str(e).splitlines()[0][:100]
        print(f"  {match:<40} -> FAILED: {msg}")

print("\nDone. Models marked OK are usable on this key's tier.")

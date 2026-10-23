"""Deploy beam_agent/ to Agent Platform (formerly Vertex AI) in the professor's project.

From the repository root:

    .venv-deploy/bin/python deploy/agent_platform.py --project PROJECT_ID
    .venv-deploy/bin/python deploy/agent_platform.py --project PROJECT_ID --update AGENT_ID

The first form creates the agent; the second ships new code to the same agent,
so its Playground page and its chat history stay where they are.

.venv-deploy holds google-cloud-aiplatform, which the agent's own environment
does not need and whose dependencies would move that environment's pins:

    /usr/local/bin/python3.12 -m venv .venv-deploy
    .venv-deploy/bin/pip install "google-adk[a2a]==2.8.0" "google-cloud-aiplatform[agent_engines]==2.6.0"

It signs in as whichever account gcloud has active, with a short-lived token
rather than application default credentials, so the machine's own default login
is never touched. Everything that costs money is set in
beam_agent/.agent_engine_config.json, and is checked here before anything is sent.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AGENT_DIR = REPO / "beam_agent"
CONFIG = AGENT_DIR / ".agent_engine_config.json"
ADK_VERSION = "2.8.0"

# The deploy builds its container from ADK's template, which starts from Python
# 3.11; the pinned numpy and scipy need 3.12, the version the tests run on.
BASE_IMAGE = ("FROM python:3.11-slim", "FROM python:3.12-slim")


def gcloud(*args) -> str:
    return subprocess.run(["gcloud", *args], check=True, capture_output=True, text=True).stdout.strip()


def cheap_settings_problems(config: dict) -> list:
    """What in the agent's settings would cost more than intended, or [] if nothing."""
    problems = []
    if config.get("min_instances") != 0:
        problems.append(
            f"min_instances is {config.get('min_instances')!r}: anything but 0 keeps an "
            "instance running, and billing, around the clock"
        )
    if not isinstance(config.get("max_instances"), int) or config["max_instances"] > 2:
        problems.append(f"max_instances is {config.get('max_instances')!r}: at most 2 for one user")
    limits = config.get("resource_limits") or {}
    if limits.get("cpu") not in ("1", "2"):
        problems.append(f"resource_limits.cpu is {limits.get('cpu')!r}: 1 or 2 is plenty")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", required=True, help="Google Cloud project id")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--update", metavar="AGENT_ID", help="the numeric id of the agent to update")
    args = parser.parse_args(argv)

    config = json.loads(CONFIG.read_text())
    problems = cheap_settings_problems(config)
    if problems:
        print("Refusing to deploy:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1

    account = gcloud("auth", "list", "--filter=status:ACTIVE", "--format=value(account)")
    print(f"account: {account}\nproject: {args.project}, region: {args.region}")
    print(
        f"settings: {config['min_instances']} instances kept running, at most "
        f"{config['max_instances']}, {config['resource_limits']['cpu']} CPU and "
        f"{config['resource_limits']['memory']} each"
    )

    import google.auth
    from google.oauth2.credentials import Credentials

    credentials = Credentials(token=gcloud("auth", "print-access-token"), quota_project_id=args.project)
    google.auth.default = lambda *a, **k: (credentials, args.project)

    from google.adk.cli import cli_deploy

    old, new = BASE_IMAGE
    if old not in cli_deploy._DOCKERFILE_TEMPLATE:
        print("ADK's container template changed: check its Python version before deploying.", file=sys.stderr)
        return 1
    cli_deploy._DOCKERFILE_TEMPLATE = cli_deploy._DOCKERFILE_TEMPLATE.replace(old, new)

    cli_deploy.to_agent_engine(
        agent_folder=str(AGENT_DIR),
        project=args.project,
        region=args.region,
        agent_engine_id=args.update,
        adk_version=ADK_VERSION,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

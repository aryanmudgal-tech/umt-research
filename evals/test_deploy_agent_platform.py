"""Offline tests for the Agent Platform deploy script's cost guard.

The guard is the last thing between a typo in the agent's settings and an
instance billing around the clock, so it is tested on its own. No API calls,
no gcloud: only the pure check is exercised.
"""

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("agent_platform_deploy", REPO / "deploy" / "agent_platform.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)

GOOD = {"min_instances": 0, "max_instances": 1, "resource_limits": {"cpu": "1", "memory": "2Gi"}}


def test_the_committed_settings_pass_the_guard():
    assert deploy.cheap_settings_problems(json.loads(deploy.CONFIG.read_text())) == []


def test_an_instance_kept_running_is_refused():
    for value in (1, None):
        problems = deploy.cheap_settings_problems({**GOOD, "min_instances": value})
        assert problems and "around the clock" in problems[0]


def test_many_instances_or_large_machines_are_refused():
    assert deploy.cheap_settings_problems({**GOOD, "max_instances": 100})
    assert deploy.cheap_settings_problems({**GOOD, "resource_limits": {"cpu": "4", "memory": "4Gi"}})
    assert deploy.cheap_settings_problems({k: v for k, v in GOOD.items() if k != "max_instances"})


def test_the_container_is_moved_to_the_tested_python():
    old, new = deploy.BASE_IMAGE
    assert old.endswith("3.11-slim") and new.endswith("3.12-slim")

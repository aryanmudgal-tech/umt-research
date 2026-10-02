"""Offline tests for web/storage.py: where runs live, on disk or in a bucket.

Both stores sit behind one interface, so every test runs against the local
folder store and against the Cloud Storage store backed by an in-memory fake
client. No network.
"""

import sys
import time
from pathlib import Path

import pytest
from google.api_core.exceptions import NotFound

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from web import storage
from web.storage import GCSStore, LocalStore, new_run_id, valid_run_id


class _Blob:
    def __init__(self, objects, name):
        self.objects, self.name = objects, name

    def upload_from_string(self, data, content_type=None):
        self.objects[self.name] = data.encode() if isinstance(data, str) else bytes(data)

    def download_as_bytes(self):
        if self.name not in self.objects:
            raise NotFound(self.name)
        return self.objects[self.name]


class _Bucket:
    def __init__(self, objects):
        self.objects = objects

    def blob(self, name):
        return _Blob(self.objects, name)


class FakeClient:
    def __init__(self):
        self.objects = {}

    def bucket(self, name):
        return _Bucket(self.objects)

    def list_blobs(self, bucket_name, prefix=""):
        return [_Blob(self.objects, n) for n in sorted(self.objects) if n.startswith(prefix)]


@pytest.fixture(params=["local", "gcs"])
def store(request, tmp_path):
    if request.param == "local":
        return LocalStore(tmp_path / "runs")
    return GCSStore("bucket", client=FakeClient())


def test_run_ids_are_unique_sortable_and_safe():
    ids = [new_run_id() for _ in range(50)]
    assert len(set(ids)) == 50
    assert all(valid_run_id(i) for i in ids)
    for bad in ("../etc", "20261002-151530-zzzzzz", "", "a/b"):
        assert not valid_run_id(bad)


def test_files_round_trip(store):
    run = new_run_id()
    store.put(run, "report.md", "# Report")
    store.put(run, "report.pdf", b"%PDF-1.7")
    assert store.get(run, "report.md") == b"# Report"
    assert store.get(run, "report.pdf") == b"%PDF-1.7"
    assert store.get(run, "trace.json") is None


def test_only_known_file_names_are_accepted(store):
    with pytest.raises(ValueError):
        store.put(new_run_id(), "../../secret", "x")


def test_runs_are_listed_newest_first(store, monkeypatch):
    clock = [1_000_000.0]
    monkeypatch.setattr(storage.time, "time", lambda: clock[0])
    first = store.start_run("Beam one")
    clock[0] += 60
    second = store.start_run("Beam two")
    store.finish_run(first, "passed", duration_s=40.0)

    runs = store.list_runs()
    assert [r["id"] for r in runs] == [second, first]
    assert runs[1]["status"] == "passed" and runs[1]["title"] == "Beam one"
    assert runs[1]["duration_s"] == 40.0


def test_a_run_left_running_too_long_reads_as_interrupted(store, monkeypatch):
    clock = [1_000_000.0]
    monkeypatch.setattr(storage.time, "time", lambda: clock[0])
    run = store.start_run("Beam")
    assert store.read_meta(run)["status"] == "running"
    clock[0] += storage.STALE_AFTER_S + 1
    assert store.read_meta(run)["status"] == "interrupted"
    assert store.list_runs()[0]["status"] == "interrupted"


def test_runs_in_the_last_day_are_counted(store, monkeypatch):
    clock = [1_000_000.0]
    monkeypatch.setattr(storage.time, "time", lambda: clock[0])
    store.start_run("old")
    clock[0] += 25 * 3600
    store.start_run("new one")
    store.start_run("new two")
    assert store.runs_since(24 * 3600) == 2


# ------------------------------------------------------- rename and delete


class _DeletingBlob(_Blob):
    def delete(self):
        self.objects.pop(self.name, None)


def _with_delete(client):
    client.bucket = lambda name: type("B", (), {"blob": lambda self, n: _DeletingBlob(client.objects, n)})()
    client.list_blobs = lambda bucket_name, prefix="": [
        _DeletingBlob(client.objects, n) for n in sorted(client.objects) if n.startswith(prefix)
    ]
    return client


@pytest.fixture(params=["local", "gcs"])
def full_store(request, tmp_path):
    if request.param == "local":
        return LocalStore(tmp_path / "runs")
    return GCSStore("bucket", client=_with_delete(FakeClient()))


def test_a_run_can_be_renamed(full_store):
    run = full_store.start_run("25 m beam on soil")
    full_store.finish_run(run, "passed")
    full_store.rename_run(run, "  Soil beam, first try  ")
    assert full_store.read_meta(run)["title"] == "Soil beam, first try"
    assert full_store.read_meta(run)["status"] == "passed"


def test_a_deleted_run_is_gone_but_still_counts_toward_the_cap(full_store):
    run = full_store.start_run("Beam")
    full_store.put(run, "report.md", "# Report")
    full_store.put(run, "report.pdf", b"%PDF")
    full_store.finish_run(run, "passed")

    full_store.delete_run(run)

    assert full_store.read_meta(run) is None
    assert full_store.list_runs() == []
    assert full_store.get(run, "report.md") is None and full_store.get(run, "report.pdf") is None
    assert full_store.runs_since(24 * 3600) == 1  # deleting must not reset the daily cap


def test_a_deleted_runs_marker_is_cleared_after_a_day(full_store, monkeypatch):
    clock = [1_000_000.0]
    monkeypatch.setattr(storage.time, "time", lambda: clock[0])
    run = full_store.start_run("Beam")
    full_store.delete_run(run)
    clock[0] += 25 * 3600
    assert full_store.runs_since(24 * 3600) == 0
    full_store.list_runs()  # listing sweeps old markers away
    assert full_store._read_raw(run) is None

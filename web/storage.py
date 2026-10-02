"""Where runs live: a local folder in development, a Cloud Storage bucket in production.

Each run is a folder (or object prefix) named by its run id, holding a fixed
set of files: the brief, the report in markdown and PDF, the trace, and
meta.json with the title, status and timing the past-runs list shows. Both
stores share every behaviour but the raw put/get/list, so the page cannot
tell them apart.

Cloud Run keeps no disk between requests, which is why production uses the
bucket: a run written there survives the instance going to sleep.
"""

import json
import re
import secrets
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

FILES = {
    "brief.md": "text/markdown; charset=utf-8",
    "report.md": "text/markdown; charset=utf-8",
    "report.pdf": "application/pdf",
    "trace.json": "application/json",
    "trace.html": "text/html; charset=utf-8",
    "meta.json": "application/json",
}

# A run still marked running after this long lost its page: Cloud Run gives a
# request CPU only while it is open, so a closed tab can stop a run mid-way.
STALE_AFTER_S = 20 * 60

# A deleted run leaves only a small marker for this long, so deleting runs
# cannot reset the cap on runs per 24 hours. Then the marker goes too.
TOMBSTONE_S = 24 * 3600
MAX_TITLE = 120

_RUN_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")


def new_run_id() -> str:
    """A run id that sorts by time and cannot name anything but a run."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def valid_run_id(run_id: str) -> bool:
    return bool(_RUN_ID.match(run_id or ""))


def _check(run_id, name):
    if not valid_run_id(run_id):
        raise ValueError(f"not a run id: {run_id!r}")
    if name not in FILES:
        raise ValueError(f"not a run file: {name!r}")


class RunStore:
    """Shared behaviour; subclasses supply _put, _get and _meta_paths."""

    def put(self, run_id: str, name: str, data) -> None:
        _check(run_id, name)
        self._put(run_id, name, data.encode() if isinstance(data, str) else bytes(data))

    def get(self, run_id: str, name: str):
        """The file's bytes, or None if the run has no such file."""
        _check(run_id, name)
        return self._get(run_id, name)

    def start_run(self, title: str) -> str:
        run_id = new_run_id()
        now = time.time()
        self._write_meta(run_id, {
            "id": run_id,
            "title": title,
            "status": "running",
            "created": datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds"),
            "created_ts": now,
        })
        return run_id

    def finish_run(self, run_id: str, status: str, **extra) -> None:
        meta = self._read_raw(run_id) or {"id": run_id}
        meta.update(extra, status=status)
        self._write_meta(run_id, meta)

    def read_meta(self, run_id: str):
        meta = self._read_raw(run_id)
        if not meta or meta.get("status") == "deleted":
            return None
        return self._settle(meta)

    def rename_run(self, run_id: str, title: str) -> dict:
        title = " ".join(str(title).split())[:MAX_TITLE]
        if not title:
            raise ValueError("a run needs a title")
        meta = self.read_meta(run_id)
        if meta is None:
            raise KeyError(run_id)
        raw = self._read_raw(run_id)
        raw["title"] = title
        self._write_meta(run_id, raw)
        return self.read_meta(run_id)

    def delete_run(self, run_id: str) -> None:
        """Remove a run's files, leaving a marker that still counts toward the cap."""
        meta = self._read_raw(run_id)
        if meta is None or meta.get("status") == "deleted":
            raise KeyError(run_id)
        self._delete_all(run_id)
        self._write_meta(run_id, {
            "id": run_id,
            "status": "deleted",
            "created_ts": meta.get("created_ts", time.time()),
            "deleted_ts": time.time(),
        })

    def list_runs(self, limit: int = 50) -> list:
        metas = []
        for meta in self._all_meta():
            if meta.get("status") == "deleted":
                if time.time() - meta.get("created_ts", 0.0) > TOMBSTONE_S:
                    self._delete_all(meta["id"])  # its day has passed
                continue
            metas.append(self._settle(meta))
        metas.sort(key=lambda m: m.get("created_ts", 0.0), reverse=True)
        return metas[:limit]

    def runs_since(self, seconds: float) -> int:
        """Runs started in the last `seconds`, deleted ones included."""
        cutoff = time.time() - seconds
        return sum(1 for m in self._all_meta() if m.get("created_ts", 0.0) >= cutoff)

    # ------------------------------------------------------------- helpers

    def _settle(self, meta):
        if meta.get("status") == "running" and time.time() - meta.get("created_ts", 0.0) > STALE_AFTER_S:
            meta = dict(meta, status="interrupted")
        return meta

    def _write_meta(self, run_id, meta):
        self.put(run_id, "meta.json", json.dumps(meta, indent=2))

    def _read_raw(self, run_id):
        data = self.get(run_id, "meta.json")
        return json.loads(data) if data else None

    def _all_meta(self):
        for run_id in self._run_ids():
            meta = self._read_raw(run_id)
            if meta:
                meta.setdefault("id", run_id)
                yield meta


class LocalStore(RunStore):
    """Runs as folders under root. For development and tests."""

    def __init__(self, root):
        self.root = Path(root)

    def _put(self, run_id, name, data):
        folder = self.root / run_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(data)

    def _get(self, run_id, name):
        path = self.root / run_id / name
        return path.read_bytes() if path.is_file() else None

    def _run_ids(self):
        if not self.root.is_dir():
            return []
        return [p.name for p in self.root.iterdir() if valid_run_id(p.name)]

    def _delete_all(self, run_id):
        shutil.rmtree(self.root / run_id, ignore_errors=True)


class GCSStore(RunStore):
    """Runs as objects runs/<id>/<file> in a Cloud Storage bucket."""

    def __init__(self, bucket_name: str, client=None, prefix: str = "runs"):
        if client is None:
            from google.cloud import storage as gcs

            client = gcs.Client()
        self.client = client
        self.bucket_name = bucket_name
        self.bucket = client.bucket(bucket_name)
        self.prefix = prefix

    def _path(self, run_id, name):
        return f"{self.prefix}/{run_id}/{name}"

    def _put(self, run_id, name, data):
        self.bucket.blob(self._path(run_id, name)).upload_from_string(data, content_type=FILES[name])

    def _get(self, run_id, name):
        from google.api_core.exceptions import NotFound

        try:
            return self.bucket.blob(self._path(run_id, name)).download_as_bytes()
        except NotFound:
            return None

    def _run_ids(self):
        ids = []
        for blob in self.client.list_blobs(self.bucket_name, prefix=f"{self.prefix}/"):
            parts = blob.name.split("/")
            if len(parts) == 3 and parts[2] == "meta.json" and valid_run_id(parts[1]):
                ids.append(parts[1])
        return ids

    def _delete_all(self, run_id):
        for blob in self.client.list_blobs(self.bucket_name, prefix=f"{self.prefix}/{run_id}/"):
            blob.delete()

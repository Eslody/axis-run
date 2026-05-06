"""Asynchronous ConfigMap reporter for train ETTR progress.

Only rank 0 starts a reporter. The reporter patches ConfigMap on process start,
first step, disk checkpoint persisted, and process close. There is no periodic
heartbeat or per-step update.

Note: last_ckpt_at records the most recent step completion time at checkpoint,
not the checkpoint I/O completion time, for accurate ETTR calculation.
"""

from __future__ import annotations

import atexit
import copy
import datetime as dt
import json
import logging
import os
import queue
import signal
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROGRESS_ENDPOINT_ENV = "AXIS_PROGRESS_ENDPOINT"
MAX_SEGMENTS = 200

_SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"


class _K8sConflict(Exception):
    """HTTP 409 while patching ConfigMap; caller may retry read-modify-write."""


class _InClusterConfigMapREST:
    """Minimal ConfigMap client via Kubernetes REST API (stdlib only, no kubernetes pip)."""

    def __init__(self) -> None:
        token_path = os.path.join(_SA_DIR, "token")
        ns_path = os.path.join(_SA_DIR, "namespace")
        ca_path = os.path.join(_SA_DIR, "ca.crt")
        for p in (token_path, ns_path, ca_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"missing in-cluster serviceaccount file: {p}")
        with open(token_path, encoding="utf-8") as f:
            self._token = f.read().strip()
        with open(ns_path, encoding="utf-8") as f:
            if not f.read().strip():
                raise ValueError("empty serviceaccount namespace file")
        host = os.getenv("KUBERNETES_SERVICE_HOST", "").strip()
        port = os.getenv("KUBERNETES_SERVICE_PORT", "443").strip()
        if not host:
            raise OSError("KUBERNETES_SERVICE_HOST is not set")
        self._base = f"https://{host}:{port}"
        self._ssl = ssl.create_default_context(cafile=ca_path)

    def read_config_map(self, namespace: str, name: str) -> Dict[str, Any]:
        path = (
            f"/api/v1/namespaces/{urllib.parse.quote(namespace)}"
            f"/configmaps/{urllib.parse.quote(name)}"
        )
        return self._request("GET", path, None)

    def merge_patch_config_map_data(
        self, namespace: str, name: str, data: Dict[str, str]
    ) -> None:
        path = (
            f"/api/v1/namespaces/{urllib.parse.quote(namespace)}"
            f"/configmaps/{urllib.parse.quote(name)}"
        )
        body = json.dumps({"data": data}, separators=(",", ":")).encode("utf-8")
        self._request(
            "PATCH",
            path,
            body,
            "application/merge-patch+json",
        )

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[bytes],
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = self._base + path
        req = urllib.request.Request(url, data=body, method=method)
        req.add_header("Authorization", f"Bearer {self._token}")
        if body is not None and content_type:
            req.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(req, context=self._ssl, timeout=30) as resp:
                raw = resp.read()
                if not raw:
                    return {}
                parsed = json.loads(raw.decode("utf-8"))
                return parsed if isinstance(parsed, dict) else {}
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            if exc.code == 409:
                raise _K8sConflict(f"conflict: {detail}") from exc
            raise RuntimeError(
                f"kubernetes api {method} {path} failed: http {exc.code} {detail}"
            ) from exc


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _to_rfc3339(ts: dt.datetime) -> str:
    return ts.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _from_unix(value: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc)


@dataclass
class TrainProgressState:
    schema_version: int
    job_id: str
    job_name: str
    process_uid: str
    pod_name: str
    rank: int
    started_at: str
    updated_at: str
    first_step: Optional[int] = None
    first_step_at: Optional[str] = None
    last_step: Optional[int] = None
    last_step_at: Optional[str] = None
    last_ckpt_step: Optional[int] = None
    last_ckpt_at: Optional[str] = None
    resume_from_step: Optional[int] = None
    ended_at: Optional[str] = None
    ended_reason: Optional[str] = None


class ProgressReporter:
    def __init__(
        self,
        *,
        enabled: bool,
        job_name: str,
        rank: int,
        namespace: Optional[str] = None,
        cm_name: Optional[str] = None,
        job_id: Optional[str] = None,
        pod_name: Optional[str] = None,
        resume_from_step: Optional[int] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.namespace = namespace or os.getenv("POD_NAMESPACE", "default")
        self.cm_name = cm_name or f"train-progress-{job_name}"
        resolved_job_id = job_id or os.getenv("TRAINING_PLATFORM_JOB_ID", "")
        if not resolved_job_id and self.cm_name.startswith("train-progress-"):
            resolved_job_id = self.cm_name[len("train-progress-") :]
            if resolved_job_id.startswith("job-"):
                resolved_job_id = resolved_job_id[len("job-") :]
        now = _to_rfc3339(_utc_now())
        self._state = TrainProgressState(
            schema_version=1,
            job_id=resolved_job_id,
            job_name=job_name,
            process_uid=str(uuid.uuid4()),
            pod_name=pod_name or os.getenv("POD_NAME", "") or os.getenv("HOSTNAME", ""),
            rank=int(rank),
            started_at=now,
            updated_at=now,
            resume_from_step=resume_from_step,
        )
        self._lock = threading.Lock()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._server: Optional[ThreadingHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._old_handlers: Dict[int, Any] = {}
        self._k8s_api: Optional[_InClusterConfigMapREST] = None

    @classmethod
    def for_rank(cls, *, rank: int, job_name: str) -> "ProgressReporter":
        return cls(enabled=int(rank) == 0, rank=rank, job_name=job_name)

    def start(self) -> None:
        if not self.enabled:
            return
        self._k8s_api = self._load_k8s_api()
        self._worker = threading.Thread(
            target=self._loop, name="axis-progress-reporter", daemon=True
        )
        self._worker.start()
        self._start_http_server()
        self._install_cleanup_handlers()
        self._queue.put("flush")

    def on_first_step(self, step: int, at: Optional[dt.datetime] = None) -> None:
        if not self.enabled:
            return
        event_at = _to_rfc3339(at or _utc_now())
        with self._lock:
            if self._state.first_step_at is not None:
                return
            self._state.first_step = int(step)
            self._state.first_step_at = event_at
            self._state.last_step = int(step)
            self._state.last_step_at = event_at
            self._state.updated_at = event_at
        self._queue.put("flush")

    def on_disk_ckpt_saved(self, step: int, at: Optional[dt.datetime] = None) -> None:
        """Record disk checkpoint completion.

        last_ckpt_at is set to the most recent step completion time (last_step_at)
        rather than the checkpoint I/O completion time, so that ETTR calculation
        reflects actual productive training time without checkpoint write overhead.
        """
        if not self.enabled:
            return
        event_at = _to_rfc3339(at or _utc_now())
        with self._lock:
            self._state.last_ckpt_step = int(step)
            # Use last_step_at (step completion time) for ETTR accuracy;
            # fall back to event_at only if no step has been recorded yet.
            self._state.last_ckpt_at = self._state.last_step_at or event_at
            self._state.updated_at = event_at
        self._queue.put("flush")

    def on_step_done(self, step: int, at: Optional[dt.datetime] = None) -> None:
        """Record the latest completed step without patching ConfigMap immediately."""
        if not self.enabled:
            return
        event_at = _to_rfc3339(at or _utc_now())
        with self._lock:
            self._state.last_step = max(self._state.last_step or 0, int(step))
            self._state.last_step_at = event_at
            self._state.updated_at = event_at

    def flush_and_close(self, reason: str = "normal") -> None:
        if not self.enabled or self._stop.is_set():
            return
        self._stop.set()
        now = _to_rfc3339(_utc_now())
        with self._lock:
            self._state.ended_at = now
            self._state.ended_reason = reason
            self._state.updated_at = now
        self._queue.put("flush_close")
        if self._worker is not None:
            self._worker.join(timeout=3)
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        os.environ.pop(PROGRESS_ENDPOINT_ENV, None)

    def _loop(self) -> None:
        while True:
            sig = self._queue.get()
            snapshot = self._snapshot()
            for attempt, backoff in enumerate((0.5, 1.5, 4.5)):
                try:
                    self._patch_cm(snapshot)
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt == 2:
                        logger.warning("progress patch dropped: %s", exc)
                    else:
                        time.sleep(backoff)
            if sig == "flush_close":
                return

    def _snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(asdict(self._state))

    def _start_http_server(self) -> None:
        reporter = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/progress":
                    self.send_response(404)
                    self.end_headers()
                    return
                length = int(self.headers.get("content-length", "0") or "0")
                body = self.rfile.read(length)
                try:
                    payload = json.loads(body.decode("utf-8"))
                    event = payload.get("event")
                    step = int(payload["step"])
                    event_at = _from_unix(float(payload.get("at", time.time())))
                    if event == "first_step":
                        reporter.on_first_step(step, event_at)
                    elif event == "step_done":
                        reporter.on_step_done(step, event_at)
                    elif event == "disk_ckpt_saved":
                        reporter.on_disk_ckpt_saved(step, event_at)
                    else:
                        raise ValueError(f"unknown event {event}")
                except Exception as exc:  # noqa: BLE001
                    logger.debug("invalid progress payload: %s", exc)
                    self.send_response(400)
                    self.end_headers()
                    return
                self.send_response(204)
                self.end_headers()

            def log_message(self, fmt: str, *args: Any) -> None:
                logger.debug("progress http: " + fmt, *args)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        host, port = self._server.server_address
        os.environ[PROGRESS_ENDPOINT_ENV] = f"http://{host}:{port}"
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="axis-progress-http",
            daemon=True,
        )
        self._server_thread.start()

    def _install_cleanup_handlers(self) -> None:
        atexit.register(self.flush_and_close)
        for sig in (signal.SIGTERM, signal.SIGINT):
            old = signal.getsignal(sig)
            self._old_handlers[sig] = old

            def _handler(signum: int, frame: Any, *, _old=old) -> None:
                self.flush_and_close(reason="killed")
                if callable(_old):
                    _old(signum, frame)
                elif _old == signal.SIG_DFL:
                    raise SystemExit(128 + signum)

            signal.signal(sig, _handler)

    def _load_k8s_api(self) -> Optional[_InClusterConfigMapREST]:
        try:
            return _InClusterConfigMapREST()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "progress reporter disabled: cannot init in-cluster k8s client: %s",
                exc,
            )
            return None

    def _patch_cm(self, segment: Dict[str, Any]) -> None:
        client = self._k8s_api
        if client is None:
            return
        for _ in range(3):
            cm = client.read_config_map(self.namespace, self.cm_name)
            raw_data = cm.get("data")
            data = dict(raw_data) if isinstance(raw_data, dict) else {}
            payload = _parse_progress(data.get("progress.json"))
            segments = payload.setdefault("segments", [])
            _upsert_segment(segments, segment)
            if len(segments) > MAX_SEGMENTS:
                del segments[: len(segments) - MAX_SEGMENTS]
            payload["schema_version"] = 1
            payload["job_name"] = self._state.job_name
            payload["updated_at"] = segment["updated_at"]
            data["progress.json"] = json.dumps(
                payload, separators=(",", ":"), sort_keys=True
            )
            try:
                client.merge_patch_config_map_data(self.namespace, self.cm_name, data)
                return
            except _K8sConflict:
                continue
        raise RuntimeError("conflict patching progress ConfigMap")


def _parse_progress(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {"schema_version": 1, "segments": []}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"schema_version": 1, "segments": []}
    if not isinstance(payload, dict):
        return {"schema_version": 1, "segments": []}
    if not isinstance(payload.get("segments"), list):
        payload["segments"] = []
    return payload


def _upsert_segment(segments: List[Any], segment: Dict[str, Any]) -> None:
    process_uid = segment.get("process_uid")
    for idx, existing in enumerate(segments):
        if isinstance(existing, dict) and existing.get("process_uid") == process_uid:
            segments[idx] = segment
            return
    segments.append(segment)

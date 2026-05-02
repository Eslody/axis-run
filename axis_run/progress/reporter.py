"""Asynchronous ConfigMap reporter for train ETTR progress.

Only rank 0 starts a reporter. The reporter patches ConfigMap on process start,
first step, disk checkpoint persisted, and process close. There is no periodic
heartbeat or per-step update.
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
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROGRESS_ENDPOINT_ENV = "AXIS_PROGRESS_ENDPOINT"
MAX_SEGMENTS = 200


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
        self._k8s_api = None

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
        if not self.enabled:
            return
        event_at = _to_rfc3339(at or _utc_now())
        with self._lock:
            self._state.last_ckpt_step = int(step)
            self._state.last_ckpt_at = event_at
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

    def _load_k8s_api(self):
        try:
            from kubernetes import client, config

            config.load_incluster_config()
            return client.CoreV1Api()
        except Exception as exc:  # noqa: BLE001
            logger.warning("progress reporter disabled: cannot init k8s client: %s", exc)
            return None

    def _patch_cm(self, segment: Dict[str, Any]) -> None:
        if self._k8s_api is None:
            return
        for _ in range(3):
            cm = self._k8s_api.read_namespaced_config_map(
                name=self.cm_name, namespace=self.namespace
            )
            data = dict(cm.data or {})
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
            body = {"data": data}
            try:
                self._k8s_api.patch_namespaced_config_map(
                    name=self.cm_name, namespace=self.namespace, body=body
                )
                return
            except Exception as exc:  # noqa: BLE001
                if getattr(exc, "status", None) != 409:
                    raise
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

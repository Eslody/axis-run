"""Lightweight client used inside trainer processes.

The client talks to the local axis-run parent process. It never imports the
Kubernetes client, so user training code only pays a best-effort HTTP call.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)

ENDPOINT_ENV = "AXIS_PROGRESS_ENDPOINT"


def on_first_step(step: int, at: Optional[float] = None) -> None:
    _post("first_step", step=step, at=at)


def on_disk_ckpt_saved(step: int, at: Optional[float] = None) -> None:
    # This must be called only after a disk checkpoint is durable.
    _post("disk_ckpt_saved", step=step, at=at)


def on_step_done(step: int, at: Optional[float] = None) -> None:
    # Record the latest completed step; reporter flushes on checkpoint events.
    _post("step_done", step=step, at=at)


def report_ckpt_overhead(seconds: float, is_disk: bool = False) -> None:
    """Report checkpoint operation overhead (seconds). Internal use by axis-run."""
    _post("ckpt_overhead", seconds=seconds, is_disk=is_disk)


def _post(event: str, *, step: int = 0, at: Optional[float] = None, **kwargs: Any) -> None:
    endpoint = os.getenv(ENDPOINT_ENV, "")
    if not endpoint:
        return
    payload = json.dumps(
        {
            "event": event,
            "step": int(step),
            "at": float(at if at is not None else time.time()),
            **kwargs,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/progress",
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=0.5).close()
    except (OSError, urllib.error.URLError) as exc:
        logger.debug("drop progress event %s: %s", event, exc)

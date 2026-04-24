"""FlashCheckpointHelper 的单元测试。

使用 pytest monkeypatch 注入 fake DdpCheckpointer / StorageType，避免依赖
真实 dlrover。重点验证：
    - 目录规则 ``{root}/{epoch}_{step}``（step==0 时退化为 ``{root}/{epoch}``）。
    - save_memory / save_disk 对应的 storage_type。
    - global_step 线性组合不会回退。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

from axis_run.checkpoint import helper as helper_mod


class _FakeStorageType:
    MEMORY = "MEMORY"
    DISK = "DISK"


class _FakeCheckpointer:
    last_instance = None  # type: ignore[assignment]

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.saved: List[Dict[str, Any]] = []
        self.loaded_from = None
        self.load_return: Any = {"model": "x"}
        _FakeCheckpointer.last_instance = self

    def save_checkpoint(self, step, state_dict, path, storage_type):
        self.saved.append(
            {
                "step": step,
                "state_dict": state_dict,
                "path": path,
                "storage_type": storage_type,
            }
        )

    def load_checkpoint(self, resume_path=""):
        self.loaded_from = resume_path
        return self.load_return

    def wait_latest_checkpoint(self, timeout):
        self.waited_timeout = timeout


@pytest.fixture(autouse=True)
def _patch_dlrover(monkeypatch):
    monkeypatch.setattr(
        helper_mod,
        "_require_dlrover",
        lambda: (_FakeCheckpointer, _FakeStorageType),
    )
    yield
    _FakeCheckpointer.last_instance = None


def test_root_env_fallback(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AXIS_CKPT_DIR", str(tmp_path))
    h = helper_mod.FlashCheckpointHelper()
    assert h.root == str(tmp_path)


def test_raises_when_no_root(monkeypatch):
    monkeypatch.delenv("AXIS_CKPT_DIR", raising=False)
    with pytest.raises(helper_mod.CheckpointUnavailable):
        helper_mod.FlashCheckpointHelper()


def test_save_memory_path_scheme(tmp_path: Path):
    h = helper_mod.FlashCheckpointHelper(root=str(tmp_path))
    out = h.save_memory(epoch=1, step=200, state={"a": 1})
    saved = _FakeCheckpointer.last_instance.saved[0]
    assert saved["storage_type"] == _FakeStorageType.MEMORY
    assert saved["path"].endswith(os.path.join("1_200"))
    assert out == saved["path"]


def test_save_disk_step_zero_scheme(tmp_path: Path):
    h = helper_mod.FlashCheckpointHelper(root=str(tmp_path))
    h.save_disk(epoch=5, step=0, state={"a": 1})
    saved = _FakeCheckpointer.last_instance.saved[0]
    assert saved["storage_type"] == _FakeStorageType.DISK
    assert saved["path"].endswith(os.path.join("5"))  # 纯 epoch 目录


def test_explicit_path_is_respected(tmp_path: Path):
    h = helper_mod.FlashCheckpointHelper(root=str(tmp_path))
    target = str(tmp_path / "custom" / "ckpt.pt")
    h.save_disk(epoch=0, step=0, state={}, path=target)
    saved = _FakeCheckpointer.last_instance.saved[0]
    assert saved["path"] == target


def test_global_step_monotonic(tmp_path: Path):
    h = helper_mod.FlashCheckpointHelper(root=str(tmp_path))
    h.save_memory(0, 10, {})
    h.save_memory(0, 100, {})
    h.save_memory(1, 0, {})
    h.save_memory(1, 50, {})
    steps = [r["step"] for r in _FakeCheckpointer.last_instance.saved]
    assert steps == sorted(steps)
    assert steps[0] < steps[1] < steps[2] < steps[3]


def test_load_returns_none_on_empty(tmp_path: Path):
    h = helper_mod.FlashCheckpointHelper(root=str(tmp_path))
    _FakeCheckpointer.last_instance = None
    # 先实例化后触发 last_instance
    _FakeCheckpointer.last_instance  # noqa: B018
    # 需要先实例化，再调用 load
    _FakeCheckpointer.last_instance = None
    h2 = helper_mod.FlashCheckpointHelper(root=str(tmp_path))
    _FakeCheckpointer.last_instance.load_return = None  # type: ignore[union-attr]
    assert h2.load() is None


def test_load_returns_state_dict(tmp_path: Path):
    h = helper_mod.FlashCheckpointHelper(root=str(tmp_path))
    _FakeCheckpointer.last_instance.load_return = {"model": "weights"}
    assert h.load() == {"model": "weights"}


def test_wait_latest_checkpoint_forwarded(tmp_path: Path):
    h = helper_mod.FlashCheckpointHelper(root=str(tmp_path))
    h.wait_latest_checkpoint(timeout=42)
    assert _FakeCheckpointer.last_instance.waited_timeout == 42

"""FlashCheckpointHelper 的单元测试。

使用 pytest monkeypatch 注入 fake DdpCheckpointer / StorageType，避免依赖
真实 dlrover。重点验证：
    - 默认使用 dlrover 自己的 ``{root}/{global_step}/rank_<rank>.pt`` 布局。
    - save_memory / save_disk 对应的 storage_type。
    - global_step 线性组合不会回退。
"""

from __future__ import annotations

import json
import sys
import types
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
    assert saved["path"] == ""
    assert out == str(tmp_path / "1000000200")


def test_save_disk_step_zero_scheme(tmp_path: Path):
    h = helper_mod.FlashCheckpointHelper(root=str(tmp_path))
    h.save_disk(epoch=5, step=0, state={"a": 1})
    saved = _FakeCheckpointer.last_instance.saved[0]
    assert saved["storage_type"] == _FakeStorageType.DISK
    assert saved["path"] == ""
    latest = json.loads((tmp_path / helper_mod.AXIS_LATEST_FILE_NAME).read_text())
    assert latest["layout"] == "dlrover_default"
    assert latest["epoch"] == 5
    assert latest["step"] == 0
    assert latest["path"] == str(tmp_path / "5000000000")
    assert latest["checkpoints"][str(saved["step"])] == str(
        tmp_path / "5000000000"
    )


def test_explicit_path_is_respected(tmp_path: Path):
    h = helper_mod.FlashCheckpointHelper(root=str(tmp_path))
    target = str(tmp_path / "custom" / "ckpt.pt")
    h.save_disk(epoch=0, step=0, state={}, path=target)
    saved = _FakeCheckpointer.last_instance.saved[0]
    assert saved["path"] == target
    latest = json.loads((tmp_path / helper_mod.AXIS_LATEST_FILE_NAME).read_text())
    assert latest["layout"] == "explicit_path"


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


def test_load_unwraps_dlrover_model_states(tmp_path: Path):
    h = helper_mod.FlashCheckpointHelper(root=str(tmp_path))
    _FakeCheckpointer.last_instance.load_return = {
        "model_states": {"model": "weights"}
    }
    assert h.load() == {"model": "weights"}


def test_load_uses_axis_latest_with_committed_step(tmp_path: Path):
    h = helper_mod.FlashCheckpointHelper(root=str(tmp_path))
    h.save_disk(epoch=0, step=18, state={"a": 1})
    (tmp_path / "dlrover_latest.txt").write_text("18")

    _FakeCheckpointer.last_instance.load_return = {
        "model_states": {"step": 18}
    }

    assert h.load() == {"step": 18}
    assert _FakeCheckpointer.last_instance.loaded_from == str(
        tmp_path / "18" / "rank_0.pt"
    )


def test_infers_global_shard_num_from_distributed_world(
    tmp_path: Path, monkeypatch
):
    fake_torch = types.ModuleType("torch")
    fake_dist = types.ModuleType("torch.distributed")
    fake_dist.is_available = lambda: True  # type: ignore[attr-defined]
    fake_dist.is_initialized = lambda: True  # type: ignore[attr-defined]
    fake_dist.get_world_size = lambda: 16  # type: ignore[attr-defined]
    fake_torch.distributed = fake_dist  # type: ignore[attr-defined]
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)

    h = helper_mod.FlashCheckpointHelper(root=str(tmp_path))
    assert h._global_shard_num == 2
    assert _FakeCheckpointer.last_instance.kwargs["global_shard_num"] == 2


def test_load_explicit_directory_resume_path(tmp_path: Path):
    h = helper_mod.FlashCheckpointHelper(root=str(tmp_path))
    resume_dir = tmp_path / "0_18"
    resume_dir.mkdir()
    h.load(resume_path=str(resume_dir))
    assert _FakeCheckpointer.last_instance.loaded_from == str(
        resume_dir / "rank_0.pt"
    )


def test_wait_latest_checkpoint_forwarded(tmp_path: Path):
    h = helper_mod.FlashCheckpointHelper(root=str(tmp_path))
    h.wait_latest_checkpoint(timeout=42)
    assert _FakeCheckpointer.last_instance.waited_timeout == 42

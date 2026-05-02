"""FlashCheckpointHelper：封装 dlrover DdpCheckpointer 的简易用户 API。

这个 helper 的目标是让用户侧代码不需要关心 dlrover flash checkpoint 的细节，
只关注 "我要保存 epoch/step 的 state_dict"，就像普通 ``torch.save`` 一样简单。

典型使用：

    from axis_run.checkpoint import FlashCheckpointHelper

    # init 时 root 取自 AXIS_CKPT_DIR（axis-run launcher 会自动设置）。
    helper = FlashCheckpointHelper()

    for epoch in range(num_epochs):
        for step, batch in enumerate(loader):
            train_step(...)
            if step % 100 == 0:
                # 仅写共享内存；秒级完成；故障后进程重启可直接从 SHM 恢复。
                helper.save_memory(epoch=epoch, step=step, state={
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                })
            if step % 2000 == 0:
                # 异步落盘；rank0 保证持久化；换节点后可从盘恢复。
                helper.save_disk(epoch=epoch, step=step, state={...})

        helper.save_disk(epoch=epoch, step=0, state={...})  # epoch 末

    # 重启时：
    state = helper.load()
    if state is not None:
        model.load_state_dict(state["model"])

默认目录规则交给 dlrover 管理：``{root}/{global_step}/rank_<rank>.pt``。
这样多节点 DDP 下每个保存 rank 会写独立 shard，避免共享同一个 path。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from axis_run.progress import client as progress_client

logger = logging.getLogger(__name__)

DEFAULT_CKPT_ENV_KEY = "AXIS_CKPT_DIR"
AXIS_LATEST_FILE_NAME = "axis_latest.json"
MODEL_STATES_NAME = "model_states"


class CheckpointUnavailable(RuntimeError):
    """当 ckpt_dir 未配置或 dlrover flash checkpoint 不可用时抛出。"""


def _require_dlrover():
    try:
        from dlrover.trainer.torch.flash_checkpoint.ddp import DdpCheckpointer
        from dlrover.trainer.torch.flash_checkpoint.checkpointer import (
            StorageType,
        )
    except Exception as e:  # pragma: no cover
        raise CheckpointUnavailable(
            "dlrover flash checkpoint unavailable; ensure dlrover fork is"
            " installed in the user image."
        ) from e
    return DdpCheckpointer, StorageType


class FlashCheckpointHelper:
    """封装 DdpCheckpointer 的轻量 API。

    Args:
        root: checkpoint 根目录。为 None 时读取 ``AXIS_CKPT_DIR`` 环境变量；都为空则
            构造时抛 :class:`CheckpointUnavailable`。
        shard_num_per_node: 与 dlrover ``local_shard_num`` 语义一致，默认 1（每节点
            整体一个 shard）。FSDP 分片训练时可设为每节点 rank 数。
        global_shard_num: dlrover 全局 shard 数；默认自动按 DDP 节点数推导。
        comm_backend: 参与 shm 同步的 comm backend，默认跟随 DDP 主 pg。
        save_timeout: 每次 disk 保存 rank 间等待的超时（秒），默认用 dlrover 缺省。
        replica_count: 副本数，非 0 时 dlrover 会在多节点间互备 ckpt，适合多副本存储。
    """

    def __init__(
        self,
        root: Optional[str] = None,
        shard_num_per_node: int = 1,
        global_shard_num: Optional[int] = None,
        comm_backend: str = "",
        save_timeout: Optional[int] = None,
        replica_count: int = 0,
    ) -> None:
        self._root = root or os.getenv(DEFAULT_CKPT_ENV_KEY, "")
        if not self._root:
            raise CheckpointUnavailable(
                f"checkpoint root is not set; pass root=... or export "
                f"{DEFAULT_CKPT_ENV_KEY}"
            )
        os.makedirs(self._root, exist_ok=True)

        DdpCheckpointer, StorageType = _require_dlrover()
        self._StorageType = StorageType

        resolved_global_shard_num = (
            global_shard_num
            if global_shard_num is not None
            else self._infer_global_shard_num(shard_num_per_node)
        )
        kwargs: Dict[str, Any] = dict(
            checkpoint_dir=self._root,
            local_shard_num=shard_num_per_node,
            global_shard_num=resolved_global_shard_num,
            comm_backend=comm_backend,
            replica_count=replica_count,
        )
        if save_timeout is not None:
            kwargs["save_timeout"] = save_timeout
        self._checkpointer = DdpCheckpointer(**kwargs)
        self._global_shard_num = resolved_global_shard_num

    # -- Public API -------------------------------------------------------

    @property
    def root(self) -> str:
        return self._root

    def save_memory(
        self,
        epoch: int,
        step: int,
        state: Dict[str, Any],
        path: Optional[str] = None,
    ) -> str:
        """仅写入共享内存。秒级返回；进程崩溃后可由新 agent 从 SHM 恢复。"""
        global_step, target_path, checkpoint_path = self._resolve_step_and_path(
            epoch, step, path
        )
        self._checkpointer.save_checkpoint(
            global_step,
            state,
            checkpoint_path,
            storage_type=self._StorageType.MEMORY,
        )
        logger.info(
            "flash ckpt [memory] saved: step=%d path=%s", global_step, target_path
        )
        return target_path

    def save_disk(
        self,
        epoch: int,
        step: int,
        state: Dict[str, Any],
        path: Optional[str] = None,
    ) -> str:
        """异步写入磁盘。rank0 负责等待所有 rank 落盘完成。"""
        global_step, target_path, checkpoint_path = self._resolve_step_and_path(
            epoch, step, path
        )
        self._checkpointer.save_checkpoint(
            global_step,
            state,
            checkpoint_path,
            storage_type=self._StorageType.DISK,
        )
        self._write_latest_metadata(
            epoch,
            step,
            global_step,
            target_path,
            default_layout=checkpoint_path == "",
        )
        progress_client.on_disk_ckpt_saved(global_step)
        logger.info(
            "flash ckpt [disk] saved: step=%d path=%s", global_step, target_path
        )
        return target_path

    def load(self, resume_path: str = "") -> Optional[Dict[str, Any]]:
        """加载最新 checkpoint。若无历史则返回 None。"""
        resume_path = self._resolve_resume_path(resume_path)
        try:
            state = self._checkpointer.load_checkpoint(resume_path=resume_path)
        except FileNotFoundError:
            return None
        if not state:
            return None
        if isinstance(state, dict) and set(state.keys()) == {MODEL_STATES_NAME}:
            return state[MODEL_STATES_NAME]
        return state

    def wait_latest_checkpoint(self, timeout: int = 1800) -> None:
        """阻塞等待最近一次异步 disk 保存真正落盘完成。"""
        self._checkpointer.wait_latest_checkpoint(timeout=timeout)

    # -- Internal ---------------------------------------------------------

    def _resolve_step_and_path(
        self, epoch: int, step: int, explicit_path: Optional[str]
    ) -> tuple[int, str, str]:
        """生成 dlrover 需要的 (global_step, per-rank path)。

        - global_step：我们用 ``epoch * 1e9 + step`` 的线性组合保持单调递增，
          便于 dlrover engine 内部按 step 做排序/清理。
        - 默认不传自定义 path 给 dlrover，让它按 ``{root}/{global_step}/rank_<r>.pt``
          管理多节点 shard 和 stage marker。
        - 显式 path 仍然透传，供单机/自定义场景使用。
        """
        global_step = self._compose_global_step(epoch, step)
        if explicit_path:
            return (global_step, explicit_path, explicit_path)

        ckpt_dir = os.path.join(self._root, str(global_step))
        return (global_step, ckpt_dir, "")

    def _resolve_resume_path(self, resume_path: str) -> str:
        """返回 dlrover load_checkpoint 期望的具体文件路径。"""
        if resume_path:
            return self._normalize_resume_path(resume_path)

        latest = self._read_latest_metadata()
        if not latest:
            return ""
        if latest.get("layout") == "dlrover_default":
            committed_step = self._read_committed_global_step()
            if committed_step is None:
                return ""
            return self._normalize_resume_path(
                os.path.join(self._root, str(committed_step))
            )
        committed_step = self._read_committed_global_step()
        if committed_step is not None:
            checkpoints = latest.get("checkpoints", {})
            if isinstance(checkpoints, dict):
                path = checkpoints.get(str(committed_step), "")
                if isinstance(path, str) and path:
                    return self._normalize_resume_path(path)
        path = latest.get("path", "")
        if not isinstance(path, str) or not path:
            return ""
        return self._normalize_resume_path(path)

    def _normalize_resume_path(self, path: str) -> str:
        if os.path.isdir(path) or not self._is_checkpoint_file_path(path):
            return os.path.join(path, self._rank_checkpoint_name())
        return path

    @staticmethod
    def _is_checkpoint_file_path(path: str) -> bool:
        return path.endswith((".bin", ".pt", ".pth", ".safetensors"))

    def _rank_checkpoint_name(self) -> str:
        return "rank_0.pt"

    def _write_latest_metadata(
        self,
        epoch: int,
        step: int,
        global_step: int,
        path: str,
        default_layout: bool = False,
    ) -> None:
        latest_path = os.path.join(self._root, AXIS_LATEST_FILE_NAME)
        latest = self._read_latest_metadata()
        checkpoints = latest.get("checkpoints", {})
        if not isinstance(checkpoints, dict):
            checkpoints = {}
        checkpoints[str(global_step)] = path
        metadata = {
            "version": 1,
            "layout": "dlrover_default" if default_layout else "explicit_path",
            "epoch": int(epoch),
            "step": int(step),
            "global_step": int(global_step),
            "path": path,
            "checkpoints": checkpoints,
        }
        tmp_path = f"{latest_path}.{os.getpid()}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, latest_path)

    def _read_latest_metadata(self) -> Dict[str, Any]:
        latest_path = os.path.join(self._root, AXIS_LATEST_FILE_NAME)
        try:
            with open(latest_path, "r", encoding="utf-8") as f:
                latest = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}
        if not isinstance(latest, dict):
            return {}
        return latest

    def _read_committed_global_step(self) -> Optional[int]:
        tracker_path = os.path.join(self._root, "dlrover_latest.txt")
        try:
            with open(tracker_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except OSError:
            return None
        if not content:
            return None
        try:
            return int(content)
        except ValueError:
            return None

    @staticmethod
    def _compose_global_step(epoch: int, step: int) -> int:
        # epoch 末传 step=0 的时候，保留当 epoch 最大值 + 1，避免和下个 epoch 的
        # 0-step 冲突；做法是把 epoch 做主序列、step 做从序列。
        return int(epoch) * 1_000_000_000 + int(step)

    @staticmethod
    def _infer_global_shard_num(shard_num_per_node: int) -> int:
        """DDP full checkpoint 每个节点的 local shard rank 都会保存一份。"""
        shard_num_per_node = max(1, int(shard_num_per_node))
        try:
            import torch.distributed as dist

            if not (dist.is_available() and dist.is_initialized()):
                return shard_num_per_node
            world_size = dist.get_world_size()
            local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "0") or "0")
            if local_world_size <= 0:
                local_world_size = world_size
            node_num = max(1, world_size // local_world_size)
            return max(shard_num_per_node, node_num * shard_num_per_node)
        except Exception:
            return shard_num_per_node

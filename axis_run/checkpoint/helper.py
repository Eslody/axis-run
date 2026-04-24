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

目录规则（模仿 wall-x）：``{root}/{epoch}_{step}/``。当 ``step == 0`` 时目录名
只含 epoch，与 wall-x 对齐，便于阅读。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_CKPT_ENV_KEY = "AXIS_CKPT_DIR"


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
        global_shard_num: dlrover 全局 shard 数，默认 1。
        comm_backend: 参与 shm 同步的 comm backend，默认跟随 DDP 主 pg。
        save_timeout: 每次 disk 保存 rank 间等待的超时（秒），默认用 dlrover 缺省。
        replica_count: 副本数，非 0 时 dlrover 会在多节点间互备 ckpt，适合多副本存储。
    """

    def __init__(
        self,
        root: Optional[str] = None,
        shard_num_per_node: int = 1,
        global_shard_num: int = 1,
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

        kwargs: Dict[str, Any] = dict(
            checkpoint_dir=self._root,
            local_shard_num=shard_num_per_node,
            global_shard_num=global_shard_num,
            comm_backend=comm_backend,
            replica_count=replica_count,
        )
        if save_timeout is not None:
            kwargs["save_timeout"] = save_timeout
        self._checkpointer = DdpCheckpointer(**kwargs)

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
        global_step, target_path = self._resolve_step_and_path(
            epoch, step, path
        )
        self._checkpointer.save_checkpoint(
            global_step,
            state,
            target_path,
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
        global_step, target_path = self._resolve_step_and_path(
            epoch, step, path
        )
        self._checkpointer.save_checkpoint(
            global_step,
            state,
            target_path,
            storage_type=self._StorageType.DISK,
        )
        logger.info(
            "flash ckpt [disk] saved: step=%d path=%s", global_step, target_path
        )
        return target_path

    def load(self, resume_path: str = "") -> Optional[Dict[str, Any]]:
        """加载最新 checkpoint。若无历史则返回 None。"""
        try:
            state = self._checkpointer.load_checkpoint(resume_path=resume_path)
        except FileNotFoundError:
            return None
        if not state:
            return None
        return state

    def wait_latest_checkpoint(self, timeout: int = 1800) -> None:
        """阻塞等待最近一次异步 disk 保存真正落盘完成。"""
        self._checkpointer.wait_latest_checkpoint(timeout=timeout)

    # -- Internal ---------------------------------------------------------

    def _resolve_step_and_path(
        self, epoch: int, step: int, explicit_path: Optional[str]
    ) -> tuple[int, str]:
        """生成 dlrover 需要的 (global_step, per-rank path)。

        - global_step：我们用 ``epoch * 1e9 + step`` 的线性组合保持单调递增，
          便于 dlrover engine 内部按 step 做排序/清理。
        - 目录：``{root}/{epoch}_{step}`` 或 ``{root}/{epoch}``（step==0）。
        - 文件：每 rank 一个 ``.pt``（dlrover 会在文件名后插入 rank，但 engine
          也接受纯文件路径；这里按照 wall-x 风格直接给目录级 path，
          dlrover 内部会拼 ``rank_<r>.pt``）。
        """
        if explicit_path:
            return (self._compose_global_step(epoch, step), explicit_path)

        if step == 0:
            ckpt_dir = os.path.join(self._root, f"{epoch}")
        else:
            ckpt_dir = os.path.join(self._root, f"{epoch}_{step}")
        return (self._compose_global_step(epoch, step), ckpt_dir)

    @staticmethod
    def _compose_global_step(epoch: int, step: int) -> int:
        # epoch 末传 step=0 的时候，保留当 epoch 最大值 + 1，避免和下个 epoch 的
        # 0-step 冲突；做法是把 epoch 做主序列、step 做从序列。
        return int(epoch) * 1_000_000_000 + int(step)

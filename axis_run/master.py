"""AxisMaster：只在 rank 0 上启动 dlrover LocalJobMaster 子进程。

和 dlrover ``_launch_dlrover_local_master`` 的区别：

    1. 支持 ``--node_num > 1``。原函数硬编码 ``node_num=1`` 只适用于
       standalone，本类透传真实的 ``nnodes``，使 LocalJobMaster 的
       ``RendezvousManager`` 能正确等待 N 个 agent 加入。该能力依赖
       dlrover fork 的 LocalJobMaster 二开（``set_target_worker_num(worker_count)``）。

    2. 绑定监听端口由调用方传入（对应 ``AXIS_MASTER_PORT``），不调用
       ``find_free_port`` 随机分配，因为其它 Pod 需要通过 headless service
       +固定端口才能找到 master。

    3. 启动后通过 TCP 连接探测 ``127.0.0.1:port`` 作为就绪信号；若超时
       则抛 ``MasterUnavailableError``，launcher 会捕获并 ``sys.exit(1)``。
"""

from __future__ import annotations

import os
import socket
import sys
import time
from typing import Optional


class MasterUnavailableError(RuntimeError):
    """dlrover master 在约定超时内不可达，视为 axis-run 启动失败。"""


class AxisMaster:
    """负责 rank 0 上的 dlrover LocalJobMaster 子进程生命周期。"""

    def __init__(
        self,
        port: int,
        node_num: int,
        job_name: str,
        ready_timeout: int = 300,
    ) -> None:
        self._port = int(port)
        self._node_num = int(node_num)
        self._job_name = job_name
        self._ready_timeout = int(ready_timeout)
        self._handler = None  # type: Optional[object]

    @property
    def port(self) -> int:
        return self._port

    @property
    def node_num(self) -> int:
        return self._node_num

    def start(self) -> None:
        """启动 dlrover Master 子进程并阻塞等待 gRPC 端口就绪。"""
        from torch.distributed.elastic.multiprocessing.api import (
            SubprocessHandler,
        )

        from dlrover.trainer.torch.utils import version_less_than_230

        cmd = os.getenv("PYTHON_EXEC", sys.executable)
        args = (
            "-u",
            "-m",
            "dlrover.python.master.main",
            "--port",
            str(self._port),
            "--node_num",
            str(self._node_num),
            "--job_name",
            self._job_name,
            "--platform",
            "local",
        )

        # torch < 2.3 的 SubprocessHandler 参数个数不同，与 dlrover 原实现保持一致。
        if version_less_than_230():
            self._handler = SubprocessHandler(cmd, args, {}, "", "")
        else:
            self._handler = SubprocessHandler(cmd, args, {}, "", "", 0)

        if not self._wait_ready(self._port, self._ready_timeout):
            raise MasterUnavailableError(
                f"dlrover master on 127.0.0.1:{self._port} is not ready "
                f"within {self._ready_timeout}s"
            )

    def stop(self) -> None:
        """停止 master 子进程。重复调用是安全的。"""
        if self._handler is None:
            return
        try:
            self._handler.close()
        except Exception:
            pass
        finally:
            self._handler = None

    @staticmethod
    def _wait_ready(port: int, timeout: int) -> bool:
        """通过 TCP 连接本地端口判断 master gRPC 是否就绪。

        连接成功立即返回 True；连接失败则指数退避重试，直到超时。
        """
        deadline = time.time() + timeout
        backoff = 0.5
        while time.time() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                try:
                    s.connect(("127.0.0.1", port))
                    return True
                except (ConnectionRefusedError, OSError, socket.timeout):
                    pass
            time.sleep(backoff)
            backoff = min(backoff * 1.5, 3.0)
        return False

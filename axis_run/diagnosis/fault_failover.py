"""FaultConfigFailover：基于 job-fault-configmap 的分级 failover 策略。

对接 dlrover ``DynamicAgentFailoverExtension`` 接口。当 worker 进程失败，
``DiagnosisAgent.diagnose_training_failure`` 会先调用本类的
``get_user_failover_strategy``，再根据返回值决定发送 ``RESTART_WORKER``（进程级
重启，保留 SHM / flash checkpoint 内存态）还是 ``RELAUNCH_WORKER``（Pod 退出，
JobSet 重建 Pod，从磁盘 checkpoint 恢复）。

分级规则（见 plan 第 3.6 节）：

    severity == "fatal"  -> NODE_FAILOVER
    severity in ("ok", "warn", 缺省) -> NORMAL_FAILOVER

读取源：``<fault_config_dir>/nodes.json``（由 fault-restart-controller 写入）。
通过 ``NODE_NAME`` 环境变量匹配当前 Pod 所在节点的条目；若 configmap 缺失或
读取失败，一律降级为 ``NORMAL_FAILOVER``（不强制换节点，避免 configmap 问题
放大故障面）。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

SEVERITY_OK = "ok"
SEVERITY_WARN = "warn"
SEVERITY_FATAL = "fatal"
_DEFAULT_FAULT_CONFIG_DIR = "/etc/training-platform/fault"


try:  # 运行期（用户训练镜像里）一定有 dlrover。
    from dlrover.python.common.enums import FailoverStrategy as _FailoverStrategy
    from dlrover.python.elastic_agent.torch.dynamic_failover import (
        DynamicAgentFailoverExtension as _DynamicAgentFailoverExtension,
    )

    _DLROVER_AVAILABLE = True
except Exception:  # pragma: no cover - 在无 dlrover 的单测环境下走这里
    _FailoverStrategy = None  # type: ignore[assignment]

    class _DynamicAgentFailoverExtension:  # type: ignore[no-redef]
        """轻量 stub：仅供在无 dlrover 的单测场景下测试读 configmap 的代码路径。"""

        def __init__(self, *_args, **_kwargs) -> None:
            pass

    _DLROVER_AVAILABLE = False


class FaultConfigFailover(_DynamicAgentFailoverExtension):  # type: ignore[misc]
    """读取 job-fault-configmap 的 severity，决定 failover 等级。"""

    def __init__(
        self,
        fault_config_dir: Optional[str] = None,
        node_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._dir = (
            fault_config_dir
            or os.getenv("AXIS_FAULT_CONFIG_DIR")
            or _DEFAULT_FAULT_CONFIG_DIR
        )
        self._node_name = node_name or os.getenv("NODE_NAME", "")

    def get_user_failover_strategy(self, failure_info):  # type: ignore[override]
        """供 dlrover DiagnosisAgent 调用。读取本节点 severity。"""
        if not _DLROVER_AVAILABLE:
            raise RuntimeError(
                "FaultConfigFailover.get_user_failover_strategy requires dlrover"
                " at runtime; current process has no dlrover available."
            )
        severity = self.read_node_severity()
        if severity == SEVERITY_FATAL:
            logger.warning(
                "FaultConfigFailover: node=%s severity=fatal -> NODE_FAILOVER",
                self._node_name,
            )
            return _FailoverStrategy.NODE_FAILOVER

        # ok / warn / 缺省：都走 NORMAL_FAILOVER。
        # 注：保留进程重启是保护 SHM checkpoint + avoid 换节点带来的 rdzv/store 重建。
        logger.info(
            "FaultConfigFailover: node=%s severity=%s -> NORMAL_FAILOVER",
            self._node_name,
            severity,
        )
        return _FailoverStrategy.NORMAL_FAILOVER

    def read_node_severity(self) -> str:
        """读取 nodes.json 中本节点 severity。异常统一返回 "ok"。

        单独暴露为 public 方法，便于 Diagnostician 复用同一读取逻辑。
        """
        if not self._node_name:
            logger.debug("NODE_NAME not set; severity defaults to ok")
            return SEVERITY_OK

        nodes_file = os.path.join(self._dir, "nodes.json")
        if not os.path.exists(nodes_file):
            return SEVERITY_OK
        try:
            with open(nodes_file, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "failed to read %s: %s; treat as severity=ok", nodes_file, e
            )
            return SEVERITY_OK

        if not isinstance(entries, list):
            logger.warning(
                "unexpected nodes.json structure: %r; treat as severity=ok",
                type(entries).__name__,
            )
            return SEVERITY_OK

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("node_name") == self._node_name:
                sev = entry.get("severity", SEVERITY_OK) or SEVERITY_OK
                return str(sev).lower()
        return SEVERITY_OK

    @property
    def fault_config_dir(self) -> str:
        return self._dir

    @property
    def node_name(self) -> str:
        return self._node_name

"""FaultConfigFailover：基于 job-fault-configmap 的分级 failover 策略。

对接 dlrover ``DynamicAgentFailoverExtension`` 接口。当 worker 进程失败，
``DiagnosisAgent.diagnose_training_failure`` 会先调用本类的
``get_user_failover_strategy``，再根据返回值决定发送 ``RESTART_WORKER``（进程级
重启，保留 SHM / flash checkpoint 内存态）还是 ``RELAUNCH_WORKER``（Pod 退出，
JobSet 重建 Pod，从磁盘 checkpoint 恢复）。

分级规则（见 plan 第 3.6 节）：

    severity == "fatal"  -> NODE_FAILOVER
    severity in ("ok", "warn", 缺省) -> NORMAL_FAILOVER

读取源优先级：

    1) ``<fault_config_dir>/pods.json``：通过 ``POD_NAME`` 匹配当前 Pod 条目。
    2) ``<fault_config_dir>/nodes.json``：兼容旧版，通过 ``NODE_NAME`` 匹配节点条目。

若 configmap 缺失或读取失败，一律降级为 ``NORMAL_FAILOVER``（不强制换节点，
避免 configmap 问题放大故障面）。
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
        pod_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._dir = (
            fault_config_dir
            or os.getenv("AXIS_FAULT_CONFIG_DIR")
            or _DEFAULT_FAULT_CONFIG_DIR
        )
        self._node_name = node_name or os.getenv("NODE_NAME", "")
        self._pod_name = pod_name or os.getenv("POD_NAME", "")

    def get_user_failover_strategy(self, failure_info):  # type: ignore[override]
        """供 dlrover DiagnosisAgent 调用。读取本节点 severity。"""
        if not _DLROVER_AVAILABLE:
            raise RuntimeError(
                "FaultConfigFailover.get_user_failover_strategy requires dlrover"
                " at runtime; current process has no dlrover available."
            )
        severity = self.read_severity()
        if severity == SEVERITY_FATAL:
            logger.warning(
                "FaultConfigFailover: pod=%s node=%s severity=fatal -> NODE_FAILOVER",
                self._pod_name,
                self._node_name,
            )
            return _FailoverStrategy.NODE_FAILOVER

        # ok / warn / 缺省：都走 NORMAL_FAILOVER。
        # 注：保留进程重启是保护 SHM checkpoint + avoid 换节点带来的 rdzv/store 重建。
        logger.info(
            "FaultConfigFailover: pod=%s node=%s severity=%s -> NORMAL_FAILOVER",
            self._pod_name,
            self._node_name,
            severity,
        )
        return _FailoverStrategy.NORMAL_FAILOVER

    def read_severity(self) -> str:
        """优先读取 pod 级 severity，缺失时回退旧版 node 级 severity。"""
        pod_severity = self.read_pod_severity()
        if pod_severity is not None:
            return pod_severity
        return self.read_node_severity()

    def read_pod_severity(self) -> Optional[str]:
        """读取 pods.json 中本 Pod severity；文件或匹配缺失时返回 None。"""
        if not self._pod_name:
            logger.debug("POD_NAME not set; skip pods.json")
            return None

        pods_file = os.path.join(self._dir, "pods.json")
        entries = self._load_entries(pods_file)
        if entries is None:
            return None

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("pod_name") == self._pod_name:
                sev = entry.get("severity", SEVERITY_OK) or SEVERITY_OK
                return str(sev).lower()
        return SEVERITY_OK

    def read_node_severity(self) -> str:
        """读取 nodes.json 中本节点 severity。异常统一返回 "ok"。

        保留为 public 方法，兼容旧测试和旧版 job-fault ConfigMap。
        """
        if not self._node_name:
            logger.debug("NODE_NAME not set; severity defaults to ok")
            return SEVERITY_OK

        nodes_file = os.path.join(self._dir, "nodes.json")
        entries = self._load_entries(nodes_file)
        if entries is None:
            return SEVERITY_OK

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("node_name") == self._node_name:
                sev = entry.get("severity", SEVERITY_OK) or SEVERITY_OK
                return str(sev).lower()
        return SEVERITY_OK

    def _load_entries(self, path: str) -> Optional[list]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("failed to read %s: %s; ignore this file", path, e)
            return None

        if not isinstance(entries, list):
            logger.warning(
                "unexpected %s structure: %r; ignore this file",
                os.path.basename(path),
                type(entries).__name__,
            )
            return None
        return entries

    @property
    def fault_config_dir(self) -> str:
        return self._dir

    @property
    def node_name(self) -> str:
        return self._node_name

    @property
    def pod_name(self) -> str:
        return self._pod_name

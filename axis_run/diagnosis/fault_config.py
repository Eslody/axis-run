"""FaultConfigDiagnostician：周期性扫描 job-fault-configmap 主动决策换节点。

本 Diagnostician 通过 ``DLROVER_EXTENSION_DIAGNOSTICIAN`` 注册到 dlrover
``DiagnosisAgent`` 的周期调度中（``time_interval=30s``）。每次触发 diagnose：

    1) observe：读 ``<fault_config_dir>/nodes.json``，若发现本节点 severity=fatal，
       返回非空 Observation。否则返回 None（NoAction）。
    2) resolve：产出 ``NodeAction(action_type=RELAUNCH_WORKER)``，上报给
       dlrover action queue；agent 侧 ``_invoke_run`` 下一轮会 stop_workers 并
       把 WorkerState 置为 FAILED，从而退出进程、让 JobSet 换 Pod。

和 ``FaultConfigFailover`` 的区别：
    - FaultConfigFailover 在 worker **已经失败** 后被 ``diagnose_training_failure``
      触发，用来决定 dlrover 内部的 failover 级别（进程重启 vs 换节点）。
    - FaultConfigDiagnostician 是 **主动** 巡检，即使 worker 进程当前还活着，
      只要 configmap 说本节点 fatal（比如 RDMA Device removed、GPU UC），就
      主动触发 Pod 失败，避免用坏卡的节点继续训练浪费时间。

两者互补：一个负责"进程崩了怎么办"，一个负责"进程没崩但节点废了怎么办"。
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

try:
    from dlrover.python.diagnosis.common.constants import DiagnosisActionType
    from dlrover.python.diagnosis.common.diagnosis_action import (
        DiagnosisAction,
        NoAction,
        NodeAction,
    )
    from dlrover.python.diagnosis.common.diagnostician import (
        Diagnostician,
        DiagnosisObservation,
    )

    _DLROVER_AVAILABLE = True
except Exception:  # pragma: no cover
    class Diagnostician:  # type: ignore[no-redef]
        """轻量 stub，供无 dlrover 环境下的单测。"""

        def __init__(self, *_args, **_kwargs) -> None:
            pass

    class DiagnosisObservation:  # type: ignore[no-redef]
        def __init__(self, observation: str = "", extra_infos=None) -> None:
            self._observation = observation
            self._extra_infos = extra_infos or {}

        @property
        def observation(self) -> str:
            return self._observation

        @property
        def extra_infos(self):  # type: ignore[override]
            return self._extra_infos

    class DiagnosisAction:  # type: ignore[no-redef]
        pass

    class NoAction(DiagnosisAction):  # type: ignore[no-redef]
        pass

    class NodeAction(DiagnosisAction):  # type: ignore[no-redef]
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class DiagnosisActionType:  # type: ignore[no-redef]
        RELAUNCH_WORKER = "relaunch_worker"

    _DLROVER_AVAILABLE = False

from axis_run.diagnosis.fault_failover import (
    SEVERITY_FATAL,
    FaultConfigFailover,
)

logger = logging.getLogger(__name__)

FAULT_OBSERVATION_FATAL = "AxisFaultConfigFatal"


class FaultConfigDiagnostician(Diagnostician):  # type: ignore[misc]
    """主动巡检 job-fault-configmap，遇到 fatal 触发 RELAUNCH_WORKER。"""

    def __init__(
        self,
        fault_config_dir: Optional[str] = None,
        node_name: Optional[str] = None,
        node_rank: Optional[int] = None,
    ) -> None:
        super().__init__()
        # 复用 FaultConfigFailover 的 nodes.json 读取逻辑，避免在两处重复解析。
        self._reader = FaultConfigFailover(
            fault_config_dir=fault_config_dir, node_name=node_name
        )
        raw_rank = (
            node_rank
            if node_rank is not None
            else os.getenv("PET_NODE_RANK") or os.getenv("NODE_RANK") or "0"
        )
        try:
            self._node_rank = int(raw_rank)
        except (TypeError, ValueError):
            self._node_rank = 0

    def observe(self, **_kwargs) -> Optional[DiagnosisObservation]:  # type: ignore[override]
        """每轮读本节点 severity；fatal 时返回带节点信息的 Observation。"""
        severity = self._reader.read_node_severity()
        if severity != SEVERITY_FATAL:
            return None

        extra = {
            "node_name": self._reader.node_name,
            "severity": severity,
            "fault_config_dir": self._reader.fault_config_dir,
        }
        return DiagnosisObservation(
            observation=FAULT_OBSERVATION_FATAL, extra_infos=extra
        )

    def resolve(
        self,
        problem: DiagnosisObservation,
        **_kwargs,
    ) -> List[DiagnosisAction]:  # type: ignore[override]
        """把 fatal 问题转成 RELAUNCH_WORKER 动作。"""
        if not problem or problem.observation != FAULT_OBSERVATION_FATAL:
            return [NoAction()]

        extras = problem.extra_infos or {}
        reason = (
            f"AxisFaultConfig: node {extras.get('node_name', 'unknown')} "
            f"severity=fatal; requesting RELAUNCH_WORKER"
        )
        logger.warning(reason)

        # NodeType.WORKER 在 dlrover 常量里；为避免引入过重的依赖，直接用字面量。
        # 见 dlrover.python.common.constants.NodeType.WORKER == "worker"。
        action = NodeAction(
            node_id=self._node_rank,
            node_type="worker",
            reason=reason,
            action_type=DiagnosisActionType.RELAUNCH_WORKER,
        )
        return [action]

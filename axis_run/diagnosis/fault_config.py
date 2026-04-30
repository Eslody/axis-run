"""FaultConfigDiagnostician：周期性扫描 job-fault-configmap 主动决策换节点。

本 Diagnostician 通过 ``DLROVER_EXTENSION_DIAGNOSTICIAN`` 注册到 dlrover
``DiagnosisAgent`` 的周期调度中（``time_interval=30s``）。每次触发 diagnose：

    1) observe：读取 ``<fault_config_dir>/fault.json`` 顶层 ``overall_severity``；
       ok 时短路，非 ok 时通过 ``POD_NAME`` 匹配稀疏 ``pods`` 条目。若当前
       Pod severity=fatal/reset，返回非空 Observation。否则返回 None（NoAction）。
    2) resolve：fatal 产出 ``NodeAction(action_type=RELAUNCH_WORKER)``，reset
       产出 ``NodeAction(action_type=RESTART_WORKER)``，上报给
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
import time
from typing import List, Optional

try:
    from dlrover.python.diagnosis.common.constants import (
        DiagnosisActionType,
        DiagnosisConstant,
    )
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
        RESTART_WORKER = "restart_worker"
        RELAUNCH_WORKER = "relaunch_worker"

    class DiagnosisConstant:  # type: ignore[no-redef]
        LOCAL_INSTANCE = -3

    _DLROVER_AVAILABLE = False

from axis_run.diagnosis.fault_failover import (
    SEVERITY_FATAL,
    SEVERITY_RESET,
    FaultConfigFailover,
)

logger = logging.getLogger(__name__)

FAULT_OBSERVATION_FATAL = "AxisFaultConfigFatal"
FAULT_OBSERVATION_RESET = "AxisFaultConfigReset"
DEFAULT_RESET_WAIT_SECONDS = 300


class FaultConfigDiagnostician(Diagnostician):  # type: ignore[misc]
    """主动巡检 job-fault-configmap，遇到 fatal 触发 RELAUNCH_WORKER。"""

    def __init__(
        self,
        fault_config_dir: Optional[str] = None,
        node_name: Optional[str] = None,
        node_rank: Optional[int] = None,
    ) -> None:
        super().__init__()
        # 复用 FaultConfigFailover 的 job-fault 读取逻辑，避免在两处重复解析。
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
        self._last_reset_observation_at = -float(DEFAULT_RESET_WAIT_SECONDS)

    def observe(self, **_kwargs) -> Optional[DiagnosisObservation]:  # type: ignore[override]
        """每轮读本 Pod severity；fatal/reset 时返回带 Pod/节点信息的 Observation。"""
        severity = self._reader.read_severity()
        if severity not in (SEVERITY_FATAL, SEVERITY_RESET):
            return None
        if severity == SEVERITY_RESET:
            now = time.monotonic()
            if now - self._last_reset_observation_at < DEFAULT_RESET_WAIT_SECONDS:
                return None
            self._last_reset_observation_at = now

        extra = {
            "pod_name": self._reader.pod_name,
            "node_name": self._reader.node_name,
            "severity": severity,
            "fault_config_dir": self._reader.fault_config_dir,
        }
        if severity == SEVERITY_RESET:
            extra["wait_seconds"] = DEFAULT_RESET_WAIT_SECONDS
        return DiagnosisObservation(
            observation=(
                FAULT_OBSERVATION_FATAL
                if severity == SEVERITY_FATAL
                else FAULT_OBSERVATION_RESET
            ),
            extra_infos=extra,
        )

    def resolve(
        self,
        problem: DiagnosisObservation,
        **_kwargs,
    ) -> List[DiagnosisAction]:  # type: ignore[override]
        """把 fatal/reset 问题转成本地 worker 动作。"""
        if not problem or problem.observation not in (
            FAULT_OBSERVATION_FATAL,
            FAULT_OBSERVATION_RESET,
        ):
            return [NoAction()]

        extras = problem.extra_infos or {}
        action_type = (
            DiagnosisActionType.RELAUNCH_WORKER
            if problem.observation == FAULT_OBSERVATION_FATAL
            else DiagnosisActionType.RESTART_WORKER
        )
        wait_seconds = int(extras.get("wait_seconds") or 0)
        reason = (
            f"AxisFaultConfig: pod {extras.get('pod_name', 'unknown')} "
            f"node {extras.get('node_name', 'unknown')} "
            f"severity={extras.get('severity', 'unknown')}; requesting {action_type}"
        )
        logger.warning(reason)

        # NodeType.WORKER 在 dlrover 常量里；为避免引入过重的依赖，直接用字面量。
        # 见 dlrover.python.common.constants.NodeType.WORKER == "worker"。
        action = NodeAction(
            node_id=self._node_rank,
            node_type="worker",
            reason=reason,
            instance=DiagnosisConstant.LOCAL_INSTANCE,
            action_type=action_type,
            expired_time_period=(wait_seconds + 60) * 1000 if wait_seconds else 0,
            wait_seconds=wait_seconds,
        )
        if wait_seconds:
            # DLRover NodeAction 目前不暴露 executable_time_period 入参，这里设置
            # 基类字段，让 DiagnosisActionQueue 在 wait_seconds 后再交给 agent 执行。
            setattr(action, "_executable_time_period", wait_seconds)
        return [action]

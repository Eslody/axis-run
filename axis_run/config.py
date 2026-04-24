"""构建 AxisLaunchConfig。

本模块对标 dlrover ``_elastic_config_from_args``，但精简并强制了 axis-run 的约束：

    - 强制 ``rdzv_backend = "dlrover-master"`` 与 ``rdzv_endpoint = ""``，
      使 agent 走 dlrover MasterRendezvousHandler，而非 c10d。
    - 额外注入 ``dynamic_failover_extension = FaultConfigFailover(...)``，
      使 agent 在进程失败时按 ``job-fault-configmap.severity`` 做分级决策。
    - 不启用 torchrun fallback，不调用 ``_merge_elastic_config_from_master``
      以外的动态注入逻辑（保留 dlrover 原有的 master 下发配置合并）。

调用方预期：launcher 在 ``ElasticLaunch(config=...)(cmd, *cmd_args)`` 时传入。
"""

from __future__ import annotations

import argparse
import logging
from typing import Callable, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def build_axis_config(
    args: argparse.Namespace,
    fault_config_dir: str,
) -> Tuple[object, Union[Callable, str], List[str]]:
    """根据 argparse 结果构建 AxisLaunchConfig 等价对象。

    返回 ``(elastic_config, cmd, cmd_args)`` 三元组，直接供
    ``dlrover.python.elastic_agent.torch.training.ElasticLaunch`` 使用。

    Args:
        args: 经过 ``env_resolver.parse_args`` 解析后的命名空间。
        fault_config_dir: ``job-fault-configmap`` 挂载目录，用于构造
            ``FaultConfigFailover`` 与 ``FaultConfigDiagnostician``。

    Raises:
        ImportError: dlrover 未安装或版本过低（axis-run 需要 dlrover fork
            axis-run-main 分支）。
    """
    # 延迟 import：允许单元测试在没有 dlrover 运行时依赖时直接 mock。
    from dlrover.python.common.constants import Accelerators
    from dlrover.python.elastic_agent.torch.training import (
        ElasticLaunchConfig,
    )
    from torch.distributed.run import config_from_args

    base_config, cmd, cmd_args = config_from_args(args)
    logger.info("axis-run base LaunchConfig: %s", base_config.__dict__)
    elastic_config = ElasticLaunchConfig(**base_config.__dict__)

    # ---- 透传 torchrun 扩展参数（与 dlrover 对齐）----
    elastic_config.setup_log(
        getattr(args, "log_dir", None) or "",
        getattr(args, "redirects", None),
        getattr(args, "tee", None),
    )
    elastic_config.precheck = getattr(args, "precheck", 0)
    elastic_config.network_check = getattr(args, "network_check", False)
    elastic_config.comm_perf_test = getattr(args, "comm_perf_test", False)
    elastic_config.numa_affinity = getattr(args, "numa_affinity", False)
    elastic_config.membind_policy = getattr(args, "membind_policy", "none")
    elastic_config.auto_tunning = getattr(args, "auto_tunning", False)
    elastic_config.auto_config = getattr(args, "auto_config", False)
    elastic_config.accelerator = getattr(
        args, "accelerator", Accelerators.NVIDIA_GPU
    )
    elastic_config.exclude_straggler = getattr(args, "exclude_straggler", False)
    elastic_config.set_node_unit(getattr(args, "node_unit", 1))
    elastic_config.training_port = getattr(args, "training_port", 60000)
    elastic_config.save_at_breakpoint = getattr(
        args, "save_at_breakpoint", False
    )

    # ---- axis-run 强制约定 ----
    # rdzv_backend/endpoint 必须走 dlrover master，不能落回 c10d。
    elastic_config.rdzv_backend = "dlrover-master"
    elastic_config.rdzv_endpoint = ""
    # 统一 join_timeout，避免用户未显式配置时 rdzv 无限等待。
    join_timeout = elastic_config.rdzv_configs.get("join_timeout", 600)
    elastic_config.rdzv_configs["timeout"] = join_timeout

    # ---- 注入 FaultConfig 驱动的 failover 扩展 ----
    # 用代码直连的方式（而不是 DLROVER_EXTENSION_DYNAMIC_FAILOVER env）
    # 使 axis-run 的行为完全确定性，避免部署环境被意外篡改 env 导致回退。
    elastic_config.dynamic_failover_extension = _build_fault_failover(
        fault_config_dir
    )

    # dlrover 原实现里还有 _merge_elastic_config_from_master；axis-run 保守起见
    # 不主动调用它，因为在 agent 初始化前 master 可能还没 get_elastic_run_config，
    # 对应的配置下发逻辑后续按需开启。
    return elastic_config, cmd, cmd_args


def _build_fault_failover(fault_config_dir: str):
    """惰性构造 FaultConfigFailover，避免在导入期就触发 dlrover import。"""
    # 延迟 import：axis_run.diagnosis 依赖 dlrover.python.elastic_agent.torch.dynamic_failover。
    from axis_run.diagnosis.fault_failover import FaultConfigFailover

    return FaultConfigFailover(fault_config_dir=fault_config_dir)


def export_extension_diagnostician_env(
    module_path: str = "axis_run.diagnosis.fault_config",
    class_name: str = "FaultConfigDiagnostician",
) -> None:
    """把 FaultConfigDiagnostician 登记到 ``DLROVER_EXTENSION_DIAGNOSTICIAN`` 环境变量。

    dlrover fork 里的 ``DiagnosisAgent.__init__`` 会在 agent 创建时读取该变量，
    动态 import 类并通过 ``register_diagnostician`` 加入周期调度。

    本函数需要在 ``ElasticLaunch(...)`` 触发 agent 初始化之前调用；否则 singleton
    已经初始化，环境变量不再生效。
    """
    import os

    from dlrover.python.common.constants import NodeEnv

    spec = f"{module_path}::{class_name}"
    existing = os.environ.get(NodeEnv.DLROVER_EXTENSION_DIAGNOSTICIAN, "")
    if not existing:
        os.environ[NodeEnv.DLROVER_EXTENSION_DIAGNOSTICIAN] = spec
    elif spec not in existing.split(","):
        os.environ[NodeEnv.DLROVER_EXTENSION_DIAGNOSTICIAN] = (
            existing + "," + spec
        )

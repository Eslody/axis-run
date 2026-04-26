"""axis-run 参数与环境变量解析。

沿用 torchrun 的完整参数集（通过 ``torch.distributed.run.get_args_parser``），
在其上追加 axis-run 专用参数（以 ``--`` 开头，不破坏 torchrun 兼容）。

设计要点：
    1. axis-run 专用参数使用 ``argparse`` 原生 ``action`` 行为（``store_true`` /
       ``store`` 等），不依赖 dlrover 的 ``env`` / ``check_env``，避免与 PyTorch
       argparse 耦合。
    2. 对 ``PET_NNODES`` / ``PET_NODE_RANK`` / ``PET_MASTER_ADDR`` /
       ``PET_MASTER_PORT`` 的读取走原生 torchrun 行为（这些参数定义里已带 env 绑定），
       我们只是在 launcher 里再做一次兜底读取。
    3. ``AXIS_MASTER_PORT`` 是 axis-run 引入的新环境变量，默认 50001。
       dlrover Master gRPC 使用它，**不复用** ``PET_MASTER_PORT``，避免与 DDP
       c10d store 端口冲突。
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

# 默认值常量
DEFAULT_FAULT_CONFIG_DIR = "/etc/training-platform/fault"
DEFAULT_AXIS_MASTER_PORT = 50001


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _add_axis_arguments(parser: argparse.ArgumentParser) -> None:
    """在 parser 上追加 axis-run 专用参数。

    这些参数不会传给用户脚本（argparse.REMAINDER 之前消费）。
    """
    group = parser.add_argument_group("axis-run 扩展参数")

    group.add_argument(
        "--fault-config",
        "--fault_config",
        dest="fault_config",
        type=str,
        default=os.getenv("AXIS_FAULT_CONFIG_DIR", DEFAULT_FAULT_CONFIG_DIR),
        help=(
            "job-fault-* ConfigMap 挂载到 Pod 的目录，内含 summary.json / "
            "nodes.json。默认 /etc/training-platform/fault。"
        ),
    )
    group.add_argument(
        "--ckpt-dir",
        "--ckpt_dir",
        dest="ckpt_dir",
        type=str,
        default=os.getenv("AXIS_CKPT_DIR", ""),
        help=(
            "Flash Checkpoint 根目录。若为空则用户脚本需自行关闭 flash checkpoint。"
            "设置后会同时导出到 AXIS_CKPT_DIR 环境变量，供 FlashCheckpointHelper 读取。"
        ),
    )
    group.add_argument(
        "--axis-master-port",
        "--axis_master_port",
        dest="axis_master_port",
        type=int,
        default=int(os.getenv("AXIS_MASTER_PORT", DEFAULT_AXIS_MASTER_PORT)),
        help=(
            "rank 0 上 dlrover LocalJobMaster 的 gRPC 监听端口。"
            "其它 rank 通过 PET_MASTER_ADDR + axis_master_port 连接到 master。"
            "默认从环境变量 AXIS_MASTER_PORT 读取，未设置则用 50001。"
        ),
    )
    group.add_argument(
        "--master-ready-timeout",
        "--master_ready_timeout",
        dest="master_ready_timeout",
        type=int,
        default=int(os.getenv("AXIS_MASTER_READY_TIMEOUT", "300")),
        help=(
            "等待 dlrover master gRPC 端口就绪的超时秒数。超时后 axis-run 直接 "
            "sys.exit(1) 让 Pod 失败，不做 torchrun fallback。默认 300 秒。"
        ),
    )
    group.add_argument(
        "--job-name",
        "--job_name",
        dest="axis_job_name",
        type=str,
        default=os.getenv("JOB_NAME", ""),
        help=(
            "dlrover Master 的 job_name，用于日志区分。默认取 JOB_NAME 环境变量，"
            "未设置时 launcher 会生成一个带时间戳的缺省值。"
        ),
    )
    group.add_argument(
        "--save-at-breakpoint",
        "--save_at_breakpoint",
        dest="save_at_breakpoint",
        action="store_true",
        default=_env_bool("AXIS_SAVE_AT_BREAKPOINT", False),
        help=(
            "训练进程失败时，让 dlrover agent 将 shared memory 中的最新 "
            "Flash Checkpoint 同步到持久化存储。也可通过 "
            "AXIS_SAVE_AT_BREAKPOINT=true 开启。"
        ),
    )


def _get_torch_parser() -> argparse.ArgumentParser:
    """获取 torchrun 原生 parser，并增加 allow_abbrev=False 防止与 axis 参数歧义。"""
    from torch.distributed.run import get_args_parser

    parser = get_args_parser()
    parser.allow_abbrev = False
    return parser


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """解析 axis-run 命令行。与 torchrun 完全兼容，另加 axis 扩展参数。

    用户脚本和脚本参数由 torchrun 原生 parser 的 ``training_script`` 和
    ``training_script_args``（REMAINDER）收集。
    """
    parser = _get_torch_parser()
    _add_axis_arguments(parser)
    return parser.parse_args(argv)


def resolve_node_topology(args: argparse.Namespace) -> Tuple[int, int, str, int]:
    """从 argparse + 环境变量确定节点拓扑。

    优先级：CLI 参数 > ``PET_*`` env > argparse 默认值。Kubeflow Trainer 侧会
    通过 Downward API 注入 ``PET_NNODES`` / ``PET_NODE_RANK`` / ``PET_MASTER_ADDR``
    / ``PET_MASTER_PORT``，用户也可以在 ``axis-run`` 命令行显式覆盖。

    Returns:
        (nnodes, node_rank, master_addr, pet_master_port)
        - ``nnodes``：解析后的整型，支持 torchrun 的 ``MIN:MAX`` 写法时取 ``MAX``。
        - ``node_rank``：本 Pod 的 node rank（0 表示本节点需要拉起 master）。
        - ``master_addr``：其它 rank 连接 master 时的 DNS 主机（一般是 headless
          service 指向的 rank-0 Pod）。
        - ``pet_master_port``：PyTorch DDP 自己用的 c10d 端口（dlrover agent
          内部使用，与 axis master port **不同**）。
    """
    # nnodes
    raw_nnodes = str(args.nnodes) if args.nnodes else os.getenv("PET_NNODES", "1")
    nnodes = _parse_nnodes_max(raw_nnodes)

    # node_rank
    raw_node_rank = args.node_rank if args.node_rank is not None else os.getenv(
        "PET_NODE_RANK", "0"
    )
    try:
        node_rank = int(raw_node_rank)
    except (TypeError, ValueError):
        node_rank = 0

    # master_addr
    master_addr = args.master_addr or os.getenv("PET_MASTER_ADDR", "")

    # pet_master_port（c10d 用，非 axis master）
    raw_pet_port = args.master_port or os.getenv("PET_MASTER_PORT", "29500")
    try:
        pet_master_port = int(raw_pet_port)
    except (TypeError, ValueError):
        pet_master_port = 29500

    return nnodes, node_rank, master_addr, pet_master_port


def _parse_nnodes_max(raw: str) -> int:
    """解析 nnodes 字符串。torchrun 支持 ``MIN:MAX`` 形式（elastic scaling）。

    axis-run 在 local master 模式下以 MAX 为准（master target_worker_num 使用它）。
    """
    raw = str(raw).strip()
    if not raw:
        return 1
    if ":" in raw:
        parts = raw.split(":")
        try:
            return max(int(p) for p in parts if p)
        except ValueError:
            return 1
    try:
        return int(raw)
    except ValueError:
        return 1

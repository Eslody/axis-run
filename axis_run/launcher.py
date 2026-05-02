"""axis-run launcher: 以 dlrover local master 为底座启动 elastic agent。

和 ``dlrover-run``（``dlrover.trainer.torch.elastic_run.run``）对比的关键区别：

    1. 不做 torchrun fallback。任何 axis-run 自身阶段失败（master 不可达、
       import 错误、等）都以 ``sys.exit(1)`` 退出，让 JobSet 按 maxRestarts
       处理；不会静默回退到 c10d/torchrun。

    2. rank 0 上通过 :class:`axis_run.master.AxisMaster` 启动一个 dlrover
       LocalJobMaster 子进程，绑定 ``AXIS_MASTER_PORT``，node_num 取自
       ``PET_NNODES``（真实节点数，不再硬编码 1）。

    3. 通过 ``dlrover_master_addr = rank0-IP : AXIS_MASTER_PORT`` 设定给
       所有 rank 的 ``DLROVER_MASTER_ADDR``；dlrover agent 以此 addr 走
       MasterRendezvousHandler。``PET_MASTER_ADDR`` + ``AXIS_MASTER_PORT``
       组合在 headless service 下能被任意 rank 的 DNS 解析命中 rank 0。

    4. 构建 config 前把 ``FaultConfigDiagnostician`` 注册到
       ``DLROVER_EXTENSION_DIAGNOSTICIAN``（dlrover fork 的扩展点），
       并把 ``FaultConfigFailover`` 塞到 ``elastic_config.dynamic_failover_extension``。
       Agent 初始化时 dlrover 会自动 import 两者。
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import socket
import sys
import time
from typing import Optional

from axis_run.env_resolver import parse_args, resolve_node_topology

logger = logging.getLogger("axis_run")
logging.basicConfig(
    level=os.getenv("AXIS_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


def run() -> None:
    """axis-run 入口。永远不会返回正常值；成功时由 dlrover agent 决定退出码。"""
    args = parse_args()
    nnodes, node_rank, master_addr, _pet_master_port = resolve_node_topology(args)

    axis_master_port = int(args.axis_master_port)
    ready_timeout = int(args.master_ready_timeout)
    fault_config_dir = args.fault_config or ""
    ckpt_dir = args.ckpt_dir or ""

    job_name = _resolve_job_name(args)
    logger.info(
        "axis-run start: nnodes=%s node_rank=%s master_addr=%s "
        "axis_master_port=%s job_name=%s fault_config=%s ckpt_dir=%s",
        nnodes,
        node_rank,
        master_addr,
        axis_master_port,
        job_name,
        fault_config_dir,
        ckpt_dir,
    )

    master = None
    progress_reporter = None
    try:
        dlrover_master_addr = _prepare_dlrover_master(
            node_rank=node_rank,
            nnodes=nnodes,
            master_addr=master_addr,
            axis_master_port=axis_master_port,
            job_name=job_name,
            ready_timeout=ready_timeout,
        )
        master = dlrover_master_addr.master  # 可能为 None（非 rank 0）

        # 1) 广播 dlrover master addr 给本 Pod 的 agent。
        _set_master_env(dlrover_master_addr.addr, job_name)

        from axis_run.progress.reporter import ProgressReporter

        progress_reporter = ProgressReporter.for_rank(rank=node_rank, job_name=job_name)
        progress_reporter.start()

        # 2) 再次探测 master 是否可达（rank 0 已在 master.start 内等待，rank>0
        #    这里需要等 master DNS 生效 + 端口就绪）。
        if not _check_master_reachable(dlrover_master_addr.addr, ready_timeout):
            _fail(
                f"dlrover master {dlrover_master_addr.addr} unavailable after "
                f"{ready_timeout}s"
            )

        # 3) 导出 AXIS_CKPT_DIR 给用户脚本（FlashCheckpointHelper 从这里读）。
        if ckpt_dir:
            os.environ["AXIS_CKPT_DIR"] = ckpt_dir
            logger.info("AXIS_CKPT_DIR=%s", ckpt_dir)

        # 4) 注册 FaultConfigDiagnostician 到 dlrover 的扩展环境变量。
        from axis_run.config import (
            build_axis_config,
            export_extension_diagnostician_env,
        )

        if fault_config_dir:
            os.environ["AXIS_FAULT_CONFIG_DIR"] = fault_config_dir
            export_extension_diagnostician_env()

        # 5) 构建 ElasticLaunchConfig 并启动 agent。
        config, cmd, cmd_args = build_axis_config(args, fault_config_dir)

        # 6) 启动 agent；launch_agent 内部会把 worker 跑到结束或触发 failover。
        from dlrover.trainer.torch.elastic_run import ElasticLaunch

        ElasticLaunch(
            config=config,
            entrypoint=cmd,
            use_dlrover_launch=True,
        )(*cmd_args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("axis-run failed: %s", exc)
        _fail(f"axis-run crashed: {exc}")
    finally:
        if progress_reporter is not None:
            try:
                progress_reporter.flush_and_close()
            except Exception:
                pass
        if master is not None:
            try:
                master.stop()
            except Exception:
                pass


class _MasterInfo:
    """启动期的临时包装：master 实例（rank 0 有，其它 rank 为 None）+ 可达地址。"""

    __slots__ = ("master", "addr")

    def __init__(self, master, addr: str):
        self.master = master
        self.addr = addr


def _prepare_dlrover_master(
    node_rank: int,
    nnodes: int,
    master_addr: str,
    axis_master_port: int,
    job_name: str,
    ready_timeout: int,
) -> _MasterInfo:
    """根据 node_rank 决定是启动本地 master 还是连远端 master。"""
    if node_rank == 0:
        # rank 0 拉起 LocalJobMaster 子进程。
        from axis_run.master import AxisMaster, MasterUnavailableError

        master = AxisMaster(
            port=axis_master_port,
            node_num=nnodes,
            job_name=job_name,
            ready_timeout=ready_timeout,
        )
        try:
            master.start()
        except MasterUnavailableError as e:
            _fail(f"failed to start dlrover master on rank 0: {e}")
        # rank 0 的 agent 连本机 loopback，最稳。
        return _MasterInfo(master=master, addr=f"127.0.0.1:{axis_master_port}")

    if not master_addr:
        _fail(
            "PET_MASTER_ADDR/--master-addr not set; cannot locate dlrover master"
        )

    # rank>0 通过 headless service DNS（PET_MASTER_ADDR）连到 rank 0 master。
    return _MasterInfo(master=None, addr=f"{master_addr}:{axis_master_port}")


def _set_master_env(addr: str, job_name: str) -> None:
    """把 master 地址和 job name 写到 dlrover 约定的环境变量。"""
    from dlrover.python.common.constants import NodeEnv

    os.environ[NodeEnv.DLROVER_MASTER_ADDR] = addr
    # dlrover 使用 NodeEnv.JOB_NAME（即 ELASTIC_JOB_NAME）读取 job name；
    # 同时为了向后兼容也把 JOB_NAME 写上。
    os.environ[NodeEnv.JOB_NAME] = job_name
    os.environ.setdefault("JOB_NAME", job_name)


def _check_master_reachable(addr: str, timeout: int) -> bool:
    """TCP 层面探测 master addr 可达。超时失败。"""
    try:
        host, port_str = addr.split(":", 1)
        port = int(port_str)
    except ValueError:
        logger.error("invalid master addr %s", addr)
        return False

    deadline = time.time() + timeout
    backoff = 0.5
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            try:
                s.connect((host, port))
                logger.info("dlrover master %s is reachable", addr)
                return True
            except (socket.gaierror, ConnectionRefusedError, OSError, socket.timeout) as e:
                logger.debug("waiting for master %s: %s", addr, e)
        time.sleep(backoff)
        backoff = min(backoff * 1.5, 3.0)
    return False


def _resolve_job_name(args) -> str:
    """解析 job name。优先命令行 > JOB_NAME env > 生成带时间戳的 fallback。"""
    name: Optional[str] = getattr(args, "axis_job_name", "") or os.getenv(
        "JOB_NAME", ""
    )
    if name:
        return name
    return "axis-job-" + _dt.datetime.now().strftime("%Y%m%d%H%M%S")


def _fail(msg: str) -> None:
    """统一失败出口。不做 fallback，直接退出。"""
    logger.error(msg)
    sys.exit(1)

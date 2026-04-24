"""axis-run: torchrun-compatible launcher on top of dlrover elastic agent.

本包提供 axis-run CLI：
    axis-run [torchrun-style args] [--fault-config DIR] [--ckpt-dir DIR] \
             train_script.py [script args...]

典型使用与 torchrun 完全一致，区别仅在于：
    - rank 0 Pod 上会启动 dlrover LocalJobMaster 子进程（bind AXIS_MASTER_PORT）
    - 所有 rank 使用 dlrover-master rendezvous（替代 c10d）
    - 支持 FaultConfigFailover / FaultConfigDiagnostician 基于 job-fault-configmap
      进行分级 failover（warn 进程级重启 / fatal 换节点）
    - 支持 FlashCheckpointHelper（内存 + 异步磁盘）

任何 axis-run 启动阶段错误都会以 sys.exit(1) 退出，不会 fallback 到 torchrun。
"""

__version__ = "0.1.0"

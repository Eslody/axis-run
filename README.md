# axis-run

`axis-run` 是一个与 `torchrun` 命令行完全兼容的分布式训练启动器，在 `dlrover` 的 elastic agent 之上叠加了一套基于 `job-fault-configmap` 的分级容错能力：

- **进程级重启**（`severity=warn`）：只重启本机 worker，不换节点，保留内存 / NCCL 连接。
- **节点级重建**（`severity=fatal`）：通过 `RELAUNCH_WORKER` 让 JobSet 重新调度当前 Pod。
- **Flash Checkpoint**：封装 `dlrover` 的 `DdpCheckpointer`，提供 `save_memory` / `save_disk` 两种写入模式，默认目录由 `AXIS_CKPT_DIR` 指定。
- **rank 0 本地 master**：rank 0 Pod 上内嵌 `LocalJobMaster` 子进程，其他 rank 通过 `AXIS_MASTER_PORT`（默认 `50001`）连接，不引入独立 Deployment / Operator。

本仓库把 [dlrover](https://github.com/intelligent-machine-learning/dlrover) 的源码整体 vendor 进来，并打了 3 处补丁（见下方 **3 处二开**）。`pip install axis-run` 会同时把 `axis_run` 与 `dlrover` 两个命名空间装入环境。

---

## 1. 仓库布局

```
axis-run/
├── axis_run/               # 本项目自研代码（torchrun 兼容层 + 容错扩展）
│   ├── checkpoint/         # FlashCheckpointHelper，包装 dlrover DdpCheckpointer
│   ├── diagnosis/          # FaultConfigFailover / FaultConfigDiagnostician
│   ├── config.py           # 参数解析（torchrun 风格）
│   ├── env_resolver.py     # 从环境变量解析节点拓扑 / rank
│   ├── launcher.py         # 主启动入口，串起 master / agent / diagnostician
│   ├── main.py             # pip console_script 入口
│   └── master.py           # LocalJobMaster 子进程守护
├── dlrover/                # vendor 自 intelligent-machine-learning/dlrover（含 3 处补丁）
├── docs/                   # 集成测试/设计文档
├── scripts/                # dlrover_run_affinity.sh 等脚本（与上游一致）
├── tests/                  # pytest 单元测试（不依赖真实 dlrover runtime）
├── Dockerfile              # 纯 axis-run 基础镜像（无 torch / cuda）
├── Dockerfile.user         # 示例：叠加到 PyTorch 训练镜像
├── LICENSE
├── pyproject.toml
└── README.md
```

---

## 2. 快速开始

### 2.1 本地开发

```bash
# 推荐用 venv
python -m venv .venv && source .venv/bin/activate

# 装编辑态 + 测试依赖（dlrover 及其依赖会随 axis-run 一并安装）
pip install -e '.[test]'

# 跑单测（预期 26 passed，tests 里不依赖 dlrover runtime，使用 stub）
pytest tests -q

# CLI 自检（需要本地已装 torch；axis-run 复用 torch.distributed.run 的参数解析）
pip install '.[torch]'
axis-run --help
```

> 注意：
> - `axis-run` 运行时依赖 `torch`，单元测试则用 stub 跳过真实依赖；
> - 基础 `Dockerfile` 不含 `torch`，只做 `import` 级自检，CLI 自检放在 `Dockerfile.user`（叠加到 PyTorch 基础镜像后再跑）。

### 2.2 在 PyTorch 镜像里使用

```bash
# 把仓库拷到构建上下文后：
docker build -f Dockerfile.user \
    --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:24.10-py3 \
    -t my-train:v1 .

# 训练脚本用法与 torchrun 一致：
axis-run \
    --nnodes=$PET_NNODES \
    --nproc-per-node=$PET_NPROC_PER_NODE \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$PET_MASTER_ADDR:$PET_MASTER_PORT \
    train.py --epoch 10
```

### 2.3 关键环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `AXIS_MASTER_PORT` | `50001` | rank 0 上 LocalJobMaster 监听的 gRPC 端口 |
| `AXIS_FAULT_CONFIG_DIR` | `/etc/training-platform/fault` | 挂载 `job-fault-configmap` 的目录 |
| `AXIS_CKPT_DIR` | _（无，可选）_ | `FlashCheckpointHelper` 默认保存根目录 |
| `JOB_NAME` | _（由 kubeflow-trainer 注入）_ | Diagnostician 报告日志用 |
| `NODE_NAME` | _（由 Downward API 注入）_ | 用于匹配 `job-fault-configmap/nodes.json` 当前节点 |
| `PET_*` | 由 JobSet 注入 | 与 `torchrun` 完全一致，不做改动 |

---

## 3. 与 training-platform / kubeflow-trainer 的集成

上游 kubeflow-trainer 的 torch 插件会在 TrainJob 上：

1. 注入上述 `AXIS_*` / `JOB_NAME` / `NODE_NAME` 环境变量；
2. 把 `job-fault-<TrainJob.Name>` ConfigMap 挂到 `AXIS_FAULT_CONFIG_DIR`；
3. 保留所有原 `PET_*` 变量，因此 `axis-run` 命令行参数与 `torchrun` 保持一致。

Snapshot-agent / fault-restart-controller 负责**写入** `job-fault-configmap`。`axis-run` 只**读取**它，由 `FaultConfigFailover`（被动决策）和 `FaultConfigDiagnostician`（主动轮询）共同消费：

- `warn` → `FaultConfigFailover` 返回 `NORMAL_FAILOVER` → agent 只重启 worker 进程；
- `fatal` → `FaultConfigDiagnostician` 主动 enqueue `RELAUNCH_WORKER` → JobSet 重建 Pod。

详见 [`docs/INTEGRATION_TEST.md`](./docs/INTEGRATION_TEST.md)。

---

## 4. 3 处二开（dlrover 的改动点）

所有补丁都已 vendor 在 `dlrover/` 目录里，行号以当前仓库为准。如需跟随上游升级，请在 merge 后重新 apply 这 3 处改动。

| # | 文件 | 修改内容 | 目的 |
| --- | --- | --- | --- |
| 1 | `dlrover/python/common/constants.py` | 新增 `NodeEnv.DLROVER_EXTENSION_DIAGNOSTICIAN = "DLROVER_EXTENSION_DIAGNOSTICIAN"` | 允许外部通过环境变量注册自定义 Diagnostician（`axis_run.diagnosis.fault_config.FaultConfigDiagnostician`）|
| 2 | `dlrover/python/elastic_agent/diagnosis/diagnosis_agent.py` | 新增 `_load_extension_diagnosticians()` 并在 `__init__` 中调用 | 运行时从 `DLROVER_EXTENSION_DIAGNOSTICIAN` 环境变量动态加载 Diagnostician 实例 |
| 3 | `dlrover/python/master/local_master.py` | `LocalJobMaster.__init__` 使用 CLI 传入的 `node_num` 调用 `set_target_worker_num(worker_count)`，并启用 `start_metric_collect` / `start_observing` | 让本地 master 支持多节点场景（原版硬编码 worker_count=1）|

快速定位：

```bash
grep -rn "DLROVER_EXTENSION_DIAGNOSTICIAN" dlrover
grep -rn "_load_extension_diagnosticians" dlrover
grep -rn "set_target_worker_num(worker_count)" dlrover
```

### 4.1 跟随上游升级流程

1. 在 `dlrover` 子目录之外 checkout 上游新版本到临时目录；
2. 用 `rsync -av --delete <upstream>/dlrover/ dlrover/`（保留 `.git` 不要 rsync）；
3. 对照上表的 3 处补丁重新 apply（可用 `git diff` 比对旧 commit）；
4. 跑 `pytest tests -q` + `docker build .` 回归。

> 不建议把 `dlrover/` 做成 git submodule：我们有补丁要提交，且希望 `pip install git+axis-run` 能直接拿到完整代码；submodule 会让 CI 与 pip 安装流程变复杂。

---

## 5. 测试

```bash
pip install -e '.[test]'
pytest tests -q
# 预期：26 passed
```

测试不依赖真实 `dlrover` runtime（用 monkeypatch 桩掉 `MasterClient` 等），方便在没有 GPU / K8s 的机器上做 CI。

集成测试（需要真实 K8s 环境）参考 [`docs/INTEGRATION_TEST.md`](./docs/INTEGRATION_TEST.md)：
- 单节点 smoke；
- 多节点 rendezvous；
- warn / fatal 故障注入；
- Flash Checkpoint 恢复。

---

## 6. License

本项目遵循 Apache License 2.0（见 [`LICENSE`](./LICENSE)），其中 `dlrover/` 目录下的代码版权归 [Ant Group / DLRover Authors](https://github.com/intelligent-machine-learning/dlrover) 所有，以相同 License 继续分发。

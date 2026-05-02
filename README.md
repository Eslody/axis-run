# axis-run

`axis-run` 是一个与 `torchrun` 命令行完全兼容的分布式训练启动器，在 `dlrover` 的 elastic agent 之上叠加了一套基于 `job-fault-configmap` 的分级容错能力：

- **进程级重启**（`severity=reset`）：延迟 300s 后只重启本机 worker，不换节点，给热复位故障留恢复窗口。
- **节点级重建**（`severity=fatal`）：通过 `RELAUNCH_WORKER` 让 JobSet 重新调度当前 Pod。
- **Flash Checkpoint**：封装 `dlrover` 的 `DdpCheckpointer`，提供 `save_memory` / `save_disk` 两种写入模式，默认目录由 `AXIS_CKPT_DIR` 指定。
- **rank 0 本地 master**：rank 0 Pod 上内嵌 `LocalJobMaster` 子进程，其他 rank 通过 `AXIS_MASTER_PORT`（默认 `50001`）连接，不引入独立 Deployment / Operator。
- **不依赖 Kubernetes Python client**：`axis-run` 只跑 dlrover local master / elastic agent；Pod 重建、调度、ConfigMap 写入等 Kubernetes 操作全部由 training-platform / tp-plugin 平台层完成。

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
│   ├── compat.py           # PyTorch elastic / ElasticLaunchConfig 签名兼容层
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

# 跑单测（含 compat 层用例；tests 里不依赖真实 dlrover runtime，使用 stub）
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
| `AXIS_PROGRESS_ENDPOINT` | _（由 axis-run 启动时注入）_ | 训练脚本调用 `axis_run.progress` SDK 上报 first_step / disk ckpt 的本地 HTTP endpoint |
| `JOB_NAME` | _（由 kubeflow-trainer 注入）_ | Diagnostician 报告日志用 |
| `POD_NAME` | _（由 Downward API 注入）_ | 用于匹配 `job-fault-configmap/fault.json` 中的当前 Pod |
| `NODE_NAME` | _（由 Downward API 注入）_ | Diagnostician 日志与 fault.json 节点信息排查用 |
| `POD_NAMESPACE` | _（由 Downward API 注入）_ | rank0 reporter patch `train-progress-<JOB_NAME>` ConfigMap 时使用 |
| `PET_*` | 由 JobSet 注入 | 与 `torchrun` 完全一致，不做改动 |

### 2.4 版本兼容策略

`axis-run` 复用了 **PyTorch Elastic** 与 **torch.distributed.run** 的部分内部 API（例如 rank0 拉起 dlrover master 时使用的 `SubprocessHandler`，以及把 `config_from_args` 的结果塞进 DLRover 的 `ElasticLaunchConfig`）。这些 API 在不同 `torch` 小版本间可能增删构造参数。

本仓库用 [`axis_run/compat.py`](./axis_run/compat.py) 在运行时通过 `inspect.signature` 适配，而不是仅靠 `torch.__version__` 字符串分支：

- **`create_subprocess_handler`**：兼容 5 / 6 / 7+ 参数的 `SubprocessHandler`（含新版 `numa_options`）。
- **`create_elastic_launch_config` / `filter_kwargs_for_ctor`**：构造 `ElasticLaunchConfig` 时过滤掉 torch `LaunchConfig` 里 DLRover 尚未识别的字段；若目标类带 `**kwargs` 则原样透传。

**部署建议**

- **默认不要把 `torch` 写进 axis-run 的安装依赖**：训练镜像已固定 PyTorch/CUDA 时，只执行 `pip install axis-run` 或 `pip install -e .`，避免安装 axis-run 时顺带升级 torch。
- **默认不要安装 `kubernetes` Python 包**：axis-run 的 runtime 不直接调用 K8s API；K8s 行为由平台层控制。vendored DLRover 里的 Kubernetes 模块只保留给上游兼容，不在 `platform=local` 路径加载。
- **`pip install '.[torch]'`** 仅适合本机做 `axis-run --help` / 冒烟，**不要用于生产镜像**（会拉取大量 CUDA 相关 wheel）。
- **发布与验收**：按「axis-run 的 git tag + 训练镜像 torch 版本」做组合登记；升级 torch 后至少跑 `pytest tests -q`、容器内 `axis-run --help`、以及你关心的断点续训 / 单机 elastic 用例（见 [`docs/INTEGRATION_TEST.md`](./docs/INTEGRATION_TEST.md) 的 Phase 0）。

---

## 3. 与 training-platform / kubeflow-trainer 的集成

上游 kubeflow-trainer 的 torch 插件会在 TrainJob 上：

1. 注入上述 `AXIS_*` / `JOB_NAME` / `POD_NAME` / `NODE_NAME` 环境变量；
2. 把 `job-fault-<TrainJob.Name>` ConfigMap 挂到 `AXIS_FAULT_CONFIG_DIR`；
3. 保留所有原 `PET_*` 变量，因此 `axis-run` 命令行参数与 `torchrun` 保持一致。

Snapshot-agent / fault-restart-controller 负责**写入** `job-fault-configmap/fault.json`。`axis-run` 只**读取**它，由 `FaultConfigFailover`（被动决策）和 `FaultConfigDiagnostician`（主动轮询）共同消费：

- `overall_severity=ok` → 直接返回 ok，不下钻 `jobs.joblist`；
- `overall_severity=warn|reset|fatal` → 在 `jobs.joblist[0].statuses[0].pods[]` 中按 `POD_NAME` 匹配，再通过 `node.severity` 得到当前 Pod 严重度；
- 当前 Pod 不在稀疏 `pods` 列表中 → 视为 ok。

四档语义：

| severity | axis-run 行为 |
| --- | --- |
| `ok` | 不动作 |
| `warn` | 仅记录；如果训练进程自身失败，`FaultConfigFailover` 返回 `NORMAL_FAILOVER` |
| `reset` | `FaultConfigDiagnostician` enqueue 本地 `RESTART_WORKER`，默认 300s 后重启 worker 进程 |
| `fatal` | `FaultConfigDiagnostician` enqueue 本地 `RELAUNCH_WORKER`，让 JobSet 重建 Pod |

详见 [`docs/INTEGRATION_TEST.md`](./docs/INTEGRATION_TEST.md)。

### 3.1 ETTR 训练进度上报

`axis_run.progress.reporter` 只在 rank0 启用，启动后通过本地 HTTP endpoint 接收训练脚本事件，再异步 patch
`train-progress-<JOB_NAME>` ConfigMap。训练主循环不直接依赖 Kubernetes client。

训练脚本或 checkpoint helper 只需要调用轻量 SDK：

```python
from axis_run.progress import on_disk_ckpt_saved, on_first_step

on_first_step(step)

# 必须在 disk checkpoint 完全持久化后调用；memory/SHM checkpoint 不调用。
on_disk_ckpt_saved(step)
```

上报约束：

- `on_first_step` 只记录首次 step 完成时间；
- `on_disk_ckpt_saved` 只在 disk checkpoint durable 后触发 ConfigMap 更新；
- reporter 没有 `on_step` API，也没有周期 heartbeat；
- 进程退出时通过 `atexit` / SIGTERM best-effort 写一次 `ended_at/ended_reason`。

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
# 预期：32 passed（含 tests/test_compat.py 兼容层用例）
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

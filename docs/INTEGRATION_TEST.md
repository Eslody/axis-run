# axis-run 联调清单

本文档描述 axis-run 从单节点 smoke test 到多节点故障注入的完整验证步骤。
所有步骤都假定你已经：

1. 构建并推送 `axis-run` 基础镜像：
   ```bash
   # 在独立 axis-run 仓库根目录执行
   docker build -t <registry>/axis-run:0.1.0 .
   ```
2. 构建并推送用户训练镜像（参考仓库根目录的 `Dockerfile.user`）。
3. `training-platform-plugin/charts/fault-restart-controller` 与
   `training-platform-plugin/charts/snapshot-agent` 已部署，对应 ConfigMap
   `job-fault-<TrainJob.Name>` 由 controller 正常创建。
4. Kubeflow Trainer 组件已部署，并且最新版 `torch` plugin 已把
   `AXIS_MASTER_PORT / AXIS_FAULT_CONFIG_DIR / NODE_NAME / JOB_NAME` 注入
   到 trainer container。

> axis-run runtime 不需要 `kubernetes` Python 包。Pod 重建、换节点、ConfigMap 写入等 Kubernetes 操作全部由 training-platform / tp-plugin 平台层完成；axis-run 只读取已挂载的故障信息并向 dlrover local master/agent 做本地决策。

## Phase 0: 最小兼容检查（Pod / 裸机均可）

在进入 K8s 全流程前，用当前训练镜像里的 **torch + axis-run** 快速确认兼容层与 CLI 正常。

```bash
# 1) 记录 torch 版本（升级 torch 后务必重跑本 Phase）
python -c "import torch; print('torch', torch.__version__)"

# 2) 参数解析与入口（需已安装 torch）
axis-run --help | head -20

# 3) 可选：单机 1 proc + 故意崩溃 + elastic 重启（需自备最小训练脚本与 ckpt 目录）
# mkdir -p /workspace/ckpt-axis-test
# axis-run --standalone --nnodes=1 --nproc-per-node=1 --max-restarts=3 \
#   --ckpt-dir /workspace/ckpt-axis-test /path/to/test_resume.py --steps 20 --crash-step 5
```

**验证点**

- 无 `SubprocessHandler.__init__() missing ... numa_options` 等启动期异常。
- `axis-run --help` 能打印出 torchrun 风格参数及 `--fault-config` / `--ckpt-dir` 等扩展项。

## Phase 1: 单节点 smoke test

目标：确认 axis-run 本身能跑通单节点 1 proc。

```bash
# 启动一个 TrainJob，nnodes=1，nproc-per-node=1，command=["axis-run"]，
# args=["--ckpt-dir=/mnt/ckpt/smoke", "train.py"]。
kubectl apply -f smoke-trainjob.yaml
kubectl logs -f <trainjob-pod> -c node
```

**验证点**

- 日志中出现：
  ```
  axis-run start: nnodes=1 node_rank=0 master_addr=...
  dlrover master 127.0.0.1:50001 is reachable
  ```
- `dlrover.python.master.main` 作为子进程存在，`ps -ef | grep dlrover`。
- 训练能正常退出（不因 dlrover master 异常而崩）。
- `AXIS_CKPT_DIR=/mnt/ckpt/smoke` 被导出。

## Phase 2: 双节点 rdzv

目标：确认 `LocalJobMaster` 多节点场景（`node_num=2`）rendezvous 成功。

```bash
kubectl apply -f two-node-trainjob.yaml
```

**验证点**

- rank 0 Pod 日志：`axis-run start: nnodes=2 node_rank=0`。
- rank 1 Pod 日志：`axis-run start: nnodes=2 node_rank=1`
  `dlrover master <svc>:50001 is reachable`。
- dlrover master 日志：`RendezvousManager ... 2 nodes joined`，没有
  `set_target_worker_num=1` 导致的 "waiting for worker".
- 训练步骤输出 `world_size=2`，梯度同步无超时。
- `kubectl port-forward <rank0-pod> 50001:50001` 可以 `curl`
  （RPC 探活）到 master。

## Phase 3: 故障注入 (fatal) → NODE_FAILOVER

目标：fault-restart-controller 写 `severity=fatal` 后，axis-run 触发 Pod 重建。

```bash
# 模拟 snapshot-agent 上报 RDMA device removed，或直接写 CM。
kubectl -n <ns> create configmap job-fault-<trainjob-name> \
    --from-literal=nodes.json='[{"node_name":"<rank1-node>","severity":"fatal"}]' \
    --dry-run=client -o yaml | kubectl apply -f -
```

**验证点**

- 30s 内（Diagnostician 周期），rank 1 Pod 日志出现：
  ```
  AxisFaultConfig: node <rank1-node> severity=fatal; requesting RELAUNCH_WORKER
  ```
- 或进程已崩溃时，`FaultConfigFailover: severity=fatal -> NODE_FAILOVER`。
- rank 1 Pod 退出、JobSet 将其重建到**新节点**上。
- 新 Pod 日志再次与 master rendezvous 成功。

## Phase 4: 故障注入 (warn) → NORMAL_FAILOVER

目标：severity=warn 时进程级重启，不换 Pod，SHM 能留住。

```bash
kubectl -n <ns> create configmap job-fault-<trainjob-name> \
    --from-literal=nodes.json='[{"node_name":"<rank1-node>","severity":"warn"}]' \
    --dry-run=client -o yaml | kubectl apply -f -

# 手动 kill -9 rank 1 的 trainer 进程以触发 failover。
kubectl exec <rank1-pod> -c node -- bash -c "pkill -9 -f 'train.py'"
```

**验证点**

- rank 1 Pod **不重建**（`kubectl get pod` 的 Pod UID / RestartCount 可见）。
- trainer 容器内进程被 dlrover agent 重新拉起。
- `FlashCheckpointHelper.save_memory(...)` 写入的 SHM ckpt 能被新进程直接读出
  （日志里出现 `flash ckpt [memory] saved` 后重启进程仍能 `helper.load()`）。

## Phase 5: Flash Checkpoint 恢复

目标：`save_disk` 后换节点能从磁盘恢复。

```python
# train.py 示例关键片段
from axis_run.checkpoint import FlashCheckpointHelper
helper = FlashCheckpointHelper()

for epoch in range(10):
    for step, batch in enumerate(loader):
        ...
        if step % 500 == 0:
            helper.save_disk(epoch, step, {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
            })

# 在脚本 main() 开头：
state = helper.load()
if state:
    model.load_state_dict(state["model"])
```

**验证点**

- 磁盘目录 `$AXIS_CKPT_DIR/0_500/rank_*.pt` 存在。
- 触发 Phase 3 的 fatal 换节点。
- 新 Pod 启动时 `helper.load()` 返回非空字典，`global_step` 接续。
- loss 曲线无明显断点（step 跳跃应在 ±save_interval 范围内）。

## 回归检查

以下行为必须在每次迭代后通过：

1. `axis-run --help` 返回和 `torchrun --help` 相同字段 + axis 扩展参数。
2. 从成功运行中 `kubectl delete configmap job-fault-<trainjob-name>` 后，
   diagnostician 静默（不继续产生 action）；训练正常结束。
3. axis-run 启动阶段任何报错（master 起不来、dlrover import 失败）都只会
   `exit 1`，**不会**自动走 torchrun。

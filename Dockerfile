# syntax=docker/dockerfile:1.6
#
# axis-run 基础镜像
# -----------------
# 不包含 torch / CUDA，仅提供 axis-run + 内嵌的 dlrover（含 3 处二开）。
# 适用于：
#   1. 在 CI 中做打包/自检；
#   2. 作为其它 PyTorch 训练镜像的层（用 --from 复制 /usr/local）。
#
# 使用：
#   docker build -t axis-run:<ver> .
#   docker run --rm axis-run:<ver> --help

ARG PY_BASE=python:3.10-slim

# ---------- 1. builder：本地源码构建 wheel 并安装到独立前缀 ----------
FROM ${PY_BASE} AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/src

# 只拷必要内容，避免把 .git / docs 等带进镜像（配合 .dockerignore）
COPY pyproject.toml LICENSE README.md ./
COPY axis_run ./axis_run
COPY dlrover ./dlrover
COPY scripts ./scripts

# 所有依赖一次性安装进 /opt/py-deps（后面整块拷到 runtime）
RUN pip install --prefix=/opt/py-deps .

# ---------- 2. runtime：干净的小镜像，只保留已安装的包 ----------
FROM ${PY_BASE} AS runtime

# 把 builder 产物整块 mv 到系统 site-packages（Python slim 里的 /usr/local）
COPY --from=builder /opt/py-deps /usr/local

# 基本自检：两个命名空间都能 import；
# 注意：axis-run --help 需要 torch（复用 torch.distributed.run 的 argparse），
# 本基础镜像不装 torch，所以不在这里做 CLI 级自检。
RUN python -c "import axis_run, dlrover; print('axis-run', axis_run.__version__)"

ENTRYPOINT ["axis-run"]
CMD ["--help"]

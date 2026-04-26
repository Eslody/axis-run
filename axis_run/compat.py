"""PyTorch / DLRover 版本兼容层。

axis-run 复用了 torch.distributed.elastic 与 torch.distributed.run 的内部 API，
这些 API 在不同 PyTorch 版本间可能增删构造参数。这里用 ``inspect.signature`` 做
运行时适配，避免仅靠 ``torch.__version__`` 字符串判断。
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Mapping, Tuple, Type


def _torch_version_string() -> str:
    try:
        import torch

        return str(torch.__version__)
    except Exception:  # pragma: no cover
        return "unknown"


def create_subprocess_handler(
    handler_cls: Type[Any],
    cmd: str,
    args: Tuple[Any, ...],
    env: Mapping[str, str],
    stdout: str,
    stderr: str,
) -> Any:
    """构造 ``torch.distributed.elastic.multiprocessing.api.SubprocessHandler``。

    已知变体：
        - 5 个业务参数：cmd, args, env, stdout, stderr
        - 6 个：末尾多 ``local_rank_id`` 或 ``redirects``（历史上用 ``0``）
        - 7+：含 ``numa_options`` 等；``numa_options`` 传 ``None`` 即可满足签名

    Args:
        handler_cls: 一般为 ``SubprocessHandler`` 类本身。
        cmd: 可执行文件路径（通常为 ``sys.executable``）。
        args: argv tuple（不含 cmd）。
        env: 子进程环境。
        stdout / stderr: elastic 对 stdio 的占位（与 dlrover 原实现一致用 ``""``）。

    Raises:
        TypeError: 签名无法映射到上述约定时抛出，并附带 torch 版本与 signature。
    """
    sig = inspect.signature(handler_cls.__init__)
    params = [
        (name, p)
        for name, p in sig.parameters.items()
        if name not in ("self", "cls")
    ]

    def _value_for(name: str, p: inspect.Parameter) -> Any:
        if name in ("cmd", "entrypoint"):
            return cmd
        if name == "args":
            return args
        if name == "env":
            return dict(env)
        if name == "stdout":
            return stdout
        if name == "stderr":
            return stderr
        if name == "stdin":
            return None
        if name == "numa_options":
            return None
        if name in ("local_rank_id", "redirects"):
            return 0
        if name == "preexec_fn":
            return None
        if p.default is not inspect.Parameter.empty:
            return p.default
        raise TypeError(
            f"Unsupported SubprocessHandler.__init__ parameter {name!r} "
            f"for torch {_torch_version_string()}; signature={sig}"
        )

    kwargs: Dict[str, Any] = {}
    for name, p in params:
        if p.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        kwargs[name] = _value_for(name, p)

    try:
        return handler_cls(**kwargs)
    except TypeError:
        # 仅位置形参的老实现：按历史固定顺序尝试。
        pos_args = (cmd, args, dict(env), stdout, stderr)
        n = len(params)
        if n == len(pos_args):
            return handler_cls(*pos_args)
        if n == len(pos_args) + 1:
            return handler_cls(*pos_args, 0)
        if n >= len(pos_args) + 2:
            return handler_cls(*pos_args, 0, None)
        raise TypeError(
            f"SubprocessHandler positional fallback failed for torch "
            f"{_torch_version_string()}; signature={sig}"
        ) from None


def filter_kwargs_for_ctor(cls: Type[Any], data: Mapping[str, Any]) -> Dict[str, Any]:
    """只保留 ``cls.__init__`` 接受的字段，避免 torch LaunchConfig 新增字段导致
    ``ElasticLaunchConfig(**base_config.__dict__)`` 构造失败。

    若目标 ``__init__`` 带 ``**kwargs``，则原样返回 ``dict(data)``。
    """
    sig = inspect.signature(cls.__init__)
    if any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    ):
        return dict(data)

    allowed = {
        name
        for name, p in sig.parameters.items()
        if name not in ("self", "cls")
        and p.kind
        not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    }
    return {k: v for k, v in data.items() if k in allowed}


def create_elastic_launch_config(elastic_launch_config_cls: Type[Any], base_config: Any):
    """用 ``torch.distributed.run`` 产出的 LaunchConfig 构造 DLRover ``ElasticLaunchConfig``。"""
    raw = dict(base_config.__dict__)
    kwargs = filter_kwargs_for_ctor(elastic_launch_config_cls, raw)
    return elastic_launch_config_cls(**kwargs)

"""`torchrun` console_scripts 入口：安装 axis-run 后用 pip 注册的脚本覆盖 PyTorch 自带的 ``torchrun``。

行为与直接执行 ``axis-run`` 相同（见 :func:`axis_run.launcher.run`），便于沿用现有 ``torchrun ... train.py`` 启动脚本而无需改命令名。

若需要**原生** PyTorch ``torch.distributed.run``（不经 dlrover LocalJobMaster），设置环境变量 ``AXIS_TORCHRUN_LEGACY=1``（或 ``true`` / ``yes``）。
"""

from __future__ import annotations

import os


def _legacy_torchrun_requested() -> bool:
    raw = os.environ.get("AXIS_TORCHRUN_LEGACY", "")
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _invoke_legacy_torchrun() -> None:
    """调用原生 ``torch.distributed.run``，便于单测 mock。"""
    from torch.distributed.run import main as torch_main

    torch_main()


def main() -> None:
    """由 ``[project.scripts]`` 中的 ``torchrun = ...`` 调用。"""
    if _legacy_torchrun_requested():
        _invoke_legacy_torchrun()
        return

    from axis_run.launcher import run

    run()


if __name__ == "__main__":
    main()

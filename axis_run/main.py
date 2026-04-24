"""axis-run CLI 入口。

与 setup.py 中的 console_scripts 绑定，实际启动逻辑在 axis_run.launcher.run 中。
任何启动阶段的异常都会被 launcher 捕获并以 sys.exit(1) 退出。
"""

from axis_run.launcher import run


def entrypoint() -> None:
    """pip 安装后由 `axis-run` 命令调用。"""
    run()


if __name__ == "__main__":
    entrypoint()

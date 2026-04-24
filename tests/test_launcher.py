"""axis-run launcher 的轻量单元测试（不依赖真实 dlrover）。

重点覆盖：
    - rank 0 拉起 master（AxisMaster.start 被调用），并调用 ElasticLaunch。
    - rank > 0 不拉起 master；master_addr 缺失时 sys.exit(1)。
    - master 不可达时 sys.exit(1)，不调用 ElasticLaunch。
    - ckpt_dir 被正确导出到 AXIS_CKPT_DIR。

这里不做跨进程验证；ElasticLaunch 本身的行为交给上游 dlrover 的测试。
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from axis_run import launcher as launcher_mod


class _FakeMaster:
    started = False
    stopped = False

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def start(self) -> None:
        _FakeMaster.started = True

    def stop(self) -> None:
        _FakeMaster.stopped = True


class _FakeMasterError(RuntimeError):
    pass


class _ElasticLaunchSpy:
    calls: Dict[str, Any] = {}

    def __init__(self, config, entrypoint, use_dlrover_launch):
        _ElasticLaunchSpy.calls = dict(
            config=config,
            entrypoint=entrypoint,
            use_dlrover_launch=use_dlrover_launch,
            cmd_args=None,
        )

    def __call__(self, *args):
        _ElasticLaunchSpy.calls["cmd_args"] = args


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    _FakeMaster.started = False
    _FakeMaster.stopped = False
    _ElasticLaunchSpy.calls = {}
    monkeypatch.delenv("AXIS_CKPT_DIR", raising=False)
    monkeypatch.delenv("AXIS_FAULT_CONFIG_DIR", raising=False)
    yield


def _install_common_stubs(monkeypatch, *, master_reachable: bool):
    monkeypatch.setattr(
        launcher_mod,
        "_check_master_reachable",
        lambda addr, timeout: master_reachable,
    )

    # axis_run.master.AxisMaster 只有 rank 0 会 import，因此在 module 级
    # 替换 import。使用一个假的 module 对象注入 sys.modules。
    fake_master_module = SimpleNamespace(
        AxisMaster=_FakeMaster,
        MasterUnavailableError=_FakeMasterError,
    )
    sys.modules["axis_run.master"] = fake_master_module

    # axis_run.config 中我们只关心 build_axis_config 与
    # export_extension_diagnostician_env 被调用，不关心返回值结构。
    def _fake_build(args, fault_config_dir):
        return ({"cfg": True, "fault": fault_config_dir}, "cmd", ["arg"])

    fake_config_module = SimpleNamespace(
        build_axis_config=_fake_build,
        export_extension_diagnostician_env=lambda: os.environ.setdefault(
            "DLROVER_EXTENSION_DIAGNOSTICIAN",
            "axis_run.diagnosis.fault_config::FaultConfigDiagnostician",
        ),
    )
    sys.modules["axis_run.config"] = fake_config_module

    # dlrover.trainer.torch.elastic_run.ElasticLaunch
    fake_elastic_run = SimpleNamespace(ElasticLaunch=_ElasticLaunchSpy)
    sys.modules.setdefault("dlrover", SimpleNamespace())
    sys.modules.setdefault("dlrover.trainer", SimpleNamespace())
    sys.modules.setdefault("dlrover.trainer.torch", SimpleNamespace())
    sys.modules["dlrover.trainer.torch.elastic_run"] = fake_elastic_run

    # dlrover.python.common.constants.NodeEnv.*
    fake_node_env = SimpleNamespace(
        DLROVER_MASTER_ADDR="DLROVER_MASTER_ADDR",
        JOB_NAME="ELASTIC_JOB_NAME",
    )
    fake_dlrover_constants = SimpleNamespace(NodeEnv=fake_node_env)
    sys.modules.setdefault("dlrover.python", SimpleNamespace())
    sys.modules.setdefault("dlrover.python.common", SimpleNamespace())
    sys.modules["dlrover.python.common.constants"] = fake_dlrover_constants


def _fake_args(node_rank=0, master_addr="master.host", **overrides):
    base = dict(
        nnodes="2",
        node_rank=node_rank,
        master_addr=master_addr,
        master_port="29500",
        axis_master_port=50001,
        master_ready_timeout=5,
        fault_config="/etc/training-platform/fault",
        ckpt_dir="/mnt/ckpt",
        axis_job_name="axis-test",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_rank0_happy_path(monkeypatch):
    _install_common_stubs(monkeypatch, master_reachable=True)
    args = _fake_args(node_rank=0)
    monkeypatch.setattr(launcher_mod, "parse_args", lambda: args)
    monkeypatch.setattr(
        launcher_mod,
        "resolve_node_topology",
        lambda _args: (2, 0, "master.host", 29500),
    )

    launcher_mod.run()

    assert _FakeMaster.started is True
    assert _FakeMaster.stopped is True
    assert os.environ.get("AXIS_CKPT_DIR") == "/mnt/ckpt"
    assert os.environ.get("AXIS_FAULT_CONFIG_DIR") == (
        "/etc/training-platform/fault"
    )
    assert _ElasticLaunchSpy.calls["cmd_args"] == ("arg",)
    # DLROVER_EXTENSION_DIAGNOSTICIAN 已被设置
    assert "FaultConfigDiagnostician" in os.environ.get(
        "DLROVER_EXTENSION_DIAGNOSTICIAN", ""
    )


def test_rank_gt0_requires_master_addr(monkeypatch):
    _install_common_stubs(monkeypatch, master_reachable=True)
    args = _fake_args(node_rank=1, master_addr="")
    monkeypatch.setattr(launcher_mod, "parse_args", lambda: args)
    monkeypatch.setattr(
        launcher_mod,
        "resolve_node_topology",
        lambda _args: (2, 1, "", 29500),
    )
    with pytest.raises(SystemExit) as exc:
        launcher_mod.run()
    assert exc.value.code == 1
    assert _FakeMaster.started is False
    # 没走到 ElasticLaunch
    assert _ElasticLaunchSpy.calls == {}


def test_master_unreachable_exits(monkeypatch):
    _install_common_stubs(monkeypatch, master_reachable=False)
    args = _fake_args(node_rank=0)
    monkeypatch.setattr(launcher_mod, "parse_args", lambda: args)
    monkeypatch.setattr(
        launcher_mod,
        "resolve_node_topology",
        lambda _args: (2, 0, "master.host", 29500),
    )
    with pytest.raises(SystemExit) as exc:
        launcher_mod.run()
    assert exc.value.code == 1
    # 即使 master 不可达，rank 0 上已启动的 master 子进程也必须 stop。
    assert _FakeMaster.stopped is True
    assert _ElasticLaunchSpy.calls == {}

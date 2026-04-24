"""FaultConfigDiagnostician 的单元测试。

不依赖 dlrover 运行时：``fault_config.py`` 在 import 期 guard 了 dlrover，
这里只校验 observe/resolve 两步的逻辑，确认 fatal 会出 NodeAction，其它
情况出 NoAction。
"""

from __future__ import annotations

import json
from pathlib import Path

from axis_run.diagnosis.fault_config import (
    FAULT_OBSERVATION_FATAL,
    FaultConfigDiagnostician,
    DiagnosisActionType,
    NoAction,
    NodeAction,
)


def _write_nodes(tmpdir: Path, entries):
    (tmpdir / "nodes.json").write_text(json.dumps(entries), encoding="utf-8")


def test_observe_none_when_ok(tmp_path: Path):
    _write_nodes(tmp_path, [{"node_name": "n1", "severity": "ok"}])
    d = FaultConfigDiagnostician(
        fault_config_dir=str(tmp_path), node_name="n1", node_rank=2
    )
    assert d.observe() is None


def test_observe_none_when_warn(tmp_path: Path):
    _write_nodes(tmp_path, [{"node_name": "n1", "severity": "warn"}])
    d = FaultConfigDiagnostician(
        fault_config_dir=str(tmp_path), node_name="n1", node_rank=0
    )
    assert d.observe() is None


def test_observe_fatal(tmp_path: Path):
    _write_nodes(tmp_path, [{"node_name": "n1", "severity": "fatal"}])
    d = FaultConfigDiagnostician(
        fault_config_dir=str(tmp_path), node_name="n1", node_rank=3
    )
    problem = d.observe()
    assert problem is not None
    assert problem.observation == FAULT_OBSERVATION_FATAL
    assert problem.extra_infos.get("node_name") == "n1"


def test_resolve_produces_node_action(tmp_path: Path):
    _write_nodes(tmp_path, [{"node_name": "n1", "severity": "fatal"}])
    d = FaultConfigDiagnostician(
        fault_config_dir=str(tmp_path), node_name="n1", node_rank=3
    )
    problem = d.observe()
    actions = d.resolve(problem)
    assert len(actions) == 1
    action = actions[0]
    assert isinstance(action, NodeAction)
    # 无 dlrover 时 NodeAction 走 stub，kwargs 里记录所有字段。
    kwargs = getattr(action, "kwargs", None)
    if kwargs is not None:
        assert kwargs["action_type"] == DiagnosisActionType.RELAUNCH_WORKER
        assert kwargs["node_id"] == 3
        assert kwargs["node_type"] == "worker"


def test_resolve_on_empty_problem_returns_no_action(tmp_path: Path):
    d = FaultConfigDiagnostician(
        fault_config_dir=str(tmp_path), node_name="n1", node_rank=0
    )
    actions = d.resolve(None)  # type: ignore[arg-type]
    assert len(actions) == 1
    assert isinstance(actions[0], NoAction)


def test_node_rank_from_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PET_NODE_RANK", "5")
    d = FaultConfigDiagnostician(fault_config_dir=str(tmp_path), node_name="n1")
    assert d._node_rank == 5  # noqa: SLF001  访问内部变量仅用于测试
    monkeypatch.delenv("PET_NODE_RANK", raising=False)

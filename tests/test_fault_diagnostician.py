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


def _fault_doc(overall: str, pods: list[dict] | None = None) -> dict:
    return {
        "schema": "v3",
        "overall_severity": overall,
        "updated_at": 1714000000,
        "jobs": {
            "total_restart_times": 3,
            "left_restart_times": 2,
            "final_completion_time": 0,
            "joblist": [
                {
                    "namespace": "ns",
                    "name": "job",
                    "retart_cnt": 0,
                    "create_time": 1713999900,
                    "completion_time": 0,
                    "abnormalnodes": ["n1"] if overall != "ok" else [],
                    "statuses": [
                        {
                            "timestamp": 1714000000,
                            "status": "Running",
                            "pods": pods or [],
                        }
                    ],
                }
            ],
        },
    }


def _pod(name: str, healthy_id: int) -> dict:
    return {
        "name": name,
        "status": "Running",
        "node": {
            "name": "n1",
            "healthyID": healthy_id,
            "resources": [],
        },
    }


def _write_fault(tmpdir: Path, doc: dict):
    (tmpdir / "fault.json").write_text(json.dumps(doc), encoding="utf-8")


def test_observe_none_when_ok(tmp_path: Path):
    _write_fault(tmp_path, _fault_doc("ok", [_pod("p0", 500)]))
    d = FaultConfigDiagnostician(
        fault_config_dir=str(tmp_path), node_name="n1", node_rank=2
    )
    assert d.observe() is None


def test_observe_none_when_warn(tmp_path: Path):
    _write_fault(tmp_path, _fault_doc("warn", [_pod("p0", 400)]))
    d = FaultConfigDiagnostician(
        fault_config_dir=str(tmp_path), node_name="n1", node_rank=0
    )
    assert d.observe() is None


def test_observe_fatal(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("POD_NAME", "p0")
    _write_fault(tmp_path, _fault_doc("fatal", [_pod("p0", 500)]))
    d = FaultConfigDiagnostician(
        fault_config_dir=str(tmp_path), node_name="n1", node_rank=3
    )
    problem = d.observe()
    assert problem is not None
    assert problem.observation == FAULT_OBSERVATION_FATAL
    assert problem.extra_infos.get("node_name") == "n1"
    assert problem.extra_infos.get("pod_name") == "p0"
    monkeypatch.delenv("POD_NAME", raising=False)


def test_resolve_produces_node_action(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("POD_NAME", "p0")
    _write_fault(tmp_path, _fault_doc("fatal", [_pod("p0", 500)]))
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
    monkeypatch.delenv("POD_NAME", raising=False)


def test_observe_none_when_current_pod_not_sparse_list(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("POD_NAME", "p1")
    _write_fault(tmp_path, _fault_doc("fatal", [_pod("p0", 500)]))
    d = FaultConfigDiagnostician(
        fault_config_dir=str(tmp_path), node_name="n1", node_rank=3
    )
    assert d.observe() is None
    monkeypatch.delenv("POD_NAME", raising=False)


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

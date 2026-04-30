"""FaultConfigFailover 的单元测试。

这些测试不依赖真实 dlrover 运行时，仅覆盖 fault.json 的顶层 fast path 与
Pod 稀疏匹配逻辑。FailoverStrategy 映射逻辑被 guard（需要 dlrover）且语义简单，
这里不 mock dlrover 的全链路测试，在真实集成测试里覆盖。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from axis_run.diagnosis.fault_failover import (
    SEVERITY_FATAL,
    SEVERITY_OK,
    SEVERITY_WARN,
    FaultConfigFailover,
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
                    "retart_cnt": 1,
                    "create_time": 1713999900,
                    "completion_time": 0,
                    "abnormalnodes": ["node-a"] if overall != SEVERITY_OK else [],
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
            "name": "node-a",
            "healthyID": healthy_id,
            "resources": [],
        },
    }


def _write_fault(tmpdir: Path, doc: dict) -> Path:
    p = tmpdir / "fault.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _write_pods(tmpdir: Path, entries) -> Path:
    p = tmpdir / "pods.json"
    p.write_text(json.dumps(entries), encoding="utf-8")
    return p


def test_severity_defaults_to_ok_when_no_file(tmp_path: Path):
    failover = FaultConfigFailover(
        fault_config_dir=str(tmp_path), node_name="node-a", pod_name="pod-a"
    )
    assert failover.read_severity() == SEVERITY_OK


def test_overall_ok_short_circuits_without_pod_scan(tmp_path: Path):
    _write_fault(tmp_path, _fault_doc(SEVERITY_OK, [_pod("pod-a", 500)]))
    f = FaultConfigFailover(
        fault_config_dir=str(tmp_path), node_name="node-a", pod_name="pod-a"
    )
    assert f.read_severity() == SEVERITY_OK


def test_fatal_pod_matches_by_name(tmp_path: Path):
    _write_fault(tmp_path, _fault_doc(SEVERITY_FATAL, [_pod("pod-a", 500)]))
    f = FaultConfigFailover(
        fault_config_dir=str(tmp_path), node_name="node-a", pod_name="pod-a"
    )
    assert f.read_severity() == SEVERITY_FATAL


def test_warn_pod_matches_by_name(tmp_path: Path):
    _write_fault(tmp_path, _fault_doc(SEVERITY_WARN, [_pod("pod-a", 400)]))
    f = FaultConfigFailover(
        fault_config_dir=str(tmp_path), node_name="node-a", pod_name="pod-a"
    )
    assert f.read_severity() == SEVERITY_WARN


def test_sparse_pods_missing_current_pod_returns_ok(tmp_path: Path):
    _write_fault(tmp_path, _fault_doc(SEVERITY_FATAL, [_pod("pod-a", 500)]))
    f = FaultConfigFailover(
        fault_config_dir=str(tmp_path), node_name="node-a", pod_name="pod-b"
    )
    assert f.read_severity() == SEVERITY_OK


def test_malformed_json_returns_ok(tmp_path: Path):
    (tmp_path / "fault.json").write_text("not-json", encoding="utf-8")
    f = FaultConfigFailover(
        fault_config_dir=str(tmp_path), node_name="node-a", pod_name="pod-a"
    )
    assert f.read_severity() == SEVERITY_OK


def test_non_dict_json_returns_ok(tmp_path: Path):
    (tmp_path / "fault.json").write_text(json.dumps([]), encoding="utf-8")
    f = FaultConfigFailover(
        fault_config_dir=str(tmp_path), node_name="node-a", pod_name="pod-a"
    )
    assert f.read_severity() == SEVERITY_OK


def test_env_fallback(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("POD_NAME", "pod-a")
    _write_fault(tmp_path, _fault_doc(SEVERITY_FATAL, [_pod("pod-a", 500)]))
    f = FaultConfigFailover(fault_config_dir=str(tmp_path))
    assert f.pod_name == "pod-a"
    assert f.read_severity() == SEVERITY_FATAL
    monkeypatch.delenv("POD_NAME", raising=False)


def test_missing_pod_name_returns_overall_severity(tmp_path: Path):
    _write_fault(tmp_path, _fault_doc(SEVERITY_FATAL, [_pod("pod-a", 500)]))
    f = FaultConfigFailover(fault_config_dir=str(tmp_path), node_name="node-a")
    assert f.read_severity() == SEVERITY_FATAL

"""FaultConfigFailover 的单元测试。

这些测试不依赖真实 dlrover 运行时，仅覆盖 read_node_severity 这一纯文件 IO
逻辑。FailoverStrategy 映射逻辑被 guard（需要 dlrover）且语义简单，这里不做
mock dlrover 的全链路测试，在真实集成测试里覆盖。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from axis_run.diagnosis.fault_failover import (
    SEVERITY_FATAL,
    SEVERITY_OK,
    SEVERITY_WARN,
    FaultConfigFailover,
)


def _write_nodes(tmpdir: Path, entries) -> Path:
    p = tmpdir / "nodes.json"
    p.write_text(json.dumps(entries), encoding="utf-8")
    return p


def test_severity_defaults_to_ok_when_no_file(tmp_path: Path):
    failover = FaultConfigFailover(
        fault_config_dir=str(tmp_path), node_name="node-a"
    )
    assert failover.read_node_severity() == SEVERITY_OK


def test_severity_defaults_to_ok_when_node_name_missing(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("NODE_NAME", raising=False)
    _write_nodes(tmp_path, [{"node_name": "node-a", "severity": "fatal"}])
    failover = FaultConfigFailover(fault_config_dir=str(tmp_path), node_name="")
    assert failover.read_node_severity() == SEVERITY_OK


def test_severity_parses_fatal(tmp_path: Path):
    _write_nodes(
        tmp_path,
        [
            {"node_name": "node-a", "severity": "ok"},
            {"node_name": "node-b", "severity": "fatal"},
        ],
    )
    f = FaultConfigFailover(fault_config_dir=str(tmp_path), node_name="node-b")
    assert f.read_node_severity() == SEVERITY_FATAL


def test_severity_parses_warn(tmp_path: Path):
    _write_nodes(tmp_path, [{"node_name": "node-a", "severity": "WARN"}])
    f = FaultConfigFailover(fault_config_dir=str(tmp_path), node_name="node-a")
    assert f.read_node_severity() == SEVERITY_WARN


def test_unknown_node_returns_ok(tmp_path: Path):
    _write_nodes(tmp_path, [{"node_name": "node-a", "severity": "fatal"}])
    f = FaultConfigFailover(fault_config_dir=str(tmp_path), node_name="node-b")
    assert f.read_node_severity() == SEVERITY_OK


def test_malformed_json_returns_ok(tmp_path: Path):
    (tmp_path / "nodes.json").write_text("not-json", encoding="utf-8")
    f = FaultConfigFailover(fault_config_dir=str(tmp_path), node_name="node-a")
    assert f.read_node_severity() == SEVERITY_OK


def test_non_list_json_returns_ok(tmp_path: Path):
    (tmp_path / "nodes.json").write_text(
        json.dumps({"node_name": "node-a", "severity": "fatal"}), encoding="utf-8"
    )
    f = FaultConfigFailover(fault_config_dir=str(tmp_path), node_name="node-a")
    assert f.read_node_severity() == SEVERITY_OK


def test_env_fallback(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NODE_NAME", "node-c")
    _write_nodes(tmp_path, [{"node_name": "node-c", "severity": "fatal"}])
    f = FaultConfigFailover(fault_config_dir=str(tmp_path))
    assert f.node_name == "node-c"
    assert f.read_node_severity() == SEVERITY_FATAL
    monkeypatch.delenv("NODE_NAME", raising=False)

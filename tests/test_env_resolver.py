"""axis-run command line argument parsing tests."""

from __future__ import annotations

from axis_run.env_resolver import parse_args


def test_parse_save_at_breakpoint_flag():
    args = parse_args(
        [
            "--nnodes",
            "2",
            "--nproc-per-node",
            "8",
            "--save-at-breakpoint",
            "/tmp/train.py",
            "--mode",
            "flash",
        ]
    )

    assert args.save_at_breakpoint is True
    assert args.training_script == "/tmp/train.py"
    assert args.training_script_args == ["--mode", "flash"]


def test_parse_save_at_breakpoint_env(monkeypatch):
    monkeypatch.setenv("AXIS_SAVE_AT_BREAKPOINT", "true")

    args = parse_args(["/tmp/train.py"])

    assert args.save_at_breakpoint is True


"""axis_run.torchrun_shim：环境分支单测（不 import torch.distributed.run）。"""

from __future__ import annotations

from unittest.mock import patch


def test_legacy_env_calls_torch_run_main(monkeypatch):
    monkeypatch.setenv("AXIS_TORCHRUN_LEGACY", "1")
    with patch("axis_run.torchrun_shim._invoke_legacy_torchrun") as legacy:
        with patch("axis_run.launcher.run") as axis_run:
            from axis_run.torchrun_shim import main

            main()
            legacy.assert_called_once()
            axis_run.assert_not_called()


def test_default_calls_axis_launcher(monkeypatch):
    monkeypatch.delenv("AXIS_TORCHRUN_LEGACY", raising=False)
    with patch("axis_run.torchrun_shim._invoke_legacy_torchrun") as legacy:
        with patch("axis_run.launcher.run") as axis_run:
            from axis_run.torchrun_shim import main

            main()
            axis_run.assert_called_once()
            legacy.assert_not_called()

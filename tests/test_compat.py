"""axis_run.compat：SubprocessHandler 签名适配与 ElasticLaunchConfig 参数过滤。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from axis_run.compat import (
    create_elastic_launch_config,
    create_subprocess_handler,
    filter_kwargs_for_ctor,
)


class Handler5:
    last_args: tuple | None = None

    def __init__(self, cmd, args, env, stdout, stderr):
        Handler5.last_args = (cmd, args, env, stdout, stderr)

    def close(self) -> None:
        pass


class Handler6:
    last_args: tuple | None = None

    def __init__(self, cmd, args, env, stdout, stderr, local_rank_id):
        Handler6.last_args = (cmd, args, env, stdout, stderr, local_rank_id)

    def close(self) -> None:
        pass


class Handler7:
    last_args: tuple | None = None

    def __init__(self, cmd, args, env, stdout, stderr, local_rank_id, numa_options):
        Handler7.last_args = (
            cmd,
            args,
            env,
            stdout,
            stderr,
            local_rank_id,
            numa_options,
        )

    def close(self) -> None:
        pass


@pytest.mark.parametrize(
    "cls,expected_tail",
    [
        (Handler5, ()),
        (Handler6, (0,)),
        (Handler7, (0, None)),
    ],
)
def test_create_subprocess_handler_variants(cls, expected_tail):
    cls.last_args = None
    cmd = "/usr/bin/python3"
    args = ("-m", "mod")
    env = {"A": "1"}
    h = create_subprocess_handler(cls, cmd, args, env, "", "")
    assert h is not None
    assert cls.last_args is not None
    assert cls.last_args[0] == cmd
    assert cls.last_args[1] == args
    assert cls.last_args[2] == {"A": "1"}
    assert cls.last_args[3] == ""
    assert cls.last_args[4] == ""
    if expected_tail:
        assert cls.last_args[5:] == expected_tail


def test_filter_kwargs_for_ctor_drops_unknown():
    class Target:
        def __init__(self, a: int, b: str = "x"):
            self.got = (a, b)

    filtered = filter_kwargs_for_ctor(Target, {"a": 1, "b": "y", "extra": 99})
    assert filtered == {"a": 1, "b": "y"}
    t = Target(**filtered)
    assert t.got == (1, "y")


def test_filter_kwargs_for_ctor_var_kw_passes_all():
    class TargetKw:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    data = {"a": 1, "noise": 2}
    filtered = filter_kwargs_for_ctor(TargetKw, data)
    t = TargetKw(**filtered)
    assert t.kwargs == data


def test_create_elastic_launch_config_filters():
    class FakeElastic:
        def __init__(self, min_nodes: int = 1, max_nodes: int = 1):
            self.min_nodes = min_nodes
            self.max_nodes = max_nodes

    base = SimpleNamespace(min_nodes=2, max_nodes=4, future_torch_only_field=123)
    cfg = create_elastic_launch_config(FakeElastic, base)
    assert cfg.min_nodes == 2
    assert cfg.max_nodes == 4
    assert not hasattr(cfg, "future_torch_only_field")

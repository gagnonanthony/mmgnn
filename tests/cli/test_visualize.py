"""Smoke tests for the ``visualize`` CLI entry point."""


def test_help_option(script_runner):
    ret = script_runner.run(["visualize", "--help"])
    assert ret.success

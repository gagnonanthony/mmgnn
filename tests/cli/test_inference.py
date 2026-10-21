"""Smoke tests for the ``inference`` CLI entry point."""


def test_help_option(script_runner):
    ret = script_runner.run(["inference", "--help"])
    assert ret.success

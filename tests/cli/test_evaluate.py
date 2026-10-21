"""Smoke tests for the ``evaluate`` CLI entry point."""


def test_help_option(script_runner):
    ret = script_runner.run(["evaluate", "--help"])
    assert ret.success

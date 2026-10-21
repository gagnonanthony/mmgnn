"""Smoke tests for the ``train`` CLI entry point."""


def test_help_option(script_runner):
    ret = script_runner.run(["train", "--help"])
    assert ret.success

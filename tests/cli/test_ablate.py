"""Smoke tests for the ``ablate`` CLI entry point."""


def test_help_option(script_runner):
    ret = script_runner.run(["ablate", "--help"])
    assert ret.success

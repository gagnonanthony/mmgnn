# Python 3.12 Project Conventions

## Principles

- Write code that is obvious to read, not clever to write.
- Every function does one thing. If you need the word "and" to describe it, split it.
- Fail fast, fail loud. No silent swallowing of errors.
- Design new library functions so they can be tested without a CLI. Existing command `main()` functions stay scripts; their behavior is tested through the functions they call.

---

## Style & Structure

- PEP 8 enforced via `black`, `isort`, and `flake8`. Runtime target: Python 3.12 (`requires-python >= 3.12`).
- Naming: No single-letter variable names. Use descriptive names that convey intent (e.g., `result` not `r`, `directory` not `d`).
- Trailing commas in all multi-line structures.
- Absolute imports only. No wildcard imports. No unused imports.
- Remove dead code — no commented-out blocks, no `TODO` without a linked issue.
- Project layout: source in `src/<package>/`, tests in `tests/` mirroring `src`, shared fixtures in `conftest.py`.

---

## Type Hints (Mandatory on all public interfaces)

- Modern 3.12 syntax: `list[str]`, `X | None`, type aliases. Inline generics (`def f[T](x: T) -> T:`) are 3.12 syntax and are optional here.
- No legacy imports from `typing` for builtins (`List`, `Dict`, `Optional`, `Tuple`).
- No `from __future__ import annotations` — unnecessary on 3.11+ where modern syntax works natively.
- Use `TypeIs` over `TypeGuard`. Use `TypeVar` defaults where appropriate.
- `# type: ignore` must include error code: `# type: ignore[attr-defined]`.

---

## Functions & Design

- ≤20 lines per function. Return early to reduce nesting. Exceptions require a comment explaining why.
- 3+ parameters → use keyword-only args (`*`). For 2-param functions, use `*` when argument order is non-obvious or types are identical. Exceptions require a comment explaining why.
- Never use mutable defaults (`def f(items=[]):`). Use `None` and assign inside.
- Google-style docstrings on public functions. Don't restate type hints in docstrings.
- Pure functions where possible. Keep I/O and side effects at the boundaries.
- Prefer composition over inheritance. `@dataclass` or Pydantic models for data containers.
- Anything that is common and repetitive across functions should be extracted.

---

## Error Handling & Logging

- Catch specific exceptions only. No bare `except:`. No `except Exception: pass`.
- Chain with `from e` when wrapping. Handle errors at one layer, not multiple.
- `logging` module only, never `print()`. Lazy formatting: `logger.info("user %s", uid)` — no f-strings in log calls.
- Never log secrets, tokens, or credentials. Sanitize before logging.

---

## Security & Config

- All secrets via environment variables or `.env`. Never hardcode. Never commit `.env`.
- Validate config at startup with `os.environ["KEY"]` (fails loud) — not deep in call stacks.
- Don't use `assert` for runtime validation — stripped with `-O`. Use `if`/`raise ValueError`.
- `.gitignore`: `.env`, `__pycache__/`, `*.pyc`, `.mypy_cache/`, `.pytest_cache/`, `htmlcov/`, `dist/`.

---

## Testing

- **Arrange → Act → Assert**. Blank lines between the three parts are optional.
- Name: `test_<behavior>`, grouped in classes. Put the condition in the name when it is the point of the test (`test_missing_raises`). A `_when_` clause is not required.
- Mock at boundaries (external services, I/O), never internal implementation.
- `pytest.raises(SpecificError, match="...")` for exception tests.
- Use `@pytest.mark.parametrize` only when several inputs share one assertion, as in `tests/models/test_blocks.py`. `ids=` is optional. Do not fold separate tests into a parameter table.
- Test error paths and edge cases on library functions. CLI entry points (`train`, `evaluate`, `inference`, `ablate`, `visualize`) are smoke-tested with `--help` only; do not drive `main()` for those paths.
- Do not test a constant's membership, or a dataclass or `StrEnum` that has no logic. Do test config validation: required keys, filled defaults, coercion, and rejected values.
- Never use `==` for float comparison — use `pytest.approx()`. Floating point arithmetic is imprecise.
- Shared fixtures in `conftest.py`. Factory fixtures for test data with variations.
- Coverage is measured with `pytest-cov` when you ask for it. It is not a default `pytest` flag, and there is no fail-under gate.
- This package has no async tests. Do not add `asyncio_mode` or `@pytest.mark.asyncio`.

---

## Tooling

Dev tools are `pytest`, `pytest-cov`, `pytest-console-scripts`, `black`, `isort`, `flake8`, and `mypy`. There is no Ruff config and no MyPy strict config. `mypy` is available, but it is not a required check.

```toml
[project]
requires-python = ">=3.12"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

---

## Reference

Library-specific patterns and gotchas: see `library-guidelines.md`
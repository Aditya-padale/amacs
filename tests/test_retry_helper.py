"""Unit tests for unified sync and async retry helper with table of errors."""

from __future__ import annotations

import pytest

from amacs.exceptions import (
    LLMProviderError,
    RetryExhaustedError,
    arun_with_retry,
    run_with_retry,
    should_retry_exception,
)


class CustomHTTPError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class CustomRateLimitError(Exception):
    pass


class CustomAuthError(Exception):
    pass


@pytest.mark.parametrize(
    "exc,expected_retryable",
    [
        (CustomHTTPError(400, "Bad Request"), False),
        (CustomHTTPError(401, "Unauthorized"), False),
        (CustomHTTPError(403, "Forbidden"), False),
        (CustomHTTPError(404, "Not Found"), False),
        (CustomHTTPError(429, "Rate Limit Exceeded"), True),
        (CustomHTTPError(500, "Internal Server Error"), True),
        (CustomHTTPError(502, "Bad Gateway"), True),
        (CustomHTTPError(503, "Service Unavailable"), True),
        (CustomRateLimitError("Rate limit reached"), True),
        (CustomAuthError("Invalid key"), False),
        (LLMProviderError("429 Too Many Requests"), True),
        (LLMProviderError("401 Unauthorized"), False),
        (LLMProviderError("Authentication failed"), False),
        (LLMProviderError("Connection timeout"), True),
    ],
)
def test_error_classification_table(exc: Exception, expected_retryable: bool) -> None:
    assert should_retry_exception(exc) is expected_retryable


def test_sync_retry_success_after_failure() -> None:
    calls = 0

    def fn() -> str:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise CustomHTTPError(429, "Rate limited")
        return "success"

    res = run_with_retry(fn, max_attempts=3)
    assert res == "success"
    assert calls == 2


def test_sync_retry_fatal_error_fails_fast() -> None:
    calls = 0

    def fn() -> None:
        nonlocal calls
        calls += 1
        raise CustomHTTPError(401, "Unauthorized")

    with pytest.raises(CustomHTTPError):
        run_with_retry(fn, max_attempts=3)
    assert calls == 1


def test_sync_retry_exhausted_raises_retry_exhausted() -> None:
    calls = 0

    def fn() -> None:
        nonlocal calls
        calls += 1
        raise CustomHTTPError(500, "Internal Error")

    with pytest.raises(RetryExhaustedError):
        run_with_retry(fn, max_attempts=2)
    assert calls == 2


@pytest.mark.asyncio
async def test_async_retry_success_after_failure() -> None:
    calls = 0

    async def afn() -> str:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise CustomHTTPError(429, "Rate limited")
        return "async_success"

    res = await arun_with_retry(afn, max_attempts=3)
    assert res == "async_success"
    assert calls == 2


@pytest.mark.asyncio
async def test_async_retry_fatal_error_fails_fast() -> None:
    calls = 0

    async def afn() -> None:
        nonlocal calls
        calls += 1
        raise CustomHTTPError(403, "Forbidden")

    with pytest.raises(CustomHTTPError):
        await arun_with_retry(afn, max_attempts=3)
    assert calls == 1

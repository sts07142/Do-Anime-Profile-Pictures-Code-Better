"""Shared rate-limit helpers for GitHub REST & GraphQL APIs.

Provides a unified wait strategy:
  - REST 403 with X-RateLimit-Remaining == 0 → sleep until reset
  - REST 429 → honour Retry-After header
  - REST 403 "secondary rate limit" → honour Retry-After (default 65 s)
  - GraphQL 200 with errors[].type == RATE_LIMIT → query /rate_limit for reset
  - Exponential backoff for transient failures
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger("github.ratelimit")

MAX_WAIT = 3700


def _parse_reset_wait(headers: dict, default: int = 60) -> int:
    reset_ts = int(headers.get("X-RateLimit-Reset", 0))
    if reset_ts:
        return max(reset_ts - int(time.time()) + 5, 10)
    return default


def classify_rest_response(status: int, headers: dict, body: str = "") -> str | None:
    """Return a rate-limit action string or None if the response is not rate-limited.

    Returns:
        "primary"   – primary rate limit hit (wait until reset)
        "secondary" – secondary / abuse rate limit
        "retry"     – 429 Too Many Requests
        "not_found" – 404 (skip)
        "search_cap"– 422 search limit
        None        – no rate-limit issue
    """
    if status == 422:
        return "search_cap"
    if status == 404:
        return "not_found"
    if status == 429:
        return "retry"
    if status == 403:
        remaining = int(headers.get("X-RateLimit-Remaining", 1))
        if remaining == 0:
            return "primary"
        body_lower = body.lower() if body else ""
        if any(k in body_lower for k in ("rate limit", "abuse", "secondary")):
            return "secondary"
    return None


def compute_wait(kind: str, headers: dict, attempt: int = 0) -> int:
    """Compute seconds to wait based on rate-limit kind."""
    if kind == "primary":
        return min(_parse_reset_wait(headers), MAX_WAIT)
    if kind == "secondary":
        return int(headers.get("Retry-After", 65))
    if kind == "retry":
        return int(headers.get("Retry-After", 60))
    return min(2 ** attempt * 5, 120)


def sync_wait(kind: str, headers: dict, attempt: int = 0):
    """Block the current thread for the appropriate duration."""
    wait = compute_wait(kind, headers, attempt)
    log.info("[%s] waiting %ds …", kind, wait)
    time.sleep(wait)


async def async_wait(kind: str, headers: dict, attempt: int = 0):
    """Sleep asynchronously for the appropriate duration."""
    import asyncio
    wait = compute_wait(kind, headers, attempt)
    log.info("[%s] waiting %ds …", kind, wait)
    await asyncio.sleep(wait)

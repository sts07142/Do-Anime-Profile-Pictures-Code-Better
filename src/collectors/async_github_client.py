"""Async GitHub REST API client for parallel data collection."""

import logging
import asyncio

import aiohttp

from .rate_limit import classify_rest_response, async_wait

log = logging.getLogger("github.async_client")


class AsyncGitHubClient:
    """Async wrapper around GitHub REST API with rate-limit handling."""

    BASE_URL = "https://api.github.com"

    def __init__(self, token: str, concurrency: int = 15):
        self.token = token
        self.semaphore = asyncio.Semaphore(concurrency)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"token {self.token}",
                    "Accept": "application/vnd.github.v3+json",
                }
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def get(self, endpoint: str, params: dict | None = None) -> dict | list | None:
        url = endpoint if endpoint.startswith("https://") else f"{self.BASE_URL}{endpoint}"
        session = await self._get_session()

        async with self.semaphore:
            for attempt in range(5):
                async with session.get(url, params=params) as resp:
                    headers = dict(resp.headers)
                    kind = classify_rest_response(resp.status, headers)

                    if kind in ("search_cap", "not_found"):
                        return None

                    if kind is None and resp.status == 403:
                        body = await resp.text()
                        kind = classify_rest_response(resp.status, headers, body)

                    if kind in ("primary", "secondary", "retry"):
                        await async_wait(kind, headers, attempt)
                        continue
                    if resp.status in (403, 404):
                        return None
                    resp.raise_for_status()
                    return await resp.json()
        return None

    async def get_paginated(
        self, endpoint: str, params: dict | None = None, max_pages: int = 5,
    ) -> list:
        params = params or {}
        params.setdefault("per_page", 100)
        results: list = []
        url = endpoint if endpoint.startswith("https://") else f"{self.BASE_URL}{endpoint}"
        session = await self._get_session()

        for _ in range(max_pages):
            data = None
            link_header = ""
            async with self.semaphore:
                for attempt in range(3):
                    async with session.get(url, params=params) as resp:
                        headers = dict(resp.headers)
                        kind = classify_rest_response(resp.status, headers)

                        if kind in ("primary", "secondary", "retry"):
                            await async_wait(kind, headers, attempt)
                            continue
                        if resp.status in (403, 404):
                            return results
                        resp.raise_for_status()
                        data = await resp.json()
                        link_header = resp.headers.get("Link", "")
                    break
                else:
                    break

            if not data:
                break
            results.extend(data)

            next_url = None
            if link_header:
                for part in link_header.split(","):
                    if 'rel="next"' in part:
                        next_url = part.split(";")[0].strip().strip("<>")
                        break
            if not next_url:
                break
            url = next_url
            params = {}

        return results

    async def get_user(self, username: str) -> dict | None:
        return await self.get(f"/users/{username}")

    async def get_user_repos(self, username: str, max_pages: int = 5) -> list:
        return await self.get_paginated(
            f"/users/{username}/repos",
            params={"type": "owner", "sort": "updated"},
            max_pages=max_pages,
        )

    async def get_user_starred(self, username: str, max_pages: int = 1) -> list:
        return await self.get_paginated(f"/users/{username}/starred", max_pages=max_pages)

    async def get_user_orgs(self, username: str) -> list:
        return await self.get_paginated(f"/users/{username}/orgs", max_pages=1)

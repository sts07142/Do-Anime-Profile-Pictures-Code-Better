"""GitHub REST API client with automatic rate-limit handling and pagination."""

import logging

import requests

from .rate_limit import classify_rest_response, sync_wait

log = logging.getLogger("github.client")


class GitHubClient:
    """Synchronous wrapper around GitHub REST API."""

    BASE_URL = "https://api.github.com"

    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
            }
        )

    def _handle_rate_limit(self, resp: requests.Response) -> bool:
        """Return True if we waited and should retry."""
        kind = classify_rest_response(
            resp.status_code, dict(resp.headers), resp.text,
        )
        if kind in ("primary", "secondary", "retry"):
            sync_wait(kind, dict(resp.headers))
            return True
        return False

    def get(self, endpoint: str, params: dict | None = None) -> dict | list | None:
        url = endpoint if endpoint.startswith("https://") else f"{self.BASE_URL}{endpoint}"
        for attempt in range(5):
            resp = self.session.get(url, params=params)
            if self._handle_rate_limit(resp):
                continue
            if resp.status_code in (404, 422):
                return None
            if resp.status_code == 403:
                return None
            resp.raise_for_status()
            return resp.json()
        return None

    def get_paginated(
        self, endpoint: str, params: dict | None = None, max_pages: int = 5,
    ) -> list:
        params = params or {}
        params.setdefault("per_page", 100)
        results: list = []
        url = endpoint if endpoint.startswith("https://") else f"{self.BASE_URL}{endpoint}"

        for _ in range(max_pages):
            for _ in range(3):
                resp = self.session.get(url, params=params)
                if self._handle_rate_limit(resp):
                    continue
                break
            else:
                break

            if resp.status_code == 404:
                break
            resp.raise_for_status()

            data = resp.json()
            if not data:
                break
            results.extend(data)

            next_url = resp.links.get("next", {}).get("url")
            if not next_url:
                break
            url = next_url
            params = {}

        return results

    def get_rate_limit(self) -> dict:
        resp = self.session.get(f"{self.BASE_URL}/rate_limit")
        resp.raise_for_status()
        core = resp.json()["resources"]["core"]
        return {
            "remaining": core["remaining"],
            "limit": core["limit"],
            "reset": core["reset"],
        }

    # ── Convenience methods ──────────────────────────────────

    def search_repos(self, query: str, per_page: int = 30, page: int = 1) -> dict:
        return self.get(
            "/search/repositories",
            params={"q": query, "per_page": per_page, "page": page, "sort": "stars"},
        )

    def search_users(self, query: str, per_page: int = 30, page: int = 1) -> dict:
        return self.get(
            "/search/users",
            params={"q": query, "per_page": per_page, "page": page, "sort": "joined"},
        )

    def get_user(self, username: str) -> dict | None:
        return self.get(f"/users/{username}")

    def get_user_repos(self, username: str, max_pages: int = 5) -> list:
        return self.get_paginated(
            f"/users/{username}/repos",
            params={"type": "owner", "sort": "updated"},
            max_pages=max_pages,
        )

    def get_user_starred(self, username: str, max_pages: int = 3) -> list:
        return self.get_paginated(f"/users/{username}/starred", max_pages=max_pages)

    def get_user_orgs(self, username: str) -> list:
        return self.get_paginated(f"/users/{username}/orgs", max_pages=1)

    def get_repo_contributors(self, owner: str, repo: str, max_pages: int = 3) -> list:
        return self.get_paginated(
            f"/repos/{owner}/{repo}/contributors", max_pages=max_pages,
        )

    def get_org_members(self, org: str, max_pages: int = 3) -> list:
        return self.get_paginated(f"/orgs/{org}/members", max_pages=max_pages)

"""Async enricher - parallel user enrichment using asyncio."""

import asyncio
import json
from pathlib import Path
from collections import Counter

from tqdm import tqdm

from .async_github_client import AsyncGitHubClient


def _load_json(path: Path) -> dict | list:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _top_n(counter: Counter, n: int = 3) -> list[dict]:
    total = sum(counter.values())
    if total == 0:
        return []
    return [
        {"name": name, "count": count, "ratio": round(count / total, 3)}
        for name, count in counter.most_common(n)
    ]


class AsyncUserEnricher:
    """Parallel enrichment using asyncio."""

    def __init__(self, token: str, data_dir: Path, concurrency: int = 15):
        self.token = token
        self.data_dir = data_dir
        self.concurrency = concurrency
        self.sampled_path = data_dir / "raw" / "sampled_users.json"
        self.enriched_path = data_dir / "raw" / "enriched_users.json"
        self.sampled = _load_json(self.sampled_path)
        self.enriched = _load_json(self.enriched_path)

    def _build_layer1(self, user_data: dict, group: str) -> dict:
        return {
            "user_id": user_data["id"],
            "username": user_data["login"],
            "avatar_url": user_data["avatar_url"],
            "sampling_group": group,
            "created_at": user_data["created_at"],
            "followers": user_data["followers"],
            "following": user_data["following"],
            "public_repos": user_data["public_repos"],
            "company": user_data.get("company"),
            "bio": user_data.get("bio"),
            "location": user_data.get("location"),
        }

    def _build_repo_profile(self, repos: list) -> dict:
        own_repos = [r for r in repos if not r.get("fork", False)]

        lang_counter = Counter()
        topic_counter = Counter()
        total_stars = 0
        total_forks = 0
        latest_push = None

        for repo in own_repos:
            lang = repo.get("language")
            if lang:
                lang_counter[lang] += 1
            for topic in repo.get("topics", []):
                topic_counter[topic] += 1
            total_stars += repo.get("stargazers_count", 0)
            total_forks += repo.get("forks_count", 0)
            pushed = repo.get("pushed_at")
            if pushed and (latest_push is None or pushed > latest_push):
                latest_push = pushed

        return {
            "primary_languages": _top_n(lang_counter),
            "repo_topics": _top_n(topic_counter, n=10),
            "repo_count": len(own_repos),
            "total_stars_received": total_stars,
            "total_forks_received": total_forks,
            "latest_push": latest_push,
        }

    def _build_starred_profile(self, starred: list) -> dict:
        lang_counter = Counter()
        topic_counter = Counter()
        for repo in starred:
            lang = repo.get("language")
            if lang:
                lang_counter[lang] += 1
            for topic in repo.get("topics", []):
                topic_counter[topic] += 1
        return {
            "starred_languages": _top_n(lang_counter, n=5),
            "starred_topics": _top_n(topic_counter, n=10),
            "starred_count": len(starred),
        }

    def _build_org_profile(self, orgs: list) -> list[dict]:
        return [{"login": org["login"], "id": org["id"]} for org in orgs]

    def _compute_activity_grade(self, created_at: str, repo_profile: dict) -> str:
        if created_at and created_at >= "2026-01-01":
            return "new"
        latest = repo_profile.get("latest_push")
        repo_count = repo_profile.get("repo_count", 0)
        if repo_count == 0:
            return "dormant"
        if latest and latest < "2025-03-15":
            return "dormant"
        total_stars = repo_profile.get("total_stars_received", 0)
        if repo_count >= 50 or (repo_count >= 10 and total_stars >= 100):
            return "high"
        if latest and latest >= "2025-09-15":
            return "active"
        return "low"

    async def _enrich_one(
        self, client: AsyncGitHubClient, user_id: str, username: str, group: str
    ) -> tuple[str, dict | None]:
        """Enrich a single user. Returns (user_id, result_or_None)."""
        try:
            user_data = await client.get_user(username)
            if not user_data:
                return user_id, None

            layer1 = self._build_layer1(user_data, group)

            # Fetch repos, starred, orgs in parallel for this user
            repos, starred, orgs = await asyncio.gather(
                client.get_user_repos(username, max_pages=5),
                client.get_user_starred(username, max_pages=1),
                client.get_user_orgs(username),
            )

            repo_profile = self._build_repo_profile(repos)
            starred_profile = self._build_starred_profile(starred)
            org_profile = self._build_org_profile(orgs)
            activity_grade = self._compute_activity_grade(
                layer1["created_at"], repo_profile
            )

            return user_id, {
                **layer1,
                "activity_grade": activity_grade,
                "repos": repo_profile,
                "starred": starred_profile,
                "orgs": org_profile,
            }
        except Exception as e:
            print(f"[Error] {username}: {e}")
            return user_id, None

    async def enrich_all(self, limit: int | None = None):
        to_process = [
            (uid, info)
            for uid, info in self.sampled.items()
            if uid not in self.enriched
        ]
        if limit:
            to_process = to_process[:limit]

        print(f"Enriching {len(to_process)} users ({len(self.enriched)} already done), concurrency={self.concurrency}")

        client = AsyncGitHubClient(self.token, concurrency=self.concurrency)
        pbar = tqdm(total=len(to_process), desc="Enriching")
        save_interval = 100
        completed = 0

        # Process in batches to allow periodic saving
        batch_size = save_interval
        for i in range(0, len(to_process), batch_size):
            batch = to_process[i : i + batch_size]
            tasks = [
                self._enrich_one(client, uid, info["username"], info["group"])
                for uid, info in batch
            ]

            results = await asyncio.gather(*tasks)

            for uid, result in results:
                if result:
                    self.enriched[uid] = result
                completed += 1

            pbar.update(len(batch))
            self.save()

        pbar.close()
        await client.close()
        print(f"Done. Total enriched: {len(self.enriched)}")

    def save(self):
        _save_json(self.enriched, self.enriched_path)

    def summary(self) -> dict:
        grades = Counter(u.get("activity_grade", "unknown") for u in self.enriched.values())
        groups = Counter(u.get("sampling_group", "unknown") for u in self.enriched.values())
        return {"total": len(self.enriched), "by_grade": dict(grades), "by_group": dict(groups)}

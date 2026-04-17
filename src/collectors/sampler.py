"""Stratified sampling of GitHub users according to 6 sampling groups."""

import json
import random
from pathlib import Path

from .github_client import GitHubClient

# 8 target languages for language community stratum
LANGUAGES = ["Python", "JavaScript", "Go", "Rust", "Java", "TypeScript", "C++", "Kotlin"]

# Priority order for dedup (higher index = lower priority)
GROUP_PRIORITY = [
    "org_member",
    "popular_oss",
    "language_community",
    "new_user",
    "long_term_user",
    "general_active",
]

DEFAULT_PER_GROUP = 1_700
# GitHub Search API returns max 1000 results (10 pages x 100 per page)
MAX_SEARCH_PAGES = 10


def _save_progress(users: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def _load_progress(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


class StratifiedSampler:
    """Collects GitHub usernames into 6 sampling groups with dedup."""

    def __init__(self, client: GitHubClient, data_dir: Path, per_group: int = DEFAULT_PER_GROUP):
        self.client = client
        self.target_per_group = per_group
        self.progress_path = data_dir / "raw" / "sampled_users.json"
        # {user_id: {"username": ..., "group": ...}}
        self.users = _load_progress(self.progress_path)
        self._seen_ids: set[str] = {uid for uid in self.users}

    def _add_user(self, user_id: int, username: str, group: str) -> bool:
        """Add user if not already seen. Returns True if added."""
        uid = str(user_id)
        if uid in self._seen_ids:
            return False
        self._seen_ids.add(uid)
        self.users[uid] = {"username": username, "group": group}
        return True

    def _group_count(self, group: str) -> int:
        return sum(1 for u in self.users.values() if u["group"] == group)

    def _group_full(self, group: str) -> bool:
        return self._group_count(group) >= self.target_per_group

    def save(self):
        _save_progress(self.users, self.progress_path)

    def collect_popular_oss(self):
        """Stratum 1: Contributors of popular repos (star>=1000, pushed in 2025)."""
        group = "popular_oss"
        if self._group_full(group):
            print(f"[{group}] Already full ({self._group_count(group)})")
            return

        print(f"[{group}] Collecting...")
        # Search for popular repos
        for page in range(1, 11):  # up to 10 pages of repos
            if self._group_full(group):
                break
            result = self.client.search_repos(
                "stars:>1000 pushed:>2025-01-01", per_page=30, page=page
            )
            if not result or "items" not in result:
                break

            for repo in result["items"]:
                if self._group_full(group):
                    break
                owner = repo["owner"]["login"]
                name = repo["name"]
                contributors = self.client.get_repo_contributors(owner, name, max_pages=2)
                for user in contributors:
                    if self._group_full(group):
                        break
                    self._add_user(user["id"], user["login"], group)

            self.save()
            print(f"  [{group}] {self._group_count(group)}/{self.target_per_group}")

    def collect_language_community(self):
        """Stratum 2: Top repo contributors per language."""
        group = "language_community"
        if self._group_full(group):
            print(f"[{group}] Already full ({self._group_count(group)})")
            return

        print(f"[{group}] Collecting...")
        per_lang = self.target_per_group // len(LANGUAGES)  # ~212 per language

        for lang in LANGUAGES:
            if self._group_full(group):
                break
            lang_count = 0
            for page in range(1, 6):
                if lang_count >= per_lang or self._group_full(group):
                    break
                result = self.client.search_repos(
                    f"language:{lang} stars:>100 pushed:>2025-01-01",
                    per_page=10, page=page,
                )
                if not result or "items" not in result:
                    break

                for repo in result["items"]:
                    if lang_count >= per_lang:
                        break
                    owner = repo["owner"]["login"]
                    name = repo["name"]
                    contributors = self.client.get_repo_contributors(
                        owner, name, max_pages=1
                    )
                    for user in contributors:
                        if lang_count >= per_lang or self._group_full(group):
                            break
                        if self._add_user(user["id"], user["login"], group):
                            lang_count += 1

            self.save()
            print(f"  [{group}] {lang}: +{lang_count}, total {self._group_count(group)}/{self.target_per_group}")

    def collect_general_active(self):
        """Stratum 3: General active users. Split by follower range to avoid dedup overlap."""
        group = "general_active"
        if self._group_full(group):
            print(f"[{group}] Already full ({self._group_count(group)})")
            return

        print(f"[{group}] Collecting...")
        # Split by follower ranges to get diverse users not already captured
        follower_ranges = [
            "followers:0..5",
            "followers:6..20",
            "followers:21..50",
            "followers:51..200",
            "followers:>200",
        ]
        for f_range in follower_ranges:
            if self._group_full(group):
                break
            page = 1
            while not self._group_full(group) and page <= MAX_SEARCH_PAGES:
                result = self.client.search_users(
                    f"type:user repos:>3 {f_range}",
                    per_page=100, page=page,
                )
                if not result or "items" not in result:
                    break

                for user in result["items"]:
                    if self._group_full(group):
                        break
                    self._add_user(user["id"], user["login"], group)

                self.save()
                print(f"  [{group}] {f_range} p{page}: {self._group_count(group)}/{self.target_per_group}")
                page += 1

    def collect_new_users(self):
        """Stratum 4: Users created after 2025-01-01."""
        group = "new_user"
        if self._group_full(group):
            print(f"[{group}] Already full ({self._group_count(group)})")
            return

        print(f"[{group}] Collecting...")
        page = 1
        while not self._group_full(group) and page <= MAX_SEARCH_PAGES:
            result = self.client.search_users(
                "type:user created:>2025-01-01",
                per_page=100, page=page,
            )
            if not result or "items" not in result:
                break

            for user in result["items"]:
                if self._group_full(group):
                    break
                self._add_user(user["id"], user["login"], group)

            self.save()
            print(f"  [{group}] {self._group_count(group)}/{self.target_per_group}")
            page += 1

    def collect_long_term_users(self):
        """Stratum 5: Users created before 2015-01-01. Split by year range to bypass 1000 result limit."""
        group = "long_term_user"
        if self._group_full(group):
            print(f"[{group}] Already full ({self._group_count(group)})")
            return

        print(f"[{group}] Collecting...")
        # Split into year ranges to get more than 1000 results
        year_ranges = [
            "2008-01-01..2009-12-31",
            "2010-01-01..2011-12-31",
            "2012-01-01..2013-12-31",
            "2014-01-01..2014-12-31",
        ]
        for year_range in year_ranges:
            if self._group_full(group):
                break
            page = 1
            while not self._group_full(group) and page <= MAX_SEARCH_PAGES:
                result = self.client.search_users(
                    f"type:user created:{year_range} repos:>0",
                    per_page=100, page=page,
                )
                if not result or "items" not in result:
                    break

                for user in result["items"]:
                    if self._group_full(group):
                        break
                    self._add_user(user["id"], user["login"], group)

                self.save()
                print(f"  [{group}] {year_range} p{page}: {self._group_count(group)}/{self.target_per_group}")
                page += 1
            print(f"  [{group}] {self._group_count(group)}/{self.target_per_group}")

    def collect_org_members(self):
        """Stratum 6: Organization members by org size."""
        group = "org_member"
        if self._group_full(group):
            print(f"[{group}] Already full ({self._group_count(group)})")
            return

        print(f"[{group}] Collecting...")
        # Sample orgs by size tier (~425 users each)
        org_tiers = {
            "large": ["google", "microsoft", "facebook", "amazon", "apple"],
            "mid": ["vercel", "supabase", "toss", "hashicorp", "grafana"],
            "small": ["anthropics", "langchain-ai", "pydantic", "astral-sh", "ruff-lang"],
            "non_corp": ["rust-lang", "golang", "python", "apache", "mozilla"],
        }
        per_tier = self.target_per_group // len(org_tiers)  # ~425

        for tier_name, orgs in org_tiers.items():
            if self._group_full(group):
                break
            tier_count = 0
            random.shuffle(orgs)
            for org_name in orgs:
                if tier_count >= per_tier or self._group_full(group):
                    break
                members = self.client.get_org_members(org_name, max_pages=3)
                for user in members:
                    if tier_count >= per_tier or self._group_full(group):
                        break
                    if self._add_user(user["id"], user["login"], group):
                        tier_count += 1

            self.save()
            print(f"  [{group}] {tier_name}: +{tier_count}, total {self._group_count(group)}/{self.target_per_group}")

    def collect_all(self):
        """Run all sampling strata in priority order."""
        # Collect in priority order (hardest to get first)
        self.collect_org_members()
        self.collect_popular_oss()
        self.collect_language_community()
        self.collect_new_users()
        self.collect_long_term_users()
        self.collect_general_active()

        print("\n=== Sampling Summary ===")
        for group in GROUP_PRIORITY:
            print(f"  {group}: {self._group_count(group)}")
        print(f"  Total: {len(self.users)}")

    def summary(self) -> dict:
        counts = {g: self._group_count(g) for g in GROUP_PRIORITY}
        counts["total"] = len(self.users)
        return counts

"""Stratified sampling of GitHub users according to 6 sampling groups."""

import json
import random
from pathlib import Path

from .github_client import GitHubClient
from . import trendshift_scraper

# 8 target languages for language community stratum
LANGUAGES = ["Python", "JavaScript", "Go", "Rust", "Java", "TypeScript", "C++", "Kotlin"]

# Priority order for dedup (higher index = lower priority).
# ml_ai and trending are placed first so users with that identity
# are not absorbed into the broader popular_oss / language_community
# pools when they overlap.
GROUP_PRIORITY = [
    "ml_ai",
    "trending",
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

# ML/AI sampling: fixed seed repos per sub-tier.
# Sub-tiers split the per-group budget evenly when collected.
ML_AI_TIERS: dict[str, list[str]] = {
    "core_ml": [
        "huggingface/transformers",
        "pytorch/pytorch",
        "tensorflow/tensorflow",
        "keras-team/keras",
        "scikit-learn/scikit-learn",
        "jax-ml/jax",
    ],
    "llm": [
        "langchain-ai/langchain",
        "ggerganov/llama.cpp",
        "hiyouga/LLaMA-Factory",
        "vllm-project/vllm",
        "ollama/ollama",
        "huggingface/text-generation-inference",
    ],
    "generative": [
        "Stability-AI/stablediffusion",
        "comfyanonymous/ComfyUI",
        "AUTOMATIC1111/stable-diffusion-webui",
        "lucidrains/imagen-pytorch",
    ],
}

# Trending sampling: non-AI topic slugs to scrape from trendshift.
# AI-related topics are intentionally excluded — those contributors are
# claimed by ml_ai (which has higher dedup priority), so scraping them
# would waste HTTP calls.
TRENDING_TOPIC_WHITELIST: list[str] = [
    "fintech",
    "self-hosted",
    "workflow-automation",
    "devtools",
    "database",
    "game-dev",
    "web-framework",
    "crypto",
    "iot",
    "mobile-app",
]


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
        self.trendshift_snapshot_dir = data_dir / "raw" / "trendshift_snapshots"
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
            "followers:0..2",
            "followers:3..5",
            "followers:6..10",
            "followers:11..20",
            "followers:21..50",
            "followers:51..100",
            "followers:101..200",
            "followers:201..500",
            "followers:501..1000",
            "followers:>1000",
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
        """Stratum 4: Users created after 2025-01-01.

        Split by quarter to bypass the 1000-results-per-query Search API cap.
        """
        group = "new_user"
        if self._group_full(group):
            print(f"[{group}] Already full ({self._group_count(group)})")
            return

        print(f"[{group}] Collecting...")
        date_ranges = [
            "2025-01-01..2025-03-31",
            "2025-04-01..2025-06-30",
            "2025-07-01..2025-09-30",
            "2025-10-01..2025-12-31",
            "2026-01-01..2026-03-31",
            ">2026-04-01",
        ]
        for d_range in date_ranges:
            if self._group_full(group):
                break
            page = 1
            while not self._group_full(group) and page <= MAX_SEARCH_PAGES:
                result = self.client.search_users(
                    f"type:user created:{d_range}",
                    per_page=100, page=page,
                )
                if not result or "items" not in result:
                    break

                for user in result["items"]:
                    if self._group_full(group):
                        break
                    self._add_user(user["id"], user["login"], group)

                self.save()
                print(f"  [{group}] {d_range} p{page}: {self._group_count(group)}/{self.target_per_group}")
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
            "2008-01-01..2008-12-31",
            "2009-01-01..2009-12-31",
            "2010-01-01..2010-12-31",
            "2011-01-01..2011-12-31",
            "2012-01-01..2012-12-31",
            "2013-01-01..2013-12-31",
            "2014-01-01..2014-06-30",
            "2014-07-01..2014-12-31",
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
            "large": ["google", "microsoft", "facebook", "amazon", "apple",
                      "netflix", "uber", "airbnb", "twitter", "linkedin"],
            "mid": ["vercel", "supabase", "toss", "hashicorp", "grafana",
                    "stripe", "shopify", "datadog", "cloudflare", "figma"],
            "small": ["anthropics", "langchain-ai", "pydantic", "astral-sh",
                      "ruff-lang", "huggingface", "ollama", "modal-labs",
                      "denoland", "bunjs"],
            "non_corp": ["rust-lang", "golang", "python", "apache", "mozilla",
                         "kubernetes", "tensorflow", "pytorch", "nodejs", "openssl"],
        }
        per_tier = self.target_per_group // len(org_tiers)

        for tier_name, orgs in org_tiers.items():
            if self._group_full(group):
                break
            tier_count = 0
            random.shuffle(orgs)
            for org_name in orgs:
                if tier_count >= per_tier or self._group_full(group):
                    break
                members = self.client.get_org_members(org_name, max_pages=5)
                for user in members:
                    if tier_count >= per_tier or self._group_full(group):
                        break
                    if self._add_user(user["id"], user["login"], group):
                        tier_count += 1

            self.save()
            print(f"  [{group}] {tier_name}: +{tier_count}, total {self._group_count(group)}/{self.target_per_group}")

    def collect_ml_ai(self):
        """Stratum 7: Contributors of core ML / LLM / generative-AI repos.

        Sub-tiers (core_ml, llm, generative) split the per_group budget evenly.
        Note: when re-running against an existing sampled_users.json, users
        already labeled with another group are NOT re-classified into ml_ai —
        only newly-seen users go in. Delete sampled_users.json for a clean run.
        """
        group = "ml_ai"
        if self._group_full(group):
            print(f"[{group}] Already full ({self._group_count(group)})")
            return

        print(f"[{group}] Collecting...")
        per_tier = self.target_per_group // len(ML_AI_TIERS)

        for tier_name, repos in ML_AI_TIERS.items():
            if self._group_full(group):
                break
            tier_count = 0
            for repo_full in repos:
                if tier_count >= per_tier or self._group_full(group):
                    break
                owner, name = repo_full.split("/", 1)
                contributors = self.client.get_repo_contributors(
                    owner, name, max_pages=5,
                )
                for user in contributors:
                    if tier_count >= per_tier or self._group_full(group):
                        break
                    if self._add_user(user["id"], user["login"], group):
                        tier_count += 1

            self.save()
            print(
                f"  [{group}] {tier_name}: +{tier_count}, "
                f"total {self._group_count(group)}/{self.target_per_group}",
            )

    def collect_trending(self):
        """Stratum 8: Contributors of trendshift.io trending repos.

        Sub-tiers: daily_main (front page), all_time (history page),
        topics (non-AI topic pages). Each fetched repo list is also
        written to data/raw/trendshift_snapshots/ for reproducibility.

        trendshift has no official API; if its HTML structure changes
        the fetch_* helpers return [] and this method falls through
        cleanly without affecting other strata.
        """
        group = "trending"
        if self._group_full(group):
            print(f"[{group}] Already full ({self._group_count(group)})")
            return

        print(f"[{group}] Collecting...")
        per_subtier = self.target_per_group // 3

        sub_sources: list[tuple[str, list[dict]]] = []

        main_repos = trendshift_scraper.fetch_main_trending()
        if main_repos:
            trendshift_scraper.write_snapshot(
                self.trendshift_snapshot_dir, "daily_main", main_repos,
            )
        sub_sources.append(("daily_main", main_repos))

        alltime_repos = trendshift_scraper.fetch_all_time_trending()
        if alltime_repos:
            trendshift_scraper.write_snapshot(
                self.trendshift_snapshot_dir, "all_time", alltime_repos,
            )
        sub_sources.append(("all_time", alltime_repos))

        topic_repos: list[dict] = []
        for topic in TRENDING_TOPIC_WHITELIST:
            page = trendshift_scraper.fetch_topic_trending(topic)
            if page:
                trendshift_scraper.write_snapshot(
                    self.trendshift_snapshot_dir, f"topic:{topic}", page,
                )
            topic_repos.extend(page)
        sub_sources.append(("topics", topic_repos))

        for subtier, repos in sub_sources:
            if self._group_full(group):
                break
            sub_count = 0
            for repo in repos:
                if sub_count >= per_subtier or self._group_full(group):
                    break
                contributors = self.client.get_repo_contributors(
                    repo["owner"], repo["name"], max_pages=4,
                )
                for user in contributors:
                    if sub_count >= per_subtier or self._group_full(group):
                        break
                    if self._add_user(user["id"], user["login"], group):
                        sub_count += 1

            self.save()
            print(
                f"  [{group}] {subtier}: +{sub_count}, "
                f"total {self._group_count(group)}/{self.target_per_group}",
            )

    def collect_all(self):
        """Run all sampling strata in priority order."""
        self.collect_ml_ai()
        self.collect_trending()
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

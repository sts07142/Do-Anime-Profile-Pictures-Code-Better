"""Scrapes trendshift.io for trending GitHub repositories.

trendshift has no official API. The site is a Next.js / React Server
Components app, so the actual trending repo data is embedded as escaped
JSON inside <script>self.__next_f.push(...)</script> payloads (the
`full_name` field). A few sponsored repos appear as plain anchor tags.
This module parses both signals; the patterns below are the single
point of failure if trendshift changes its markup.
"""

import re

from bs4 import BeautifulSoup

# Escaped JSON pattern as it appears in the RSC payload:
#   \"full_name\":\"owner/repo\"
# Char class excludes "  \  /  so we stop cleanly at the next \".
RSC_FULL_NAME_RE = re.compile(
    r'\\"full_name\\":\\"([^"\\/]+)/([^"\\/]+)\\"'
)

# Plain anchor link to a GitHub repo.
GITHUB_LINK_RE = re.compile(r"^https?://github\.com/([^/?#]+)/([^/?#]+)/?$")

# Trendshift's own GitHub identities — exclude from results.
TRENDSHIFT_OWN_OWNERS = {"trendshift-labs", "liweiyi88"}


def parse_repo_links(html: str) -> list[dict]:
    """Parse trendshift HTML and return [{owner, name}, ...].

    Tries two paths in order:
      1) RSC JSON payload (the bulk of real trending data).
      2) Plain GitHub anchor tags (sponsored / fallback).
    Deduplicates while preserving discovery order.
    """
    seen: set[tuple[str, str]] = set()
    repos: list[dict] = []

    for m in RSC_FULL_NAME_RE.finditer(html):
        owner, name = m.group(1), m.group(2)
        if owner.lower() in TRENDSHIFT_OWN_OWNERS:
            continue
        key = (owner.lower(), name.lower())
        if key in seen:
            continue
        seen.add(key)
        repos.append({"owner": owner, "name": name})

    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        m = GITHUB_LINK_RE.match(a["href"])
        if not m:
            continue
        owner, name = m.group(1), m.group(2)
        if owner.lower() in TRENDSHIFT_OWN_OWNERS:
            continue
        key = (owner.lower(), name.lower())
        if key in seen:
            continue
        seen.add(key)
        repos.append({"owner": owner, "name": name})

    return repos


import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

log = logging.getLogger("trendshift.scraper")

USER_AGENT = (
    "do-anime-pfp-research "
    "(https://github.com/sts07142/Do-Anime-Profile-Pictures-Code-Better)"
)
REQUEST_TIMEOUT = 15
SLEEP_BETWEEN_REQUESTS = 1.0

MAIN_URL = "https://trendshift.io/"
ALL_TIME_URL = "https://trendshift.io/github-trending-repositories"
TOPIC_URL_TEMPLATE = "https://trendshift.io/topics/{topic}"


def _fetch_html(url: str) -> str | None:
    """GET url with our User-Agent. Returns HTML text or None on error."""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        log.warning("trendshift fetch failed: %s (%s)", url, e)
        return None
    finally:
        time.sleep(SLEEP_BETWEEN_REQUESTS)


def _fetch_and_parse(url: str, source_label: str) -> list[dict]:
    html = _fetch_html(url)
    if not html:
        return []
    try:
        repos = parse_repo_links(html)
    except Exception as e:
        log.warning("trendshift parse failed for %s: %s", source_label, e)
        return []
    for r in repos:
        r["source"] = source_label
    return repos


def fetch_main_trending() -> list[dict]:
    return _fetch_and_parse(MAIN_URL, "daily_main")


def fetch_all_time_trending() -> list[dict]:
    return _fetch_and_parse(ALL_TIME_URL, "all_time")


def fetch_topic_trending(topic: str) -> list[dict]:
    url = TOPIC_URL_TEMPLATE.format(topic=topic)
    repos = _fetch_and_parse(url, f"topic:{topic}")
    for r in repos:
        r["topic"] = topic
    return repos


_FILENAME_SAFE_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def write_snapshot(
    snapshot_dir: Path, source: str, repos: list[dict],
) -> Path:
    """Write a snapshot JSON file. Creates dir if missing. Returns the path."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    fetched_at = now.isoformat().replace("+00:00", "Z")
    safe_label = _FILENAME_SAFE_RE.sub("_", source)[:40]
    filename = f"{now.strftime('%Y%m%d-%H%M%S')}_{safe_label}.json"
    path = snapshot_dir / filename
    payload = {"fetched_at": fetched_at, "source": source, "repos": repos}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return path

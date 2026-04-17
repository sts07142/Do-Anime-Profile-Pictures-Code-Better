"""GitHub GraphQL — contributionsCollection collector.

Collects each user's last-year commit/PR/issue/review contribution counts
and saves to data/raw/contributions.json.

Resumable: already-collected users are skipped on restart.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm_asyncio

from .rate_limit import MAX_WAIT

log = logging.getLogger("github.contributions")

GRAPHQL_URL = "https://api.github.com/graphql"

QUERY = """
query($login: String!) {
  user(login: $login) {
    contributionsCollection {
      totalCommitContributions
      totalPullRequestContributions
      totalIssueContributions
      totalPullRequestReviewContributions
      restrictedContributionsCount
      contributionCalendar { totalContributions }
    }
  }
}
"""


async def _wait_for_reset(session: aiohttp.ClientSession):
    """Query REST /rate_limit to find the GraphQL reset timestamp, then sleep."""
    try:
        async with session.get('https://api.github.com/rate_limit',
                                timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                data = await resp.json()
                reset = data['resources']['graphql']['reset']
                wait = max(10, reset - int(time.time()) + 5)
                log.info("[graphql] waiting %ds until reset…", wait)
                await asyncio.sleep(min(wait, MAX_WAIT))
                return
    except Exception:
        pass
    await asyncio.sleep(60)


async def fetch_one(session: aiohttp.ClientSession, login: str, uid: str,
                     semaphore: asyncio.Semaphore, max_retries: int = 3):
    payload = {"query": QUERY, "variables": {"login": login}}
    for attempt in range(max_retries):
        async with semaphore:
            try:
                async with session.post(GRAPHQL_URL, json=payload,
                                          timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    # REST-style 403/429
                    if resp.status in (403, 429):
                        await _wait_for_reset(session)
                        continue
                    data = await resp.json()
                    # GraphQL rate limit (returns 200 with errors.RATE_LIMIT)
                    errors = data.get('errors', [])
                    if errors and any(e.get('type') == 'RATE_LIMIT' for e in errors):
                        await _wait_for_reset(session)
                        continue
                    if errors:
                        # NOT_FOUND or other — skip this user
                        return uid, None
                    user = data.get('data', {}).get('user')
                    if not user:
                        return uid, None
                    cc = user['contributionsCollection']
                    return uid, {
                        'commits': cc.get('totalCommitContributions', 0),
                        'prs': cc.get('totalPullRequestContributions', 0),
                        'issues': cc.get('totalIssueContributions', 0),
                        'reviews': cc.get('totalPullRequestReviewContributions', 0),
                        'restricted': cc.get('restrictedContributionsCount', 0),
                        'total': cc.get('contributionCalendar', {}).get('totalContributions', 0),
                    }
            except Exception:
                await asyncio.sleep(2 ** attempt)
    return uid, None


async def collect_all(token: str, enriched_path: Path, output_path: Path,
                       concurrency: int = 10):
    with open(enriched_path, encoding='utf-8') as f:
        enriched = json.load(f)

    if output_path.exists():
        with open(output_path) as f:
            done = json.load(f)
    else:
        done = {}

    targets = [(uid, info['username']) for uid, info in enriched.items()
               if uid not in done and info.get('username')]
    print(f"Total {len(enriched)} users | already collected {len(done)} | remaining {len(targets)}")

    if not targets:
        print("All done.")
        return done

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v4+json",
    }
    semaphore = asyncio.Semaphore(concurrency)

    batch = 200  # periodic save interval
    async with aiohttp.ClientSession(headers=headers) as session:
        for i in range(0, len(targets), batch):
            chunk = targets[i:i + batch]
            coros = [fetch_one(session, login, uid, semaphore) for uid, login in chunk]
            results = await tqdm_asyncio.gather(*coros, desc=f'{i+len(chunk)}/{len(targets)}')
            for uid, data in results:
                if data is not None:
                    done[uid] = data
            # periodic checkpoint
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(done, f, ensure_ascii=False)
            print(f"  progress saved: {len(done)} users")

    return done

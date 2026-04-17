"""Async avatar image downloader."""

import asyncio
import json
from pathlib import Path

import aiohttp
from tqdm import tqdm

DEFAULT_PRE_CLASSIFIED_PATH = (
    Path(__file__).parent.parent.parent / "data" / "processed" / "pre_classified.json"
)


async def download_avatars(
    enriched_path: Path,
    output_dir: Path,
    concurrency: int = 30,
    size: int = 640,
    pre_classified_path: Path = DEFAULT_PRE_CLASSIFIED_PATH,
):
    """Download avatar images for all enriched users.

    Args:
        enriched_path: Path to enriched_users.json
        output_dir: Directory to save images (data/raw/avatars/)
        concurrency: Max parallel downloads
        size: Image size (GitHub supports ?s=N)
        pre_classified_path: Path to pre_classified.json; uids marked
            "default_avatar" are skipped. If the file does not exist,
            prefilter is run automatically to generate it.
    """
    with open(enriched_path) as f:
        users = json.load(f)

    if not pre_classified_path.exists():
        print(f"No pre-classified file at {pre_classified_path}; running prefilter first")
        from .prefilter import run_prefilter
        await run_prefilter(enriched_path, pre_classified_path)

    with open(pre_classified_path) as f:
        pre_classified = json.load(f)
    skip_uids = {uid for uid, label in pre_classified.items()
                 if label == "default_avatar"}
    print(f"Skipping {len(skip_uids)} pre-classified default avatars")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out already downloaded and pre-classified defaults
    to_download = []
    for uid, info in users.items():
        if uid in skip_uids:
            continue
        img_path = output_dir / f"{uid}.png"
        if not img_path.exists():
            url = info["avatar_url"]
            # Append size parameter
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}s={size}"
            to_download.append((uid, url, img_path))

    already = len(users) - len(to_download) - len(skip_uids)
    print(f"Downloading {len(to_download)} avatars ({already} already exist)")

    if not to_download:
        return

    semaphore = asyncio.Semaphore(concurrency)
    failed = []

    async with aiohttp.ClientSession() as session:
        pbar = tqdm(total=len(to_download), desc="Downloading")

        async def _download_one(uid: str, url: str, path: Path):
            async with semaphore:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status == 200:
                            data = await resp.read()
                            path.write_bytes(data)
                        else:
                            failed.append((uid, resp.status))
                except Exception as e:
                    failed.append((uid, str(e)))
                finally:
                    pbar.update(1)

        tasks = [_download_one(uid, url, path) for uid, url, path in to_download]
        await asyncio.gather(*tasks)
        pbar.close()

    if failed:
        print(f"Failed: {len(failed)} downloads")
    print(f"Done. Total images: {len(list(output_dir.glob('*.png')))}")

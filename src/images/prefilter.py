"""Pre-filter default avatars before YOLOv8 classification.

Two-step detection:
  1. Gravatar URL  -> "default_avatar" (instant, no download)
  2. Identicon check via pixel analysis (tiny image download, s=40)
     - GitHub identicons: 5x5 symmetric grid, <= 3 distinct colors at 10x10

Saves data/processed/pre_classified.json:
  {uid: "default_avatar"} for detected defaults
  -> remaining uids go to YOLOv8 anime face detection
"""

import asyncio
import json
from io import BytesIO
from pathlib import Path

import aiohttp
import numpy as np
from PIL import Image
from tqdm import tqdm


def _is_gravatar(url: str) -> bool:
    return "gravatar.com" in url


def _is_identicon(img_bytes: bytes) -> bool:
    """Return True if the image looks like a GitHub identicon."""
    try:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        # Resize to 10x10 with NEAREST (no anti-aliasing) to count raw colors
        small = img.resize((10, 10), Image.NEAREST)
        arr = np.array(small)  # (10, 10, 3)
        unique_colors = len({tuple(px) for px in arr.reshape(-1, 3)})
        if unique_colors > 4:
            return False

        # Check left-right symmetry on a 70x70 grid
        mid = img.resize((70, 70), Image.NEAREST)
        a = np.array(mid)
        left = a[:, :35]
        right = a[:, 35:][:, ::-1]
        symmetric = np.mean(np.abs(left.astype(int) - right.astype(int))) < 8
        return symmetric
    except Exception:
        return False


async def _fetch_tiny(session: aiohttp.ClientSession, uid: str, url: str,
                      semaphore: asyncio.Semaphore) -> tuple[str, bytes | None]:
    sized_url = url + ("&" if "?" in url else "?") + "s=40"
    async with semaphore:
        try:
            async with session.get(sized_url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    return uid, await resp.read()
                return uid, None
        except Exception:
            return uid, None


async def run_prefilter(
    enriched_path: Path,
    output_path: Path,
    concurrency: int = 50,
    avatar_dir: Path | None = None,
):
    with open(enriched_path, encoding="utf-8") as f:
        users = json.load(f)

    # Load existing progress
    if output_path.exists():
        with open(output_path) as f:
            pre_classified: dict[str, str] = json.load(f)
        print(f"Resuming: {len(pre_classified)} already classified")
    else:
        pre_classified = {}

    remaining = [(uid, info["avatar_url"]) for uid, info in users.items()
                 if uid not in pre_classified]

    # Step 1: Gravatar URL check (no download needed)
    non_gravatar = []
    for uid, url in remaining:
        if _is_gravatar(url):
            pre_classified[uid] = "default_avatar"
        else:
            non_gravatar.append((uid, url))

    gravatar_count = len(remaining) - len(non_gravatar)
    print(f"Gravatar detected: {gravatar_count}")
    print(f"Need identicon check: {len(non_gravatar)}")

    # Step 2: Download tiny images and check identicon
    semaphore = asyncio.Semaphore(concurrency)
    identicon_count = 0

    async with aiohttp.ClientSession() as session:
        pbar = tqdm(total=len(non_gravatar), desc="Identicon check")
        batch_size = 200

        for i in range(0, len(non_gravatar), batch_size):
            batch = non_gravatar[i : i + batch_size]
            results = await asyncio.gather(
                *[_fetch_tiny(session, uid, url, semaphore) for uid, url in batch]
            )
            for uid, img_bytes in results:
                if img_bytes and _is_identicon(img_bytes):
                    pre_classified[uid] = "default_avatar"
                    identicon_count += 1
            pbar.update(len(batch))

        pbar.close()

    print(f"Identicon detected: {identicon_count}")
    print(f"Total default_avatar: {sum(1 for v in pre_classified.values() if v == 'default_avatar')}")
    print(f"Remaining for YOLOv8 classification: {len(users) - len(pre_classified)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(pre_classified, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output_path}")

    # Remove any already-downloaded default-avatar images so they don't
    # waste disk space or leak into YOLO classification.
    if avatar_dir is not None and avatar_dir.exists():
        deleted = 0
        for uid, label in pre_classified.items():
            if label != "default_avatar":
                continue
            img_path = avatar_dir / f"{uid}.png"
            if img_path.exists():
                img_path.unlink()
                deleted += 1
        if deleted:
            print(f"Removed {deleted} default-avatar images from {avatar_dir}")

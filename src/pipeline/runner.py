"""High-level collection pipeline: sample → enrich → prefilter → download → classify.

All steps are resumable — re-running continues from wherever the previous run
stopped. The same functions power the `collect` CLI and can be called from
notebooks directly.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from classification.classifier import ClassificationConfig, classify_avatars
from collectors.contributions import collect_all as collect_contributions_all
from collectors.enricher import AsyncUserEnricher
from collectors.github_client import GitHubClient
from collectors.sampler import GROUP_PRIORITY, StratifiedSampler
from images.downloader import download_avatars
from images.prefilter import run_prefilter

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_YOLO_MODEL = REPO_ROOT / "yolov8x6_animeface.pt"
NUM_SAMPLING_GROUPS = len(GROUP_PRIORITY)


@dataclass
class PipelineConfig:
    total: int = 10_200
    enrich_concurrency: int = 15
    enrich_limit: int | None = None
    contributions_concurrency: int = 10
    prefilter_concurrency: int = 50
    download_concurrency: int = 30
    download_size: int = 256
    skip_classify: bool = False
    yolo_model_path: Path = DEFAULT_YOLO_MODEL
    yolo_conf: float = 0.01
    yolo_iou: float = 0.6
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    clip_anime_threshold: float = 0.75
    classify_save_interval: int = 50
    device: str | None = None
    data_dir: Path = DEFAULT_DATA_DIR


def _paths(data_dir: Path) -> dict[str, Path]:
    raw = data_dir / "raw"
    proc = data_dir / "processed"
    return {
        "sampled": raw / "sampled_users.json",
        "enriched": raw / "enriched_users.json",
        "contributions": raw / "contributions.json",
        "avatars": raw / "avatars",
        "pre_classified": proc / "pre_classified.json",
        "classification_progress": proc / "classification_progress.json",
        "classified_csv": proc / "classified_3cat.csv",
    }


def load_token() -> str:
    load_dotenv(REPO_ROOT / ".env")
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN not set. Add it to .env or export it before running."
        )
    return token


def sample_users(token: str, data_dir: Path, total: int) -> dict:
    """Step 1: stratified sampling. Resumes from sampled_users.json."""
    per_group = math.ceil(total / NUM_SAMPLING_GROUPS)
    client = GitHubClient(token)
    rate = client.get_rate_limit()
    print(f"[sample] rate limit {rate['remaining']}/{rate['limit']}, "
          f"target {total} users ({per_group}/group × {NUM_SAMPLING_GROUPS} groups)")
    sampler = StratifiedSampler(client, data_dir, per_group=per_group)
    sampler.collect_all()
    return sampler.summary()


async def enrich_users(
    token: str, data_dir: Path, concurrency: int = 15, limit: int | None = None,
) -> dict:
    """Step 2: async enrichment. Resumes from enriched_users.json."""
    enricher = AsyncUserEnricher(token, data_dir, concurrency=concurrency)
    await enricher.enrich_all(limit=limit)
    return enricher.summary()


async def collect_contributions(
    token: str, data_dir: Path, concurrency: int = 10,
) -> None:
    """Step 2.5: GraphQL contributionsCollection (commits/PRs/issues/reviews).
    Resumes from contributions.json."""
    p = _paths(data_dir)
    if not p["enriched"].exists():
        raise FileNotFoundError(f"missing {p['enriched']} — run enrich first")
    await collect_contributions_all(
        token, p["enriched"], p["contributions"], concurrency=concurrency,
    )


async def prefilter_defaults(data_dir: Path, concurrency: int = 50) -> None:
    """Step 3: detect default avatars (Gravatar + Identicon). Resumes."""
    p = _paths(data_dir)
    if not p["enriched"].exists():
        raise FileNotFoundError(f"missing {p['enriched']} — run enrich first")
    await run_prefilter(
        p["enriched"], p["pre_classified"],
        concurrency=concurrency, avatar_dir=p["avatars"],
    )


async def download_all_avatars(
    data_dir: Path, concurrency: int = 30, size: int = 256,
) -> None:
    """Step 4: download non-default avatars. Skips already-downloaded files."""
    p = _paths(data_dir)
    if not p["enriched"].exists():
        raise FileNotFoundError(f"missing {p['enriched']} — run enrich first")
    await download_avatars(
        p["enriched"], p["avatars"],
        concurrency=concurrency, size=size,
        pre_classified_path=p["pre_classified"],
    )


def classify_all(cfg: PipelineConfig):
    """Step 5: YOLO + CLIP classification. Resumes from progress file."""
    p = _paths(cfg.data_dir)
    if not p["enriched"].exists():
        raise FileNotFoundError(f"missing {p['enriched']} — run enrich first")
    # YOLO weights are auto-downloaded inside classify_avatars if missing.

    cc = ClassificationConfig(
        yolo_model_path=cfg.yolo_model_path,
        yolo_conf=cfg.yolo_conf,
        yolo_iou=cfg.yolo_iou,
        clip_model_name=cfg.clip_model_name,
        clip_pretrained=cfg.clip_pretrained,
        clip_anime_threshold=cfg.clip_anime_threshold,
        save_interval=cfg.classify_save_interval,
        device=cfg.device,
    )
    return classify_avatars(
        enriched_path=p["enriched"],
        avatar_dir=p["avatars"],
        pre_classified_path=p["pre_classified"],
        output_path=p["classified_csv"],
        progress_path=p["classification_progress"],
        cfg=cc,
    )


async def run_collect(cfg: PipelineConfig, token: str | None = None) -> None:
    """Orchestrate the full pipeline end-to-end. Each step is resumable, so
    re-running after interruption picks up from the last completed item.
    """
    token = token or load_token()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n===== [1/6] sample (target total {cfg.total}) =====")
    sample_users(token, cfg.data_dir, cfg.total)

    print("\n===== [2/6] enrich =====")
    await enrich_users(
        token, cfg.data_dir,
        concurrency=cfg.enrich_concurrency, limit=cfg.enrich_limit,
    )

    print("\n===== [3/6] contributions (GraphQL commits/PRs/issues/reviews) =====")
    await collect_contributions(
        token, cfg.data_dir, concurrency=cfg.contributions_concurrency,
    )

    print("\n===== [4/6] prefilter (default avatars) =====")
    await prefilter_defaults(cfg.data_dir, concurrency=cfg.prefilter_concurrency)

    print("\n===== [5/6] download avatars =====")
    await download_all_avatars(
        cfg.data_dir,
        concurrency=cfg.download_concurrency, size=cfg.download_size,
    )

    if cfg.skip_classify:
        print("\n===== [6/6] classify — SKIPPED =====")
        return

    print("\n===== [6/6] classify (YOLOv8 + CLIP) =====")
    classify_all(cfg)
    print("\n✓ pipeline complete")


def print_status(data_dir: Path, token: str | None = None) -> None:
    """Show progress snapshot across all pipeline stages."""
    p = _paths(data_dir)

    if p["sampled"].exists():
        with open(p["sampled"]) as f:
            sampled = json.load(f)
        groups = Counter(u["group"] for u in sampled.values())
        print(f"sampled:  {len(sampled)} users")
        for g, c in sorted(groups.items()):
            print(f"  {g}: {c}")
    else:
        print("sampled:  (none)")

    if p["enriched"].exists():
        with open(p["enriched"]) as f:
            enriched = json.load(f)
        print(f"enriched: {len(enriched)} users")
    else:
        print("enriched: (none)")

    if p["contributions"].exists():
        with open(p["contributions"]) as f:
            contributions = json.load(f)
        print(f"contributions: {len(contributions)} users")
    else:
        print("contributions: (none)")

    if p["pre_classified"].exists():
        with open(p["pre_classified"]) as f:
            pre = json.load(f)
        defaults = sum(1 for v in pre.values() if v == "default_avatar")
        print(f"prefilter: {len(pre)} checked, {defaults} default avatars")
    else:
        print("prefilter: (none)")

    if p["avatars"].exists():
        count = len(list(p["avatars"].glob("*.png")))
        print(f"avatars:  {count} downloaded")
    else:
        print("avatars:  (none)")

    if p["classification_progress"].exists():
        with open(p["classification_progress"]) as f:
            prog = json.load(f)
        anime = sum(1 for v in prog.values() if v.get("face_detected") and v.get("clip_is_anime"))
        face = sum(1 for v in prog.values() if v.get("face_detected"))
        print(f"classify: {len(prog)} processed, {face} faces, {anime} anime")
    else:
        print("classify: (none)")

    token = token or os.environ.get("GITHUB_TOKEN")
    if token:
        client = GitHubClient(token)
        rate = client.get_rate_limit()
        print(f"rate limit: {rate['remaining']}/{rate['limit']}")

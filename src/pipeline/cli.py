"""`collect` CLI — unified entry for the full pipeline."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .runner import (
    DEFAULT_DATA_DIR,
    DEFAULT_YOLO_MODEL,
    PipelineConfig,
    load_token,
    print_status,
    run_collect,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="collect",
        description=(
            "End-to-end pipeline: sample → enrich → prefilter → download → "
            "classify. All steps resume from their last checkpoint."
        ),
    )

    p.add_argument("--status", action="store_true",
                   help="Show progress across all stages and exit")
    p.add_argument("--total", type=int, default=10_200,
                   help="Target total users across 6 sampling groups (default: 10200)")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help=f"Data root (default: {DEFAULT_DATA_DIR})")

    g_enrich = p.add_argument_group("enrich")
    g_enrich.add_argument("--enrich-concurrency", type=int, default=15)
    g_enrich.add_argument("--enrich-limit", type=int, default=None,
                          help="Cap enrichment count per run (default: no cap)")
    g_enrich.add_argument("--contributions-concurrency", type=int, default=10,
                          help="GraphQL contributions concurrency (default: 10)")

    g_img = p.add_argument_group("images")
    g_img.add_argument("--prefilter-concurrency", type=int, default=50)
    g_img.add_argument("--download-concurrency", type=int, default=30)
    g_img.add_argument("--download-size", type=int, default=256)

    g_cls = p.add_argument_group("classification")
    g_cls.add_argument("--skip-classify", action="store_true")
    g_cls.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO_MODEL)
    g_cls.add_argument("--yolo-conf", type=float, default=0.01)
    g_cls.add_argument("--yolo-iou", type=float, default=0.6)
    g_cls.add_argument("--clip-model", default="ViT-B-32")
    g_cls.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    g_cls.add_argument("--clip-anime-threshold", type=float, default=0.75)
    g_cls.add_argument("--device", default=None,
                       help="Override device (cpu/cuda/mps). Auto-detected if unset.")
    return p


def _cfg_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        total=args.total,
        enrich_concurrency=args.enrich_concurrency,
        enrich_limit=args.enrich_limit,
        contributions_concurrency=args.contributions_concurrency,
        prefilter_concurrency=args.prefilter_concurrency,
        download_concurrency=args.download_concurrency,
        download_size=args.download_size,
        skip_classify=args.skip_classify,
        yolo_model_path=args.yolo_model,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        clip_anime_threshold=args.clip_anime_threshold,
        device=args.device,
        data_dir=args.data_dir,
    )


def main() -> None:
    args = _build_parser().parse_args()

    if args.status:
        print_status(args.data_dir)
        return

    cfg = _cfg_from_args(args)
    token = load_token()
    asyncio.run(run_collect(cfg, token=token))


if __name__ == "__main__":
    main()

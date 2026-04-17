"""YOLOv8 + CLIP avatar classifier.

Pipeline per image:
  1. YOLOv8 anime-face detection (catches anime + human faces)
  2. For each detected bbox: crop + CLIP zero-shot (anime vs human)
  3. Final anime = YOLO detected AND CLIP agrees

Resumability: per-uid progress saved to progress_path after each save_interval.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from device import get_device


YOLO_HF_REPO = "Fuyucchi/yolov8_animeface"
YOLO_HF_FILENAME = "yolov8x6_animeface.pt"


def ensure_yolo_model(target_path: Path) -> Path:
    """Download the YOLOv8 anime-face model to target_path if it's missing."""
    if target_path.exists():
        return target_path
    from huggingface_hub import hf_hub_download

    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[yolo] downloading {YOLO_HF_FILENAME} → {target_path}")
    downloaded = hf_hub_download(
        repo_id=YOLO_HF_REPO,
        filename=YOLO_HF_FILENAME,
        local_dir=target_path.parent,
    )
    downloaded = Path(downloaded)
    if downloaded.resolve() != target_path.resolve():
        import shutil

        shutil.move(str(downloaded), target_path)
    return target_path


# Anime prompts target otaku / moe / bishoujo style Japanese anime —
# NOT western cartoons, NOT generic 2D illustration.
DEFAULT_ANIME_PROMPTS = [
    "a japanese anime character face with large expressive eyes",
    "a cute moe anime girl illustration",
    "a bishoujo anime character in japanese manga style",
    "a kawaii anime waifu character portrait",
    "an otaku style japanese anime character drawing",
]
# Non-anime prompts cover real humans AND western / 3D / non-otaku art
# so those images lose to the anime prompts only when the image is clearly
# otaku-style.
DEFAULT_HUMAN_PROMPTS = [
    "a photograph of a real human face",
    "a selfie of a real person",
    "a western cartoon character like pixar or disney",
    "a 3d rendered animated movie character",
    "a realistic digital portrait of a person",
]


@dataclass
class ClassificationConfig:
    yolo_model_path: Path
    yolo_conf: float = 0.01
    yolo_iou: float = 0.6
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    clip_anime_threshold: float = 0.75
    anime_prompts: list[str] = field(default_factory=lambda: list(DEFAULT_ANIME_PROMPTS))
    human_prompts: list[str] = field(default_factory=lambda: list(DEFAULT_HUMAN_PROMPTS))
    save_interval: int = 50
    device: str | None = None


def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _build_clip(cfg: ClassificationConfig, device: str):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.clip_model_name, pretrained=cfg.clip_pretrained, device=device,
    )
    model.train(False)
    tokenizer = open_clip.get_tokenizer(cfg.clip_model_name)

    prompts = cfg.anime_prompts + cfg.human_prompts
    with torch.no_grad():
        tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return model, preprocess, text_features


def _classify_one(
    img_path: Path,
    yolo_model,
    clip_model,
    clip_preprocess,
    text_features,
    n_anime_prompts: int,
    cfg: ClassificationConfig,
    device: str,
) -> dict:
    result = {
        "face_detected": False,
        "anime_conf": 0.0,
        "anime_faces": 0,
        "clip_is_anime": False,
        "clip_anime_score": 0.0,
        "clip_human_score": 0.0,
    }

    if not img_path.exists():
        return result

    try:
        predictions = yolo_model.predict(
            str(img_path), conf=cfg.yolo_conf, iou=cfg.yolo_iou,
            verbose=False, device=device,
        )
    except Exception:
        return result

    boxes = predictions[0].boxes
    if len(boxes) == 0:
        return result

    result["face_detected"] = True
    result["anime_conf"] = float(boxes.conf.max())
    result["anime_faces"] = int(len(boxes))

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return result

    xyxy = boxes.xyxy.cpu().numpy()
    W, H = img.size
    crops = []
    for x1, y1, x2, y2 in xyxy:
        pad = int(max(x2 - x1, y2 - y1) * 0.1)
        cx1 = max(0, int(x1) - pad)
        cy1 = max(0, int(y1) - pad)
        cx2 = min(W, int(x2) + pad)
        cy2 = min(H, int(y2) + pad)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        crops.append(clip_preprocess(img.crop((cx1, cy1, cx2, cy2))))

    if not crops:
        return result

    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = (image_features @ text_features.T) * 100.0
        probs = logits.softmax(dim=-1).cpu().numpy()

    anime_scores = probs[:, :n_anime_prompts].sum(axis=1)
    human_scores = probs[:, n_anime_prompts:].sum(axis=1)
    best_idx = int(anime_scores.argmax())
    best_anime = float(anime_scores[best_idx])
    best_human = float(human_scores[best_idx])

    result["clip_anime_score"] = best_anime
    result["clip_human_score"] = best_human
    result["clip_is_anime"] = best_anime > cfg.clip_anime_threshold
    return result


def classify_avatars(
    enriched_path: Path,
    avatar_dir: Path,
    pre_classified_path: Path,
    output_path: Path,
    progress_path: Path,
    cfg: ClassificationConfig,
) -> pd.DataFrame:
    """Classify avatars into {Default, Anime, Photo}.

    Default comes from pre_classified_path. Anime requires both YOLO face
    detection and CLIP zero-shot confirmation. Progress is saved incrementally
    so interrupted runs resume from where they left off.
    """
    enriched = _load_json(enriched_path)
    pre_classified = _load_json(pre_classified_path)
    progress = _load_json(progress_path)  # uid -> result dict

    device = cfg.device or get_device()

    default_uids = {uid for uid, label in pre_classified.items() if label == "default_avatar"}
    pending = [uid for uid in enriched if uid not in default_uids and uid not in progress]

    print(f"Classification: {len(enriched)} total, {len(default_uids)} default, "
          f"{len(progress)} already classified, {len(pending)} pending")

    if pending:
        from ultralytics import YOLO

        ensure_yolo_model(cfg.yolo_model_path)
        yolo_model = YOLO(str(cfg.yolo_model_path))
        clip_model, clip_preprocess, text_features = _build_clip(cfg, device)
        n_anime_prompts = len(cfg.anime_prompts)

        pbar = tqdm(pending, desc=f"Classify ({cfg.clip_model_name})")
        for i, uid in enumerate(pbar, start=1):
            img_path = avatar_dir / f"{uid}.png"
            progress[uid] = _classify_one(
                img_path, yolo_model, clip_model, clip_preprocess,
                text_features, n_anime_prompts, cfg, device,
            )
            if i % cfg.save_interval == 0:
                _save_json(progress, progress_path)
        _save_json(progress, progress_path)

    rows = []
    for uid, info in enriched.items():
        if uid in default_uids:
            profile_type = "Default"
            r = {}
        else:
            r = progress.get(uid, {})
            face = r.get("face_detected", False)
            clip_ok = r.get("clip_is_anime", False)
            if face and clip_ok:
                profile_type = "Anime"
            else:
                profile_type = "Photo"

        repos = info.get("repos", {})
        primary_langs = repos.get("primary_languages", [])
        top_lang = (
            max(primary_langs, key=lambda x: x.get("count", 0)).get("name")
            if primary_langs else None
        )

        rows.append({
            "uid": int(uid),
            "profile_type": profile_type,
            "is_anime": bool(r.get("face_detected", False)),
            "anime_conf": r.get("anime_conf", 0.0),
            "anime_faces": r.get("anime_faces", 0),
            "clip_is_anime": bool(r.get("clip_is_anime", False)),
            "clip_anime_score": r.get("clip_anime_score", 0.0),
            "clip_human_score": r.get("clip_human_score", 0.0),
            "top_language": top_lang,
            "bio": info.get("bio", ""),
            "company": info.get("company", ""),
            "location": info.get("location", ""),
            "created_at": info.get("created_at", ""),
        })

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("=== Classification results ===")
    print(df["profile_type"].value_counts())
    print(f"Saved: {output_path}")
    return df

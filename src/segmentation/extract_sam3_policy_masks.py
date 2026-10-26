#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAM3 foreground-mask extraction for intermediate diffusion images.

This version assumes NO seed folders.

Input:
  image_root/
    ClassName/
      stem/
        00000/
          tar1_step020_image.png

Output:
  output_root/
    ClassName/
      stem/
        00000/
          tar1_step020_image_mask_binary.png

Policy:
  Looks up final_ref_mask_manifest.csv by dataset/class/stem.

Prompt priority:
  manual_review_v2_prompt
  > target_mask_prompt
  > final_prompt
  > raw_prompt
  > --prompt
"""

import argparse
import csv
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image
import torch


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_candidate_image(path: Path) -> bool:
    if path.suffix.lower() not in VALID_IMAGE_EXTS:
        return False

    name = path.name.lower()

    banned = ["mask", "overlay", "seg", "alpha", "depth", "normal"]
    if any(x in name for x in banned):
        return False

    return True


def iter_step20_images_fast(root: Path) -> Iterable[Path]:
    """
    Fast fixed-depth scanner for:
      root/class/stem/index/image.png

    It yields image paths immediately.
    """
    print(f"[SCAN] {root}", flush=True)

    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue

        for stem_dir in sorted(cls_dir.iterdir()):
            if not stem_dir.is_dir():
                continue

            for idx_dir in sorted(stem_dir.iterdir()):
                if not idx_dir.is_dir():
                    continue

                matched = sorted(idx_dir.glob("*step020*image*.png"))

                if not matched:
                    matched = sorted(
                        p for p in idx_dir.iterdir()
                        if p.is_file() and is_candidate_image(p)
                    )

                for p in matched:
                    if p.is_file() and is_candidate_image(p):
                        yield p


def to_numpy_mask_tensor(masks) -> np.ndarray:
    if masks is None:
        return np.zeros((0, 1, 1), dtype=np.float32)

    if isinstance(masks, torch.Tensor):
        masks = masks.detach().float().cpu().numpy()
    else:
        masks = np.asarray(masks)

    if masks.ndim == 2:
        masks = masks[None, ...]
    elif masks.ndim == 4:
        if masks.shape[1] == 1:
            masks = masks[:, 0, :, :]
        elif masks.shape[0] == 1:
            masks = masks[0]
        else:
            n0, n1, h, w = masks.shape
            masks = masks.reshape(n0 * n1, h, w)

    if masks.ndim != 3:
        raise ValueError(f"Unexpected mask shape after conversion: {masks.shape}")

    return masks.astype(np.float32)


def to_numpy_scores(scores, n: int) -> np.ndarray:
    if scores is None:
        return np.ones((n,), dtype=np.float32)

    if isinstance(scores, torch.Tensor):
        scores = scores.detach().float().cpu().numpy()
    else:
        scores = np.asarray(scores)

    scores = scores.reshape(-1).astype(np.float32)
    if len(scores) < n:
        scores = np.concatenate([scores, np.ones((n - len(scores),), dtype=np.float32)])
    return scores[:n]


def resize_mask_if_needed(mask: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    target_w, target_h = image_size

    if (w, h) == (target_w, target_h):
        return mask

    pil = Image.fromarray(mask.astype(np.uint8) * 255)
    pil = pil.resize((target_w, target_h), resample=Image.NEAREST)
    return (np.array(pil) > 127).astype(np.uint8)


def make_union_binary_mask(
    masks,
    scores=None,
    image_size: Optional[Tuple[int, int]] = None,
    score_thresh: float = 0.0,
    mask_thresh: float = 0.5,
) -> np.ndarray:
    masks_np = to_numpy_mask_tensor(masks)

    if masks_np.shape[0] == 0:
        if image_size is None:
            return np.zeros((1, 1), dtype=np.uint8)
        return np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

    scores_np = to_numpy_scores(scores, masks_np.shape[0])
    keep = scores_np >= score_thresh

    if not np.any(keep):
        h, w = masks_np.shape[-2:]
        union = np.zeros((h, w), dtype=np.uint8)
    else:
        union = np.any(masks_np[keep] > mask_thresh, axis=0).astype(np.uint8)

    if image_size is not None:
        union = resize_mask_if_needed(union, image_size)

    return union


def build_output_path(image_path: Path, input_root: Path, output_root: Path, suffix: str) -> Path:
    rel = image_path.relative_to(input_root)
    return output_root / rel.with_name(f"{rel.stem}{suffix}.png")


def save_mask(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(out_path)


def save_overlay(image: Image.Image, mask: np.ndarray, out_path: Path, alpha: float = 0.45) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base = image.convert("RGBA")
    rgba = np.array(base).astype(np.float32)

    overlay = np.zeros_like(rgba)
    overlay[..., 0] = 255
    overlay[..., 3] = mask.astype(np.float32) * 255 * alpha

    comp = Image.alpha_composite(
        Image.fromarray(rgba.astype(np.uint8), mode="RGBA"),
        Image.fromarray(overlay.astype(np.uint8), mode="RGBA"),
    )
    comp.save(out_path)


# ============================================================
# Manifest / policy utilities
# ============================================================

def _clean_cell(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    return s


def _stem_to_str(x) -> str:
    s = _clean_cell(x)

    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass

    return s


def get_class_stem_from_image_path(image_path: Path, input_root: Path) -> Tuple[str, str]:
    """
    Expected:
      input_root/ClassName/stem/00000/tar1_step020_image.png

    rel.parts:
      ClassName, stem, 00000, image.png
    """
    rel = image_path.relative_to(input_root)
    parts = rel.parts

    if len(parts) < 4:
        raise ValueError(f"Unexpected image path layout: {image_path}")

    cls = parts[0]
    stem = _stem_to_str(parts[1])
    return cls, stem


def load_policy_manifest(csv_path: str, dataset_filter: str = "") -> Dict[tuple, Dict[str, Any]]:
    """
    Load final_ref_mask_manifest.csv.

    Lookup keys:
      1. (dataset, class, stem)
      2. ("", class, stem)

    Prompt priority:
      manual_review_v2_prompt
      > target_mask_prompt
      > final_prompt
      > raw_prompt
      > default prompt
    """
    table: Dict[tuple, Dict[str, Any]] = {}

    if not csv_path:
        return table

    csv_path = str(csv_path)
    if not Path(csv_path).is_file():
        raise FileNotFoundError(f"policy manifest not found: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            dataset = _clean_cell(row.get("dataset"))
            cls = _clean_cell(row.get("class") or row.get("Class"))
            stem = _stem_to_str(row.get("stem"))

            if not cls or not stem:
                continue

            if dataset_filter and dataset != dataset_filter:
                continue

            status = _clean_cell(row.get("status")).lower()
            action = _clean_cell(row.get("action"))
            crop_action = _clean_cell(row.get("crop_action"))
            manual_action = _clean_cell(row.get("manual_review_action"))
            manual_v2_action = _clean_cell(row.get("manual_review_v2_action"))

            prompt = (
                _clean_cell(row.get("manual_review_v2_prompt"))
                or _clean_cell(row.get("target_mask_prompt"))
                or _clean_cell(row.get("final_prompt"))
                or _clean_cell(row.get("raw_prompt"))
            )

            policy = {
                "dataset": dataset,
                "class": cls,
                "stem": stem,
                "status": status,
                "action": action,
                "crop_action": crop_action,
                "manual_review_action": manual_action,
                "manual_review_v2_action": manual_v2_action,
                "prompt": prompt,
                "raw_prompt": _clean_cell(row.get("raw_prompt")),
                "final_prompt": _clean_cell(row.get("final_prompt")),
                "target_mask_prompt": _clean_cell(row.get("target_mask_prompt")),
                "manual_review_v2_prompt": _clean_cell(row.get("manual_review_v2_prompt")),
                "manual_review_v2_box": _clean_cell(row.get("manual_review_v2_box")),
                "manual_review_v2_roi": _clean_cell(row.get("manual_review_v2_roi")),
                "skip": (
                    status == "cl_sampling"
                    or action == "exclude_from_mgr_use_cl_sampling"
                    or manual_action == "exclude_from_mgr_use_cl_sampling"
                    or manual_v2_action == "exclude_from_mgr_use_cl_sampling"
                ),
            }

            table[(dataset, cls, stem)] = policy
            table[("", cls, stem)] = policy

    print(f"[POLICY] loaded {len(table)} lookup keys from {csv_path}", flush=True)
    return table


def lookup_policy(
    policy_table: Dict[tuple, Dict[str, Any]],
    dataset: str,
    cls: str,
    stem: str,
    default_prompt: str,
) -> Dict[str, Any]:
    stem = _stem_to_str(stem)

    policy = (
        policy_table.get((dataset, cls, stem))
        or policy_table.get(("", cls, stem))
    )

    if policy is None:
        return {
            "dataset": dataset,
            "class": cls,
            "stem": stem,
            "status": "manifest_missing",
            "action": "",
            "crop_action": "",
            "manual_review_action": "",
            "manual_review_v2_action": "",
            "prompt": default_prompt,
            "skip": False,
            "manual_review_v2_box": "",
            "manual_review_v2_roi": "",
        }

    if not policy.get("prompt"):
        policy = dict(policy)
        policy["prompt"] = default_prompt

    return policy


def parse_normalized_cxcywh_box(
    box_text: str,
    image_size: Tuple[int, int],
) -> Optional[Tuple[float, float, float, float]]:
    """
    Parse normalized box string:
      "cx,cy,w,h"

    Return pixel xyxy:
      (x1, y1, x2, y2)
    """
    box_text = _clean_cell(box_text)
    if not box_text:
        return None

    try:
        vals = [float(v.strip()) for v in box_text.split(",")]
        if len(vals) != 4:
            return None

        cx, cy, bw, bh = vals
        img_w, img_h = image_size

        x1 = (cx - bw / 2.0) * img_w
        y1 = (cy - bh / 2.0) * img_h
        x2 = (cx + bw / 2.0) * img_w
        y2 = (cy + bh / 2.0) * img_h

        x1 = max(0.0, min(float(img_w - 1), x1))
        y1 = max(0.0, min(float(img_h - 1), y1))
        x2 = max(0.0, min(float(img_w - 1), x2))
        y2 = max(0.0, min(float(img_h - 1), y2))

        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2, y2)

    except Exception:
        return None


def run_sam3_with_policy(
    processor,
    image: Image.Image,
    prompt: str,
    policy: Dict[str, Any],
):
    """
    Run SAM3 according to policy.

    Main path:
      processor.set_image(image)
      processor.set_text_prompt(state=state, prompt=prompt)

    If manifest says manual_review_v2_action == box_prompt_refine
    and the installed SAM3 processor has a compatible box-prompt method,
    this tries to use the box. If unsupported, it falls back to text prompt.
    """
    state = processor.set_image(image)

    manual_v2_action = policy.get("manual_review_v2_action", "")
    box_text = policy.get("manual_review_v2_box", "")
    box_xyxy = parse_normalized_cxcywh_box(box_text, image.size)

    use_box = manual_v2_action == "box_prompt_refine" and box_xyxy is not None

    if use_box:
        for method_name in ["set_box_prompt", "add_box_prompt"]:
            if hasattr(processor, method_name):
                method = getattr(processor, method_name)
                try:
                    return method(state=state, box=box_xyxy)
                except TypeError:
                    try:
                        return method(state, box_xyxy)
                    except Exception:
                        pass
                except Exception:
                    pass

        if hasattr(processor, "set_prompts"):
            try:
                return processor.set_prompts(state=state, text=prompt, box=box_xyxy)
            except TypeError:
                try:
                    return processor.set_prompts(state=state, prompt=prompt, box=box_xyxy)
                except Exception:
                    pass
            except Exception:
                pass

        print(
            f"[WARN] box_prompt_refine requested but no compatible SAM3 box API found. "
            f"Fallback to text prompt. box={box_xyxy}, prompt='{prompt}'",
            flush=True,
        )

    return processor.set_text_prompt(state=state, prompt=prompt)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="Root folder containing class/stem/index/image folders directly. No seed folder.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="",
        help="Default: image_root.rstrip('/') + '_sam3_policy_masks'",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="object",
        help="Fallback prompt if manifest row/prompt is missing.",
    )

    parser.add_argument(
        "--policy_manifest_csv",
        type=str,
        default="",
        help="Optional policy manifest CSV containing per class/stem segmentation prompts. Leave empty to use --prompt for all images.",
    )
    parser.add_argument(
        "--policy_dataset",
        type=str,
        default="",
        help=(
            "Dataset key in manifest, e.g., imb_cub_0.01, imb_flower_0.01, cub/10shot. "
            "If empty, lookup falls back to class/stem only."
        ),
    )
    parser.add_argument(
        "--skip_cl_sampling_policy",
        action="store_true",
        help="Skip images whose manifest says cl_sampling / exclude_from_mgr_use_cl_sampling.",
    )

    parser.add_argument("--score_thresh", type=float, default=0.0)
    parser.add_argument("--mask_thresh", type=float, default=0.5)
    parser.add_argument("--save_overlay", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    image_root = Path(args.image_root).resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"image_root not found: {image_root}")

    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else Path(str(image_root).rstrip("/") + "_sam3_policy_masks").resolve()
    )

    print(f"[ROOT] image_root      = {image_root}", flush=True)
    print(f"[ROOT] output_root     = {output_root}", flush=True)
    print(f"[SAM3] fallback prompt = {args.prompt}", flush=True)
    print(f"[POLICY] manifest      = {args.policy_manifest_csv}", flush=True)
    print(f"[POLICY] dataset       = {args.policy_dataset or '<class/stem fallback>'}", flush=True)

    policy_table = load_policy_manifest(
        csv_path=args.policy_manifest_csv,
        dataset_filter=args.policy_dataset,
    )

    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except Exception as exc:
        raise RuntimeError(
            "SAM3 is not importable. Install SAM3 in the active environment "
            "or add the SAM3 repository to PYTHONPATH."
        ) from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    print("[SAM3] loading model...", flush=True)
    model = build_sam3_image_model()
    if hasattr(model, "to"):
        model = model.to(args.device)
    if hasattr(model, "eval"):
        model.eval()

    processor = Sam3Processor(model)
    print("[SAM3] model loaded.", flush=True)

    num_seen = 0
    num_saved = 0
    num_empty = 0
    num_skip = 0
    num_error = 0
    num_policy_missing = 0
    num_policy_skip = 0

    for img_path in iter_step20_images_fast(image_root):
        num_seen += 1

        mask_path = build_output_path(img_path, image_root, output_root, "_mask_binary")
        overlay_path = build_output_path(img_path, image_root, output_root, "_overlay")

        if mask_path.exists() and not args.overwrite:
            num_skip += 1
            if num_seen % 100 == 0:
                print(
                    f"[PROGRESS] seen={num_seen}, saved={num_saved}, skipped={num_skip}, "
                    f"policy_skipped={num_policy_skip}, missing_policy={num_policy_missing}, "
                    f"empty={num_empty}, errors={num_error}",
                    flush=True,
                )
            continue

        try:
            cls_name, stem = get_class_stem_from_image_path(img_path, image_root)

            policy = lookup_policy(
                policy_table=policy_table,
                dataset=args.policy_dataset,
                cls=cls_name,
                stem=stem,
                default_prompt=args.prompt,
            )

            if policy.get("status") == "manifest_missing":
                num_policy_missing += 1

            if args.skip_cl_sampling_policy and policy.get("skip", False):
                num_policy_skip += 1
                num_skip += 1
                print(
                    f"[POLICY SKIP] class={cls_name} stem={stem} "
                    f"status={policy.get('status')} "
                    f"action={policy.get('action')} "
                    f"manual_action={policy.get('manual_review_action')} "
                    f"manual_v2_action={policy.get('manual_review_v2_action')} "
                    f"path={img_path}",
                    flush=True,
                )
                continue

            policy_prompt = policy.get("prompt") or args.prompt

            image = Image.open(img_path).convert("RGB")

            with torch.inference_mode():
                if args.device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        output = run_sam3_with_policy(
                            processor=processor,
                            image=image,
                            prompt=policy_prompt,
                            policy=policy,
                        )
                else:
                    output = run_sam3_with_policy(
                        processor=processor,
                        image=image,
                        prompt=policy_prompt,
                        policy=policy,
                    )

            binary = make_union_binary_mask(
                masks=output.get("masks", None),
                scores=output.get("scores", None),
                image_size=image.size,
                score_thresh=args.score_thresh,
                mask_thresh=args.mask_thresh,
            )

            if binary.sum() == 0:
                num_empty += 1

            save_mask(binary, mask_path)

            if args.save_overlay:
                save_overlay(image, binary, overlay_path)

            num_saved += 1

            if num_seen <= 10 or num_seen % 100 == 0:
                print(
                    f"[POLICY APPLY] seen={num_seen} "
                    f"class={cls_name} stem={stem} "
                    f"prompt='{policy_prompt}' "
                    f"status={policy.get('status')} "
                    f"action={policy.get('action')} "
                    f"crop_action={policy.get('crop_action')} "
                    f"manual_v2_action={policy.get('manual_review_v2_action')} "
                    f"box={policy.get('manual_review_v2_box')}",
                    flush=True,
                )

        except Exception as exc:
            num_error += 1
            print(f"[ERROR] {img_path} | {type(exc).__name__}: {exc}", flush=True)

        if num_seen % 20 == 0:
            print(
                f"[PROGRESS] seen={num_seen}, saved={num_saved}, skipped={num_skip}, "
                f"policy_skipped={num_policy_skip}, missing_policy={num_policy_missing}, "
                f"empty={num_empty}, errors={num_error}",
                flush=True,
            )

        if args.max_images > 0 and num_seen >= args.max_images:
            print(f"[STOP] max_images reached: {args.max_images}", flush=True)
            break

    print("[DONE]", flush=True)
    print(f"  seen           : {num_seen}", flush=True)
    print(f"  saved masks    : {num_saved}", flush=True)
    print(f"  skipped        : {num_skip}", flush=True)
    print(f"  policy skipped : {num_policy_skip}", flush=True)
    print(f"  policy missing : {num_policy_missing}", flush=True)
    print(f"  empty masks    : {num_empty}", flush=True)
    print(f"  errors         : {num_error}", flush=True)
    print(f"  output_root    : {output_root}", flush=True)


if __name__ == "__main__":
    main()
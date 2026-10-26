# -*- coding: utf-8 -*-
"""
Sampler for MGR Attention + Repulsive Noise Optimization (RNO).

Official implementation of "Improving Diffusion-Based Augmentation for
Data-Scarce Image Classification via Mask-Guided Reference Attention and
Repulsive Noise Optimization" (WACV 2027).

This script implements:
  1) Mask-Guided Reference (MGR) Attention
  2) Repulsive Noise Optimization (RNO)
"""

import os
import re
import random
import argparse
import csv
import json
import glob
from typing import List, Dict

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDIMScheduler

from pipelines.pipeline_sd15_MGR_RNO import StableDiffusionPipeline15MGR
from utils_sd15 import memory_efficient, init_latent


def _norm_name(s: str) -> str:
    return re.sub(r'[\s_/]+', ' ', s).strip().lower()

def _prompt_meta_path(root: str, cls: str, stem: str, idx: int) -> str:
    return os.path.join(root, cls, stem, f"{idx:05d}", "prompt_meta.json")

def _save_prompt_meta(path: str, **meta):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def _load_prompt_meta(path: str):
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@torch.no_grad()
def load_diffmix_embeddings(embed_path: str,
                            text_encoder: CLIPTextModel,
                            tokenizer: CLIPTokenizer):
    if not embed_path or not os.path.isfile(embed_path):
        print(f"[ft-db] no textual inversion embeddings found at {embed_path}; using raw class names")
        return {}
    ckpt = torch.load(embed_path, map_location='cpu')
    learned_embeds_dict = ckpt["learned_embeds_dict"]
    name2placeholder = ckpt["name2placeholder"]

    tokens = list(learned_embeds_dict.keys())
    to_add = [t for t in tokens if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
    if to_add:
        tokenizer.add_tokens(to_add)
        text_encoder.resize_token_embeddings(len(tokenizer))

    emb = text_encoder.get_input_embeddings()
    for token, vec in learned_embeds_dict.items():
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid != tokenizer.unk_token_id:
            emb.weight.data[tid] = vec.to(device=emb.weight.device, dtype=emb.weight.dtype)

    name2placeholder = {k.replace('/',' ').replace('_',' '): v for k, v in name2placeholder.items()}
    return name2placeholder

# =============== Per-class suffix diversification manager ===============
class SuffixManager:
    """
    - Load suffix candidates for the given dataset/shot setting.
    - Build a shuffled per-class queue and return suffixes sequentially without repetition until exhausted.
    - Refill the queue by reshuffling after exhaustion, following a round-robin style.
    """
    def __init__(self, datasets: str, shot: str,
                 suffix_root: str = "assets/suffix"):
        self.datasets = datasets
        self.shot = shot
        self.suffix_root = suffix_root
        self._cands: List[str] = self._load_candidates()
        self._per_class_queues: Dict[str, List[str]] = {}

    def _load_candidates(self) -> List[str]:
        try:
            if "imb" in self.datasets:
                sp = os.path.join(self.suffix_root, self.datasets, "suffix.txt")
            else:
                sp = os.path.join(self.suffix_root, self.datasets, f"{self.shot}.txt")

            if os.path.isfile(sp):
                with open(sp, "r") as f:
                    cands = [" " + line.strip() for line in f if line.strip()]
                return cands
        except Exception:
            pass
        return []  # If empty, run without suffixes.

    def next(self, class_key: str) -> str:
        if not self._cands:
            return ""  # No candidates; do not use a suffix.
        q = self._per_class_queues.get(class_key)
        if not q:
            q = random.sample(self._cands, len(self._cands))  # shuffled copy
            self._per_class_queues[class_key] = q
        # Pop and return; refill immediately after exhaustion.
        suffix = q.pop()
        if not q:
            self._per_class_queues[class_key] = random.sample(self._cands, len(self._cands))
        return suffix
# ==============================================================

def build_ref_prompt_from_dataset(datasets: str,
                                  shot: str,
                                  class_dir: str,
                                  name2placeholder: dict,
                                  fallback_name: str = "",
                                  suffix_root: str = "assets/suffix",
                                  suffix_manager: SuffixManager = None) -> str:
    import difflib
    parent = os.path.basename(os.path.normpath(class_dir))
    candidates = [parent] + ([fallback_name] if fallback_name else [])

    norm_map = {_norm_name(k): v for k, v in name2placeholder.items()}
    placeholder = None
    for c in candidates:
        k = _norm_name(c)
        if k in norm_map:
            placeholder = norm_map[k]; break
    if placeholder is None:
        keys = list(norm_map.keys())
        close = difflib.get_close_matches(_norm_name(parent), keys, n=1, cutoff=0.6)
        if close: placeholder = norm_map[close[0]]
    if placeholder is None:
        placeholder = parent

    if any(key in datasets for key in ["ISIC-2019-LT", "DermaMNIST-LT", "OrganCMNIST-LT", "OrganSMNIST-LT", "TissueMNIST"]):
        prefix = f"an image of {placeholder} {parent.replace('_', ' ')}"
    elif "cub" in datasets:
        prefix = f"a photo of a {placeholder} bird"
    elif "flower" in datasets:
        prefix = f"a photo of a {placeholder} flower"
    elif "aircraft" in datasets:
        prefix = f"a photo of a {placeholder} aircraft"
    elif "car" in datasets:
        prefix = f"a photo of a {placeholder} car"
    elif "pet" in datasets:
        prefix = f"a photo of a {placeholder} animal"
    else:
        prefix = f"a photo of a {placeholder}"

    # --- Suffix selection for per-class prompt diversity. ---
    if suffix_manager is not None:
        suffix = suffix_manager.next(parent)
    else:
        # Keep the previous behavior: random suffix selection.
        suffix = ""
        try:
            sp = os.path.join(suffix_root, datasets, "suffix.txt") if "imb" in datasets \
                 else os.path.join(suffix_root, datasets, f"{shot}.txt")
            if os.path.isfile(sp):
                with open(sp, "r") as f:
                    cands = [" " + line.strip() for line in f if line.strip()]
                if cands:
                    suffix = random.choice(cands)
        except Exception:
            pass

    return prefix + suffix, prefix

def list_images(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]

def corresponding_image(class_dir: str, stem: str) -> str:
    # try exact stem match; else fallback to first image
    imgs = list_images(class_dir)
    for p in imgs:
        if os.path.splitext(os.path.basename(p))[0] == stem:
            return p
    return imgs[0] if imgs else ""


def create_number_list(n):
    return list(range(max(0, int(n))))

def create_nested_list(t):
    return [[0, t]]

_REF_MASK_MANIFEST_CACHE = {}

# =====================
# Reference-mask manifest utilities
# =====================
# This release uses the reference-mask manifest only.
# Expected manifest columns:
#     dataset, class, stem, status, ref_image_path, ref_mask_path
#
# The path columns must be repository-relative paths, e.g.:
#     assets/ref_masks/<dataset>/<class>/<stem>/input_image.png
#     assets/ref_masks/<dataset>/<class>/<stem>/mask_binary.png
#
# Absolute paths are intentionally not supported in the sampler.

_REQUIRED_MANIFEST_COLUMNS = {
    "dataset",
    "class",
    "stem",
    "ref_image_path",
    "ref_mask_path",
}


def _manifest_repo_root(manifest_path: str) -> str:
    """
    Infer the repository root from the manifest location.

    If the manifest is under <repo_root>/assets/, this returns <repo_root>.
    Otherwise, it falls back to the manifest directory.
    """
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    if os.path.basename(manifest_dir) == "assets":
        return os.path.dirname(manifest_dir)
    return manifest_dir


def _resolve_manifest_path(path: str, manifest_path: str) -> str:
    """
    Resolve a manifest path relative to the repository root.

    Manifest entries are expected to be repository-relative. Absolute paths inside the CSV are rejected.
    """
    if path is None:
        return ""

    raw = str(path).strip()
    if not raw:
        return ""

    raw = raw.replace("\\", "/")

    if os.path.isabs(raw):
        raise ValueError(
            "The reference-mask manifest contains an absolute path. "
            "Manifest path columns must be repository-relative. "
            f"Offending path: {raw}"
        )

    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    repo_root = _manifest_repo_root(manifest_path)

    candidates = [
        os.path.abspath(os.path.join(repo_root, raw)),
        os.path.abspath(os.path.join(manifest_dir, raw)),
        os.path.abspath(raw),
    ]

    for cand in candidates:
        if os.path.isfile(cand):
            return cand

    # Return the canonical repo-root candidate even if it does not exist so the
    # later error message is deterministic and easy to debug.
    return candidates[0]


def _display_manifest_path(path: str, manifest_path: str) -> str:
    """Return a repo-relative display path for logs when possible."""
    if not path:
        return ""
    repo_root = _manifest_repo_root(manifest_path)
    try:
        return os.path.relpath(path, repo_root).replace("\\", "/")
    except Exception:
        return os.path.basename(path)


def _load_ref_mask_manifest(manifest_path: str):
    """
    Load the reference-mask manifest.

    Required columns:
        dataset, class, stem, ref_image_path, ref_mask_path

    Optional columns:
        status
    """
    if not manifest_path:
        return {}

    manifest_path = os.path.abspath(os.path.expanduser(manifest_path))

    if manifest_path in _REF_MASK_MANIFEST_CACHE:
        return _REF_MASK_MANIFEST_CACHE[manifest_path]

    rows = {}
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Reference-mask manifest not found: {manifest_path}"
        )

    with open(manifest_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing_cols = sorted(_REQUIRED_MANIFEST_COLUMNS - fieldnames)
        if missing_cols:
            raise ValueError(
                "The reference-mask manifest is missing required columns: "
                f"{missing_cols}. Expected columns: "
                "dataset, class, stem, status, ref_image_path, ref_mask_path."
            )

        for row in reader:
            dataset = (row.get("dataset") or "").strip()
            cls = (row.get("class") or "").strip()
            stem = str(row.get("stem") or "").strip()

            if not (dataset and cls and stem):
                continue

            ref_image_rel = (row.get("ref_image_path") or "").strip()
            ref_mask_rel = (row.get("ref_mask_path") or "").strip()

            if not ref_image_rel or not ref_mask_rel:
                raise ValueError(
                    "Manifest row is missing ref_image_path/ref_mask_path: "
                    f"dataset={dataset}, class={cls}, stem={stem}"
                )

            ref_image_abs = _resolve_manifest_path(ref_image_rel, manifest_path)
            ref_mask_abs = _resolve_manifest_path(ref_mask_rel, manifest_path)

            clean_row = {
                "dataset": dataset,
                "class": cls,
                "stem": stem,
                "status": (row.get("status") or "ready").strip().lower(),
                "ref_image_path": ref_image_abs,
                "ref_mask_path": ref_mask_abs,
                "ref_image_path_display": ref_image_rel,
                "ref_mask_path_display": ref_mask_rel,
            }
            rows[(dataset, cls, stem)] = clean_row

    print(
        f"[REF MASK MANIFEST] loaded {len(rows)} entries from "
        f"{_display_manifest_path(manifest_path, manifest_path)}",
        flush=True,
    )
    _REF_MASK_MANIFEST_CACHE[manifest_path] = rows
    return rows


def load_ref_from_manifest(manifest_path: str,
                           dataset_key: str,
                           cls_name: str,
                           stem: str):
    """Load one reference image and mask from the manifest."""
    rows = _load_ref_mask_manifest(manifest_path)
    key = (dataset_key, cls_name, str(stem))
    row = rows.get(key)

    if not row:
        print(
            f"[MANIFEST MISS] reference mask row not found: "
            f"dataset={dataset_key}, class={cls_name}, stem={stem}",
            flush=True,
        )
        return None, None, None

    status = str(row.get("status", "ready")).strip().lower()
    if status and status not in {"ready", "ok", "valid"}:
        print(
            f"[MANIFEST SKIP] reference mask row is not usable: "
            f"dataset={dataset_key}, class={cls_name}, stem={stem}, status={status}",
            flush=True,
        )
        return None, None, None

    img_path = row.get("ref_image_path", "")
    mask_path = row.get("ref_mask_path", "")

    img_exists = bool(img_path and os.path.isfile(img_path))
    mask_exists = bool(mask_path and os.path.isfile(mask_path))

    if not img_exists or not mask_exists:
        print(
            f"[MANIFEST PATH MISSING] reference image/mask path does not exist:\n"
            f"  dataset={dataset_key}\n"
            f"  class={cls_name}\n"
            f"  stem={stem}\n"
            f"  status={status}\n"
            f"  ref_image_path={row.get('ref_image_path_display', '')}\n"
            f"  img_exists={img_exists}\n"
            f"  ref_mask_path={row.get('ref_mask_path_display', '')}\n"
            f"  mask_exists={mask_exists}",
            flush=True,
        )
        return None, None, None

    try:
        ref_img = Image.open(img_path).convert("RGB")
        ref_mask = np.array(Image.open(mask_path).convert("L"))
        print(
            f"[MANIFEST LOAD OK] reference image/mask loaded: "
            f"dataset={dataset_key}, class={cls_name}, stem={stem}",
            flush=True,
        )
        return ref_img, ref_mask, img_path
    except Exception as exc:
        print(
            f"[MANIFEST LOAD ERROR] failed to load reference image/mask: "
            f"dataset={dataset_key}, class={cls_name}, stem={stem}, "
            f"err={type(exc).__name__}: {exc}",
            flush=True,
        )
        return None, None, None


def build_train_index_from_manifest(manifest_path: str, dataset_key: str):
    """
    Build a lightweight train index from the ref-mask manifest.

    This is useful for a release package where the real train directory is not
    shipped separately and assets/ref_masks/<dataset>/<class>/<stem>/input_image.png
    already contains the reference/train image used by MGR.

    Returns:
        dict[class_name][stem] = ref_image_abs_path
    """
    rows = _load_ref_mask_manifest(manifest_path)
    class_to_stems = {}
    for (dataset, cls, stem), row in rows.items():
        if dataset != dataset_key:
            continue
        status = str(row.get("status", "ready")).strip().lower()
        if status and status not in {"ready", "ok", "valid"}:
            continue
        img_path = row.get("ref_image_path", "")
        mask_path = row.get("ref_mask_path", "")
        if not (img_path and os.path.isfile(img_path)):
            continue
        if not (mask_path and os.path.isfile(mask_path)):
            continue
        class_to_stems.setdefault(cls, {})[str(stem)] = img_path
    return class_to_stems


# =====================
# Main
# =====================
def main():
    ap = argparse.ArgumentParser(
        description="Generate synthetic samples with MGR and optional RNO only."
    )

    # Dataset / checkpoint paths
    ap.add_argument('--train_dir', type=str, required=True,
                    help='Original train image directory. If missing, the sampler uses ref_mask_manifest as the train index.')
    ap.add_argument('--datasets', type=str, default='imb_cub_0.01')
    ap.add_argument('--shot', type=str, default='5shot')
    ap.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5')
    ap.add_argument('--ft_db_root', type=str, required=True,
                    help='LoRA/Textual-Inversion checkpoint directory.')
    ap.add_argument('--result_dir', type=str, required=True)

    # Sampling settings
    ap.add_argument('--guidance_scale', type=float, default=7.0)
    ap.add_argument('--num_inference_steps', type=int, default=50)
    ap.add_argument('--expansion_rate', type=int, default=5)
    ap.add_argument('--generation_mode', type=str, default='expansion',
                    choices=['expansion', 'normalization'])
    ap.add_argument('--class_quota_csv', type=str, default='',
                    help='Optional CSV with class,quota columns. Overrides generation_mode.')
    ap.add_argument('--max_classes', type=int, default=0)
    ap.add_argument('--only_classes', type=str, default='',
                    help='Comma-separated class directory names to sample.')
    ap.add_argument('--target_batch_size', type=int, default=4,
                    help='Number of target samples generated with one reference branch.')
    ap.add_argument('--vary_target_suffixes', action='store_true', default=True,
                    help='Use different suffix prompts for target branches in the same batch.')
    ap.add_argument('--reference_prompt_no_suffix', action='store_true',
                    help='Use the suffix-free class prompt for the reference branch.')

    # MGR settings
    ap.add_argument('--use_mgr', action='store_true', default=True,
                    help='Enable Mask-Guided Reference Attention. Default: enabled.')
    ap.add_argument('--no_mgr', dest='use_mgr', action='store_false',
                    help='Disable MGR for sanity/debug runs.')
    ap.add_argument('--step_start_mgr', type=int, default=20,
                    help='First denoising-step index where mask-guided MGR is applied.')
    ap.add_argument('--mgr_block_from', type=str, default='up2_1',
                    choices=['up2_1', 'up2_2', 'up1_only', 'up2_only', 'mid_only', 'down_only'])
    ap.add_argument('--temperature', type=float, default=0.5)
    ap.add_argument('--tar_mask_ema', type=float, default=0.0)
    ap.add_argument('--save_intermediates', action='store_true')
    ap.add_argument('--mode', type=str, choices=['sampling', 'debug'], default='sampling')
    ap.add_argument('--save_seg_masks', action='store_true')

    # Reference mask / target-mask dump-resume path
    ap.add_argument('--ref_mask_manifest', type=str, required=True,
                    help='CSV manifest for precomputed reference images and masks.')
    ap.add_argument('--require_ref_seg', action='store_true')
    ap.add_argument('--clipseg_fallback_prompt', type=str, default='',
                    help='Optional CLIPSeg fallback prompt. If empty, a dataset-level default is used.')
    ap.add_argument('--target_step_dump_root', type=str, default='')
    ap.add_argument('--dump_target_step_only', action='store_true')
    ap.add_argument('--precomputed_target_mask_root', type=str, default='')
    ap.add_argument('--resume_target_latent_root', type=str, default='')
    ap.add_argument('--skip_missing_resume_latents', action='store_true')
    ap.add_argument('--prompt_meta_root', type=str, default='')
    ap.add_argument('--ref_kv_cache_mode', type=str, default='off', choices=['off', 'save', 'load'])
    ap.add_argument('--ref_kv_cache_root', type=str,
                    default='outputs/ref_kv_cache')

    # RNO settings. RNO is disabled unless --use_rno is passed.
    ap.add_argument('--use_rno', action='store_true',
                    help='Enable reference-guided initial-noise optimization.')
    ap.add_argument('--rno_num_opt_steps', type=int, default=3)
    ap.add_argument('--rno_lr', type=float, default=0.01)
    ap.add_argument('--rno_temperature', type=float, default=0.07)
    ap.add_argument('--rno_gamma', type=float, default=1.0)
    ap.add_argument('--rno_pool_size', type=int, default=16)
    ap.add_argument('--rno_clip', type=float, default=None)
    ap.add_argument('--rno_kl_weight', type=float, default=0.0)
    ap.add_argument('--rno_ref_repel_weight', type=float, default=1.0,
                    help='Weight for repelling target noise from the reference branch.')
    ap.add_argument('--rno_positive_weight', type=float, default=0.0,
                    help='0.0 gives negative-only RNO; 1.0 keeps the RNO positive anchor.')
    ap.add_argument('--rno_negative_prompt_no_suffix', action='store_true',
                    help='Use suffix-free class prompt as the negative prompt pool.')
    ap.add_argument('--rno_fixed_neg_anchor', action='store_true')
    ap.add_argument('--rno_neg_anchor_extra', type=int, default=0)
    ap.add_argument('--rno_force_per_sample', action='store_true')
    ap.add_argument('--rno_exclude_self_neg', action='store_true')

    # Runtime / sharding
    ap.add_argument('--N_proc', type=int, default=1)
    ap.add_argument('--cur_proc', type=int, default=0)
    ap.add_argument('--base_seed', type=int, default=int(os.environ.get('SAMPLING_BASE_SEED', '42')))
    ap.add_argument('--seed_stride', type=int, default=int(os.environ.get('SAMPLING_SEED_STRIDE', '100003')))
    ap.add_argument('--suffix_root', type=str, default='assets/suffix')
    ap.add_argument('--color_cal_start_t', type=int, default=150)
    ap.add_argument('--color_cal_window_size', type=int, default=50)

    args = ap.parse_args()


    seed_offset = int(os.environ.get('SAMPLING_SEED_OFFSET', '0'))
    seed = int(args.base_seed) + int(args.cur_proc) * int(args.seed_stride) + seed_offset
    print(f'[SEED] deterministic seed={seed}', flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    assert torch.cuda.is_available(), 'CUDA not available'
    torch.cuda.set_device(0)
    device = torch.device('cuda')
    torch_dtype = torch.float16

    save_seg_masks = bool(args.save_seg_masks or args.mode == 'debug')
    print(f'[SEG-MASK SAVE] save_seg_masks={save_seg_masks} mode={args.mode}')

    pipe = StableDiffusionPipeline15MGR.from_pretrained(
        args.model_id, safety_checker=None, torch_dtype=torch_dtype
    ).to(device)
    memory_efficient(pipe, device)

    pipe.scheduler = DDIMScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe.scheduler.fix_traj_t_start = args.color_cal_start_t
    pipe.scheduler.fix_traj_t_end = args.color_cal_start_t - args.color_cal_window_size

    pipe.load_lora_weights(os.path.join(args.ft_db_root, 'pytorch_lora_weights.safetensors'))
    ft_step = re.findall(r'checkpoint-(\d+)', args.ft_db_root)
    if ft_step:
        embed_path = os.path.join(args.ft_db_root, f'learned_embeds-steps-{ft_step[0]}.bin')
    else:
        embed_candidates = [
            os.path.join(args.ft_db_root, 'learned_embeds-steps-last.bin'),
            os.path.join(args.ft_db_root, 'learned_embeds-steps-35000.bin'),
        ]
        embed_path = next((p for p in embed_candidates if os.path.isfile(p)), embed_candidates[0])
    name2placeholder = load_diffmix_embeddings(embed_path, pipe.text_encoder, pipe.tokenizer)

    pipe.activate_mgr(
        use_mgr=args.use_mgr,
        temperature=args.temperature,
        mgr_block_from=args.mgr_block_from,
        unet_chunk_size=2,
        num_inference_steps=args.num_inference_steps,
        step_start_mgr=args.step_start_mgr,
    )
    print(
        f'Activated MGR: mask-guided MGR {args.step_start_mgr}..{args.num_inference_steps-1}'
    )

    # The original research code used --train_dir to list classes/stems and to
    # compute generation quotas. For this release, the
    # selected train/reference images can also be indexed directly from
    # assets/ref_masks via the manifest.  If --train_dir is missing, we fall back
    # to the manifest index automatically.
    use_manifest_train_index = not (args.train_dir and os.path.isdir(args.train_dir))
    manifest_train_index = None
    if use_manifest_train_index:
        print(
            f'[TRAIN INDEX] train_dir not found; using ref_mask_manifest as train index: {args.train_dir}',
            flush=True,
        )
        manifest_train_index = build_train_index_from_manifest(args.ref_mask_manifest, args.datasets)
        classes = sorted(manifest_train_index.keys())
    else:
        classes = [
            d for d in sorted(os.listdir(args.train_dir))
            if os.path.isdir(os.path.join(args.train_dir, d))
        ]

    if args.only_classes:
        allow = {c.strip() for c in args.only_classes.split(',') if c.strip()}
        classes = [c for c in classes if c in allow]
    if args.max_classes and len(classes) > args.max_classes:
        classes = classes[:args.max_classes]
    if not classes:
        raise RuntimeError('No classes selected.')

    os.makedirs(args.result_dir, exist_ok=True)

    if args.clipseg_fallback_prompt:
        clip_prompt_default = args.clipseg_fallback_prompt
    elif 'flower' in args.datasets:
        clip_prompt_default = 'flower'
    elif 'cub' in args.datasets:
        clip_prompt_default = 'bird'
    elif 'pet' in args.datasets:
        clip_prompt_default = 'animal'
    elif 'car' in args.datasets:
        clip_prompt_default = 'car'
    elif 'aircraft' in args.datasets:
        clip_prompt_default = 'aircraft'
    else:
        clip_prompt_default = 'object'

    suffix_manager = SuffixManager(args.datasets, args.shot, args.suffix_root)

    # Compute per-class generation quotas.
    if use_manifest_train_index:
        orig_counts = [len(manifest_train_index.get(cls, {})) for cls in classes]
    else:
        orig_counts = [len(list_images(os.path.join(args.train_dir, cls))) for cls in classes]
    max_orig = max(orig_counts) if orig_counts else 0
    class_quota = None
    if args.class_quota_csv:
        class_quota = {}
        with open(args.class_quota_csv, newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                cls_name = (row.get('class') or row.get('Class') or '').strip()
                quota_raw = row.get('quota') or row.get('Quota') or row.get('n_add') or row.get('count') or '0'
                if cls_name:
                    class_quota[cls_name] = max(0, int(float(quota_raw)))
        print(f'[class quota] loaded {len(class_quota)} rows; total={sum(class_quota.values())}')

    class_img_counts = []
    for cls, n_orig in zip(classes, orig_counts):
        if class_quota is not None:
            count = class_quota.get(cls, 0)
        elif args.generation_mode == 'expansion':
            count = n_orig * int(args.expansion_rate)
        else:
            count = max(0, max_orig - n_orig)
        class_img_counts.append(count)

    total_imgs = sum(class_img_counts)
    imgs_per_proc = total_imgs // max(1, args.N_proc)
    proc_classes = []
    start = 0
    for proc_id in range(args.N_proc):
        proc_cls = []
        proc_imgs = 0
        while start < len(classes) and (proc_imgs < imgs_per_proc or proc_id == args.N_proc - 1):
            proc_cls.append(classes[start])
            proc_imgs += class_img_counts[start]
            start += 1
        proc_classes.append(proc_cls)

    if args.cur_proc < 0 or args.cur_proc >= len(proc_classes):
        raise ValueError(f'cur_proc={args.cur_proc} must be in [0, {len(proc_classes)-1}]')
    my_classes = proc_classes[args.cur_proc]
    print(f'[PROC] cur_proc={args.cur_proc}, N_proc={args.N_proc}, assigned={len(my_classes)} classes')

    csv_path = os.path.join(args.result_dir, f'prompt_log_proc{args.cur_proc:02d}-of-{args.N_proc:02d}procs.csv')
    csv_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a', newline='', encoding='utf-8') as csvf:
        writer = csv.writer(csvf)
        if not csv_exists:
            writer.writerow(['class', 'index', 'stem', 'image_path', 'prompt', 'ref_prompt', 'target_prompt', 'pipe_prompt'])

        total_imgs_to_generate = sum(class_img_counts[classes.index(c)] for c in my_classes)
        with tqdm(total=total_imgs_to_generate, desc=f'[Proc {args.cur_proc}]') as total_pbar:
            for ci, cls in enumerate(tqdm(my_classes, desc=f'Proc {args.cur_proc} classes')):
                if use_manifest_train_index:
                    class_dir = cls
                    img_by_stem = dict(manifest_train_index.get(cls, {}))
                    img_list = list(img_by_stem.values())
                else:
                    class_dir = os.path.join(args.train_dir, cls)
                    img_list = list_images(class_dir)
                    img_by_stem = {
                        os.path.splitext(os.path.basename(p))[0]: p
                        for p in img_list
                    }

                if not img_list:
                    print(f'[SKIP {cls}] no images found')
                    continue

                out_dir = os.path.join(args.result_dir, cls)
                os.makedirs(out_dir, exist_ok=True)

                if class_quota is not None:
                    target_cnt = class_quota.get(cls, 0)
                elif args.generation_mode == 'expansion':
                    target_cnt = len(img_list) * int(args.expansion_rate)
                else:
                    target_cnt = max(0, max_orig - len(img_list))

                existed = len(list_images(out_dir))
                if existed >= target_cnt:
                    print(f'[{ci:03d}] {cls}: already satisfied ({existed}/{target_cnt}), skip.')
                    continue

                stems = sorted(img_by_stem.keys())
                K = max(1, len(stems))
                if args.generation_mode == 'expansion' and class_quota is None:
                    stem_quota = {s: int(args.expansion_rate) for s in stems}
                else:
                    base = target_cnt // K
                    rem = target_cnt % K
                    stem_quota = {s: base for s in stems}
                    for s in stems[:rem]:
                        stem_quota[s] += 1

                # Load reference image/mask assets once per stem.
                # This release uses only the manifest.
                # The manifest should contain repo-relative or absolute paths to
                # the selected reference image and mask for each (dataset, class, stem).
                stem_assets = {}
                for stem in stems:
                    ref_img_pre, ref_mask_pre, ref_img_path = load_ref_from_manifest(
                        args.ref_mask_manifest, args.datasets, cls, stem
                    )
                    if ref_img_pre is None or ref_mask_pre is None:
                        if args.require_ref_seg:
                            print(f'[WARN] required ref segmentation missing -> skip {cls}/{stem}')
                            continue
                        fallback_img_path = img_by_stem.get(stem, img_list[0])
                        print(f'[WARN] {cls}/{stem}: ref seg not found -> original image + full mask')
                        ref_img_pre = Image.open(fallback_img_path).convert('RGB')
                        ref_mask_pre = np.full((ref_img_pre.height, ref_img_pre.width), 255, dtype=np.uint8)
                        ref_img_path = fallback_img_path

                    stem_assets[stem] = {
                        'image': ref_img_pre.resize((512, 512), resample=Image.BICUBIC),
                        'mask': cv2.resize(ref_mask_pre, (512, 512), interpolation=cv2.INTER_NEAREST),
                        'ref_img_path': ref_img_path,
                    }

                if not stem_assets:
                    print(f'[WARN] {cls}: no usable reference image found -> skip class')
                    continue

                existing_counts = {}
                for p in list_images(out_dir):
                    bn = os.path.basename(p)
                    stem_name = os.path.splitext(bn.split('_', 1)[1] if '_' in bn else bn)[0]
                    existing_counts[stem_name] = existing_counts.get(stem_name, 0) + 1

                tar_seeds = create_number_list(max(1, int(args.target_batch_size)))
                saved = existed
                remaining_cnt = max(0, target_cnt - existed)

                with tqdm(total=remaining_cnt, desc=f'Class {cls}', position=1, leave=False) as pbar:
                    keep_generating = True
                    while keep_generating:
                        keep_generating = False
                        for stem in stems:
                            if (saved - existed) >= remaining_cnt:
                                break

                            have = existing_counts.get(stem, 0)
                            need = max(0, stem_quota.get(stem, 0) - have)
                            if need <= 0 or stem not in stem_assets:
                                continue
                            keep_generating = True

                            real_img = stem_assets[stem]['image']
                            refer_mask = stem_assets[stem]['mask']
                            ref_img_path = stem_assets[stem]['ref_img_path']

                            ref_prompt, inf_prompt = build_ref_prompt_from_dataset(
                                datasets=args.datasets,
                                shot=args.shot,
                                class_dir=class_dir,
                                name2placeholder=name2placeholder,
                                fallback_name=os.path.splitext(os.path.basename(img_list[0]))[0],
                                suffix_manager=suffix_manager,
                            )

                            pipe_prompt = inf_prompt if args.reference_prompt_no_suffix else ref_prompt
                            target_prompts_for_batch = None
                            pipe_prompt_for_batch = pipe_prompt
                            if args.vary_target_suffixes:
                                target_prompts_for_batch = [
                                    inf_prompt + (suffix_manager.next(cls) if suffix_manager is not None else '')
                                    for _ in range(len(tar_seeds))
                                ]
                                pipe_prompt_for_batch = [pipe_prompt] + target_prompts_for_batch

                            meta_path = _prompt_meta_path(args.prompt_meta_root, cls, stem, saved)
                            prompt_meta = _load_prompt_meta(meta_path)
                            if prompt_meta:
                                ref_prompt = prompt_meta.get('ref_prompt', ref_prompt)
                                inf_prompt = prompt_meta.get('target_prompt', inf_prompt)
                                pipe_prompt = prompt_meta.get('pipe_prompt', pipe_prompt)
                                target_prompts_for_batch = prompt_meta.get('target_prompts_for_batch', target_prompts_for_batch)
                                pipe_prompt_for_batch = [pipe_prompt] + target_prompts_for_batch if target_prompts_for_batch else pipe_prompt
                            elif args.target_step_dump_root:
                                _save_prompt_meta(
                                    _prompt_meta_path(args.target_step_dump_root, cls, stem, saved),
                                    dataset=args.datasets, cls=cls, stem=stem, index=saved,
                                    ref_prompt=ref_prompt, target_prompt=inf_prompt,
                                    target_prompts_for_batch=target_prompts_for_batch,
                                    pipe_prompt=pipe_prompt,
                                    reference_prompt_no_suffix=int(args.reference_prompt_no_suffix),
                                    clip_prompt=clip_prompt_default,
                                )

                            latents = [init_latent(model=pipe, device_name=device, dtype=torch_dtype, seed=random.getrandbits(64))]
                            for _ in range(len(tar_seeds)):
                                latents.append(init_latent(model=pipe, device_name=device, dtype=torch_dtype, seed=random.getrandbits(64)))
                            latents = torch.cat(latents, dim=0)

                            resume_target_latent_dir = (
                                os.path.join(args.resume_target_latent_root, cls, stem, f'{saved:05d}')
                                if args.resume_target_latent_root else None
                            )
                            if (args.skip_missing_resume_latents and resume_target_latent_dir and
                                    not glob.glob(os.path.join(resume_target_latent_dir, 'target_latents_step*.pt'))):
                                saved += len(tar_seeds)
                                existing_counts[stem] = existing_counts.get(stem, 0) + len(tar_seeds)
                                pbar.update(len(tar_seeds))
                                total_pbar.update(len(tar_seeds))
                                continue

                            out = pipe(
                                prompt=pipe_prompt_for_batch,
                                target_prompt=([inf_prompt] + target_prompts_for_batch if target_prompts_for_batch else inf_prompt),
                                image=real_img,
                                ref_mask=refer_mask,
                                latents=latents,
                                num_images_per_prompt=1,
                                guidance_scale=args.guidance_scale,
                                num_inference_steps=args.num_inference_steps,
                                output_type='pil',
                                synth_save_dir=out_dir,
                                save_intermediates=args.save_intermediates,
                                save_seg_masks=save_seg_masks,
                                cur_tar_idx=saved,
                                ref_img_path=ref_img_path,
                                ref_kv_cache_mode=args.ref_kv_cache_mode,
                                ref_kv_cache_root=args.ref_kv_cache_root,
                                ref_kv_cache_id=f'{args.datasets}/{cls}/{stem}',
                                target_step_dump_dir=(
                                    os.path.join(args.target_step_dump_root, cls, stem, f'{saved:05d}')
                                    if args.target_step_dump_root else None
                                ),
                                dump_target_step_only=args.dump_target_step_only,
                                precomputed_target_mask_dir=(
                                    os.path.join(args.precomputed_target_mask_root, cls, stem, f'{saved:05d}')
                                    if args.precomputed_target_mask_root else None
                                ),
                                resume_target_latent_dir=resume_target_latent_dir,
                                clip_prompt=clip_prompt_default,
                                tar_mask_ema=args.tar_mask_ema,
                                use_mgr=args.use_mgr,
                                use_rno=args.use_rno,
                                rno_num_opt_steps=args.rno_num_opt_steps,
                                rno_lr=args.rno_lr,
                                rno_temperature=args.rno_temperature,
                                rno_gamma=args.rno_gamma,
                                rno_pool_size=args.rno_pool_size,
                                rno_clip=args.rno_clip,
                                rno_kl_weight=args.rno_kl_weight,
                                rno_ref_repel_weight=args.rno_ref_repel_weight,
                                rno_positive_weight=args.rno_positive_weight,
                                rno_negative_prompt=(inf_prompt if args.rno_negative_prompt_no_suffix else None),
                                rno_fixed_neg_anchor=args.rno_fixed_neg_anchor,
                                rno_neg_anchor_extra=args.rno_neg_anchor_extra,
                                rno_force_per_sample=args.rno_force_per_sample,
                                rno_exclude_self_neg=args.rno_exclude_self_neg,
                                rno_optimize_start_idx=1,
                                return_target_only=True,
                            )

                            if args.dump_target_step_only:
                                saved += len(tar_seeds)
                                existing_counts[stem] = existing_counts.get(stem, 0) + len(tar_seeds)
                                pbar.update(len(tar_seeds))
                                total_pbar.update(len(tar_seeds))
                                continue

                            for local_img_idx, img in enumerate(out.images):
                                if (saved - existed) >= remaining_cnt:
                                    break
                                save_path = os.path.join(out_dir, f'{saved + 1:05d}_{stem}.png')
                                img.save(save_path)
                                target_prompt_for_image = (
                                    target_prompts_for_batch[local_img_idx]
                                    if target_prompts_for_batch and local_img_idx < len(target_prompts_for_batch)
                                    else inf_prompt
                                )
                                writer.writerow([
                                    cls, saved + 1, stem, save_path, target_prompt_for_image,
                                    ref_prompt, target_prompt_for_image, pipe_prompt_for_batch,
                                ])
                                saved += 1
                                existing_counts[stem] = existing_counts.get(stem, 0) + 1
                                pbar.update(1)
                                total_pbar.update(1)

                print(f'[{ci:03d}/{len(my_classes):03d}] {cls}: saved {saved} images to {out_dir}')

    print(f'[PROC {args.cur_proc}] Done.')


if __name__ == '__main__':
    main()

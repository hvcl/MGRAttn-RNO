# Adapted from Visual Style Prompting (https://github.com/naver-ai/Visual-Style-Prompting),
# licensed under the Apache License, Version 2.0. Modified for MGR Attention and RNO.

from __future__ import annotations

import math
import os
from typing import List, Optional

import torch
from einops import rearrange


def swapping_attention(key: torch.Tensor, value: torch.Tensor, chunk_size: int = 2):
    """
    Full-range K/V swapping: force K/V of all frames to use the reference frame at index 0.
    """
    chunk_length = key.size(0) // chunk_size  # F
    reference_image_index = [0] * chunk_length

    key = rearrange(key, "(b f) d c -> b f d c", f=chunk_length)
    key = key[:, reference_image_index]  # ref to all
    key = rearrange(key, "b f d c -> (b f) d c")

    value = rearrange(value, "(b f) d c -> b f d c", f=chunk_length)
    value = value[:, reference_image_index]  # ref to all
    value = rearrange(value, "b f d c -> (b f) d c")

    return key, value

class CrossFrameAttnProcessor:
    """
    Mask-guided cross-frame attention processor.

    - Operates only on self-attention.
    - Applies only within activate_step_indices.
    - Uses K/V tokens from reference foreground regions and updates only target foreground tokens.
    - If masks are unavailable, the processor can fall back to full-range swapping when enabled.
    """

    def __init__(
        self,
        unet_chunk_size: int = 2,             # B, usually 2 for [reference=0, target=1]
        attn_map_save_steps: list = None,
        activate_step_indices: list = None,   # e.g., [[38, 50]]
        do_swap_fallback: bool = True,
        max_inference_step: int = None,       # e.g., reset to -1 after steps 0..49 when max_inference_step=50
        layer_name: str = "",
        out_dir: str = "",
        use_mgr: bool = False,
        temperature: float = 0.5,
    ):
        self.layer_name = layer_name
        self.out_dir = out_dir
        self.use_mgr = use_mgr
        self.bank_mode = "mgr"

        self.unet_chunk_size = unet_chunk_size
        self.attn_map_save_steps = attn_map_save_steps or []
        self.activate_step_indices = activate_step_indices
        self.do_swap_fallback = do_swap_fallback
        self.max_inference_step = max_inference_step
        self.temperature = temperature

        self._raw_ref_mask: torch.Tensor = None  # (H,W) float[0,1]
        self._raw_tar_mask: torch.Tensor = None  # (H,W) float[0,1]

        self._raw_ref_mask_cond: torch.Tensor = None  # (H,W) float[0,1]
        self._raw_tar_mask_cond: torch.Tensor = None  # (H,W) float[0,1]
        self.ref_kv_cache_mode = "off"
        self.ref_kv_cache_root = None
        self.ref_kv_cache_id = None
        self._mgr_first_apply_logged = False
        self._mgr_missing_mask_logged = False

    @staticmethod
    def _as_mask_batch(mask):
        """
        Normalize mask input to [N, H, W].

        Accepts:
            [H, W]       -> [1, H, W]
            [1, H, W]    -> [1, H, W]
            [T, H, W]    -> [T, H, W]
            list of masks -> [T, H, W]

        This is required for batch-5 MGR:
            target masks = [mask_t1, mask_t2, mask_t3, mask_t4]
        """
        if mask is None:
            return None

        if isinstance(mask, (list, tuple)):
            masks = []
            for m in mask:
                mb = CrossFrameAttnProcessor._as_mask_batch(m)
                if mb is None:
                    continue
                if mb.ndim == 3 and mb.shape[0] == 1:
                    mb = mb[0]
                masks.append(mb)
            if len(masks) == 0:
                return None
            mask = torch.stack(masks, dim=0)

        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)

        mask = mask.float()
        if mask.max() > 1.5:
            mask = mask / 255.0

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 3:
            # [N,H,W] already. If [H,W,1], convert to [1,H,W].
            if mask.shape[-1] == 1 and mask.shape[0] != 1:
                mask = mask.permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

        return mask.clamp(0, 1)

    # Inject conditional masks from the pipeline.
    # ref_mask_cond: [H, W] or [1, H, W]
    # tar_mask_cond: [H, W] or [T, H, W], where T is the number of target images.
    def set_raw_masks(self, ref_mask_cond, tar_mask_cond):
        self._raw_ref_mask_cond = self._as_mask_batch(ref_mask_cond)
        self._raw_tar_mask_cond = self._as_mask_batch(tar_mask_cond)
        self.masks_ready = self._raw_ref_mask_cond is not None and self._raw_tar_mask_cond is not None

    def set_ref_kv_cache(self, mode: str = "off", root: str = None, cache_id: str = None):
        self.ref_kv_cache_mode = mode or "off"
        self.ref_kv_cache_root = root
        self.ref_kv_cache_id = cache_id

    @staticmethod
    def _safe_cache_name(name: str) -> str:
        return str(name).replace("/", "__").replace("\\", "__").replace(":", "_")

    def _ref_kv_cache_path(self, step: int, token_len: int, channel_dim: int):
        if self.ref_kv_cache_mode == "off" or not self.ref_kv_cache_root or not self.ref_kv_cache_id:
            return None
        layer = self._safe_cache_name(self.layer_name or "unknown_layer")
        cache_dir = os.path.join(self.ref_kv_cache_root, self.ref_kv_cache_id)
        return os.path.join(cache_dir, f"{layer}_step{int(step):03d}_D{int(token_len)}_C{int(channel_dim)}.pt")

    def _save_ref_kv_cache(self, step: int, k_ref: torch.Tensor, v_ref: torch.Tensor):
        path = self._ref_kv_cache_path(step, k_ref.shape[-2], k_ref.shape[-1])
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fg_ratio = None
        cache_content = "full_ref_kv"
        if self._raw_ref_mask_cond is not None:
            ref_m = self._resize_mask_tokens(
                self._raw_ref_mask_cond,
                k_ref.shape[-2],
                device=k_ref.device,
                dtype=k_ref.dtype,
            )
            if ref_m is not None:
                # Save a full-shape tensor for load-time compatibility, but zero out
                # non-foreground reference tokens. MGR still uses the mask at
                # attention time, so this cache is both compact in semantics and
                # safe to drop into the existing K/V path.
                ref_m = ref_m[:1].expand(k_ref.shape[0], -1, -1)
                k_ref = k_ref * ref_m
                v_ref = v_ref * ref_m
                fg_ratio = float(ref_m.detach().float().mean().cpu().item())
                cache_content = "masked_ref_fg_kv_full_shape"
        payload = {
            "layer_name": self.layer_name,
            "step": int(step),
            "shape": tuple(k_ref.shape),
            "cache_content": cache_content,
            "fg_token_ratio": fg_ratio,
            "k_ref": k_ref.detach().to("cpu", dtype=torch.float16),
            "v_ref": v_ref.detach().to("cpu", dtype=torch.float16),
        }
        torch.save(payload, path)

    def _load_ref_kv_cache(self, step: int, like_ref: torch.Tensor):
        path = self._ref_kv_cache_path(step, like_ref.shape[-2], like_ref.shape[-1])
        if not path or not os.path.isfile(path):
            return None, None
        payload = torch.load(path, map_location=like_ref.device)
        k_ref = payload.get("k_ref", None)
        v_ref = payload.get("v_ref", None)
        if k_ref is None or v_ref is None:
            return None, None
        k_ref = k_ref.to(device=like_ref.device, dtype=like_ref.dtype)
        v_ref = v_ref.to(device=like_ref.device, dtype=like_ref.dtype)
        if tuple(k_ref.shape) != tuple(like_ref.shape) or tuple(v_ref.shape) != tuple(like_ref.shape):
            print(
                f"[WARN] ref-KV cache shape mismatch for {self.layer_name} step={step}: "
                f"cache_k={tuple(k_ref.shape)} expected={tuple(like_ref.shape)}. Ignore cache."
            )
            return None, None
        return k_ref, v_ref

    @staticmethod
    def _resize_mask_tokens(raw_mask: torch.Tensor, token_len: int, device, dtype):
        """
        raw_mask: [N,H,W] or [H,W] float[0,1]
        token_len: D (= h*w)
        return: [N,D,1] float mask
        """
        raw_mask = CrossFrameAttnProcessor._as_mask_batch(raw_mask)
        if raw_mask is None:
            return None

        N = raw_mask.shape[0]
        h = int(math.sqrt(token_len))
        w = h if h > 0 else 1
        if h * w != token_len:
            h = max(1, int(round(math.sqrt(token_len))))
            w = max(1, token_len // h)

        m = raw_mask.to(device=device, dtype=dtype).unsqueeze(1)  # [N,1,H,W]
        m = torch.nn.functional.interpolate(m, size=(h, w), mode="nearest")
        m = m.reshape(N, h * w, 1)  # [N,D,1]
        return m.to(device=device, dtype=dtype).clamp(0, 1)

    def mgr_ref_fg_tar_bg_update_fg(
        self,
        k_bfdc: torch.Tensor,
        v_bfdc: torch.Tensor,
        q_bfdc: torch.Tensor,
        ref_m_bfd1: torch.Tensor,
        tar_m_tfd1: torch.Tensor,
        ref_idx: int = 0,
        tar_idx: int = 1,
        target_indices: Optional[List[int]] = None,
        temp: float = 1.0,
        mix_v: float = 1.0,
        mix_q: float = 0.0,
        debug: bool = False,
        bg_preserve: bool = True,
        save_v_feature: bool = True,
        save_dir: str = "./mgr_v_features",
        step: int = 0,
    ):
        """
        Multi-target MGR for batch-5 layout.

        Expected hidden layout after rearrange:
            k_bfdc/v_bfdc/q_bfdc: [B, F, D, C]

        For CFG batch-5:
            B = 2  # uncond / cond branches
            F = 5  # [ref, tar1, tar2, tar3, tar4]

        Masks:
            ref_m_bfd1 : [1,D,1] or [B,D,1]
            tar_m_tfd1 : [T,D,1], T = F-1 target masks

        Alignment rule:
            frame 1 -> tar_m_tfd1[0]
            frame 2 -> tar_m_tfd1[1]
            frame 3 -> tar_m_tfd1[2]
            frame 4 -> tar_m_tfd1[3]
        """
        B, F, D, C = v_bfdc.shape

        if target_indices is None:
            # By default, all frames except ref_idx are treated as targets.
            target_indices = [f for f in range(F) if f != ref_idx]


        # ref mask: [1,D,1] or [B,D,1] -> [B,D]
        ref_fg_mask = (ref_m_bfd1 > 0.5).squeeze(-1)
        if ref_fg_mask.ndim == 1:
            ref_fg_mask = ref_fg_mask.unsqueeze(0)
        if ref_fg_mask.shape[0] == 1:
            ref_fg_mask = ref_fg_mask.expand(B, -1)
        elif ref_fg_mask.shape[0] != B:
            raise ValueError(
                f"ref mask batch mismatch: ref_fg_mask={tuple(ref_fg_mask.shape)}, B={B}"
            )

        # target masks: [T,D,1] or [D,1] -> [T,D]
        tar_fg_masks = (tar_m_tfd1 > 0.5).squeeze(-1)
        if tar_fg_masks.ndim == 1:
            tar_fg_masks = tar_fg_masks.unsqueeze(0)

        num_target_masks = tar_fg_masks.shape[0]
        num_targets = len(target_indices)
        if num_target_masks not in (1, num_targets):
            # Keep running with nearest valid mapping, but make the issue visible.
            if debug:
                print(
                    f"[WARN] target mask count ({num_target_masks}) != target frame count ({num_targets}). "
                    "Using min(target_local_idx, num_target_masks-1)."
                )

        scale = (float(C) ** 0.5) * max(self.temperature, 1e-6)

        k_out = k_bfdc.clone()
        v_out = v_bfdc.clone()
        q_out = q_bfdc.clone()

        k_ref = k_bfdc[:, ref_idx]  # [B,D,C]
        v_ref = v_bfdc[:, ref_idx]

        for local_target_idx, frame_idx in enumerate(target_indices):
            if frame_idx < 0 or frame_idx >= F or frame_idx == ref_idx:
                continue

            # target frame <-> corresponding target mask alignment
            mask_idx = 0 if num_target_masks == 1 else min(local_target_idx, num_target_masks - 1)
            tar_fg_mask = tar_fg_masks[mask_idx]  # [D]
            tar_bg_mask = ~tar_fg_mask

            k_tar = k_bfdc[:, frame_idx]  # [B,D,C]
            v_tar = v_bfdc[:, frame_idx]
            q_tar = q_bfdc[:, frame_idx]
            v_tar_new = v_tar.clone()

            for b in range(B):
                cur_ref_fg = ref_fg_mask[b]  # [D]
                cur_tar_fg = tar_fg_mask     # [D], same mask for uncond/cond branch
                cur_tar_bg = tar_bg_mask

                tar_bg_idx = cur_tar_bg.nonzero(as_tuple=False).squeeze(-1)
                tar_fg_idx = cur_tar_fg.nonzero(as_tuple=False).squeeze(-1)
                ref_fg_idx = cur_ref_fg.nonzero(as_tuple=False).squeeze(-1)

                # If any required set is empty, keep original target V for stability.
                if tar_fg_idx.numel() == 0 or ref_fg_idx.numel() == 0:
                    continue

                if tar_bg_idx.numel() > 0:
                    K_bank = torch.cat(
                        [k_tar[b, tar_bg_idx, :], k_ref[b, ref_fg_idx, :]],
                        dim=0,
                    )
                    V_bank = torch.cat(
                        [v_tar[b, tar_bg_idx, :], v_ref[b, ref_fg_idx, :]],
                        dim=0,
                    )
                else:
                    K_bank = k_ref[b, ref_fg_idx, :]
                    V_bank = v_ref[b, ref_fg_idx, :]

                Q_fg = q_tar[b, tar_fg_idx, :].to(K_bank.dtype)
                logits = (Q_fg @ K_bank.transpose(0, 1)) / scale
                probs = torch.softmax(logits, dim=-1)
                ctx_fg = (probs @ V_bank).to(v_tar_new.dtype)

                # Update only FG tokens of the corresponding target frame.
                v_tar_new[b, tar_fg_idx, :] = (
                    (1.0 - mix_v) * v_tar[b, tar_fg_idx, :] + mix_v * ctx_fg
                )

                if bg_preserve:
                    v_tar_new[b, cur_tar_bg, :] = v_tar[b, cur_tar_bg, :]


            v_out[:, frame_idx] = v_tar_new

        return q_out, k_out, v_out

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        # per-layer step/state
        if not hasattr(attn, "attn_map"):
            setattr(attn, "attn_map", {})
            setattr(attn, "inference_step", 0)
        else:
            attn.inference_step += 1

        bsz, seq_len, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, bsz)

        # QKV
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        key   = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # MGR gate: one mask-guided interval only.
        mgr_gate = False
        if self.activate_step_indices:
            mgr_start, mgr_end = self.activate_step_indices[0]
            if mgr_start <= attn.inference_step <= mgr_end:
                mgr_gate = True

        # Skip MGR if masks are not ready.
        masks_ready = getattr(self, "masks_ready", False)

        if self.use_mgr and (not is_cross_attention) and mgr_gate and not masks_ready and not self._mgr_missing_mask_logged:
            print(
                "[MGR-APPLY][WARN] "
                f"layer={self.layer_name} step={attn.inference_step} "
                "MGR gate is active but masks_ready=False; MGR is skipped.",
                flush=True,
            )
            self._mgr_missing_mask_logged = True

        # ------------------------------------------------------------------
        # 1) MGR active interval: mask-guided reference value update
        # ------------------------------------------------------------------
        if self.use_mgr and (not is_cross_attention) and mgr_gate and masks_ready:
            B = self.unet_chunk_size
            F = key.size(0) // B
            D = key.size(1)

            if F >= 2 and \
                (self._raw_ref_mask_cond is not None) and \
                (self._raw_tar_mask_cond is not None):
                
                # (B,F,D,C)
                k_bfdc = rearrange(key,   "(b f) d c -> b f d c", b=B, f=F)
                v_bfdc = rearrange(value, "(b f) d c -> b f d c", b=B, f=F)
                q_bfdc = rearrange(query, "(b f) d c -> b f d c", b=B, f=F)

                if self.ref_kv_cache_mode == "save":
                    self._save_ref_kv_cache(attn.inference_step, k_bfdc[:, 0], v_bfdc[:, 0])
                elif self.ref_kv_cache_mode == "load":
                    cached_k_ref, cached_v_ref = self._load_ref_kv_cache(attn.inference_step, k_bfdc[:, 0])
                    if cached_k_ref is not None and cached_v_ref is not None:
                        k_bfdc[:, 0] = cached_k_ref
                        v_bfdc[:, 0] = cached_v_ref

                
                # ref mask: [1,H,W] -> [1,D,1]
                # target masks: [T,H,W] -> [T,D,1], T should be F-1 for batch-5
                ref_m_bfd1 = self._resize_mask_tokens(
                    self._raw_ref_mask_cond, D, device=key.device, dtype=key.dtype
                )
                tar_m_tfd1 = self._resize_mask_tokens(
                    self._raw_tar_mask_cond, D, device=key.device, dtype=key.dtype
                )

                target_indices = list(range(1, F))  # batch-5: [1,2,3,4]

                if not self._mgr_first_apply_logged:
                    ref_cov = float((ref_m_bfd1 > 0.5).float().mean().detach().cpu().item()) if ref_m_bfd1 is not None else -1.0
                    tar_cov = float((tar_m_tfd1 > 0.5).float().mean().detach().cpu().item()) if tar_m_tfd1 is not None else -1.0
                    print(
                        "[MGR-APPLY] "
                        f"layer={self.layer_name} step={attn.inference_step} "
                        f"use_mgr={self.use_mgr} "
                        f"B={B} F={F} D={D} C={key.shape[-1]} "
                        f"ref_mask_shape={tuple(ref_m_bfd1.shape) if ref_m_bfd1 is not None else None} "
                        f"tar_mask_shape={tuple(tar_m_tfd1.shape) if tar_m_tfd1 is not None else None} "
                        f"ref_fg_ratio={ref_cov:.4f} tar_fg_ratio={tar_cov:.4f} "
                        f"targets={target_indices}",
                        flush=True,
                    )
                    self._mgr_first_apply_logged = True

                if tar_m_tfd1 is not None and tar_m_tfd1.shape[0] not in (1, len(target_indices)):
                    print(
                        f"[WARN] {self.layer_name}: target mask count={tar_m_tfd1.shape[0]} "
                        f"but target frame count={len(target_indices)}. "
                        "Masks will be mapped by min(index, last_mask)."
                    )

                q_bfdc, k_bfdc, v_bfdc = self.mgr_ref_fg_tar_bg_update_fg(
                        k_bfdc, v_bfdc, q_bfdc, ref_m_bfd1, tar_m_tfd1,
                        ref_idx=0,
                        target_indices=target_indices,
                        debug=False,
                        save_dir=self.out_dir,
                        save_v_feature=False,
                        step=attn.inference_step,
                        bg_preserve=False,
                    )
                    
                # Restore the original tensor layout.
                key   = rearrange(k_bfdc, "b f d c -> (b f) d c")
                value = rearrange(v_bfdc, "b f d c -> (b f) d c")
                query = rearrange(q_bfdc, "b f d c -> (b f) d c")
                    
                # ----- Standard attention computation -----
                query = attn.head_to_batch_dim(query)
                key   = attn.head_to_batch_dim(key)
                value = attn.head_to_batch_dim(value)

                attention_probs = attn.get_attention_scores(query, key, attention_mask)

                    # Save attention maps for debugging.
                if attn.inference_step in self.attn_map_save_steps:
                        attn.attn_map[attn.inference_step] = attention_probs.detach().cpu().clone()

                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = attn.batch_to_head_dim(hidden_states)
                
                # print(f"Step {attn.inference_step}: MGR applied with mgr_ref_fg_tar_bg_update_fg.")

        # ------------------------------------------------------------------
        # 2) Default path: cross-attention, inactive interval, or missing masks
        #    -> Standard attention without MGR updates.
        else:
            q = attn.head_to_batch_dim(query)
            k = attn.head_to_batch_dim(key)
            v = attn.head_to_batch_dim(value)

            attn_probs = attn.get_attention_scores(q, k, attention_mask)
            hidden_states = torch.bmm(attn_probs, v)
            hidden_states = attn.batch_to_head_dim(hidden_states)

        # ------------------------------------------------------------------
        # Common output projection.
        # ------------------------------------------------------------------
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # Optional step reset.
        if self.max_inference_step is not None:
            if attn.inference_step >= (self.max_inference_step - 1):
                attn.inference_step = -1

        return hidden_states

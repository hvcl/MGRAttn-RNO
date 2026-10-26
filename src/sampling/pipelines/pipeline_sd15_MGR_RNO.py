# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from the diffusers StableDiffusionPipeline to add MGR Attention and RNO.

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
# For faster inference
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    LoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

from pipelines.inverted_ve_pipeline_MGR_RNO import CrossFrameAttnProcessor
from PIL import Image as PILImage
import os
import subprocess
import numpy as np
from PIL import Image
import cv2
import glob

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Attention processor candidates for Stable Diffusion 1.5.
SD15_ACTIVATE_LAYER_CANDIDATE = [
    'down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor',
    'down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor',
    'down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor',
    'down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor',
    
    'down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor',
    'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor',
    'down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor',
    'down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor',
    
    'mid_block.attentions.0.transformer_blocks.0.attn1.processor',
    'mid_block.attentions.0.transformer_blocks.0.attn2.processor',

    'up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor',
    'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor',
    'up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor',
    'up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor',
    'up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor',
    'up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor',
    
    'up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor',
    'up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor',
    'up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor',
    'up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor',
    'up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor',
    'up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor',
    
    'up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor',
    'up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor',
    'up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor',
    'up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor',
    'up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor',
    'up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor',
]

up1_only = {
    'up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor',
    'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor',
    'up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor',
    'up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor',
    'up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor',
    'up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor',
}

up2_only = {
    'up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor',
    'up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor',
    'up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor',
    'up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor',
    'up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor',
    'up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor',
}

# Helper set of layer names grouped by UNet blocks.
UP2_12_UP3_NAMES = {
    # 'mid_block.attentions.0.transformer_blocks.0.attn1.processor',
    # 'mid_block.attentions.0.transformer_blocks.0.attn2.processor',
    # 'up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor',
    # 'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor',
    # 'up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor',
    # 'up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor',
    # 'up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor',
    # 'up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor',    
    # 'up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor',
    # 'up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor',
    # 'up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor',
    # 'up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor',
    'up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor',
    'up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor',
    'up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor',
    'up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor',
    'up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor',
    'up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor',
    'up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor',
    'up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor',
}

up2_1 = {
    'up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor',
    'up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor',
    'up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor',
    'up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor',
    'up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor',
    'up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor',
    'up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor',
    'up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor',
    'up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor',
    'up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor',    
}

up2_2 = {
    'up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor',
    'up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor',
    'up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor',
    'up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor',
    'up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor',
    'up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor',
    'up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor',
    'up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor',    
}

mid_only = {
    'mid_block.attentions.0.transformer_blocks.0.attn1.processor',
    'mid_block.attentions.0.transformer_blocks.0.attn2.processor',
}

down_only = {
    'down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor',
    'down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor',
    'down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor',
    'down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor',
    
    'down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor',
    'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor',
    'down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor',
    'down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor',
}

# MID_NAMES = {
#     'mid_block.attentions.0.transformer_blocks.0.attn1.processor',
#     'mid_block.attentions.0.transformer_blocks.0.attn2.processor',
# }

# =========================================================
# Original rno implementation
# Diverse Text-to-Image Generation via Contrastive Noise Optimization
# All helper names are prefixed with orig_ to avoid confusion.
# =========================================================


def _ranges(a, b):  # inclusive range [a, b]
    return [[int(a), int(b)]]

def _concat(*xs):
    out = []
    for x in xs:
        out += x
    return out

## rno utilities ##
def _get_alpha_cumprod(scheduler, t, device=None, dtype=None):
    if isinstance(t, torch.Tensor):
        t_idx = int(t.flatten()[0].item())
    else:
        t_idx = int(t)

    alphas_cumprod = scheduler.alphas_cumprod
    if not torch.is_tensor(alphas_cumprod):
        alphas_cumprod = torch.tensor(alphas_cumprod, device=device, dtype=dtype)
    else:
        alphas_cumprod = alphas_cumprod.to(device=device, dtype=dtype)

    return alphas_cumprod[t_idx]


def tweedie_x0_stopgrad(z_t, eps_pred, scheduler, t):
    """
    Algorithm 1 line 6:
    x0_hat = (z_t - sqrt(1 - alpha_bar_t) * stopgrad(eps_theta(z_t, t, c))) / sqrt(alpha_bar_t)
    """
    alpha_bar_t = _get_alpha_cumprod(
        scheduler, t, device=z_t.device, dtype=z_t.dtype
    )
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)

    while sqrt_alpha_bar.ndim < z_t.ndim:
        sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(0)
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(0)

    eps_pred = eps_pred.detach()  # stop-gradient on model path
    x0_hat = (z_t - sqrt_one_minus_alpha_bar * eps_pred) / (sqrt_alpha_bar + 1e-8)
    return x0_hat


def adaptive_pool_and_normalize(x, pool_size=16):
    """
    Algorithm 1 line 10-11:
    DownSample(...; w) -> Normalize(...)
    """
    x = F.adaptive_avg_pool2d(x, (pool_size, pool_size))
    x = x.flatten(1)
    x = F.normalize(x, p=2, dim=1)
    return x


def rno_loss_exact(
    x_cur,
    x_ref,
    temperature=0.07,
    gamma=1.0,
    x_extra_neg=None,
    extra_negative_weight=1.0,
):
    """
    Algorithm 1 line 13 / Eq. (7)

    numerator   : exp(sim(x_cur_i, x_ref_i) / (gamma * tau))
    denominator : sum_j exp(sim(x_cur_i, x_cur_j) / tau)
                  plus optional extra negative terms.
    """
    # cosine similarity because x_cur/x_ref are already L2-normalized
    sim_cur_ref = torch.sum(x_cur * x_ref, dim=1) / (gamma * temperature)     # [B]
    sim_cur_cur = torch.matmul(x_cur, x_cur.T) / temperature                  # [B, B]

    denom_terms = [sim_cur_cur]
    if x_extra_neg is not None and float(extra_negative_weight or 0.0) > 0.0:
        sim_extra = torch.matmul(x_cur, x_extra_neg.T) / temperature          # [B, R]
        if float(extra_negative_weight) != 1.0:
            sim_extra = sim_extra + torch.log(
                torch.tensor(float(extra_negative_weight), device=x_cur.device, dtype=x_cur.dtype)
            )
        denom_terms.append(sim_extra)

    log_denom = torch.logsumexp(torch.cat(denom_terms, dim=1), dim=1)          # [B]
    loss = (-sim_cur_ref + log_denom).mean()
    return loss


def gaussian_prior_kl_for_noise(z):
    """
    Appendix KL regularization for keeping optimized initial noises close to N(0, I).
    """
    flat = z.float().flatten(1)
    mu = flat.mean(dim=1)
    sigma = flat.std(dim=1, unbiased=False).clamp_min(1e-8)
    kl = torch.log(1.0 / sigma) + 0.5 * (sigma.square() + mu.square()) - 0.5
    return kl.mean()


def _repeat_cond_embed_to_batch(cond_embed, batch_size):
    """
    cond_embed: [1, L, D] or [B, L, D]
    return: [batch_size, L, D]
    """
    if cond_embed.shape[0] == batch_size:
        return cond_embed
    if cond_embed.shape[0] == 1:
        return cond_embed.repeat(batch_size, 1, 1)
    raise ValueError(
        f"Cannot match cond_embed batch={cond_embed.shape[0]} to target batch={batch_size}"
    )


def _disable_cross_frame_processors(unet):
    proc_states = []
    for _, module in unet.named_modules():
        proc = getattr(module, "processor", None)
        if isinstance(proc, CrossFrameAttnProcessor):
            proc_states.append(
                (
                    proc,
                    getattr(proc, "use_mgr", None),
                    getattr(proc, "masks_ready", None),
                    list(getattr(proc, "activate_step_indices", [])),
                )
            )
            proc.use_mgr = False
            proc.masks_ready = False
            proc.activate_step_indices = []
    return proc_states


def _restore_cross_frame_processors(proc_states):
    for proc, use_mgr_state, masks_ready_state, activate_steps_state in proc_states:
        if use_mgr_state is not None:
            proc.use_mgr = use_mgr_state
        if masks_ready_state is not None:
            proc.masks_ready = masks_ready_state
        proc.activate_step_indices = activate_steps_state




def rno_optimize_initial_noise_latents(
    pipeline,
    z_t,
    t,
    pos_embed,
    neg_embed=None,
    num_opt_steps=3,
    lr=0.01,
    temperature=0.07,
    gamma=1.0,
    pool_size=16,
    optimize_start_idx=1,
    noise_clip=None,
    kl_weight=0.0,
    ref_repel_weight=0.0,
    fixed_neg_anchor=False,
    neg_anchor_extra=0,
    force_per_sample=False,
    exclude_self_neg=False,
    positive_weight=1.0,
):
    """
    rno-style initial noise optimization.

    z_t layout may be [ref, target_1, ... target_B]. Only target latents are
    optimized. The anchor x_ref is the Tweedie clean estimate from the original
    target noise; x_cur is the Tweedie estimate from the optimized noise.
    """
    if z_t.shape[0] <= optimize_start_idx + 1:
        return z_t

    device = z_t.device
    dtype = z_t.dtype
    z_ref = z_t[:optimize_start_idx].detach()
    z_base = z_t[optimize_start_idx:].detach()
    B = z_base.shape[0]
    pos_base = _repeat_cond_embed_to_batch(pos_embed.detach(), B).to(device=device, dtype=dtype)
    pos_ref = _repeat_cond_embed_to_batch(
        pos_embed.detach()[:1], max(1, optimize_start_idx)
    ).to(device=device, dtype=dtype)
    neg_base = None
    neg_ref = None
    if neg_embed is not None:
        neg_base = _repeat_cond_embed_to_batch(neg_embed.detach(), B).to(device=device, dtype=dtype)
        neg_ref = _repeat_cond_embed_to_batch(
            neg_embed.detach()[:1], max(1, optimize_start_idx)
        ).to(device=device, dtype=dtype)

    proc_states = _disable_cross_frame_processors(pipeline.unet)

    try:
        with torch.enable_grad():
            with torch.no_grad():
                model_in = pipeline.scheduler.scale_model_input(z_base, t)
                eps_ref = pipeline.unet(
                    model_in,
                    t,
                    encoder_hidden_states=pos_base,
                    return_dict=False,
                )[0]
                x_ref = tweedie_x0_stopgrad(z_base, eps_ref, pipeline.scheduler, t)
                x_ref = adaptive_pool_and_normalize(x_ref.float(), pool_size=pool_size).detach()

                x_ref_branch_neg = None
                if float(ref_repel_weight or 0.0) > 0.0 and optimize_start_idx > 0:
                    model_in_ref_branch = pipeline.scheduler.scale_model_input(z_ref, t)
                    eps_ref_branch = pipeline.unet(
                        model_in_ref_branch,
                        t,
                        encoder_hidden_states=neg_ref if neg_ref is not None else pos_ref,
                        return_dict=False,
                    )[0]
                    x_ref_branch_neg = tweedie_x0_stopgrad(z_ref, eps_ref_branch, pipeline.scheduler, t)
                    x_ref_branch_neg = adaptive_pool_and_normalize(
                        x_ref_branch_neg.float(), pool_size=pool_size
                    ).detach()

                x_neg_anchor = None
                if neg_base is not None and fixed_neg_anchor:
                    z_neg_anchor = z_base
                    extra_neg_count = max(0, int(neg_anchor_extra or 0))
                    if extra_neg_count > 0:
                        sigma = getattr(getattr(pipeline, "scheduler", None), "init_noise_sigma", 1.0)
                        z_extra = torch.randn(
                            (extra_neg_count, *z_base.shape[1:]),
                            device=device,
                            dtype=dtype,
                        ) * float(sigma)
                        z_neg_anchor = torch.cat([z_base, z_extra], dim=0)
                    neg_anchor_embed = _repeat_cond_embed_to_batch(
                        neg_embed.detach()[:1], z_neg_anchor.shape[0]
                    ).to(device=device, dtype=dtype)
                    model_in_neg_anchor = pipeline.scheduler.scale_model_input(z_neg_anchor, t)
                    eps_neg_anchor = pipeline.unet(
                        model_in_neg_anchor,
                        t,
                        encoder_hidden_states=neg_anchor_embed,
                        return_dict=False,
                    )[0]
                    x_neg_anchor = tweedie_x0_stopgrad(
                        z_neg_anchor, eps_neg_anchor, pipeline.scheduler, t
                    )
                    x_neg_anchor = adaptive_pool_and_normalize(
                        x_neg_anchor.float(), pool_size=pool_size
                    ).detach()

            z_opt = z_base.clone().detach().requires_grad_(True)
            if B <= 4 and not force_per_sample:
                for _ in range(int(num_opt_steps)):
                    x_neg_all = None
                    if neg_base is not None:
                        if x_neg_anchor is not None:
                            x_neg_all = x_neg_anchor
                        else:
                            with torch.no_grad():
                                model_in_neg = pipeline.scheduler.scale_model_input(z_opt.detach(), t)
                                eps_neg = pipeline.unet(
                                    model_in_neg,
                                    t,
                                    encoder_hidden_states=neg_base,
                                    return_dict=False,
                                )[0]
                                x_neg_all = tweedie_x0_stopgrad(z_opt.detach(), eps_neg, pipeline.scheduler, t)
                                x_neg_all = adaptive_pool_and_normalize(
                                    x_neg_all.float(), pool_size=pool_size
                                ).detach()
                    model_in = pipeline.scheduler.scale_model_input(z_opt, t)
                    eps = pipeline.unet(
                        model_in,
                        t,
                        encoder_hidden_states=pos_base,
                        return_dict=False,
                    )[0]
                    x_cur = tweedie_x0_stopgrad(z_opt, eps, pipeline.scheduler, t)
                    x_cur = adaptive_pool_and_normalize(x_cur.float(), pool_size=pool_size)
                    if x_neg_all is None:
                        loss = rno_loss_exact(
                            x_cur,
                            x_ref,
                            temperature=temperature,
                            gamma=gamma,
                            x_extra_neg=x_ref_branch_neg,
                            extra_negative_weight=ref_repel_weight,
                        )
                    else:
                        sim_pos = torch.sum(x_cur * x_ref, dim=1) / (gamma * temperature)
                        denom_terms = [torch.matmul(x_cur, x_neg_all.T) / temperature]
                        if x_ref_branch_neg is not None and float(ref_repel_weight or 0.0) > 0.0:
                            sim_extra = torch.matmul(x_cur, x_ref_branch_neg.T) / temperature
                            if float(ref_repel_weight) != 1.0:
                                sim_extra = sim_extra + torch.log(
                                    torch.tensor(float(ref_repel_weight), device=x_cur.device, dtype=x_cur.dtype)
                                )
                            denom_terms.append(sim_extra)
                        loss = -float(positive_weight) * sim_pos + torch.logsumexp(
                            torch.cat(denom_terms, dim=1),
                            dim=1,
                        )
                        loss = loss.mean()
                    if float(kl_weight or 0.0) > 0.0:
                        loss = loss + float(kl_weight) * gaussian_prior_kl_for_noise(z_opt)
                    grad = torch.autograd.grad(loss, z_opt, retain_graph=False, create_graph=False)[0]

                    with torch.no_grad():
                        z_opt = z_opt - float(lr) * grad
                        if noise_clip is not None:
                            sigma = getattr(getattr(pipeline, "scheduler", None), "init_noise_sigma", 1.0)
                            z_opt = z_opt.clamp(-float(noise_clip) * float(sigma), float(noise_clip) * float(sigma))
                    z_opt.requires_grad_(True)
            else:
    
                print('Running per-sample optimization with batch size', B)
                z_opt = z_base.clone().detach()

                for _ in range(int(num_opt_steps)):

                    # ---------------------------------------------------------
                    # Prepare negative anchors.
                    # If neg_embed / x_neg_anchor is unavailable, keep x_neg_all=None
                    # so that the loss falls back to x_ref_branch_neg via rno_loss_exact.
                    # ---------------------------------------------------------
                    with torch.no_grad():
                        x_neg_all = None

                        if neg_base is not None:
                            if x_neg_anchor is not None:
                                x_neg_all = x_neg_anchor
                            else:
                                model_in_all = pipeline.scheduler.scale_model_input(z_opt.detach(), t)
                                eps_all = pipeline.unet(
                                    model_in_all,
                                    t,
                                    encoder_hidden_states=neg_base,
                                    return_dict=False,
                                )[0]
                                x_neg_all = tweedie_x0_stopgrad(
                                    z_opt.detach(),
                                    eps_all,
                                    pipeline.scheduler,
                                    t,
                                )
                                x_neg_all = adaptive_pool_and_normalize(
                                    x_neg_all.float(),
                                    pool_size=pool_size,
                                ).detach()

                    updated = []

                    for i in range(B):
                        z_i = z_opt[i:i + 1].clone().detach().requires_grad_(True)

                        model_in_i = pipeline.scheduler.scale_model_input(z_i, t)
                        eps_i = pipeline.unet(
                            model_in_i,
                            t,
                            encoder_hidden_states=pos_base[i:i + 1],
                            return_dict=False,
                        )[0]

                        x_i = tweedie_x0_stopgrad(
                            z_i,
                            eps_i,
                            pipeline.scheduler,
                            t,
                        )
                        x_i = adaptive_pool_and_normalize(
                            x_i.float(),
                            pool_size=pool_size,
                        )

                        # -----------------------------------------------------
                        # Case 1: no negative anchor pool.
                        # Use z_ref-derived x_ref_branch_neg as x_extra_neg,
                        # exactly like the batch branch:
                        #
                        #   rno_loss_exact(..., x_extra_neg=x_ref_branch_neg)
                        # -----------------------------------------------------
                        if x_neg_all is None:
                            loss = rno_loss_exact(
                                x_i,
                                x_ref[i:i + 1],
                                temperature=temperature,
                                gamma=gamma,
                                x_extra_neg=x_ref_branch_neg,
                                extra_negative_weight=ref_repel_weight,
                            )

                            # Apply an additional correction when positive_weight is used.
                            # This handles the case where rno_loss_exact does not take positive_weight directly.
                            if float(positive_weight) != 1.0:
                                sim_pos = torch.sum(
                                    x_i * x_ref[i:i + 1],
                                    dim=1,
                                ) / (gamma * temperature)

                                # The default rno_loss_exact already includes -sim_pos.
                                # To replace it with positive_weight:
                                # -sim_pos -> -positive_weight * sim_pos
                                # Add only the difference as a correction term.
                                loss = loss + (1.0 - float(positive_weight)) * sim_pos.mean()

                        # -----------------------------------------------------
                        # Case 2: negative anchor pool exists.
                        # Keep the existing behavior, while also adding x_ref_branch_neg as an extra negative.
                        # -----------------------------------------------------
                        else:
                            sim_pos = torch.sum(
                                x_i * x_ref[i:i + 1],
                                dim=1,
                            ) / (gamma * temperature)

                            if exclude_self_neg and x_neg_all.shape[0] >= B and B > 1:
                                neg_for_i = torch.cat(
                                    [x_neg_all[:i], x_neg_all[i + 1:]],
                                    dim=0,
                                )
                            else:
                                neg_for_i = x_neg_all

                            denom_terms = [
                                torch.matmul(x_i, neg_for_i.T) / temperature
                            ]

                            if x_ref_branch_neg is not None and float(ref_repel_weight or 0.0) > 0.0:
                                sim_extra = torch.matmul(
                                    x_i,
                                    x_ref_branch_neg.T,
                                ) / temperature

                                if float(ref_repel_weight) != 1.0:
                                    sim_extra = sim_extra + torch.log(
                                        torch.tensor(
                                            float(ref_repel_weight),
                                            device=x_i.device,
                                            dtype=x_i.dtype,
                                        )
                                    )

                                denom_terms.append(sim_extra)

                            loss = -float(positive_weight) * sim_pos + torch.logsumexp(
                                torch.cat(denom_terms, dim=1),
                                dim=1,
                            )
                            loss = loss.mean()

                        if float(kl_weight or 0.0) > 0.0:
                            loss = loss + float(kl_weight) * gaussian_prior_kl_for_noise(z_i)

                        grad_i = torch.autograd.grad(
                            loss,
                            z_i,
                            retain_graph=False,
                            create_graph=False,
                        )[0]

                        with torch.no_grad():
                            z_next = z_i - float(lr) * grad_i

                            if noise_clip is not None:
                                sigma = getattr(
                                    getattr(pipeline, "scheduler", None),
                                    "init_noise_sigma",
                                    1.0,
                                )
                                z_next = z_next.clamp(
                                    -float(noise_clip) * float(sigma),
                                    float(noise_clip) * float(sigma),
                                )

                        updated.append(z_next.detach())

                    z_opt = torch.cat(updated, dim=0).detach()
    finally:
        _restore_cross_frame_processors(proc_states)

    out = torch.cat([z_ref, z_opt.detach().to(dtype=dtype)], dim=0)
    del z_base, pos_base, pos_ref, z_opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out

##################################

def _extract_prev_sample(step_out, expected_channels: int):
    # SchedulerOutput-like
    if hasattr(step_out, "pred_original_sample"):
        return step_out.pred_original_sample
    if hasattr(step_out, "prev_sample"):
        return step_out.prev_sample
    if isinstance(step_out, dict):
        for key in ("pred_original_sample", "prev_sample", "prev_sample"):
            if key in step_out:
                return step_out[key]
    if isinstance(step_out, (tuple, list)) and len(step_out) > 0:
        # choose first 4D tensor-like element that matches channels
        for elem in step_out:
            if isinstance(elem, torch.Tensor) and elem.ndim == 4 and elem.shape[1] == expected_channels:
                return elem
        return step_out[0]
    raise RuntimeError(f"Unable to extract prev_sample from scheduler.step() output (type={type(step_out)})")

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

# 1. Preload the zero-shot segmentation model inside the pipeline class for reuse.
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

class StableDiffusionPipeline15MGR(
    DiffusionPipeline, FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion 1.5 with Visual Style Prompting support.
    """
    
    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]
    _exclude_from_cpu_offload = ["safety_checker"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

        self.setup_clip_models()

        # MGR start/end steps. MGR is applied only after foreground masks are attached.
        self.step_start_mgr: int = 20
        self.step_end_mgr: int = 49
        # Initialize target/reference mask caches.
        self._ref_mask_cache = None
        self._tar_mask_cache = None

    def setup_clip_models(self, input_size=352):
        # Preload the zero-shot segmentation model.
        self.clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to("cuda").eval()

        H = W = int(input_size)
        # Configure image_processor/feature_extractor size.
        if hasattr(self.clipseg_processor, "image_processor"):
            self.clipseg_processor.image_processor.size = {"height": H, "width": W}
            self.clipseg_processor.image_processor.crop_size = {"height": H, "width": W}
        elif hasattr(self.clipseg_processor, "feature_extractor"):
            self.clipseg_processor.feature_extractor.size = {"height": H, "width": W}
            self.clipseg_processor.feature_extractor.crop_size = {"height": H, "width": W}
        # Adjust positional encoding settings when available.
        if hasattr(self.clipseg_model.config, "vision_config"):
            try:
                setattr(self.clipseg_model.config.vision_config, "interpolate_pos_encoding", True)
            except Exception:
                pass

    def run_clipseg(self, pil_img, clip_prompt, prefix='cond', step=0,
                    threshold=0.45, open_kernel=0, close_kernel=5, dilate_iter=1, device='cuda',
                    save_overlay=False, save_binary=False, 
                    dir_overlay=None, dir_binary=None, 
                    fn_overlay=None, fn_binary=None, 
                    ):
        """
        Generate and optionally save masks using CLIPSeg.
        """
        if isinstance(pil_img, list):
            pil_img = pil_img[0]
        orig_w, orig_h = pil_img.size

        # Parse prompt list.
        pos_prompts = [p.strip() for p in clip_prompt.split(",") if p.strip()]
        neg_prompts = ["background"]

        # Prepare inputs.
        inputs_pos = self.clipseg_processor(
            text=pos_prompts, images=[pil_img] * len(pos_prompts),
            padding=True, truncation=True, return_tensors="pt"
        )
        if "pixel_values" in inputs_pos:
            inputs_pos["pixel_values"] = inputs_pos["pixel_values"].to(device)
        inputs_pos = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs_pos.items()}

        with torch.no_grad():
            out_pos = self.clipseg_model(**inputs_pos).logits
            if out_pos.ndim == 2:
                out_pos = out_pos.unsqueeze(0).unsqueeze(0)
            elif out_pos.ndim == 3:
                out_pos = out_pos.unsqueeze(1)
            out_pos = F.interpolate(out_pos, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            logits_pos = out_pos.squeeze(1)  # [B,H,W]
            logits_pos = logits_pos.max(dim=0).values  # ensemble by max pooling

        # Negative (optional)
        if neg_prompts:
            inputs_neg = self.clipseg_processor(
                text=neg_prompts, images=[pil_img] * len(neg_prompts),
                padding=True, truncation=True, return_tensors="pt"
            )
            inputs_neg = {k: v.to(device) for k, v in inputs_neg.items()}
            # out_neg = model(**inputs_neg, interpolate_pos_encoding=True).logits
            with torch.no_grad():
                out_neg = self.clipseg_model(**inputs_neg).logits
                if out_neg.ndim == 2:
                    out_neg = out_neg.unsqueeze(0).unsqueeze(0)   # [1,1,h,w]
                elif out_neg.ndim == 3:
                    out_neg = out_neg.unsqueeze(1)                # [B,1,h,w]
            out_neg = F.interpolate(out_neg, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            logits_neg = out_neg.squeeze(1).max(dim=0).values
        else:
            logits_neg = torch.zeros_like(logits_pos)

        # Combine positive and negative logits to suppress background responses.
        lam = 0.6
        logits_comb = logits_pos - lam * logits_neg
        prob = torch.sigmoid(logits_comb).cpu().numpy()   # [H,W]

        # Post-process masks with open, close, and dilation. This mirrors flower_seg_once.py /
        # clipseg_zeroshot_v2.py used for the saved Flower ref segOnce masks.
        mask = (prob >= threshold).astype(np.uint8) * 255
        if open_kernel > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        if close_kernel > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        if dilate_iter > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            mask = cv2.dilate(mask, k, iterations=dilate_iter)

        # Create overlay.
        # Default overlay image.
        overlay_img = None

        if save_binary:
            if os.path.exists(dir_binary) is False:
                os.makedirs(dir_binary)
            if fn_binary is None:
                fn_binary = f"step{step:02d}_{prefix}_mask_binary.png"
            Image.fromarray(mask).save(os.path.join(dir_binary, fn_binary))
        if save_overlay:
            img_np = np.asarray(pil_img).astype(np.uint8)
            if mask.shape[:2] != img_np.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (img_np.shape[1], img_np.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            overlay = img_np.copy()
            color = np.array([255, 0, 0], dtype=np.uint8)
            m = mask > 0
            overlay[m] = (0.5 * color + 0.5 * overlay[m]).astype(np.uint8)
            overlay_img = Image.fromarray(overlay)
            if os.path.exists(dir_overlay) is False:
                os.makedirs(dir_overlay)
            if fn_overlay is None:
                fn_overlay = f"step{step:02d}_{prefix}_overlay.png"
            overlay_img.save(os.path.join(dir_overlay, fn_overlay))
            pil_img.save(os.path.join(dir_overlay, "3_tar_mask_cond.png"))
        
        # Always return in-memory outputs regardless of save flags.
        return {
            "prob": prob,                              # np.float32 [H,W], 0~1
            "mask_uint8": mask,                        # np.uint8 [H,W], {0,255}
            "mask_bool": (mask > 0),                   # np.bool_ [H,W]
            "overlay": overlay_img                     # PIL.Image or None
        }

    def run_attention_saliency_mask(self, pil_img, prefix='cond', step=0,
                                    threshold=0.60, close_kernel=5, dilate_iter=1,
                                    save_overlay=False, save_binary=False,
                                    dir_overlay=None, dir_binary=None,
                                    fn_overlay=None, fn_binary=None):
        """
        Lightweight target-side attention/saliency proxy from the Tweedie x0 image.
        This is saved for mask debugging/comparison against CLIPSeg; CLIPSeg remains
        the default mask used by MGR attention unless explicitly changed elsewhere.
        """
        if isinstance(pil_img, list):
            pil_img = pil_img[0]
        img_np = np.asarray(pil_img.convert("RGB")).astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (0, 0), 3)
        high = cv2.absdiff(gray, gray_blur)
        color_std = img_np.astype(np.float32).std(axis=2)
        sal = high.astype(np.float32) + 0.35 * color_std
        sal = sal - sal.min()
        if sal.max() > 1e-6:
            sal = sal / sal.max()
        mask = (sal >= float(threshold)).astype(np.uint8) * 255
        if close_kernel > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        if dilate_iter > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, k, iterations=dilate_iter)

        overlay_img = None
        if save_binary:
            os.makedirs(dir_binary, exist_ok=True)
            if fn_binary is None:
                fn_binary = f"step{step:02d}_{prefix}_attention_saliency_binary.png"
            Image.fromarray(mask).save(os.path.join(dir_binary, fn_binary))
        if save_overlay:
            overlay = img_np.copy()
            color = np.array([0, 128, 255], dtype=np.uint8)
            m = mask > 0
            overlay[m] = (0.5 * color + 0.5 * overlay[m]).astype(np.uint8)
            overlay_img = Image.fromarray(overlay)
            os.makedirs(dir_overlay, exist_ok=True)
            if fn_overlay is None:
                fn_overlay = f"step{step:02d}_{prefix}_attention_saliency_overlay.png"
            overlay_img.save(os.path.join(dir_overlay, fn_overlay))

        return {
            "prob": sal,
            "mask_uint8": mask,
            "mask_bool": (mask > 0),
            "overlay": overlay_img,
        }

    def activate_mgr(self, *,
                     use_mgr: bool = True,
                     temperature: float = 0.5,
                     mgr_block_from: str = "",
                     unet_chunk_size: int = 2,
                     num_inference_steps: int = 50,
                     step_start_mgr: int = 20,
                     step_end_mgr: int = None,
                     bank_mode: str = "mgr",):
        """
        Activate Mask-Guided Reference Attention (MGR).

        In this release, there is no separate warm-up stage.
        MGR starts when foreground masks are available:

            step_start_mgr .. step_end_mgr

        Before step_start_mgr, the UNet uses the standard attention processor.
        """
        if mgr_block_from == "up2_1":
            mgr_blocks = up2_1
        elif mgr_block_from == "up2_2":
            mgr_blocks = up2_2
        elif mgr_block_from == "up1_only":
            mgr_blocks = up1_only
        elif mgr_block_from == "up2_only":
            mgr_blocks = up2_only
        elif mgr_block_from == "mid_only":
            mgr_blocks = mid_only
        elif mgr_block_from == "down_only":
            mgr_blocks = down_only
        else:
            print(f"[!] Unknown mgr_block_from: {mgr_block_from}")
            exit()

        if step_end_mgr is None:
            step_end_mgr = int(num_inference_steps) - 1

        self.step_start_mgr = int(step_start_mgr)
        self.step_end_mgr = int(step_end_mgr)
        self.range_mgr = [self.step_start_mgr, self.step_end_mgr]
        attn_procs = {}

        act_mgr = _ranges(self.step_start_mgr, self.step_end_mgr)

        def _iter_attn_processor_names():
            if hasattr(self.unet, "attn_processors"):
                return list(self.unet.attn_processors.keys())
            names = []
            for module_name, module in self.unet.named_modules():
                if hasattr(module, "processor"):
                    names.append(f"{module_name}.processor")
            return names

        all_layers = _iter_attn_processor_names()
        selected_layers = []
        for name in all_layers:
            if name in mgr_blocks:
                selected_layers.append(name)
                attn_procs[name] = CrossFrameAttnProcessor(
                    unet_chunk_size=unet_chunk_size,
                    attn_map_save_steps=[],
                    activate_step_indices=act_mgr,
                    layer_name=name,
                    out_dir="",
                    use_mgr=use_mgr,
                    temperature=temperature,
                )
            else:
                attn_procs[name] = AttnProcessor()
        self.unet.set_attn_processor(attn_procs)

        selected_set = set(selected_layers)
        missing_requested = sorted(set(mgr_blocks) - set(all_layers))
        print(
            "[MGR-LAYERS] "
            f"mgr_block_from={mgr_block_from} bank_mode={bank_mode} "
            f"use_mgr={use_mgr} temperature={temperature} "
            f"mgr_range={self.range_mgr} "
            f"selected={len(selected_layers)}/{len(all_layers)}",
            flush=True,
        )
        for idx, name in enumerate(selected_layers):
            print(f"[MGR-LAYERS][{idx:02d}] {name}", flush=True)
        if missing_requested:
            print(
                "[MGR-LAYERS][WARN] requested layer names not present in UNet: "
                + ", ".join(missing_requested),
                flush=True,
            )
        inactive = [name for name in all_layers if name not in selected_set]
        print(f"[MGR-LAYERS] inactive_attention_processors={len(inactive)}", flush=True)
        
    def prepare_img_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None, do_classifier_free_guidance=False):
        """Encode an image into latent space and add noise."""
        if not isinstance(image, (torch.Tensor, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image
        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator[i]) for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image), generator)

            init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        target_prompt: Optional[Union[str, List[str]]] = None,
        target_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        """
        Extended encode_prompt:
         - Accepts `target_prompt` / `target_prompt_embeds`.
         - If do_classifier_free_guidance: builds negative_prompt_embeds (if not provided).
         - Returns concatenated embeddings in order:
             [negative_prompt_embeds, negative_prompt_embeds, target_prompt_embeds, prompt_embeds]
           after applying per-sample repetition for num_images_per_prompt.
        Note: final batch-length depends on batch_size and num_images_per_prompt.
        """
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        # infer batch_size from provided inputs (prompt or prompt_embeds)
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # --- Build prompt_embeds using the standard logic. ---
        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # --- Build negative_prompt_embeds using the standard logic. ---
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure to"
                    " only forward one of the two."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
            # repeat per num_images_per_prompt to match prompt_embeds layout
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        # --- Build target_prompt_embeds for the target branch. ---
        if target_prompt is not None and target_prompt_embeds is None:
            # tokenization similar to prompt
            if isinstance(target_prompt, str):
                target_tokens = [target_prompt]
            elif isinstance(target_prompt, list):
                target_tokens = target_prompt
            else:
                raise TypeError(f"Unsupported target_prompt type: {type(target_prompt)}")

            target_in = self.tokenizer(
                target_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                target_attention_mask = target_in.attention_mask.to(device)
            else:
                target_attention_mask = None

            target_prompt_embeds = self.text_encoder(
                target_in.input_ids.to(device),
                attention_mask=target_attention_mask,
            )
            target_prompt_embeds = target_prompt_embeds[0]

        if target_prompt_embeds is not None:
            target_prompt_embeds = target_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
            # repeat per num_images_per_prompt
            target_prompt_embeds = target_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            target_prompt_embeds = target_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        # --- build final concatenation ---
        if do_classifier_free_guidance:
            # negative * 2, then target (if any), then prompt
            neg_dup = torch.cat([negative_prompt_embeds, negative_prompt_embeds], dim=0)
            parts = [neg_dup]
            if target_prompt_embeds is not None:
                parts.append(target_prompt_embeds)
            parts.append(prompt_embeds)
            final = torch.cat(parts, dim=0)
            return final
        else:
            # no guidance: if target provided, concat target then prompt, else just prompt
            if target_prompt_embeds is not None:
                return torch.cat([target_prompt_embeds, prompt_embeds], dim=0)
            return prompt_embeds

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def run_safety_checker(self, image, device, dtype):
        # For SD 1.5, implement safety checker similar to standard pipeline
        if self.safety_checker is not None:
            # Convert tensor to numpy for safety checker
            if isinstance(image, torch.Tensor):
                safety_checker_input = self.feature_extractor(
                    self.image_processor.postprocess(image, output_type="pil", do_denormalize=[True] * image.shape[0]), 
                    return_tensors="pt"
                ).to(device)
                image, has_nsfw_concept = self.safety_checker(
                    images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
                )
            else:
                has_nsfw_concept = None
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept


    @staticmethod
    def _to_mask_tensor(m):
        """
        m: str(Path) | PIL.Image | np.ndarray | torch.Tensor | list/tuple of masks
        return:
          - single mask: torch.FloatTensor [H, W] in [0, 1]
          - multi mask : torch.FloatTensor [B, H, W] in [0, 1]

        Batch-5 MGR case:
          ref mask       -> [H, W]
          target masks   -> [4, H, W]
        """
        if isinstance(m, (list, tuple)):
            masks = [StableDiffusionPipeline15MGR._to_mask_tensor(x) for x in m]
            masks = [x.squeeze(0) if torch.is_tensor(x) and x.ndim == 3 and x.shape[0] == 1 else x for x in masks]
            return torch.stack(masks, dim=0).float().clamp(0, 1)

        if isinstance(m, str):
            arr = np.array(Image.open(m).convert("L")).astype("float32")
            if arr.max() > 1.5:
                arr = arr / 255.0
            return torch.from_numpy(arr).clamp(0, 1)
        
        elif isinstance(m, Image.Image):
            arr = np.array(m.convert("L")).astype("float32")
            if arr.max() > 1.5:
                arr = arr / 255.0
            return torch.from_numpy(arr).clamp(0, 1)

        elif isinstance(m, np.ndarray):
            arr = m.astype("float32")
            if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
                arr = arr[..., 0]
            if arr.max() > 1.5:
                arr = arr / 255.0
            return torch.from_numpy(arr).clamp(0, 1)

        elif torch.is_tensor(m):
            t = m.float()
            # [1,H,W] -> [H,W], [3,H,W] -> first channel, [B,H,W] is preserved
            if t.ndim == 3:
                if t.shape[0] == 1:
                    t = t.squeeze(0)
                elif t.shape[0] in (3, 4):
                    t = t[0]
                # otherwise treat as [B,H,W] and preserve batch dimension
            elif t.ndim == 4:
                # [B,1,H,W] / [B,3,H,W] -> [B,H,W]
                if t.shape[1] in (1, 3, 4):
                    t = t[:, 0]
            if t.max() > 1.5:
                t = t / 255.0
            return t.clamp(0, 1)
        else:
            raise TypeError(f"Unsupported mask type: {type(m)}")


    def _attach_mgr_masks_to_unet(self):
        """
        Attach masks to every CrossFrameAttnProcessor.

        Batch-5 layout:
            latents[0]   = reference
            latents[1:5] = four targets

        Therefore:
            ref_t_cond: [H, W]
            tar_t_cond: [4, H, W]

        The processor must use tar_t_cond[k] for target k. If the processor only
        supports a single [H,W] target mask, this function still passes [4,H,W],
        so the processor should be updated to consume batched target masks.
        """
        ref_t_cond = self._to_mask_tensor(self._ref_mask_cache['cond'])
        tar_t_cond = self._to_mask_tensor(self._tar_mask_cache['cond'])

        # Keep the metadata on the pipeline for debugging / processor-side use.
        self._num_target_masks = int(tar_t_cond.shape[0]) if torch.is_tensor(tar_t_cond) and tar_t_cond.ndim == 3 else 1

        for name, module in self.unet.named_modules():
            if hasattr(module, "processor"):
                proc = getattr(module, "processor", None)
                if isinstance(proc, CrossFrameAttnProcessor):
                    # Expose target-mask count for processors that support batch masks.
                    try:
                        setattr(proc, "num_target_masks", self._num_target_masks)
                    except Exception:
                        pass
                    if hasattr(proc, "set_raw_masks"):
                        proc.set_raw_masks(ref_t_cond, tar_t_cond)

    @staticmethod
    def apply_tangential(noise_pred_uncond, noise_pred_text):
        all_noise = torch.stack((noise_pred_text, noise_pred_uncond), dim=1).to(dtype=torch.float32)
        all_noise = all_noise.reshape(all_noise.size(0), all_noise.size(1), -1)

        U, S, Vh = torch.linalg.svd(all_noise, full_matrices=False)
        Vh = Vh.to(all_noise.device)
        Vh_modified = Vh.clone().to(all_noise.device)
        Vh_modified[:,1] = 0
        noise_null_flat = noise_pred_uncond.reshape(noise_pred_uncond.size(0), 1, -1).to(dtype=torch.float32)
        noise_null_flat = noise_null_flat.to(Vh.device)
        x_Vh = torch.matmul(noise_null_flat, Vh.transpose(-2, -1))
        x_Vh_V = torch.matmul(x_Vh, Vh_modified)
        noise_pred_uncond = x_Vh_V.reshape(*noise_pred_uncond.shape).to(noise_pred_text.dtype).to(noise_pred_text.device)

        return noise_pred_uncond


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        target_prompt: Optional[Union[str, List[str]]] = None,
        image = None,
        save_intermediates: bool = False,
        save_dir: Optional[str] = None,
        **kwargs,
    ):

        # __call__ options: MGR + RNO only.
        save_seg_masks = bool(kwargs.get("save_seg_masks", False))

        use_rno = bool(kwargs.get("use_rno", kwargs.get("use_noise_rno", False)))
        use_noise_rno = use_rno
        rno_optimize_start_idx = int(kwargs.get("rno_optimize_start_idx", 1))
        rno_temperature = float(kwargs.get("rno_temperature", kwargs.get("rno_temperature", 0.07)))
        rno_gamma = float(kwargs.get("rno_gamma", kwargs.get("rno_gamma", 1.0)))
        noise_rno_lr = float(kwargs.get("rno_lr", kwargs.get("noise_rno_lr", 0.01)))
        noise_rno_num_opt_steps = int(kwargs.get("rno_num_opt_steps", kwargs.get("noise_rno_num_opt_steps", 3)))
        noise_rno_pool_size = int(kwargs.get("rno_pool_size", kwargs.get("noise_rno_pool_size", 16)))
        noise_rno_clip = kwargs.get("rno_clip", kwargs.get("noise_rno_clip", None))
        noise_rno_kl_weight = float(kwargs.get("rno_kl_weight", kwargs.get("noise_rno_kl_weight", 0.0)) or 0.0)
        noise_rno_ref_repel_weight = float(kwargs.get("rno_ref_repel_weight", kwargs.get("noise_rno_ref_repel_weight", 0.0)) or 0.0)
        noise_rno_negative_prompt = kwargs.get("rno_negative_prompt", kwargs.get("noise_rno_negative_prompt", None))
        noise_rno_fixed_neg_anchor = bool(kwargs.get("rno_fixed_neg_anchor", kwargs.get("noise_rno_fixed_neg_anchor", False)))
        noise_rno_neg_anchor_extra = int(kwargs.get("rno_neg_anchor_extra", kwargs.get("noise_rno_neg_anchor_extra", 0)) or 0)
        noise_rno_force_per_sample = bool(kwargs.get("rno_force_per_sample", kwargs.get("noise_rno_force_per_sample", False)))
        noise_rno_exclude_self_neg = bool(kwargs.get("rno_exclude_self_neg", kwargs.get("noise_rno_exclude_self_neg", False)))
        noise_rno_positive_weight = float(kwargs.get("rno_positive_weight", kwargs.get("noise_rno_positive_weight", 0.0)))

        # Disabled prototype branches. These names remain only to keep the denoising
        # loop simple and explicit. No code path below enables ablation baselines or text-delta variants.
        rno_prompt_embeds = None
        early_latent_noise_scale = 0.0
        early_latent_noise_steps = 0
        early_text_noise_scale = 0.0
        early_text_noise_steps = 0
        early_text_noise_token_mask = None
        use_noise_rno_mid = False

        return_target_only = bool(kwargs.get("return_target_only", False))
        ref_kv_cache_mode = str(kwargs.get("ref_kv_cache_mode", "off") or "off")
        ref_kv_cache_root = kwargs.get("ref_kv_cache_root", None)
        ref_kv_cache_id = kwargs.get("ref_kv_cache_id", None)
        target_step_dump_dir = kwargs.get("target_step_dump_dir", None)
        dump_target_step_only = bool(kwargs.get("dump_target_step_only", False))
        precomputed_target_mask_dir = kwargs.get("precomputed_target_mask_dir", None)
        mask_update_mode = str(kwargs.get("mask_update_mode", "once") or "once")
        target_mask_source = str(kwargs.get("target_mask_source", "default") or "default")
        target_branch_indices = kwargs.get("target_branch_indices", None)
        if isinstance(target_branch_indices, str):
            target_branch_indices = [int(x.strip()) for x in target_branch_indices.split(',') if x.strip()]
        elif target_branch_indices is not None:
            target_branch_indices = [int(x) for x in target_branch_indices]
        resume_target_latents_path = kwargs.get("resume_target_latents_path", None)
        resume_target_latent_dir = kwargs.get("resume_target_latent_dir", None)

        # Reset state at every call to avoid leakage from the previous image.
        self._mgr_masks_attached = False
        self._mgr_masks_ready = False
        self._ref_mask_cache = None  # reset here
        self._tar_mask_cache = None  # reset here
        self._seg_masks_again = False  # reset here

        # ref mask
        self._ref_mask_cache = {
            # "uncond": kwargs.get("ref_mask", None),
            "cond":   kwargs.get("ref_mask", None),
        }

        # Reset all UNet attention-processor states and active step settings.
        for name, module in self.unet.named_modules():
            if hasattr(module, "processor"):
                proc = getattr(module, "processor", None)
                if isinstance(proc, CrossFrameAttnProcessor):
                    if hasattr(proc, "set_ref_kv_cache"):
                        proc.set_ref_kv_cache(ref_kv_cache_mode, ref_kv_cache_root, ref_kv_cache_id)
                    if hasattr(proc, "inference_step"):
                        proc.inference_step = 0
                    if hasattr(proc, "attn_map"):
                        proc.attn_map = {}
                    if hasattr(proc, "max_inference_step"):
                        proc.max_inference_step = num_inference_steps

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # --- normalize input image to RGB before preprocess ---
        if isinstance(image, PILImage.Image):
            # Convert to RGB because PNG/RGBA inputs may be provided.
            image = image.convert("RGB")

        elif isinstance(image, list):
            # Handle batched image inputs.
            image = [
                im.convert("RGB") if isinstance(im, PILImage.Image) else im
                for im in image
            ]

        elif isinstance(image, torch.Tensor):
            # Normalize possible [H, W, C] tensors to [B, C, H, W].
            if image.dim() == 3:           # [C, H, W] or [H, W, C]
                # Permute if the channel dimension is last.
                if image.shape[0] in (1,3,4):
                    pass
                else:
                    image = image.permute(2,0,1)
                image = image.unsqueeze(0)  # [1,C,H,W]
            # Drop the alpha channel for RGBA tensors.
            if image.shape[1] == 4:
                image = image[:, :3]
                
        # Image preprocessing for MGR
        if image is not None:
            z0 = self.image_processor.preprocess(image)

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        
        # Main prompt encoding
        prompt_and_target_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            target_prompt=target_prompt,
            target_prompt_embeds=None,
        )
        encoded_batch_size = batch_size * num_images_per_prompt

        def _embedding_chunks():
            if do_classifier_free_guidance:
                if target_prompt is not None:
                    neg = prompt_and_target_embeds[:encoded_batch_size]
                    target = prompt_and_target_embeds[2 * encoded_batch_size:3 * encoded_batch_size]
                    prompt_part = prompt_and_target_embeds[3 * encoded_batch_size:4 * encoded_batch_size]
                else:
                    neg = prompt_and_target_embeds[:encoded_batch_size]
                    target = None
                    prompt_part = prompt_and_target_embeds[2 * encoded_batch_size:3 * encoded_batch_size]
            else:
                if target_prompt is not None:
                    target = prompt_and_target_embeds[:encoded_batch_size]
                    prompt_part = prompt_and_target_embeds[encoded_batch_size:2 * encoded_batch_size]
                else:
                    target = None
                    prompt_part = prompt_and_target_embeds[:encoded_batch_size]
                neg = None
            return neg, target, prompt_part

        def _select_embed_rows(embed, *, active_targets_only=False):
            if embed is None:
                return None
            if embed.shape[0] == latents.shape[0]:
                if active_targets_only:
                    return embed[rno_optimize_start_idx:]
                if self._mgr_masks_attached:
                    return embed
                return embed[rno_optimize_start_idx:]
            return embed[:1]

        def _noise_rno_positive_embed():
            if rno_prompt_embeds is not None:
                return _select_embed_rows(rno_prompt_embeds, active_targets_only=True)
            if do_classifier_free_guidance and target_prompt is not None:
                _, target_chunk, _ = _embedding_chunks()
                return _select_embed_rows(target_chunk, active_targets_only=True)
            if do_classifier_free_guidance:
                _, _, prompt_chunk = _embedding_chunks()
                return _select_embed_rows(prompt_chunk, active_targets_only=True)
            _, target_chunk, prompt_chunk = _embedding_chunks()
            return _select_embed_rows(target_chunk if target_chunk is not None else prompt_chunk, active_targets_only=True)

        # Legacy rno-positive prompt override is disabled in the
        # RNO-only path.  RNO uses the current target/prompt
        # embeddings as the positive branch and `noise_rno_negative_prompt`
        # below as the negative branch.  Keeping this as None avoids the
        # undefined legacy `rno_prompt` variable.
        rno_prompt_embeds = None

        noise_rno_negative_prompt_embeds = None
        if noise_rno_negative_prompt is not None:
            if isinstance(noise_rno_negative_prompt, str):
                noise_rno_negative_tokens = [noise_rno_negative_prompt]
            elif isinstance(noise_rno_negative_prompt, list):
                noise_rno_negative_tokens = noise_rno_negative_prompt
            else:
                raise TypeError(f"Unsupported noise_rno_negative_prompt type: {type(noise_rno_negative_prompt)}")
            max_length = self.tokenizer.model_max_length
            noise_rno_negative_input = self.tokenizer(
                noise_rno_negative_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            noise_rno_negative_prompt_embeds = self.text_encoder(
                noise_rno_negative_input.input_ids.to(device),
                attention_mask=(
                    noise_rno_negative_input.attention_mask.to(device)
                    if getattr(self.text_encoder.config, "use_attention_mask", False)
                    else None
                ),
            )[0]
            noise_rno_negative_prompt_embeds = noise_rno_negative_prompt_embeds.to(
                dtype=self.text_encoder.dtype,
                device=device,
            )


        # prompt_embeds = self.encode_prompt(
        #     prompt,
        #     device,
        #     num_images_per_prompt,
        #     do_classifier_free_guidance,
        #     negative_prompt,
        #     prompt_embeds=prompt_embeds,
        #     negative_prompt_embeds=negative_prompt_embeds,
        #     lora_scale=text_encoder_lora_scale,
        # )

        # ## Target prompt encoding (for MGR)
        # if target_prompt is not None:
        #     target_prompt_embeds = self.encode_prompt(
        #         target_prompt,
        #         device,
        #         num_images_per_prompt,
        #         do_classifier_free_guidance,
        #         negative_prompt,
        #         lora_scale=text_encoder_lora_scale,
        #     )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_and_target_embeds.dtype,
            device,
            generator,
            latents,
        )

        resume_start_step = None
        resume_payload = None
        if resume_target_latents_path or resume_target_latent_dir:
            resume_path = resume_target_latents_path
            if not resume_path and resume_target_latent_dir:
                cand_paths = sorted(glob.glob(os.path.join(resume_target_latent_dir, "target_latents_step*.pt")))
                if cand_paths:
                    resume_path = cand_paths[-1]
            if not resume_path or not os.path.isfile(resume_path):
                raise FileNotFoundError(
                    f"resume target latent not found: path={resume_target_latents_path}, dir={resume_target_latent_dir}"
                )
            resume_payload = torch.load(resume_path, map_location="cpu")
            if isinstance(resume_payload, dict):
                resume_latents = resume_payload.get("latents", None)
                resume_start_step = int(resume_payload.get("step", self.step_start_mgr))
            else:
                resume_latents = resume_payload
                resume_start_step = int(self.step_start_mgr)
            if resume_latents is None:
                raise ValueError(f"resume payload has no 'latents': {resume_path}")
            resume_latents = resume_latents.to(device=device, dtype=latents.dtype)
            if target_branch_indices:
                gather_idx = torch.tensor([i - 1 for i in target_branch_indices], device=resume_latents.device, dtype=torch.long)
                if resume_latents.ndim < 1 or int(gather_idx.max().item()) >= resume_latents.shape[0]:
                    raise ValueError(
                        f"target_branch_indices={target_branch_indices} incompatible with resume latents shape={tuple(resume_latents.shape)}"
                    )
                resume_latents = resume_latents.index_select(0, gather_idx)
            ref_count_for_resume = 1
            resume_target_slice = slice(ref_count_for_resume, None)
            target_latent_shape = latents[resume_target_slice].shape
            if resume_latents.shape == target_latent_shape:
                latents = latents.clone()
                latents[resume_target_slice] = resume_latents
            elif resume_latents.shape == latents.shape:
                latents = resume_latents
            else:
                raise ValueError(
                    "resume latent shape mismatch: "
                    f"loaded={tuple(resume_latents.shape)}, target={tuple(target_latent_shape)}, full={tuple(latents.shape)}"
                )
            print(
                f"[MGR resume] loaded target latents from {resume_path}; "
                f"resume loop at step index {resume_start_step}"
            )

        # RNO: optimize target initial noises while keeping the reference row fixed.
        noise_rno_ref_latent = None
        if (
            use_noise_rno
            and image is not None
            and float(noise_rno_ref_repel_weight or 0.0) > 0.0
            and rno_optimize_start_idx > 0
        ):
            with torch.no_grad():
                z0_rno_ref = z0.to(device=device, dtype=prompt_and_target_embeds.dtype)
                z0_rno_ref = self.vae.config.scaling_factor * self.vae.encode(z0_rno_ref).latent_dist.sample()
                ref_noise = randn_tensor(
                    z0_rno_ref.shape,
                    generator=generator,
                    device=device,
                    dtype=z0_rno_ref.dtype,
                )
                noise_rno_ref_latent = self.scheduler.add_noise(z0_rno_ref, ref_noise, timesteps[0]).to(
                    device=latents.device,
                    dtype=latents.dtype,
                )

        if use_noise_rno:
            if latents.shape[0] <= rno_optimize_start_idx + 1:
                logger.warning(
                    f"RNO skipped: latent batch size={latents.shape[0]} is too small. "
                    f"Need at least {rno_optimize_start_idx + 2}."
                )
            else:
                if hasattr(self.unet, "enable_gradient_checkpointing"):
                    self.unet.enable_gradient_checkpointing()
                rno_pos_embed = _noise_rno_positive_embed()
                rno_neg_embed = _select_embed_rows(
                    noise_rno_negative_prompt_embeds,
                    active_targets_only=True,
                )
                noise_rno_latents = latents
                if noise_rno_ref_latent is not None:
                    noise_rno_latents = latents.clone()
                    noise_rno_latents[:rno_optimize_start_idx] = noise_rno_ref_latent[:rno_optimize_start_idx]
                optimized_noise_rno_latents = rno_optimize_initial_noise_latents(
                    self,
                    z_t=noise_rno_latents,
                    t=timesteps[0],
                    pos_embed=rno_pos_embed,
                    neg_embed=rno_neg_embed,
                    num_opt_steps=noise_rno_num_opt_steps,
                    lr=noise_rno_lr,
                    temperature=rno_temperature,
                    gamma=rno_gamma,
                    pool_size=noise_rno_pool_size,
                    optimize_start_idx=rno_optimize_start_idx,
                    noise_clip=noise_rno_clip,
                    kl_weight=noise_rno_kl_weight,
                    ref_repel_weight=noise_rno_ref_repel_weight,
                    fixed_neg_anchor=noise_rno_fixed_neg_anchor,
                    neg_anchor_extra=noise_rno_neg_anchor_extra,
                    force_per_sample=noise_rno_force_per_sample,
                    exclude_self_neg=noise_rno_exclude_self_neg,
                    positive_weight=noise_rno_positive_weight,
                )
                if noise_rno_ref_latent is not None:
                    latents = latents.clone()
                    latents[rno_optimize_start_idx:] = optimized_noise_rno_latents[rno_optimize_start_idx:]
                    print(
                        f"[RNO] using real reference image latent as repel branch; "
                        f"targets={latents.shape[0] - rno_optimize_start_idx} "
                        f"ref_repel={noise_rno_ref_repel_weight}"
                    )
                else:
                    latents = optimized_noise_rno_latents

        # 6. Prepare extra step kwargs. TODO: Logic should ideally be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # 7. Denoising loop
        # --- Prepare paths before entering the denoising loop. ---
        save_dir = kwargs.get("synth_save_dir") if isinstance(kwargs.get("synth_save_dir"), str) else None
        if not save_dir:
            save_dir = os.environ.get("SYNTH_SAVE_DIR") or os.getcwd()
        os.makedirs(save_dir, exist_ok=True)

        # Create the segmentation output directory outside the loop and use an absolute path.
        
        if save_seg_masks: 
            cur_tar_idx = kwargs.get("cur_tar_idx") if isinstance(kwargs.get("cur_tar_idx"), int) else 0
            seg_out_dir_abs = os.path.abspath(os.path.join(save_dir, f"mgr_segment_out_{cur_tar_idx:02d}"))
            # dir_mgr_feat = os.path.abspath(os.path.join(save_dir, f"mgr_feat/{cur_tar_idx:02d}"))
            # attn_out_dir_abs = os.path.abspath(dir_mgr_feat)
            # os.makedirs(seg_out_dir_abs, exist_ok=True)
            # for name, module in self.unet.named_modules():
            #     if hasattr(module, "processor"):
            #         proc = getattr(module, "processor", None)
            #         if isinstance(proc, CrossFrameAttnProcessor):
            #             setattr(proc, "out_dir", attn_out_dir_abs)
            #             print(f"Set out_dir for {name} to {attn_out_dir_abs}")
        else: 
            seg_out_dir_abs = None

        # prepare save dir if requested
        if save_intermediates:
            sd = seg_out_dir_abs or "./timestep_intermediates"
            os.makedirs(sd, exist_ok=True)
            os.makedirs(os.path.join(sd, "0_uncond_ref"), exist_ok=True)
            os.makedirs(os.path.join(sd, "1_uncond_ori"), exist_ok=True)
            os.makedirs(os.path.join(sd, "2_cond_ref"), exist_ok=True)
            os.makedirs(os.path.join(sd, "3_cond_ori"), exist_ok=True)
            dir_refmask_uncond_overlay = os.path.join(sd, "4_refmask_uncond_overlay")
            dir_refmask_cond_overlay = os.path.join(sd, "4_refmask_cond_overlay")
            dir_tarmask_uncond_overlay = os.path.join(sd, "5_tarmask_uncond_overlay")
            dir_tarmask_cond_overlay = os.path.join(sd, "5_tarmask_cond_overlay")
            os.makedirs(dir_refmask_uncond_overlay, exist_ok=True)
            os.makedirs(dir_refmask_cond_overlay, exist_ok=True)
        else:
            sd = None
        
        # fixed_tar_mask_paths = None
        self.tar_mask_ema = kwargs.get("tar_mask_ema", None)
        # prev_soft_tar_uncond_mask = None
        # prev_soft_tar_cond_mask   = None


        # prepare z0 as latents for reference image by encoding with vae encoder
        z0 = z0.to(device=device, dtype=prompt_and_target_embeds.dtype) # [1, 3, 512, 512]
        z0 = self.vae.config.scaling_factor * self.vae.encode(z0).latent_dist.sample()  # [1, 4, 64, 64]

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if resume_start_step is not None and i < resume_start_step:
                    progress_bar.update(1)
                    continue

                if image is not None and self._mgr_masks_attached:
                    noise = randn_tensor(z0.shape, generator=generator, device=device, dtype=z0.dtype)
                    zt = self.scheduler.add_noise(z0, noise, t)
                    latents[0] = zt[0]

                if early_latent_noise_scale > 0 and i < early_latent_noise_steps:
                    target_start = 1
                    if latents.shape[0] > target_start:
                        if i == 0:
                            print(
                                f"[EARLY-LATENT-NOISE] scale={early_latent_noise_scale} "
                                f"steps={early_latent_noise_steps} target_start={target_start} "
                                f"targets={latents.shape[0] - target_start}"
                            )
                        noise = randn_tensor(
                            latents[target_start:].shape,
                            generator=generator,
                            device=latents.device,
                            dtype=latents.dtype,
                        )
                        latents = latents.clone()
                        latents[target_start:] = latents[target_start:] + early_latent_noise_scale * noise

                # ---------------------------------------------------------
                # Batch handling for rno/MGR
                # latents[0]  = reference / anchor latent
                # latents[1:] = target latents
                # Before MGR masks are attached, denoise only target latents.
                # After MGR masks are attached, include reference + targets.
                # ---------------------------------------------------------
                ref_count = 1
                target_slice = slice(ref_count, None)

                if not self._mgr_masks_attached:
                    active_latents = latents[target_slice]
                else:
                    active_latents = latents

                latent_model_input = (
                    torch.cat([active_latents] * 2)
                    if do_classifier_free_guidance
                    else active_latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Prompt embeddings must match the currently active latent batch.
                num_active = active_latents.shape[0]
                if do_classifier_free_guidance:
                    neg_chunk, target_chunk, prompt_chunk = _embedding_chunks()
                    if target_prompt is not None and i >= len(timesteps) // 2:
                        pos_embed = target_chunk
                    else:
                        pos_embed = prompt_chunk

                    neg_rows = _select_embed_rows(neg_chunk)
                    pos_rows = _select_embed_rows(pos_embed)
                    neg_batch = _repeat_cond_embed_to_batch(neg_rows, num_active)
                    pos_batch = _repeat_cond_embed_to_batch(pos_rows, num_active)
                    current_prompt_embeds = torch.cat([neg_batch, pos_batch], dim=0)
                else:
                    _, target_chunk, prompt_chunk = _embedding_chunks()
                    if target_prompt is not None and i >= len(timesteps) // 2:
                        pos_embed = target_chunk
                    else:
                        pos_embed = prompt_chunk
                    current_prompt_embeds = _repeat_cond_embed_to_batch(
                        _select_embed_rows(pos_embed),
                        num_active,
                    )

                for name, module in self.unet.named_modules():
                    if hasattr(module, "processor"):
                        proc = getattr(module, "processor", None)
                        if isinstance(proc, CrossFrameAttnProcessor):
                            try:
                                setattr(proc, "global_step", i)
                                setattr(proc, "masks_ready", self._mgr_masks_ready)
                            except Exception:
                                pass

                # print(latent_model_input.shape)
                # print(current_prompt_embeds.shape)
                # exit()

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=current_prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    
                    if save_intermediates:
                        # save intermediate results
                        num_channels_latents = self.unet.config.in_channels
                        x0_uncond = _extract_prev_sample(
                            self.scheduler.step(noise_pred_uncond, t, active_latents, # ref_z0=None,
                                               **extra_step_kwargs, return_dict=True), 
                            num_channels_latents
                        )
                        x0_text = _extract_prev_sample(
                            self.scheduler.step(noise_pred_text, t, active_latents, # ref_z0=None,
                                               **extra_step_kwargs, return_dict=True), 
                            num_channels_latents
                        )
                        decoded_x0_uncond = self.vae.decode(x0_uncond / self.vae.config.scaling_factor, return_dict=False)[0].detach().cpu().numpy()
                        decoded_x0_text = self.vae.decode(x0_text / self.vae.config.scaling_factor, return_dict=False)[0].detach().cpu().numpy()

                        clip_decoded_x0_uncond = (decoded_x0_uncond.clip(-1, 1) + 1.0) / 2.
                        clip_decoded_x0_text = (decoded_x0_text.clip(-1, 1) + 1.0) / 2.

                        uncond_ref = clip_decoded_x0_uncond[0].transpose(1, 2, 0)
                        uncond_ori = clip_decoded_x0_uncond[1].transpose(1, 2, 0)
                        cond_ref = clip_decoded_x0_text[0].transpose(1, 2, 0)
                        cond_ori = clip_decoded_x0_text[1].transpose(1, 2, 0)

                        pil_uncond_ref = self.numpy_to_pil(uncond_ref)
                        pil_uncond_ori = self.numpy_to_pil(uncond_ori)
                        pil_cond_ref = self.numpy_to_pil(cond_ref)
                        pil_cond_ori = self.numpy_to_pil(cond_ori)

                        # save images per batch idx
                        fname = os.path.join(sd, f"0_uncond_ref/step_{i:02d}_uncond_ref.png")
                        pil_uncond_ref[0].save(fname)
                        
                        fname = os.path.join(sd, f"1_uncond_ori/step_{i:02d}_uncond_ori.png")
                        pil_uncond_ori[0].save(fname)
                        
                        fname = os.path.join(sd, f"2_cond_ref/step_{i:02d}_cond_ref.png")
                        pil_cond_ref[0].save(fname)

                        fname = os.path.join(sd, f"3_cond_ori/step_{i:02d}_cond_ori.png")
                        pil_cond_ori[0].save(fname)

                        clip_prompt = kwargs.get("clip_prompt", "object")

                        self.run_clipseg(pil_uncond_ref, clip_prompt,
                                         save_overlay=save_seg_masks, dir_overlay=dir_refmask_uncond_overlay,
                                         prefix='uncond', step=i, device=device)
                        self.run_clipseg(pil_cond_ref, clip_prompt,
                                         save_overlay=save_seg_masks, dir_overlay=dir_refmask_cond_overlay,
                                         prefix='cond', step=i, device=device)
                        self.run_clipseg(pil_uncond_ori, clip_prompt,
                                         save_overlay=save_seg_masks, dir_overlay=dir_tarmask_uncond_overlay,
                                         prefix='uncond', step=i, device=device)
                        self.run_clipseg(pil_cond_ori, clip_prompt,
                                         save_overlay=save_seg_masks, dir_overlay=dir_tarmask_cond_overlay,
                                         prefix='cond', step=i, device=device)

                        # noise_pred_uncond = self.apply_tangential(noise_pred_uncond, noise_pred_text)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                step_out = self.scheduler.step(noise_pred, t, active_latents, **extra_step_kwargs, return_dict=True)
                updated_active_latents = step_out[0]

                if not self._mgr_masks_attached:
                    # keep reference latent fixed; update only target latents
                    latents = latents.clone()
                    latents[target_slice] = updated_active_latents
                else:
                    # active_latents == latents
                    latents = updated_active_latents

                # before MGR starts, ensure batch size is at least 2 for MGR processing
                if latents.shape[0] == 1:
                    latents = latents.repeat(2, 1, 1, 1)

                # ===================== FG-MGR: create/inject masks and keep them active =====================
                # once: use masks created at step_start until the end.
                # each_step: refresh reference/target masks from the current x0 after step_start.
                should_refresh_masks = (
                    i == self.step_start_mgr
                    or (mask_update_mode == "each_step" and i > self.step_start_mgr and self._mgr_masks_attached)
                )
                if should_refresh_masks:
                    # try:
                    # 1. Create an x0 image from noise_pred_text.
                    num_channels_latents = self.unet.config.in_channels
                    x0_text = _extract_prev_sample(
                        self.scheduler.step(noise_pred_text, t, active_latents, **extra_step_kwargs, return_dict=True),
                        num_channels_latents
                    )

                    # Decode with the VAE.
                    decoded_x0_text = self.vae.decode(x0_text / self.vae.config.scaling_factor, return_dict=False)[0]

                    # Convert to PIL images.
                    pil_text = self.image_processor.postprocess(
                        decoded_x0_text,
                        output_type="pil",
                        do_denormalize=[True] * decoded_x0_text.shape[0],
                    )
                    if not isinstance(pil_text, list):
                        pil_text = [pil_text]

                    if target_step_dump_dir:
                        os.makedirs(target_step_dump_dir, exist_ok=True)
                        for tar_i, pil_text_tar in enumerate(pil_text):
                            pil_text_tar.save(
                                os.path.join(
                                    target_step_dump_dir,
                                    f"tar{tar_i+1}_step{i:03d}_image.png",
                                )
                            )
                        torch.save(
                            {
                                "step": int(i),
                                "timestep": int(t.item()) if hasattr(t, "item") else int(t),
                                "num_inference_steps": int(num_inference_steps),
                                "step_start_mgr": int(self.step_start_mgr),
                                "latents": active_latents.detach().to("cpu", dtype=torch.float16),
                            },
                            os.path.join(target_step_dump_dir, f"target_latents_step{i:03d}.pt"),
                        )
                        if dump_target_step_only:
                            return StableDiffusionPipelineOutput(
                                images=pil_text,
                                nsfw_content_detected=None,
                            )

                    # ---------------------------------------------------------
                    # Batch-5 multi-target mask generation.
                    # At first attachment, active_latents contains only targets.
                    # In each-step mode after attachment, active_latents contains
                    # [ref, tar1, tar2, tar3, tar4], so update the ref mask too.
                    # ---------------------------------------------------------
                    clip_prompt = kwargs.get("clip_prompt", "object")
                    has_ref_frame = self._mgr_masks_attached and len(pil_text) > 1
                    target_pil_text = pil_text[1:] if has_ref_frame else pil_text

                    if mask_update_mode == "each_step" and has_ref_frame:
                        ref_mask_fn = f"1_ref_attention_mask_binary_step{i:02d}.png"
                        ref_attn = self.run_attention_saliency_mask(
                            pil_text[0],
                            prefix='ref_attention',
                            step=i,
                            save_binary=save_seg_masks,
                            dir_binary=seg_out_dir_abs,
                            fn_binary=ref_mask_fn,
                            save_overlay=save_seg_masks,
                            dir_overlay=seg_out_dir_abs,
                            fn_overlay=f"1_ref_attention_mask_overlay_step{i:02d}.png",
                        )
                        self._ref_mask_cache = {'cond': ref_attn['mask_uint8']}

                    target_masks_cond = []

                    for tar_i, pil_text_tar in enumerate(target_pil_text):
                        branch_idx = int(target_branch_indices[tar_i]) if target_branch_indices else tar_i + 1
                        fn_tar_attn_mask_binary = f"2_tar{branch_idx}_attention_mask_binary.png"
                        fn_tar_attn_mask_overlay = f"2_tar{branch_idx}_attention_mask_overlay.png"
                        fn_tar_mask_cond_binary = f"3_tar{branch_idx}_mask_cond_binary.png"
                        fn_tar_mask_cond_overlay = f"3_tar{branch_idx}_mask_cond_overlay.png"

                        attn_mask = self.run_attention_saliency_mask(
                            pil_text_tar,
                            prefix=f'tar{tar_i+1}_attention',
                            step=i,
                            save_binary=save_seg_masks,
                            dir_binary=seg_out_dir_abs,
                            fn_binary=fn_tar_attn_mask_binary,
                            save_overlay=save_seg_masks,
                            dir_overlay=seg_out_dir_abs,
                            fn_overlay=fn_tar_attn_mask_overlay,
                        )

                        pre_mask = None
                        if precomputed_target_mask_dir and target_mask_source != "attention":
                            for cand_name in (
                                f"tar{branch_idx}_mask_binary.png",
                                f"tar{branch_idx}_mask.png",
                                f"tar{branch_idx}_cond_mask_binary.png",
                                f"tar{branch_idx}_cond_mask.png",
                                f"3_tar{branch_idx}_mask_cond_binary.png",
                            ):
                                cand = os.path.join(precomputed_target_mask_dir, cand_name)
                                if os.path.isfile(cand):
                                    pre_mask = np.array(Image.open(cand).convert("L")).astype(np.uint8)
                                    break

                        if target_mask_source == "attention":
                            target_masks_cond.append(attn_mask['mask_uint8'])
                            if save_seg_masks and seg_out_dir_abs:
                                Image.fromarray(attn_mask['mask_uint8']).save(
                                    os.path.join(seg_out_dir_abs, fn_tar_mask_cond_binary)
                                )
                        elif pre_mask is not None:
                            if pre_mask.shape[:2] != (pil_text_tar.height, pil_text_tar.width):
                                pre_mask = cv2.resize(
                                    pre_mask,
                                    (pil_text_tar.width, pil_text_tar.height),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                            target_masks_cond.append(pre_mask)
                            if save_seg_masks and seg_out_dir_abs:
                                Image.fromarray(pre_mask).save(
                                    os.path.join(seg_out_dir_abs, fn_tar_mask_cond_binary)
                                )
                        else:
                            seg_tar_cond = self.run_clipseg(
                                pil_text_tar,
                                clip_prompt,
                                prefix=f'tar{tar_i+1}_cond',
                                step=i,
                                save_binary=save_seg_masks,
                                dir_binary=seg_out_dir_abs,
                                fn_binary=fn_tar_mask_cond_binary,
                                save_overlay=save_seg_masks,
                                dir_overlay=seg_out_dir_abs,
                                fn_overlay=fn_tar_mask_cond_overlay,
                                device=device,
                            )
                            target_masks_cond.append(seg_tar_cond['mask_uint8'])

                    self._tar_mask_cache = {
                        # list[np.ndarray H,W] -> _to_mask_tensor -> torch.Tensor [T,H,W]
                        'cond': target_masks_cond,
                    }

                    # Inject the reference mask [H, W] and target masks [T, H, W] into the UNet.
                    self._attach_mgr_masks_to_unet()
                    self._mgr_masks_attached = True
                    self._mgr_masks_ready = True
                    for name, module in self.unet.named_modules():
                        if hasattr(module, "processor"):
                            proc = getattr(module, "processor", None)
                            if isinstance(proc, CrossFrameAttnProcessor):
                                try:
                                    setattr(proc, "masks_ready", True)
                                except Exception:
                                    pass
                elif i == self.step_end_mgr:
                    # print("MGR stage ended: removing MGR masks from UNet.")
                    self._mgr_masks_attached = False
                    self._mgr_masks_ready = False
                    for name, module in self.unet.named_modules():
                        if hasattr(module, "processor"):
                            proc = getattr(module, "processor", None)
                            if isinstance(proc, CrossFrameAttnProcessor):
                                try:
                                    setattr(proc, "masks_ready", False)
                                except Exception:
                                    pass

                progress_bar.update(1)

        progress_bar.close()
        
        decode_latents = latents
        if return_target_only and decode_latents.shape[0] > rno_optimize_start_idx:
            decode_latents = decode_latents[rno_optimize_start_idx:]

        if not output_type == "latent":
            image = self.vae.decode(decode_latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_and_target_embeds.dtype)
        else:
            image = decode_latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw_concept] * image.shape[0]
        
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

def retrieve_latents(encoder_output, generator):
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

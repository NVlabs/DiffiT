#!/usr/bin/env python3

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
DiffiT: Diffusion Vision Transformers for Image Generation.
Code by Ali Hatamizadeh. 
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Mlp, PatchEmbed


# ---------------------------------------------------------------------------
# Positional embedding utilities
# ---------------------------------------------------------------------------

def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Compute 1-D sinusoidal positional embeddings.

    Args:
        embed_dim: Output dimension for each position (must be even).
        pos:       Positions to encode, arbitrary shape – will be flattened.

    Returns:
        Embedding array of shape ``(M, embed_dim)``.
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega                          # (D/2,)

    pos = pos.reshape(-1)                                # (M,)
    out = np.einsum("m,d->md", pos, omega)               # (M, D/2)

    return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # (M, D)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    return np.concatenate([emb_h, emb_w], axis=1)                       # (H*W, D)


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
    extra_tokens: int = 0,
) -> np.ndarray:
    """
    Generate 2-D sinusoidal positional embeddings on a square grid.

    Args:
        embed_dim:    Embedding dimension.
        grid_size:    Height (= width) of the grid.
        cls_token:    If *True*, prepend ``extra_tokens`` zero rows.
        extra_tokens: Number of extra token slots to prepend.

    Returns:
        Array of shape ``[grid_size*grid_size, embed_dim]`` (or with
        ``extra_tokens`` additional leading rows when requested).
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)           # w first
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


# ---------------------------------------------------------------------------
# Timestep & label embedders
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """Embeds scalar diffusion timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        """
        Sinusoidal timestep embeddings following Vaswani et al.

        Args:
            t:          1-D tensor of ``N`` indices (may be fractional).
            dim:        Embedding dimension.
            max_period: Controls the minimum frequency.

        Returns:
            Tensor of shape ``(N, dim)``.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    Supports label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        use_cfg_embedding = int(dropout_prob > 0)
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(
        self,
        labels: torch.Tensor,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Replace a random subset of labels with the null class for CFG."""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(
        self,
        labels: torch.Tensor,
        train: bool,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


# ---------------------------------------------------------------------------
# DiffiT attention with relative position bias
# ---------------------------------------------------------------------------

class DiffiTAttention(nn.Module):
    """
    Window-based multi-head self-attention with time-embedding modulation
    and learnable relative position bias (Swin-style).
    """

    def __init__(
        self,
        dim: int,
        temb_dim: Optional[int],
        num_heads: int,
        window_size: int = 16,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Time-embedding projection ------------------------------------------
        if temb_dim is not None:
            self.qkv_temb = nn.Linear(temb_dim, dim * 3)

        # Window / relative position bias ------------------------------------
        self.window_size = (window_size, window_size)
        ws = self.window_size

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * ws[0] - 1) * (2 * ws[1] - 1), num_heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(ws[0])
        coords_w = torch.arange(ws[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flat = torch.flatten(coords, 1)

        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += ws[0] - 1
        relative_coords[:, :, 1] += ws[1] - 1
        relative_coords[:, :, 0] *= 2 * ws[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        # Linear projections -------------------------------------------------
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor]) -> torch.Tensor:
        B, N, C = x.shape

        if temb is not None:
            qkv_temb = self.qkv_temb(temb).unsqueeze(1).to(x.dtype)
            qkv = qkv_temb + self.qkv(x)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                          # each: (B, heads, N, head_dim)

        attn = (q * self.scale) @ k.transpose(-2, -1)    # (B, heads, N, N)

        # Add relative position bias
        ws = self.window_size
        rpb = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(ws[0] * ws[1], ws[0] * ws[1], -1)
        attn = attn + rpb.permute(2, 0, 1).contiguous().unsqueeze(0)

        attn = self.attn_drop(self.softmax(attn))

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class DiffiTBlock(nn.Module):
    """Single DiffiT transformer block: Norm → Attention → Norm → MLP."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = DiffiTAttention(
            dim=hidden_size,
            temb_dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), c)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Final projection layer
# ---------------------------------------------------------------------------

class FinalLayer(nn.Module):
    """Final projection: LayerNorm → SiLU → Linear  →  patch-level predictions."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.silu(self.norm_final(x)))


# ---------------------------------------------------------------------------
# DiffiT model
# ---------------------------------------------------------------------------

# Latent-size threshold that separates the two CFG strategies.
# image_size 256 → latent_size 32  (power-cosine CFG)
# image_size 512 → latent_size 64  (linear CFG)
_LATENT_SIZE_THRESHOLD = 32


class DiffiT(nn.Module):
    """
    DiffiT: Diffusion Vision Transformers for Image Generation.

    A class-conditional latent diffusion model built from DiffiTBlocks
    with window-based relative-position self-attention and sinusoidal
    time/position embeddings.
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 30,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        mask_ratio=None,
        decode_layer=None,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        # window_size must match the spatial grid so the relative
        # position bias table covers every token pair.
        window_size = input_size // patch_size  # 256→16, 512→32

        # Embedders -----------------------------------------------------------
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        # Transformer backbone ------------------------------------------------
        self.blocks = nn.ModuleList([
            DiffiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, window_size=window_size)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self._initialize_weights()

    # ----- weight init -------------------------------------------------------

    def _initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Fixed sinusoidal position embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Patch embedding projection
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Label embedding
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Timestep MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-init final projection so the model starts near identity
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # ----- helpers -----------------------------------------------------------

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rearrange patch tokens back into a spatial feature map.

        Args:
            x: ``(N, T, patch_size**2 * C)``

        Returns:
            ``(N, C, H, W)``
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, h * p)

    # ----- forward -----------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        enable_mask: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of DiffiT.

        Args:
            x: ``(N, C, H, W)`` spatial inputs (images or latent codes).
            t: ``(N,)`` diffusion timesteps.
            y: ``(N,)`` class labels.
            enable_mask: Reserved for future use.

        Returns:
            ``(N, out_C, H, W)`` noise (and optionally variance) prediction.
        """
        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t) + self.y_embedder(y, self.training)

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x)
        return self.unpatchify(x)

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: Optional[float] = None,
        diffusion_steps: int = 1000,
        scale_pow: float = 4.0,
    ) -> torch.Tensor:
        """
        Forward pass with (optional) classifier-free guidance.

        When ``cfg_scale`` is given the batch is assumed to be
        ``[cond_half, uncond_half]`` and guidance is applied to the
        predicted noise.

        CFG strategy varies by resolution:
            - **image_size 256** (``input_size <= 32``): power-cosine schedule
              that ramps the effective scale over the diffusion process.
            - **image_size 512** (``input_size > 32``): constant linear CFG
              scaling.
        """
        if cfg_scale is None:
            model_out = self.forward(x, t, y)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            return torch.cat([eps, rest], dim=1)

        # Duplicate the conditional half for paired evaluation
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)

        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

        if self.input_size <= _LATENT_SIZE_THRESHOLD:
            # ----- image_size 256: power-cosine CFG schedule -----------------
            phase = ((1 - t / diffusion_steps) ** scale_pow) * math.pi
            scale_step = 0.5 * (1 - torch.cos(phase))
            real_cfg_scale = (cfg_scale - 1) * scale_step + 1
            real_cfg_scale = real_cfg_scale[: len(x) // 2].view(-1, 1, 1, 1)
        else:
            # ----- image_size 512: constant linear CFG -----------------------
            real_cfg_scale = cfg_scale

        half_eps = uncond_eps + real_cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


# ---------------------------------------------------------------------------
# Model constructor
# ---------------------------------------------------------------------------

def Diffit(**kwargs):
    """DiffiT-XL/2 configuration (depth 28, dim 1152, patch 2, 16 heads)."""
    return DiffiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

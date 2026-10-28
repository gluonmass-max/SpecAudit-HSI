"""Frozen DOFA ViT-B with dense multi-depth outputs and LoRA on the attention projections.

Uses the official DOFA wavelength-conditioned patch embedding (Dynamic_MLP_OFA) and returns token grids
from selected blocks instead of a pooled feature. Wavelengths are passed in micrometres.
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

from .wave_dynamic_layer import Dynamic_MLP_OFA


class OFAViTDense(nn.Module):
    """Same construction as DOFA's OFAViT, so the pretrained state_dict keys match."""

    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, wv_planes=128,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.wv_planes = wv_planes
        self.patch_embed = Dynamic_MLP_OFA(wv_planes=wv_planes, inter_dim=128, kernel_size=patch_size,
                                           embed_dim=embed_dim)
        self.pretrain_grid = img_size // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.pretrain_grid ** 2 + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                                     for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

    def _interp_pos_embed(self, grid_h, grid_w):
        pe = self.pos_embed
        g = self.pretrain_grid
        if grid_h == g and grid_w == g:
            return pe
        patch_pe = pe[:, 1:].reshape(1, g, g, self.embed_dim).permute(0, 3, 1, 2)
        patch_pe = F.interpolate(patch_pe, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, self.embed_dim)
        return torch.cat([pe[:, :1], patch_pe], dim=1)

    def forward_multiscale(self, x, wave_list, out_indices=(2, 5, 8, 11)):
        """(B, C, H, W) -> list of (B, 768, H/16, W/16) maps after the requested blocks (0-based)."""
        B, _, H, W = x.shape
        grid_h, grid_w = H // self.patch_size, W // self.patch_size
        wavelist = torch.as_tensor(wave_list, device=x.device, dtype=torch.float32)
        tokens, _ = self.patch_embed(x, wavelist)
        pos_embed = self._interp_pos_embed(grid_h, grid_w)
        tokens = tokens + pos_embed[:, 1:, :]
        tokens = torch.cat([(self.cls_token + pos_embed[:, :1, :]).expand(B, -1, -1), tokens], dim=1)
        feats, out_indices = [], set(out_indices)
        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens)
            if i in out_indices:
                feats.append(tokens[:, 1:, :].transpose(1, 2).reshape(B, self.embed_dim, grid_h, grid_w))
            if i == max(out_indices):
                break
        return feats


def build_dofa_vit_base(checkpoint_path):
    model = OFAViTDense()
    sd = torch.load(checkpoint_path, map_location="cpu")
    missing, _ = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith(("head", "fc_norm"))]
    assert not bad, f"missing DOFA keys: {bad}"
    return model


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


class LoRA_QKV(nn.Module):
    """LoRA on the Q and V slices of timm's fused qkv projection; K stays frozen."""

    def __init__(self, qkv: nn.Linear, r: int, dim: int):
        super().__init__()
        self.qkv, self.dim, self.scale = qkv, dim, 1.0
        self.lora_A_q = nn.Linear(dim, r, bias=False)
        self.lora_B_q = nn.Linear(r, dim, bias=False)
        self.lora_A_v = nn.Linear(dim, r, bias=False)
        self.lora_B_v = nn.Linear(r, dim, bias=False)
        nn.init.kaiming_uniform_(self.lora_A_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_q.weight)
        nn.init.zeros_(self.lora_B_v.weight)

    def forward(self, x):
        qkv = self.qkv(x).clone()
        qkv[..., : self.dim] += self.lora_B_q(self.lora_A_q(x)) * self.scale
        qkv[..., -self.dim:] += self.lora_B_v(self.lora_A_v(x)) * self.scale
        return qkv


def inject_lora(model: OFAViTDense, r: int):
    for blk in model.blocks:
        old = blk.attn.qkv
        for p in old.parameters():
            p.requires_grad = False
        blk.attn.qkv = LoRA_QKV(old, r=r, dim=old.in_features)
    return model

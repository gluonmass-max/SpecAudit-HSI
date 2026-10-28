"""TLSA network: Band Amplitude Calibration (BAC) in front of a frozen DOFA entry, LoRA-adapted DOFA ViT-B,
a multi-depth decoder and the baseline spectral-gating branch.

BAC multiplies each input band by one learnable gain (initialised to 1) shared across images and pixels.
The spectral-gating branch receives the uncalibrated input. The second level (SGSR) is applied at inference
on the saved coarse map, see tlsa/sgsr.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bands import wavelengths_um
from .dofa_backbone import build_dofa_vit_base, freeze, inject_lora
from .losses import IOU

BCE = nn.BCELoss()
IOU_LOSS = IOU()


class SpectralGatingBranch(nn.Module):
    """Pixel-wise cosine contrast to the image-mean spectrum -> gate logits at the four decoder resolutions."""

    def __init__(self, n_levels=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )
        self.gates = nn.ModuleList([nn.Conv2d(16, 1, 1) for _ in range(n_levels)])

    def forward(self, x, base_res):
        mu = x.mean(dim=(2, 3), keepdim=True)
        s = 1.0 - (F.normalize(x, dim=1) * F.normalize(mu, dim=1)).sum(dim=1, keepdim=True)
        feat = self.stem(s)
        return [conv(F.adaptive_avg_pool2d(feat, base_res * 2 ** i)) for i, conv in enumerate(self.gates)]


class Decoder(nn.Module):
    """Four 16x16 token maps -> 128x128 saliency logits."""

    def __init__(self, in_dim=768, dec_dim=192):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.Conv2d(in_dim, dec_dim, 1), nn.BatchNorm2d(dec_dim),
                                                 nn.ReLU(inplace=True)) for _ in range(4)])
        self.fuse0 = nn.Sequential(nn.Conv2d(dec_dim * 4, dec_dim, 3, padding=1), nn.BatchNorm2d(dec_dim),
                                   nn.ReLU(inplace=True))
        self.up_convs = nn.ModuleList([nn.Sequential(nn.Conv2d(dec_dim, dec_dim, 3, padding=1),
                                                     nn.BatchNorm2d(dec_dim), nn.ReLU(inplace=True))
                                       for _ in range(3)])
        self.out_conv = nn.Conv2d(dec_dim, 1, 1)

    def forward(self, feats, gates):
        x = self.fuse0(torch.cat([p(f) for p, f in zip(self.proj, feats)], dim=1))
        x = x * (1.0 + torch.sigmoid(gates[0]))
        for conv, g in zip(self.up_convs, gates[1:]):
            x = conv(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
            x = x * (1.0 + torch.sigmoid(g))
        return self.out_conv(x)


class TLSANet(nn.Module):
    def __init__(self, dofa_ckpt, band_idx=None, lora_r=8, img_size=256, feat_layers=(2, 5, 8, 11)):
        super().__init__()
        wl = wavelengths_um(band_idx)
        self.img_size, self.feat_layers = img_size, tuple(feat_layers)
        self.backbone = build_dofa_vit_base(dofa_ckpt)
        freeze(self.backbone)
        inject_lora(self.backbone, r=lora_r)
        self.register_buffer("wl_base", torch.tensor(wl, dtype=torch.float32))
        self.band_scale = nn.Parameter(torch.ones(1, len(wl), 1, 1))           # BAC: one gain per band
        self.spectral_branch = SpectralGatingBranch()
        self.decoder = Decoder()

    def forward(self, x, gt=None):
        feats = self.backbone.forward_multiscale(x * self.band_scale, self.wl_base, out_indices=self.feat_layers)
        logits = self.decoder(feats, self.spectral_branch(x, feats[0].shape[-1]))
        logits = F.interpolate(logits, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        if self.training:
            prob = torch.sigmoid(logits)
            return logits, BCE(prob, gt) + IOU_LOSS(prob, gt)
        return logits

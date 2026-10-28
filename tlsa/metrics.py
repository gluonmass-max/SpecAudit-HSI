"""S-measure used for checkpoint selection, from the HRSSD/DSSN reference code
(MIT, see third_party_licenses/LICENSE.DSSN_HRSSD). Reported results use eval/ (official evaluator)."""
import os
import random

import numpy as np
import torch


def _object(pred, gt):
    temp = pred[gt == 1]
    x, sigma_x = temp.mean(), temp.std()
    return 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)


def _s_object(pred, gt):
    fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
    bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
    u = gt.mean()
    return u * _object(fg, gt) + (1 - u) * _object(bg, 1 - gt)


def _centroid(gt):
    rows, cols = gt.size()[-2:]
    gt = gt.view(rows, cols)
    if gt.sum() == 0:
        X = torch.eye(1, device=gt.device) * round(cols / 2)
        Y = torch.eye(1, device=gt.device) * round(rows / 2)
    else:
        total = gt.sum()
        i = torch.arange(cols, device=gt.device).float()
        j = torch.arange(rows, device=gt.device).float()
        X = torch.round((gt.sum(dim=0) * i).sum() / total)
        Y = torch.round((gt.sum(dim=1) * j).sum() / total)
    return X.long(), Y.long()


def _split(a, X, Y):
    h, w = a.size()[-2:]
    a = a.view(h, w)
    return a[:Y, :X], a[:Y, X:w], a[Y:h, :X], a[Y:h, X:w]


def _ssim(pred, gt):
    gt = gt.float()
    h, w = pred.size()[-2:]
    N = h * w
    x, y = pred.mean(), gt.mean()
    sx2 = ((pred - x) ** 2).sum() / (N - 1 + 1e-20)
    sy2 = ((gt - y) ** 2).sum() / (N - 1 + 1e-20)
    sxy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)
    alpha, beta = 4 * x * y * sxy, (x * x + y * y) * (sx2 + sy2)
    if alpha != 0:
        return alpha / (beta + 1e-20)
    return 1.0 if beta == 0 else 0


def _s_region(pred, gt):
    X, Y = _centroid(gt)
    h, w = gt.size()[-2:]
    Xf, Yf = X.float(), Y.float()
    w1 = Xf * Yf / (h * w)
    w2 = (w - Xf) * Yf / (h * w)
    w3 = Xf * (h - Yf) / (h * w)
    ws = [w1, w2, w3, 1 - w1 - w2 - w3]
    return sum(wi * _ssim(p, g) for wi, p, g in zip(ws, _split(pred, X, Y), _split(gt, X, Y)))


def smeasure(pred, gt):
    """Mean S-measure (alpha = 0.5) over a batch of (1, H, W) maps."""
    total = 0.0
    with torch.no_grad():
        for p, g in zip(pred, gt):
            y = g.mean()
            if y == 0:
                q = 1.0 - p.mean()
            elif y == 1:
                q = p.mean()
            else:
                g = (g >= 0.5).float()
                q = 0.5 * _s_object(p, g) + 0.5 * _s_region(p, g)
                q = torch.clamp(torch.as_tensor(q), min=0.0)
            total += float(q)
    return total / len(pred)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

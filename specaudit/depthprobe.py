"""DepthProbe: linear readability of the 32-band advantage at each depth of the DOFA backbone (patch level).

Features: patch-mean spectra ("in", no PCA) and the tokens after the entry (depth 0) and after selected blocks.
Patch label = foreground fraction > 0.5 in each 16 x 16 window (ignored pixels count as background).
Token features are projected with PCA (k components, at most 60 000 training tokens, fitted on training
features only). Readouts, all with covariance-scaled regularisation 1e-3 (tr(S_w)/d + 1e-12):
  global     one Fisher direction fitted on all training patches, AUC per test image
  oracle     per test image, fitted and scored on the same patches
  splithalf  per test image, random halves, each scored by the direction fitted on the other half
Only test images with at least 8 foreground and 8 background patches are used. Reported: 32b - 3b AUC, paired
over images, 10 000 bootstrap resamples.

Frozen pretrained backbone (default), or task-adapted backbones by giving both checkpoints:
  python specaudit/depthprobe.py --data_root /path/to/HRSSD --dofa_ckpt DOFA_ViT_base_e100.pth --out results/depthprobe
  python specaudit/depthprobe.py ... --ckpt_32b runs/a/model-best-val --ckpt_3b runs/b/model-best-val
Adapted checkpoints must be TLSANet state dicts with 32 and 3 (visible) input bands; checkpoints without
BAC gains are accepted and read with unit gains.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tlsa.bands import VISIBLE_3B, wavelengths_um  # noqa: E402
from tlsa.data import HRSSD  # noqa: E402
from tlsa.dofa_backbone import build_dofa_vit_base  # noqa: E402
from tlsa.model import TLSANet  # noqa: E402

CONF = {"3b": VISIBLE_3B, "32b": list(range(32))}
MIN = 8


def fisher(X0, X1, rel_eps=1e-3):
    Sw = (np.cov(X0, rowvar=False) * (len(X0) - 1) + np.cov(X1, rowvar=False) * (len(X1) - 1)) / (len(X0) + len(X1) - 2)
    Sw = np.atleast_2d(Sw); d = Sw.shape[0]
    return np.linalg.solve(Sw + rel_eps * (np.trace(Sw) / d + 1e-12) * np.eye(d), X1.mean(0) - X0.mean(0))


def auc(score, y):
    r = rankdata(score)
    P = y.sum(); N = len(y) - P
    return float((r[y == 1].sum() - P * (P + 1) / 2) / (P * N))


class Tapper:
    """Token grids of one backbone for one band configuration; scale = BAC gains (1 for the frozen model)."""

    def __init__(self, backbone, idx, scale=None):
        self.m, self.idx, self.scale = backbone, idx, scale
        self.wl = torch.tensor(wavelengths_um(idx))

    @torch.no_grad()
    def __call__(self, x, layers):
        m = self.m
        x = x[:, self.idx]
        if self.scale is not None:
            x = x * self.scale
        B, _, H, W = x.shape
        gh, gw = H // m.patch_size, W // m.patch_size
        tok, _ = m.patch_embed(x, self.wl.to(x.device))
        pe = m._interp_pos_embed(gh, gw)
        tok = torch.cat([(m.cls_token + pe[:, :1, :]).expand(B, -1, -1), tok + pe[:, 1:, :]], 1)
        outs = {0: tok[:, 1:, :].clone()} if 0 in layers else {}
        for l, blk in enumerate(m.blocks, start=1):
            tok = blk(tok)
            if l in layers:
                outs[l] = tok[:, 1:, :].clone()
            if l >= max(layers):
                break
        return outs


def collect(data, tappers, layers, size, dev):
    feats = {c: {l: [] for l in ["in"] + layers} for c in tappers}
    labels = []
    for gt, img, _, _, _ in DataLoader(data, batch_size=1, shuffle=False):
        img = F.interpolate(img.to(dev).float(), size, mode="nearest")
        gt = gt.to(dev).float(); gt = gt * (gt != -1).float()
        g = F.interpolate(gt[:, None], size, mode="nearest")
        labels.append((F.avg_pool2d(g, 16).flatten(1) > 0.5).float().cpu())
        for c, tap in tappers.items():
            feats[c]["in"].append(F.avg_pool2d(img[:, CONF[c]], 16).flatten(2).transpose(1, 2).cpu())
            for l, t in tap(img, layers).items():
                feats[c][l].append(t.cpu())
    return {c: {l: torch.cat(v).numpy() for l, v in feats[c].items()} for c in tappers}, torch.cat(labels).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--dofa_ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--ckpt_32b", default=None)
    ap.add_argument("--ckpt_3b", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); rng = np.random.RandomState(0)

    if args.ckpt_32b and args.ckpt_3b:
        layers, tag, tappers = [0, 1, 3, 6, 9, 12], "adapted", {}
        for c, ck in (("32b", args.ckpt_32b), ("3b", args.ckpt_3b)):
            net = TLSANet(args.dofa_ckpt, band_idx=CONF[c])
            missing, unexpected = net.load_state_dict(torch.load(ck, map_location="cpu"), strict=False)
            assert not unexpected and set(missing) <= {"band_scale"}, (missing, unexpected)
            net = net.to(dev).eval()
            tappers[c] = Tapper(net.backbone, CONF[c], net.band_scale)
    else:
        layers, tag = [0, 1, 2, 4, 6, 8, 10, 12], "frozen"
        model = build_dofa_vit_base(args.dofa_ckpt).to(dev).eval()
        tappers = {c: Tapper(model, idx) for c, idx in CONF.items()}

    tr = HRSSD(args.data_root, "tr"); tr.train = False
    Xtr, Ytr = collect(tr, tappers, layers, args.size, dev)
    Xte, Yte = collect(HRSSD(args.data_root, "ts"), tappers, layers, args.size, dev)

    rows = []
    for l in ["in"] + layers:
        res = {}
        for c in CONF:
            Ftr, Fte = Xtr[c][l], Xte[c][l]
            if l == "in":
                Ptr, Pte = Ftr, Fte
            else:
                Z = Ftr.reshape(-1, Ftr.shape[-1]).astype(np.float64)
                if Z.shape[0] > 60000:
                    Z = Z[rng.choice(Z.shape[0], 60000, replace=False)]
                mu = Z.mean(0)
                _, V = np.linalg.eigh(np.atleast_2d(np.cov(Z - mu, rowvar=False)))
                V = V[:, ::-1][:, :args.k]
                Ptr, Pte = (Ftr.astype(np.float64) - mu) @ V, (Fte.astype(np.float64) - mu) @ V
            A, y = Ptr.reshape(-1, Ptr.shape[-1]), Ytr.reshape(-1)
            w = fisher(A[y == 0], A[y == 1])
            g, o, s = [], [], []
            for P, yi in zip(Pte, Yte):
                if (yi == 1).sum() < MIN or (yi == 0).sum() < MIN:
                    g.append(np.nan); o.append(np.nan); s.append(np.nan); continue
                g.append(auc(P @ w, yi))
                o.append(auc(P @ fisher(P[yi == 0], P[yi == 1]), yi))
                fold, sc, ok = rng.rand(len(yi)) < 0.5, np.zeros(len(yi)), True
                for fA in (True, False):
                    a = fold == fA
                    if (yi[a] == 1).sum() < 3 or (yi[a] == 0).sum() < 3:
                        ok = False; break
                    sc[~a] = P[~a] @ fisher(P[a & (yi == 0)], P[a & (yi == 1)])
                s.append(auc(sc, yi) if ok else np.nan)
            res[c] = {"global": np.array(g), "oracle": np.array(o), "splithalf": np.array(s)}
        r = {"depth": l}
        for key in ("global", "oracle", "splithalf"):
            a, b = res["3b"][key], res["32b"][key]
            m = ~np.isnan(a) & ~np.isnan(b); d = b[m] - a[m]
            bs = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(10000)])
            r.update({f"{key}_dAUC": d.mean(), f"{key}_lo": np.percentile(bs, 2.5),
                      f"{key}_hi": np.percentile(bs, 97.5), f"{key}_n": int(m.sum())})
        rows.append(r)
        print(f"depth {str(l):>2s} | global dAUC {r['global_dAUC']:+.4f} [{r['global_lo']:+.4f}, {r['global_hi']:+.4f}]"
              f" | split-half dAUC {r['splithalf_dAUC']:+.4f} [{r['splithalf_lo']:+.4f}, {r['splithalf_hi']:+.4f}]"
              f" n={r['splithalf_n']}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, f"depthprobe_{tag}_k{args.k}_{args.size}.csv"), index=False)


if __name__ == "__main__":
    main()

"""Spectral Ledger: linear readability of the 32-band advantage at the input (pixel level).

One regularised Fisher discriminant w = (S_w + eps I)^-1 (mu_1 - mu_0) is fitted on reflectance spectra and its
projection is mapped to [0, 1] by a scaled logistic (centre = midpoint of the class-mean projections, scale =
projection standard deviation of the fitting set). The 32-band input and a three-band reference arm are compared
on the same pixels; readings differ only in fitting scope:
  in   per-image, in-sample: fitted and scored on all valid pixels of each test image
  sh   per-image, random split-half: fitted on one random half, scores the other half, both directions
  shs  per-image, spatial split-half: left/right and top/bottom halves, averaged over valid axes
  gl   global direction fitted on training pixels (at most 4000 per class per image), training-derived calibration
  gln  global direction, calibrated by the mean and std of the test image's projections (no labels)
  glc  global direction, calibrated with the test image's class-mean midpoint (uses test labels; control only)
The per-image and glc readings use test labels and are diagnostic existence probes, not predictions.
Output: per-image CSV and a table of paired 32b - 3b differences (S-measure and AUC, 10 000 bootstrap resamples).

  python specaudit/ledger.py --data_root /path/to/HRSSD --out results/ledger
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from osgeo import gdal
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tlsa.bands import BAND_CONFIGS  # noqa: E402
from tlsa.metrics import smeasure  # noqa: E402

REFS = {"visible": BAND_CONFIGS["Visible 3b"], "random": BAND_CONFIGS["Random 3b"], "spread": BAND_CONFIGS["Spread 3b"]}
READINGS = ["in", "sh", "shs", "gl", "gln", "glc"]
gdal.UseExceptions()


def fit(X0, X1, eps):
    Sw = (np.cov(X0, rowvar=False) * (len(X0) - 1) + np.cov(X1, rowvar=False) * (len(X1) - 1)) / (len(X0) + len(X1) - 2)
    w = np.linalg.solve(Sw + eps * np.eye(X0.shape[1]), X1.mean(0) - X0.mean(0))
    z0, z1 = X0 @ w, X1 @ w
    return w, 0.5 * (z0.mean() + z1.mean()), np.concatenate([z0, z1]).std() + 1e-6


def squash(z, c, sc):
    return 1.0 / (1.0 + np.exp(-(z - c) / sc * 3))


def auc(score, y):
    r = rankdata(score)
    P = y.sum(); N = len(y) - P
    return float((r[y == 1].sum() - P * (P + 1) / 2) / (P * N))


def load(split_dir, n):
    gt = gdal.Open(f"{split_dir}/label/{n}.tif").ReadAsArray().astype(np.float32)
    mask = gt != -1
    img = gdal.Open(f"{split_dir}/image/{n}.tif").ReadAsArray().astype(np.float32) / 10000.0
    return gt, mask, img.reshape(img.shape[0], -1).T, ((gt > 0.5) & mask).reshape(-1).astype(int), mask.reshape(-1)


def score(s, gt, mask, y, valid):
    S = smeasure(torch.from_numpy((s.reshape(gt.shape) * mask).astype(np.float32))[None],
                 torch.from_numpy(((gt > 0.5) & mask).astype(np.float32))[None])
    return S, auc(s[valid], y[valid])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="visible", choices=list(REFS))
    ap.add_argument("--eps", type=float, default=1e-3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(0)
    tr, ts = os.path.join(args.data_root, "tr"), os.path.join(args.data_root, "ts")
    arms = {"3": REFS[args.ref], "32": list(range(32))}

    P0, P1 = [], []
    for n in sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{tr}/label/*.tif")):
        _, _, X, y, valid = load(tr, n)
        i0, i1 = np.where(valid & (y == 0))[0], np.where(valid & (y == 1))[0]
        if len(i0) > 4000: i0 = rng.choice(i0, 4000, replace=False)
        if len(i1) > 4000: i1 = rng.choice(i1, 4000, replace=False)
        P0.append(X[i0]); P1.append(X[i1])
    P0, P1 = np.concatenate(P0), np.concatenate(P1)
    G = {k: fit(P0[:, idx], P1[:, idx], args.eps) for k, idx in arms.items()}

    recs = []
    for n in sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{ts}/label/*.tif")):
        gt, mask, X, y, valid = load(ts, n)
        H, W = gt.shape
        col, row = np.tile(np.arange(W), H), np.repeat(np.arange(H), W)
        rec = {"name": n}
        for k, idx in arms.items():
            Xs = X[:, idx]
            i0, i1 = np.where(valid & (y == 0))[0], np.where(valid & (y == 1))[0]
            if len(i0) < 20 or len(i1) < 20:
                continue
            w, c, sc = fit(Xs[i0], Xs[i1], args.eps)
            rec[f"in_S_{k}"], rec[f"in_A_{k}"] = score(squash(Xs @ w, c, sc), gt, mask, y, valid)
            fold, s = rng.rand(len(y)) < 0.5, np.zeros(len(y))
            for fA in (True, False):
                a = fold == fA
                w, c, sc = fit(Xs[a & valid & (y == 0)], Xs[a & valid & (y == 1)], args.eps)
                s[~a] = squash(Xs[~a] @ w, c, sc)
            rec[f"sh_S_{k}"], rec[f"sh_A_{k}"] = score(s, gt, mask, y, valid)
            S_ax, A_ax = [], []
            for half in (col < W // 2, row < H // 2):
                if not all(((half == fA) & valid & (y == cl)).sum() >= 20 for fA in (True, False) for cl in (0, 1)):
                    continue
                s = np.zeros(len(y))
                for fA in (True, False):
                    a = half == fA
                    w, c, sc = fit(Xs[a & valid & (y == 0)], Xs[a & valid & (y == 1)], args.eps)
                    s[~a] = squash(Xs[~a] @ w, c, sc)
                Sv, Av = score(s, gt, mask, y, valid); S_ax.append(Sv); A_ax.append(Av)
            if S_ax:
                rec[f"shs_S_{k}"], rec[f"shs_A_{k}"] = np.mean(S_ax), np.mean(A_ax)
            w, c, sc = G[k]; z = Xs @ w
            rec[f"gl_S_{k}"], rec[f"gl_A_{k}"] = score(squash(z, c, sc), gt, mask, y, valid)
            rec[f"gln_S_{k}"], rec[f"gln_A_{k}"] = score(squash(z, z[valid].mean(), z[valid].std() + 1e-6), gt, mask, y, valid)
            c2 = 0.5 * (z[i0].mean() + z[i1].mean())
            rec[f"glc_S_{k}"], rec[f"glc_A_{k}"] = score(squash(z, c2, z[valid].std() + 1e-6), gt, mask, y, valid)
        recs.append(rec)
    d = pd.DataFrame(recs)
    d.to_csv(os.path.join(args.out, f"ledger_{args.ref}.csv"), index=False)

    L = [f"| reading | n | dS (32b - 3b) [95% CI] | dAUC [95% CI] |", "|---|:-:|:-:|:-:|"]
    for r in READINGS:
        cols = [f"{r}_{m}_{k}" for m in "SA" for k in ("3", "32")]
        if not set(cols) <= set(d.columns):
            continue
        m = d[cols].dropna()
        out = []
        for met in "SA":
            diff = (m[f"{r}_{met}_32"] - m[f"{r}_{met}_3"]).values
            bs = np.array([diff[rng.randint(0, len(diff), len(diff))].mean() for _ in range(10000)])
            out.append(f"{diff.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
        L.append(f"| {r} | {len(m)} | {out[0]} | {out[1]} |")
    open(os.path.join(args.out, f"ledger_{args.ref}.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

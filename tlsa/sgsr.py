"""Self-Guided Spectral Refinement (SGSR): image-specific, training-free refinement of a coarse saliency map.

For one image:
  1. candidates from the coarse map P over all pixels: foreground P >= 0.5, background P <= 0.1;
     a class with fewer than 20 pixels is redefined by the 0.95 (foreground) or 0.30 (background) quantile of P;
  2. regularised Fisher discriminant on the candidate spectra: w = (S_w + 1e-3 I)^-1 (mu_fg - mu_bg);
  3. spectral score S = sigmoid(3 (z - a) / (sd(z) + 1e-6)), z = X w, a = midpoint of the class-mean projections;
  4. Gaussian smoothing of S (sigma = 4 pixels);
  5. geometric fusion P_hat = sqrt(P * G(S)).
If a class still has fewer than 20 pixels, S = P. No annotation is read and no network weight is updated.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

TAU_HI, TAU_LO, MIN_COUNT, Q_FG, Q_BG = 0.5, 0.1, 20, 0.95, 0.30
EPS, KAPPA, DELTA, SIGMA = 1e-3, 3.0, 1e-6, 4.0


def sgsr(P, X):
    """P: (H, W) coarse map in [0, 1]; X: (H, W, C) reflectance. Returns the refined (H, W) map."""
    shape = P.shape
    p = P.reshape(-1).astype(np.float32)
    X = X.reshape(-1, X.shape[-1]).astype(np.float32)
    fg, bg = p >= TAU_HI, p <= TAU_LO
    if fg.sum() < MIN_COUNT:
        fg = p >= np.quantile(p, Q_FG)
    if bg.sum() < MIN_COUNT:
        bg = p <= np.quantile(p, Q_BG)
    S = p
    if fg.sum() >= MIN_COUNT and bg.sum() >= MIN_COUNT:
        X0, X1 = X[bg], X[fg]
        Sw = (np.cov(X0, rowvar=False) * (len(X0) - 1) + np.cov(X1, rowvar=False) * (len(X1) - 1)) \
            / (len(X0) + len(X1) - 2)
        w = np.linalg.solve(Sw + EPS * np.eye(X.shape[1]), X1.mean(0) - X0.mean(0))
        z = X @ w
        z = z - 0.5 * (z[bg].mean() + z[fg].mean())
        S = 1.0 / (1.0 + np.exp(-KAPPA * z / (z.std() + DELTA)))
    S = gaussian_filter(S.reshape(shape), SIGMA)
    return np.clip(np.sqrt(np.clip(p.reshape(shape) * S, 0, None)), 0, 1)

"""KernelScope: geometry of the kernels that the frozen DOFA entry generates from a wavelength table.

No images, no labels, CPU only. For each band configuration the entry generates its own C kernels; each kernel is
flattened into one row of K (C x 16*16*768). Reported per configuration:
  mean cos   mean off-diagonal cosine between rows of K
  r_eff      effective rank exp(-sum_j p_j log p_j), p_j = sigma_j / sum_k sigma_k (singular values of K)
  r_eff / C
  chi        C ||k_bar||^2 / ||K||_F^2, energy fraction of the common (mean) kernel; 1 for identical kernels,
             1/C for mutually orthogonal kernels of equal norm
Matched-shape i.i.d. Gaussian matrices are reported as a geometric reference.

  python specaudit/kernelscope.py --dofa_ckpt DOFA_ViT_base_e100.pth [--table image]
--table entry (default) uses the band centres supplied to the entry in the paper; --table image uses the
nominal image band centres (see tlsa/bands.py).
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tlsa.bands import BAND_CONFIGS, ENTRY_WAVELENGTHS_NM, IMAGE_BAND_CENTRES_NM  # noqa: E402
from tlsa.wave_dynamic_layer import Dynamic_MLP_OFA, get_1d_sincos_pos_embed_from_grid_torch  # noqa: E402


def load_entry(ckpt):
    pe = Dynamic_MLP_OFA(wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=768)
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("model", sd)
    missing, _ = pe.load_state_dict({k[len("patch_embed."):]: v for k, v in sd.items()
                                     if k.startswith("patch_embed.")}, strict=False)
    assert not missing, f"entry not fully loaded: {missing}"
    return pe.eval()


@torch.no_grad()
def kernels(pe, wl_nm):
    waves = get_1d_sincos_pos_embed_from_grid_torch(pe.wv_planes, torch.tensor(wl_nm, dtype=torch.float32))
    w, _ = pe._get_weights(pe.fclayer(waves))
    return (w.reshape(w.shape[0], -1) * pe.scaler).double()


def geometry(K):
    C = K.shape[0]
    s = torch.linalg.svdvals(K); p = s / s.sum()
    r_eff = float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))
    Kn = K / K.norm(dim=1, keepdim=True); G = Kn @ Kn.t()
    cos = float((G.sum() - G.diagonal().sum()) / (C * (C - 1)))
    chi = float(C * K.mean(0).pow(2).sum() / K.pow(2).sum())
    return dict(C=C, mean_cos=cos, r_eff=r_eff, r_eff_over_C=r_eff / C, chi=chi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dofa_ckpt", required=True)
    ap.add_argument("--table", default="entry", choices=["entry", "image"])
    args = ap.parse_args()
    table = ENTRY_WAVELENGTHS_NM if args.table == "entry" else IMAGE_BAND_CENTRES_NM
    pe = load_entry(args.dofa_ckpt)
    print("| configuration | C | mean cos | r_eff | r_eff/C | chi |\n|---|:-:|:-:|:-:|:-:|:-:|")
    for name, idx in BAND_CONFIGS.items():
        g = geometry(kernels(pe, [table[i] for i in idx]))
        print(f"| {name} | {g['C']} | {g['mean_cos']:.3f} | {g['r_eff']:.2f} | {g['r_eff_over_C']:.3f} | {g['chi']:.3f} |")
    gen = torch.Generator().manual_seed(0)
    g = geometry(torch.randn(32, 16 * 16 * 768, generator=gen, dtype=torch.float64))
    print(f"| i.i.d. Gaussian reference | 32 | {g['mean_cos']:.3f} | {g['r_eff']:.2f} | {g['r_eff_over_C']:.3f} | {g['chi']:.3f} |")


if __name__ == "__main__":
    main()

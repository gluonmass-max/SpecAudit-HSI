"""TLSA inference on the HRSSD test split: network -> coarse map -> SGSR.

As in the paper, the coarse map is resized to 512 x 512 (nearest), stored as 8-bit JPEG and reloaded for SGSR;
refined maps are written as 8-bit PNG for the official evaluator (eval/main.py).

  python infer.py --data_root /path/to/HRSSD --dofa_ckpt DOFA_ViT_base_e100.pth \
                  --ckpt runs/tlsa_s7/model-best-val --out_dir preds/tlsa_s7
Outputs: <out_dir>/coarse/*.jpg (network only) and <out_dir>/tlsa/*.png (network + SGSR).
"""
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from osgeo import gdal
from PIL import Image
from torch.utils.data import DataLoader

from tlsa.data import HRSSD, IMG_SIZE
from tlsa.model import TLSANet
from tlsa.sgsr import sgsr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--dofa_ckpt", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    coarse_dir, out_dir = os.path.join(args.out_dir, "coarse"), os.path.join(args.out_dir, "tlsa")
    os.makedirs(coarse_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    net = TLSANet(args.dofa_ckpt).cuda()
    net.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    net.eval()

    data = HRSSD(args.data_root, "ts")
    with torch.no_grad():
        for gt, img, (H, W), names, _ in DataLoader(data, batch_size=8, shuffle=False):
            pred = torch.sigmoid(net(F.interpolate(img.cuda().float(), IMG_SIZE, mode="nearest")))
            pred = F.interpolate(pred, (H[0], W[0]), mode="nearest")
            for i, n in enumerate(names):
                cv2.imwrite(os.path.join(coarse_dir, n + ".jpg"), pred[i, 0].cpu().numpy() * 255)

    for n in data.names:
        P = np.asarray(Image.open(os.path.join(coarse_dir, n + ".jpg")).convert("L"), dtype=np.float32) / 255.0
        X = gdal.Open(os.path.join(args.data_root, "ts", "image", n + ".tif")).ReadAsArray().astype(np.float32)
        X = np.transpose(X, (1, 2, 0)) / 10000.0
        refined = sgsr(P, X)
        Image.fromarray(np.round(refined * 255).astype(np.uint8)).save(os.path.join(out_dir, n + ".png"))
    print(f"wrote {len(data)} maps to {coarse_dir} and {out_dir}")


if __name__ == "__main__":
    main()

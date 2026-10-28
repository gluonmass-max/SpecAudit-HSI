"""HRSSD loader, following the HRSSD/DSSN reference pipeline (MIT, see third_party_licenses/LICENSE.DSSN_HRSSD).

Expected layout: <root>/{tr,ts}/{image,label,mask}/<name>.{tif,tif,png}. Images are divided by 10000.
Training: horizontal flip (p = 0.5) and random crop of up to 1/8 per side, then nearest resize to 256.
"""
import copy
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from osgeo import gdal
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

IMG_SIZE = 256


def _flip(img, gt, tm):
    if np.random.random() < 0.5:
        r = torch.arange(img.shape[2] - 1, -1, -1)
        return img[:, :, r], gt[:, r], tm[:, r]
    return img, gt, tm


def _crop(img, gt, tm):
    H, W = gt.shape
    randw = torch.randint(W // 8 + 1, (1,)).item()
    randh = torch.randint(H // 8 + 1, (1,)).item()
    oh = 0 if randh == 0 else torch.randint(randh, (1,)).item()
    ow = 0 if randw == 0 else torch.randint(randw, (1,)).item()
    sl = (slice(oh, H + oh - randh), slice(ow, W + ow - randw))
    return img[(slice(None),) + sl], gt[sl], tm[sl]


class HRSSD(Dataset):
    def __init__(self, root, split):
        self.root, self.split, self.train = root, split, split == "tr"
        img_dir = os.path.join(root, split, "image")
        self.names = sorted(f[:-4] for f in os.listdir(img_dir) if f.endswith(".tif"))
        self.img, self.gt, self.tm = [], [], []
        for n in tqdm(self.names, desc=f"loading {split}"):
            img = gdal.Open(os.path.join(img_dir, n + ".tif")).ReadAsArray()
            gt = gdal.Open(os.path.join(root, split, "label", n + ".tif")).ReadAsArray()
            tm = cv2.imread(os.path.join(root, split, "mask", n + ".png"), 0)
            self.img.append(torch.tensor(img, dtype=torch.float) / 10000.0)
            self.gt.append(torch.tensor(gt, dtype=torch.long))
            self.tm.append(torch.tensor(tm, dtype=torch.float))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img, gt, tm = self.img[i] * self.tm[i], self.gt[i] * self.tm[i], self.tm[i]
        if self.train:
            img, gt, tm = _crop(*_flip(img, gt, tm))
            return img, gt, self.names[i], tm
        return gt, img, gt.size(), self.names[i], tm

    @staticmethod
    def collate(batch):
        img, gt, name, tm = [list(b) for b in zip(*batch)]
        for i in range(len(batch)):
            img[i] = F.interpolate(img[i][None], size=(IMG_SIZE, IMG_SIZE), mode="nearest")[0]
            gt[i] = F.interpolate(gt[i][None, None].float(), size=(IMG_SIZE, IMG_SIZE), mode="nearest")[0]
            tm[i] = F.interpolate(tm[i][None, None], size=(IMG_SIZE, IMG_SIZE), mode="nearest")[0]
        return torch.stack(img), torch.stack(gt), name, torch.stack(tm)


def train_val_split(train_data, val_frac=0.1, split_seed=0):
    """Fixed split of the official training set: names sorted, shuffled with split_seed, 10 % held out (49 images)."""
    order = sorted(range(len(train_data)), key=lambda i: train_data.names[i])
    np.random.RandomState(split_seed).shuffle(order)
    n_val = int(round(len(order) * val_frac))
    val_view = copy.copy(train_data)
    val_view.train = False
    return Subset(train_data, sorted(order[n_val:])), Subset(val_view, sorted(order[:n_val])), \
        [train_data.names[i] for i in sorted(order[:n_val])]

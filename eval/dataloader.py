import torch
from torch.utils import data
from osgeo import gdal
import os


class EvalDataset(data.Dataset):
    def __init__(self, img_root, label_root):
        lst_label = sorted(os.listdir(label_root))
        lst_pred = sorted(os.listdir(img_root))
        # Modified from the HRSSD/DSSN evaluator: accept .png as well as .jpg predictions.
        self.image_path, self.label_path = [], []
        for name in lst_label:
            stem = name.split('.')[0]
            for ext in ('.png', '.jpg'):
                if stem + ext in lst_pred:
                    self.image_path.append(os.path.join(img_root, stem + ext))
                    self.label_path.append(os.path.join(label_root, stem + '.tif'))
                    break

    def __getitem__(self, item):
        label_dataset = gdal.Open(self.label_path[item], gdal.GA_ReadOnly)
        pred = gdal.Open(self.image_path[item], gdal.GA_ReadOnly)

        gt = label_dataset.ReadAsArray()
        gt = torch.tensor(gt, dtype=torch.float32)

        pred = pred.ReadAsArray()
        pred = torch.tensor(pred, dtype=torch.float32)
        pred = pred / 255.

        mask = (gt != -1)
        mask = torch.tensor(mask, dtype=torch.float32)

        return pred, gt, mask

    def __len__(self):
        return len(self.image_path)

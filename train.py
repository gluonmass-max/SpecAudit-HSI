"""Train the TLSA network (first level: BAC + LoRA + decoder + spectral gating; DOFA weights frozen).

Recipe of the paper: 100 epochs, batch 16, NAdam lr 3e-3, no weight decay, cosine annealing to 1e-7,
element-wise gradient clamp at lr, BCE + IoU loss. The checkpoint with the highest validation S-measure
(49 images held out from the official training split) is kept as <save_dir>/model-best-val.
The validation reading is the training-time batch mean (batches of 8), as in the paper.

  python train.py --data_root /path/to/HRSSD --dofa_ckpt DOFA_ViT_base_e100.pth --save_dir runs/tlsa_s7 --seed 7
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tlsa.data import HRSSD, IMG_SIZE, train_val_split
from tlsa.metrics import set_seed, smeasure
from tlsa.model import TLSANet

EPOCHS, BATCH, LR = 100, 16, 3e-3


@torch.no_grad()
def validate(loader, net):
    net.eval()
    total, n = 0.0, 0
    for gt, img, (h, w), _, _ in loader:
        img, gt = img.cuda(), gt.cuda()
        mask = (gt != -1).float()
        pred = torch.sigmoid(net(F.interpolate(img, IMG_SIZE, mode="nearest")))
        pred = (F.interpolate(pred, (h[0], w[0]), mode="nearest") * mask.unsqueeze(1)).squeeze(1)
        total += smeasure(pred, gt * mask)
        n += 1
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--dofa_ckpt", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    train_data = HRSSD(args.data_root, "tr")
    train_set, val_set, val_names = train_val_split(train_data)
    json.dump(val_names, open(os.path.join(args.save_dir, "val_names.json"), "w"))
    train_loader = DataLoader(train_set, collate_fn=HRSSD.collate, batch_size=BATCH, shuffle=True,
                              pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=2)

    net = TLSANet(args.dofa_ckpt).cuda()
    params = [p for p in net.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in params) / 1e6:.2f} M")
    opt = torch.optim.NAdam(params, lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-7)

    best, best_epoch = -1.0, -1
    for epoch in range(1, EPOCHS + 1):
        net.train()
        loss_sum, steps = 0.0, 0
        for img, gt, _, _ in train_loader:
            img, gt = img.float().cuda(), gt.float().cuda()
            gt = gt * (gt != -1).float()                       # unlabelled pixels count as background
            _, loss = net(img, gt)
            opt.zero_grad()
            loss.backward()
            for p in params:
                if p.grad is not None:
                    p.grad.data.clamp_(-LR, LR)
            opt.step()
            loss_sum, steps = loss_sum + loss.item(), steps + 1
        sched.step()
        val_s = validate(val_loader, net)
        if val_s > best:
            best, best_epoch = val_s, epoch
            torch.save(net.state_dict(), os.path.join(args.save_dir, "model-best-val"))
        print(f"epoch {epoch:3d}  loss {loss_sum / steps:.4f}  val S {val_s:.4f}  (best {best:.4f} @ {best_epoch})")
    json.dump({"best_val_s": best, "best_epoch": best_epoch}, open(os.path.join(args.save_dir, "summary.json"), "w"))


if __name__ == "__main__":
    main()

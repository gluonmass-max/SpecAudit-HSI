"""IoU loss, as in the HRSSD/DSSN reference implementation (MIT, see third_party_licenses/LICENSE.DSSN_HRSSD)."""
import torch


class IOU(torch.nn.Module):
    def forward(self, pred, target):
        loss = 0.0
        for i in range(pred.shape[0]):
            inter = torch.sum(target[i] * pred[i])
            union = torch.sum(target[i]) + torch.sum(pred[i]) - inter
            loss = loss + (1 - inter / union)
        return loss / pred.shape[0]

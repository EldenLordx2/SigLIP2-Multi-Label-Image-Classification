import torch
import torch.nn as nn


class MultiLabelAsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, disable_focal_loss_grad=True):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_focal_loss_grad = disable_focal_loss_grad

    def forward(self, logits, targets):
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_focal_loss_grad:
                with torch.no_grad():
                    pt = xs_pos * targets + xs_neg * (1 - targets)
                    gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
                    w = torch.pow(1 - pt, gamma)
            else:
                pt = xs_pos * targets + xs_neg * (1 - targets)
                gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
                w = torch.pow(1 - pt, gamma)
            loss *= w

        return -loss

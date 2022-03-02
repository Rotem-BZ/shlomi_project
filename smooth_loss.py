import torch
from torch import nn

class SmoothLoss(nn.Module):
    def __init__(self, lamb=0.15, tau=4):
        super(SmoothLoss, self).__init__()
        self.lamb = lamb
        self.tau = tau
        self.soft = nn.LogSoftmax(dim=1)
        self.cls_loss = nn.NLLLoss(ignore_index=-100)

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        logits = self.soft(x)
        cls_loss = self.cls_loss(logits, target)
        mse_loss = 0
        for pred1, pred2 in zip(logits[:, -1], logits[:, 1:]):
            pred1.detach()
            deltas = torch.clip((pred1 - pred2)**2, max=self.tau**2)
            mse_loss += torch.sum(deltas)
        return cls_loss + self.lamb * mse_loss / torch.prod(torch.tensor(x.shape))
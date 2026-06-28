"""
Функции потерь для сегментации
"""
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dice_loss = 0
        for c in range(self.num_classes):
            prob_c = probs[:, c, :, :]
            target_c = targets[:, c, :, :]
            
            intersection = (prob_c * target_c).sum()
            union = prob_c.sum() + target_c.sum()
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss += 1 - dice
        return dice_loss / self.num_classes
"""
Trainer для обучения модели сегментации
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Tuple
from .losses import DiceLoss


class Trainer:
    def __init__(self, model: nn.Module, device: torch.device, class_names: List[str], 
                 learning_rate: float = 5e-5, pos_weight: float = 1.0):
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=1e-4)
        
        self.dice_loss = DiceLoss(num_classes=len(class_names))
        
        self.registered_pos_weight = torch.tensor([pos_weight]).to(device)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.registered_pos_weight)
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc="Train", leave=False)
        for batch in pbar:
            images = batch['inputs'].to(self.device)
            masks = batch['masks'].to(self.device)
            
            logits = self.model(images)
            
            loss_dice = self.dice_loss(logits, masks)
            loss_bce = self.bce_loss(logits, masks)
            loss = loss_dice + loss_bce
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_dice = 0.0
        total_iou = 0.0
        count = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Val  ", leave=False)
            for batch in pbar:
                images = batch['inputs'].to(self.device)
                masks = batch['masks'].to(self.device)
                
                logits = self.model(images)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                intersection = (preds * masks).sum(dim=(1, 2, 3))
                pred_sum = preds.sum(dim=(1, 2, 3))
                mask_sum = masks.sum(dim=(1, 2, 3))
                
                dice = (2.0 * intersection + 1e-6) / (pred_sum + mask_sum + 1e-6)
                iou = (intersection + 1e-6) / (pred_sum + mask_sum - intersection + 1e-6)
                
                total_dice += dice.mean().item()
                total_iou += iou.mean().item()
                count += 1
        
        if count == 0:
            return 0.0, 0.0
        return total_dice / count, total_iou / count
"""
Работа с данными
"""
import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Dict, Any
import torch
from config import TARGET_SIZE, CLASS_MAPPING


def create_mask(row: pd.Series, classes: List[str], target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Создает маску (H, W) для одного класса или (C, H, W) для списка классов. В коде используется (C, H, W)."""
    height = int(row["height"])
    width = int(row["width"])
    findings = row["findings"]
    num_classes = len(classes)
    
    label_to_class = {cls: i for i, cls in enumerate(classes)}
    mask = np.zeros((num_classes, height, width), dtype=np.uint8)
    
    if isinstance(findings, list):
        for finding in findings:
            label = finding.get("label")
            points_str = finding.get("points")
            
            if label not in CLASS_MAPPING:
                continue
                
            target_label = CLASS_MAPPING[label]
            
            if target_label in label_to_class and points_str:
                class_idx = label_to_class[target_label]
                try:
                    pts = [list(map(float, p.split(","))) for p in points_str.split(";") if "," in p]
                    if len(pts) < 3:
                        continue
                    points_np = np.array(pts, dtype=np.int32)
                    cv2.fillPoly(mask[class_idx], [points_np], color=1)
                except Exception:
                    continue

    if target_size:
        t_h, t_w = target_size
        if height != t_h or width != t_w:
            resized_mask = np.zeros((num_classes, t_h, t_w), dtype=np.uint8)
            for ch in range(num_classes):
                resized_mask[ch] = cv2.resize(mask[ch], (t_w, t_h), interpolation=cv2.INTER_NEAREST)
            mask = resized_mask
            
    return mask.astype(np.float32)


def get_train_augmentations(num_classes: int) -> A.Compose:
    """Возвращает композицию аугментаций для обучения."""
    targets_dict = {f"mask_ch{i}": "mask" for i in range(num_classes)}
    return A.Compose([
        A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_REPLICATE),
        A.Affine(translate_percent=(-0.1, 0.1), p=0.5, border_mode=cv2.BORDER_REPLICATE),
        A.HorizontalFlip(p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ], additional_targets=targets_dict, is_check_shapes=False)


class MyDataset(Dataset):
    """Dataset для сегментации с поддержкой инверсии и аугментаций."""
    def __init__(
        self,
        df: pd.DataFrame,
        classes: List[str],
        transform: Optional[A.Compose] = None,
        target_size: tuple = TARGET_SIZE,
        use_inversion: bool = True,
        is_train: bool = True
    ):
        self.df = df.reset_index(drop=True)
        self.classes = classes
        self.transform = transform
        self.target_size = target_size
        self.num_classes = len(classes)
        self.use_inversion = use_inversion and is_train
        self.is_train = is_train
        
        if self.use_inversion:
            self.indices = []
            for idx in range(len(df)):
                self.indices.append((idx, False))
                self.indices.append((idx, True))
        else:
            self.indices = [(idx, False) for idx in range(len(df))]
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def _load_image(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        if img is None:
            h, w = self.target_size
            return np.zeros((h, w, 3), dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img
    
    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        if img.size == 0:
            return img
        img_uint8 = (img * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        return result
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        orig_idx, is_inverted = self.indices[index]
        row = self.df.iloc[orig_idx]
        img_path = row["final_file_path"]
        
        img = self._load_image(img_path)
        
        if is_inverted:
            img = 1.0 - img
        
        img = self._apply_clahe(img)
        
        mask = create_mask(row, self.classes, target_size=self.target_size)
        
        h, w = img.shape[:2]
        t_h, t_w = self.target_size
        if h != t_h or w != t_w:
            img = cv2.resize(img, (t_w, t_h), interpolation=cv2.INTER_LINEAR)
        
        if self.is_train and self.transform:
            aug_input = {"image": img}
            for c in range(self.num_classes):
                aug_input[f"mask_ch{c}"] = mask[c]
            
            augmented = self.transform(**aug_input)
            img = augmented["image"]
            mask = np.stack([augmented[f"mask_ch{c}"] for c in range(self.num_classes)], axis=0)
        
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).float()
        
        return {
            "inputs": img_tensor,
            "masks": mask_tensor,
            "image_path": img_path,
            "is_inverted": is_inverted
        }
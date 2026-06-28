"""
Метрики для оценки качества сегментации
"""
import random
import numpy as np
import cv2
import pandas as pd
from torch.utils.data import Sampler
from typing import Tuple, List


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def match_objects(pred_mask: np.ndarray, gt_mask: np.ndarray, 
                  iou_thresh: float = 0.5) -> Tuple[int, int, int, List[float]]:
    def get_components(mask):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        components = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 20:
                components.append(labels == i)
        return components

    pred_comps = get_components(pred_mask)
    gt_comps = get_components(gt_mask)
    
    n_pred = len(pred_comps)
    n_gt = len(gt_comps)
    
    if n_pred == 0 and n_gt == 0:
        return 0, 0, 0, []
    if n_pred == 0 or n_gt == 0:
        return 0, n_pred, n_gt, []
        
    iou_matrix = np.zeros((n_gt, n_pred))
    for i, gt_c in enumerate(gt_comps):
        for j, pred_c in enumerate(pred_comps):
            iou_matrix[i, j] = calculate_iou(gt_c, pred_c)
            
    pairs = []
    for i in range(n_gt):
        for j in range(n_pred):
            if iou_matrix[i, j] >= iou_thresh:
                pairs.append((iou_matrix[i, j], i, j))
    
    pairs.sort(key=lambda x: x[0], reverse=True)
    
    matched_gt = set()
    matched_pred = set()
    matched_ious = []
    
    for iou_val, gt_idx, pred_idx in pairs:
        if gt_idx not in matched_gt and pred_idx not in matched_pred:
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
            matched_ious.append(iou_val)
            
    return len(matched_ious), n_pred, n_gt, matched_ious


class BatchSampler(Sampler):
    def __init__(self, df: pd.DataFrame, class_name: str, batch_size: int,
                 small_neg_ratio: float = 0.1,
                 medium_neg_ratio: float = 0.3,
                 large_neg_ratio: float = 0.5):
        self.df = df.reset_index(drop=True)
        self.class_name = class_name
        self.batch_size = batch_size
        
        size_map = {
            'Патологическая тень_очаг': 'medium',
            'Плевральный выпот': 'large',
            'Просветление': 'medium',
            'Прочая ненорма': 'medium',
        }
        self.class_size = size_map.get(class_name, 'medium')
        
        neg_ratio_map = {'small': small_neg_ratio, 'medium': medium_neg_ratio, 'large': large_neg_ratio}
        self.neg_ratio = neg_ratio_map[self.class_size]
        
        self.pos_indices_original = self.df[self.df[class_name] == 1].index.tolist()
        self.neg_indices_original = self.df[self.df[class_name] == 0].index.tolist()
        
        self.pos_indices = []
        for idx in self.pos_indices_original:
            self.pos_indices.extend([idx * 2, idx * 2 + 1])
        
        self.neg_indices = []
        for idx in self.neg_indices_original:
            self.neg_indices.extend([idx * 2, idx * 2 + 1])

    def __iter__(self):
        random.shuffle(self.pos_indices)
        batches = []
        
        for i in range(0, len(self.pos_indices), self.batch_size):
            batch_pos = self.pos_indices[i:i + self.batch_size]
            if len(batch_pos) == 0:
                break
            
            p_count = len(batch_pos)
            if self.neg_ratio >= 1.0:
                n_count = p_count
            else:
                n_count = int(p_count * self.neg_ratio / (1.0 - self.neg_ratio))
            
            max_n = self.batch_size - p_count
            n_count = min(n_count, max_n, len(self.neg_indices))
            
            batch = batch_pos[:]
            if n_count > 0 and len(self.neg_indices) > 0:
                selected_neg = random.sample(self.neg_indices, n_count)
                batch.extend(selected_neg)
            
            while len(batch) < self.batch_size and len(self.neg_indices) > 0:
                nxt = random.choice(self.neg_indices)
                if nxt not in batch:
                    batch.append(nxt)
                elif len(self.neg_indices) < 10:
                    batch.append(nxt)
                    break
            
            batches.append(batch)
        
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.pos_indices) + len(self.neg_indices)) // self.batch_size
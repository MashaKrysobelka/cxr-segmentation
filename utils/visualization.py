"""
Функции для визуализации масок и контуров
"""
import os
import cv2
import numpy as np
import torch
from typing import Dict, Tuple, List
from config import COLORS, CLASS_MAPPING, DEVICE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def load_and_preprocess_image(img_path: str) -> np.ndarray:
    """Загрузка + CLAHE"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def draw_all_gt_contours(image: np.ndarray, findings: List, 
                         colors_map: Dict[str, Tuple[int, int, int]] = COLORS,
                         black_thickness: int = 10,
                         color_thickness: int = 6) -> np.ndarray:
    """Рисует контуры ВСЕХ патологий из разметки"""
    img_out = image.copy()
    h, w = img_out.shape[:2]
    class_masks = {cls: np.zeros((h, w), dtype=np.uint8) for cls in colors_map.keys()}
    
    if not findings:
        return img_out

    for finding in findings:
        label = finding.get('label')
        points_str = finding.get('points')
        
        if label not in CLASS_MAPPING or not points_str:
            continue
            
        target_class = CLASS_MAPPING[label]
        if target_class not in colors_map:
            continue
            
        pts = []
        for p in points_str.split(';'):
            if ',' in p:
                x, y = map(float, p.split(','))
                pts.append([int(x), int(y)])
        
        if len(pts) >= 3:
            temp_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(temp_mask, [np.array(pts, dtype=np.int32)], 1)
            class_masks[target_class] = cv2.bitwise_or(class_masks[target_class], temp_mask)

    for cls_name, mask in class_masks.items():
        if mask.sum() == 0:
            continue
        color = colors_map[cls_name]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                cv2.drawContours(img_out, [cnt], -1, (0, 0, 0), black_thickness)
                cv2.drawContours(img_out, [cnt], -1, color, color_thickness)
                
    return img_out


def draw_pred_overlay(image: np.ndarray, pred_prob: np.ndarray, 
                      target_class_name: str, 
                      colors_map: Dict[str, Tuple[int, int, int]] = COLORS,
                      threshold: float = 0.5) -> np.ndarray:
    """Рисует полупрозрачную заливку ТОЛЬКО для целевого класса"""
    img_out = image.copy()
    pred_binary = (pred_prob > threshold).astype(np.uint8)
    if pred_binary.sum() == 0:
        return img_out
        
    color = colors_map.get(target_class_name, (0, 255, 0))
    overlay = np.zeros_like(img_out, dtype=np.float32)
    overlay[pred_binary > 0] = color
    
    contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            cv2.drawContours(img_out, [cnt], -1, (0, 0, 0), 2)
    
    alpha = 0.4
    img_out = cv2.addWeighted(img_out, 1.0, (overlay * alpha).astype(np.uint8), 1.0, 0)
    
    return img_out


def create_legend(output_path: str = None, colors_map: Dict[str, Tuple[int, int, int]] = COLORS) -> np.ndarray:
    """
    Создает PNG-изображение с легендой по цветам классов.
    """
    class_names = list(colors_map.keys())
    colors_hex = []
    for cls in class_names:
        rgb = colors_map[cls]
        hex_c = '#%02x%02x%02x' % rgb
        colors_hex.append(hex_c)

    fig, ax = plt.subplots(figsize=(14, 2.5))
    ax.axis('off')
    
    patches = [mpatches.Patch(color=c, label=l, edgecolor='black', linewidth=2) 
               for c, l in zip(colors_hex, class_names)]
    
    ax.legend(handles=patches, loc='center', ncol=len(class_names), 
              prop={'size': 12, 'weight': 'bold'}, 
              handlelength=2.5, handleheight=2.5, frameon=True, shadow=True)
    
    plt.suptitle('Легенда классов', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Легенда сохранена: {output_path}")
        return None
    else:
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        legend_img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
        legend_img = cv2.cvtColor(legend_img, cv2.COLOR_BGR2RGB)
        plt.close()
        return legend_img

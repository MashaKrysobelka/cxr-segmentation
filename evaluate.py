"""
Скрипт для вычисления метрик на тестовых выборках
"""
import os
import sys
import json
import pathlib
import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score
from typing import List, Dict, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEVICE, THRESHOLD, IOU_THRESHOLD, MODEL_DIR, 
    OUTPUT_DIR_OLD, OUTPUT_DIR_NEW, MERGED_CLASSES,
    CLASS_MAPPING, TEST_OLD_JSON_PATH, TEST_NEW_JSON_PATH
)
from utils import DINOSegmentationUNet, match_objects, calculate_iou


def create_mask_merged(row: pd.Series, classes: List[str], height: int, width: int) -> np.ndarray:
    num_classes = len(classes)
    mask = np.zeros((num_classes, height, width), dtype=np.uint8)
    label_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    findings = row.get('findings', [])
    if not isinstance(findings, list):
        return mask
        
    for finding in findings:
        orig_label = finding.get('label')
        points_str = finding.get('points')
        
        if orig_label not in CLASS_MAPPING or not points_str:
            continue
            
        target_label = CLASS_MAPPING[orig_label]
        
        if target_label not in label_to_idx:
            continue
            
        class_idx = label_to_idx[target_label]
        try:
            pts = [list(map(float, p.split(','))) for p in points_str.split(';') if ',' in p]
            if len(pts) < 3:
                continue
            points_np = np.array(pts, dtype=np.int32)
            cv2.fillPoly(mask[class_idx], [points_np], color=1)
        except Exception:
            continue
            
    return mask


def evaluate_dataset_merged(df: pd.DataFrame, models: Dict, output_dir: str, dataset_name: str):
    results_per_class = {cls: [] for cls in MERGED_CLASSES}
    image_level_preds = {cls: [] for cls in MERGED_CLASSES}
    image_level_gts = {cls: [] for cls in MERGED_CLASSES}
    
    print(f"\n--- Тестирование на {dataset_name} (Укрупненные классы) ---")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Inference {dataset_name}"):
        img_path = row['final_file_path']
        h, w = int(row['height']), int(row['width'])
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_input = cv2.resize(img, (518, 518), interpolation=cv2.INTER_LINEAR)
        img_input = img_input.astype(np.float32) / 255.0
        
        lab = cv2.cvtColor((img_input * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img_input = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        
        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        
        pred_masks_full = {}
        
        with torch.no_grad():
            for cls_name, model in models.items():
                logits = model(img_tensor)
                prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
                pred_bin = (prob > THRESHOLD).astype(np.uint8)
                pred_full = cv2.resize(pred_bin, (w, h), interpolation=cv2.INTER_NEAREST)
                pred_masks_full[cls_name] = pred_full
        
        gt_mask_full = create_mask_merged(row, MERGED_CLASSES, h, w)
        
        for c_idx, cls_name in enumerate(MERGED_CLASSES):
            pred_m = pred_masks_full[cls_name]
            gt_m = gt_mask_full[c_idx]
            
            n_matched, n_pred, n_gt, ious_matched = match_objects(pred_m, gt_m, iou_thresh=IOU_THRESHOLD)
            
            precision_obj = n_matched / n_pred if n_pred > 0 else (1.0 if n_gt == 0 else 0.0)
            recall_obj = n_matched / n_gt if n_gt > 0 else (1.0 if n_pred == 0 else 0.0)
            
            f1_obj = 0.0
            if precision_obj + recall_obj > 0:
                f1_obj = 2 * precision_obj * recall_obj / (precision_obj + recall_obj)
                
            mean_iou_matched = np.mean(ious_matched) if ious_matched else 0.0
            role = "positive" if n_gt > 0 else "negative"
            
            results_per_class[cls_name].append({
                "file_name": img_path,
                "role": role,
                "n_objs_ann": n_gt,
                "n_objs_pred": n_pred,
                "n_matched": n_matched,
                "precision": precision_obj,
                "recall": recall_obj,
                "f1": f1_obj,
                "mean_iou_matched": mean_iou_matched,
                "pixel_iou": calculate_iou(pred_m, gt_m)
            })
            
            gt_has_class = 1 if n_gt > 0 else 0
            pred_has_class = 1 if n_pred > 0 else 0
            
            image_level_gts[cls_name].append(gt_has_class)
            image_level_preds[cls_name].append(pred_has_class)

    print(f"\nАгрегация метрик для {dataset_name}...")

    csv_rows = []
    
    for cls_name in MERGED_CLASSES:
        data = results_per_class[cls_name]
        df_cls = pd.DataFrame(data)
        
        avg_precision = df_cls['precision'].mean()
        avg_recall = df_cls['recall'].mean()
        avg_f1 = df_cls['f1'].mean()
        avg_iou_matched = df_cls[df_cls['n_matched'] > 0]['mean_iou_matched'].mean() if (df_cls['n_matched'] > 0).any() else 0.0
        avg_pixel_iou = df_cls['pixel_iou'].mean()
        
        # Image-level метрики
        y_true = np.array(image_level_gts[cls_name])
        y_pred = np.array(image_level_preds[cls_name])
        
        img_precision = precision_score(y_true, y_pred, zero_division=0)
        img_recall = recall_score(y_true, y_pred, zero_division=0)
        img_f1 = f1_score(y_true, y_pred, zero_division=0)
        kappa = cohen_kappa_score(y_true, y_pred)
        
        csv_rows.append({
            "Class": cls_name,
            "Obj_Precision": avg_precision,
            "Obj_Recall": avg_recall,
            "Obj_F1": avg_f1,
            "Matched_IoU": avg_iou_matched,
            "Pixel_IoU": avg_pixel_iou,
            "Img_Precision": img_precision,
            "Img_Recall": img_recall,
            "Img_F1": img_f1,
            "Kappa": kappa
        })
        
        print(f"  {cls_name}: Obj_F1={avg_f1:.4f}, Img_F1={img_f1:.4f}")
    
    # === СОХРАНЯЕМ CSV ===
    df_summary = pd.DataFrame(csv_rows)
    csv_path = os.path.join(output_dir, "summary_metrics.csv")
    df_summary.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"CSV сохранён: {csv_path}")
    
    # === СОХРАНЯЕМ JSON ===
    json_output_path = os.path.join(output_dir, "per_class_results.json")
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(results_per_class, f, ensure_ascii=False, indent=2)
    
    print(f"Результаты сохранены в {output_dir}")


def main():
    print("Загрузка разметки...")
    with open(TEST_OLD_JSON_PATH, 'r', encoding='utf-8') as f:
        test_old_data = json.load(f)
    with open(TEST_NEW_JSON_PATH, 'r', encoding='utf-8') as f:
        test_new_data = json.load(f)

    test_old_df = pd.DataFrame(list(test_old_data.values()))
    test_new_df = pd.DataFrame(list(test_new_data.values()))

    for df in [test_old_df, test_new_df]:
        if 'final_file_path' in df.columns:
            df['file_exists'] = df['final_file_path'].apply(lambda x: pathlib.Path(x).exists())
            df = df[df['file_exists']].reset_index(drop=True)

    common_paths = set(test_old_df['final_file_path']).intersection(set(test_new_df['final_file_path']))
    test_old_df = test_old_df[test_old_df['final_file_path'].isin(common_paths)].reset_index(drop=True)
    test_new_df = test_new_df[test_new_df['final_file_path'].isin(common_paths)].reset_index(drop=True)

    print(f"Количество общих снимков для теста: {len(test_old_df)}")

    models = {}
    print("Загрузка моделей (Укрупненные классы)...")
    for cls_name in MERGED_CLASSES:
        safe_name = cls_name.replace(' ', '_')
        path = os.path.join(MODEL_DIR, f"{safe_name}.pth")
        
        if not os.path.exists(path):
            print(f"Модель не найдена: {path}")
            continue
            
        model = DINOSegmentationUNet(num_classes=1, device=DEVICE).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        models[cls_name] = model
        print(f"Загружена модель для: {cls_name}")

    if len(models) != len(MERGED_CLASSES):
        print("ВНИМАНИЕ: Загружены не все модели!")

    evaluate_dataset_merged(test_old_df, models, OUTPUT_DIR_OLD, "TEST_OLD")
    evaluate_dataset_merged(test_new_df, models, OUTPUT_DIR_NEW, "TEST_NEW")


if __name__ == "__main__":
    main()
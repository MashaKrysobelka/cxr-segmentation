"""
Основной скрипт для обучения моделей сегментации
"""
import os
import sys
import gc
import pickle
import json
import pathlib
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    SEED, DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    MODEL_DIR, MERGED_CLASSES, CLASS_WEIGHTS, CLASS_SIZE_MAP,
    ORIGINAL_CLASSES, CLASS_MAPPING,
    TRAIN_JSON_PATH, TEST_OLD_JSON_PATH, TEST_NEW_JSON_PATH
)
from utils import (
    MyDataset, get_train_augmentations, DINOSegmentationUNet, Trainer, BatchSampler
)


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_and_prepare_data():
    with open(TRAIN_JSON_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(TEST_OLD_JSON_PATH, 'r', encoding='utf-8') as f:
        test_old_data = json.load(f)
    with open(TEST_NEW_JSON_PATH, 'r', encoding='utf-8') as f:
        test_new_data = json.load(f)

    train_data = list(train_data.values())
    test_old_data = list(test_old_data.values())
    test_new_data = list(test_new_data.values())

    train_df = pd.DataFrame(train_data)
    test_old_df = pd.DataFrame(test_old_data)
    test_new_df = pd.DataFrame(test_new_data)

    for df in [train_df, test_old_df, test_new_df]:
        for lbl in ORIGINAL_CLASSES:
            df[lbl] = df['findings'].apply(
                lambda x: 1 if any(f.get('label') == lbl for f in x) else 0 # f - словарь (элемент списка x)
            )
        if 'final_file_path' in df.columns:
            df['file_exists'] = df['final_file_path'].apply(lambda x: pathlib.Path(x).exists())
            df = df[df['file_exists']].reset_index(drop=True)
            
    return train_df, test_old_df, test_new_df


def convert_to_merged(df: pd.DataFrame) -> pd.DataFrame:
    result_df = df.copy()
    for merged_cls in MERGED_CLASSES:
        result_df[merged_cls] = 0
    
    for idx, row in df.iterrows():
        findings = row['findings']
        if not isinstance(findings, list):
            continue
        for finding in findings: # findings - список словарей
            label = finding.get('label')
            if label in CLASS_MAPPING:
                merged_cls = CLASS_MAPPING[label]
                result_df.at[idx, merged_cls] = 1
                
    return result_df


def main():
    print("="*60)
    print("ОБУЧЕНИЕ 4 БИНАРНЫХ МОДЕЛЕЙ (ONE-VS-REST)")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")

    print("\nЗагрузка данных...")
    train_df, test_old_df, test_new_df = load_and_prepare_data()
    
    train_merged = convert_to_merged(train_df)
    test_old_merged = convert_to_merged(test_old_df)
    test_new_merged = convert_to_merged(test_new_df)
    
    train_merged['has_any'] = train_merged[MERGED_CLASSES].any(axis=1).astype(int)
    
    if train_merged['has_any'].nunique() > 1:
        stratify_col = train_merged['has_any']
    else:
        stratify_col = None

    train_data_merged, val_data_merged = train_test_split(
        train_merged,
        test_size=0.15,
        random_state=SEED,
        stratify=stratify_col
    )
    
    train_data_merged = train_data_merged.drop('has_any', axis=1).reset_index(drop=True)
    val_data_merged = val_data_merged.drop('has_any', axis=1).reset_index(drop=True)
    
    print(f"Train merged: {len(train_data_merged)}, Val merged: {len(val_data_merged)}")

    results_4_classes = {}
    os.makedirs(MODEL_DIR, exist_ok=True)

    for class_idx, class_name in enumerate(MERGED_CLASSES):
        print(f"\n>>> Обучение модели для класса: {class_name} <<<")
        
        weights_config = CLASS_WEIGHTS.get(class_name, {0: 1.0, 1: 1.0})
        w0, w1 = weights_config[0], weights_config[1]
        calculated_pos_weight = w1 / w0

        train_ds = MyDataset(
            df=train_data_merged,
            classes=[class_name],
            transform=get_train_augmentations(1),
            target_size=(518, 518),
            use_inversion=True,
            is_train=True
        )
        
        val_ds = MyDataset(
            df=val_data_merged,
            classes=[class_name],
            transform=None,
            target_size=(518, 518),
            use_inversion=False,
            is_train=False
        )
        
        sampler = BatchSampler(
            df=train_data_merged,
            class_name=class_name,
            batch_size=BATCH_SIZE,
            small_neg_ratio=0.1,
            medium_neg_ratio=0.3,
            large_neg_ratio=0.5
        )
        
        train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        
        model = DINOSegmentationUNet(num_classes=1, device=DEVICE).to(DEVICE)
        trainer = Trainer(model, DEVICE, class_names=[class_name], 
                         learning_rate=LEARNING_RATE, pos_weight=calculated_pos_weight)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            trainer.optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7
        )
        
        best_dice = 0.0
        best_model_state = None
        
        print(f"  Старт обучения ({EPOCHS} эпох)...")
        
        for epoch in range(EPOCHS):
            train_loss = trainer.train_epoch(train_loader)
            val_dice, val_iou = trainer.validate(val_loader)
            
            scheduler.step(val_dice)
            
            is_best = False
            if val_dice > best_dice:
                best_dice = val_dice
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                is_best = True
            
            if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
                marker = " ദ്ദി" if is_best else ""
                print(f"Epoch {epoch+1}: loss={train_loss:.4f}, val_dice={val_dice:.4f}{marker}")
        
        safe_name = class_name.replace(' ', '_')
        save_path = os.path.join(MODEL_DIR, f"{safe_name}.pth")
        torch.save(best_model_state, save_path)
        print(f"  Модель сохранена: {save_path} (Best Dice: {best_dice:.4f})")
        
        results_4_classes[class_name] = {
            'best_dice': best_dice,
            'final_val_dice': val_dice,
            'final_val_iou': val_iou,
            'used_pos_weight': calculated_pos_weight
        }
        
        del model, trainer, train_ds, val_ds, sampler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    with open('results_4_classes.pkl', 'wb') as f:
        pickle.dump(results_4_classes, f)
    
    print("\n" + "="*60)
    print("СВОДНЫЙ РЕЗУЛЬТАТ")
    print("="*60)
    for name, res in results_4_classes.items():
        print(f"{name}: Best Dice = {res['best_dice']:.4f}, Weight = {res['used_pos_weight']:.2f}")
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")


if __name__ == "__main__":
    set_seed()
    main()
"""
Скрипт для визуализации одного изображения с опциональной разметкой
"""
import os
import sys
import torch
import argparse
import json
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import DEVICE, MODEL_DIR, COLORS
from utils import DINOSegmentationUNet, load_and_preprocess_image, draw_all_gt_contours, draw_pred_overlay, create_legend


def load_findings_from_json(json_path: str, image_path: str) -> list:
    """
    Загружает разметку для конкретного изображения из JSON-файла.
    Если image_path не найден, возвращает пустой список.
    """
    if not os.path.exists(json_path):
        print(f"JSON-файл не найден: {json_path}")
        return []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data.values():
        if item.get('final_file_path') == image_path:
            return item.get('findings', [])
    
    print(f"Изображение не найдено в JSON: {image_path}")
    return []


def main():
    parser = argparse.ArgumentParser(description='Визуализация сегментации для одного снимка')
    parser.add_argument('--image_path', type=str, required=True, help='Путь к изображению')
    parser.add_argument('--class_name', type=str, required=True, 
                        choices=['Патологическая тень_очаг', 'Плевральный выпот', 'Просветление', 'Прочая ненорма'],
                        help='Целевой класс для визуализации')
    parser.add_argument('--output_path', type=str, default=None, help='Путь для сохранения результата')
    parser.add_argument('--threshold', type=float, default=0.5, help='Порог бинаризации')
    parser.add_argument('--has_gt', action='store_true', 
                        help='Есть ли разметка врачей для этого снимка')
    parser.add_argument('--gt_json', type=str, default=None,
                        help='Путь к JSON-файлу с разметкой (например, test_set_old.json)')
    
    args = parser.parse_args()
    
    # 1. Проверка изображения
    if not os.path.exists(args.image_path):
        print(f"Файл с изображением не найден: {args.image_path}")
        return
    
    # 2. Проверка модели
    safe_name = args.class_name.replace(' ', '_')
    model_path = os.path.join(MODEL_DIR, f"{safe_name}.pth")
    
    if not os.path.exists(model_path):
        print(f"Модель не найдена: {model_path}")
        return
    
    # 3. Загрузка модели
    print(f"Загрузка модели: {model_path}")
    model = DINOSegmentationUNet(num_classes=1, device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 4. Загрузка разметки (если указана)
    findings = []
    if args.has_gt:
        if args.gt_json is None:
            print("Указан --has_gt, но не указан --gt_json. Разметка не будет отображаться.")
        else:
            findings = load_findings_from_json(args.gt_json, args.image_path)
            if findings:
                print(f"Загружено {len(findings)} объектов разметки")
            else:
                print("Разметка не найдена для этого изображения")
    
    # 5. Загрузка и предобработка изображения
    img = load_and_preprocess_image(args.image_path)
    if img is None:
        print(f"Не удалось загрузить изображение: {args.image_path}")
        return
    
    h, w = img.shape[:2]
    
    # 6. Инференс модели
    img_input = cv2.resize(img, (518, 518), interpolation=cv2.INTER_LINEAR)
    img_input = img_input.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    
    with torch.no_grad():
        logits = model(img_tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    
    pred_resized = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 7. Построение визуализации
    if args.has_gt and findings:
        # 3 изображения: оригинал | разметка (GT) (все классы) | предсказание (только для выбранного класса)
        img_original = img.copy()
        img_gt = draw_all_gt_contours(img.copy(), findings, COLORS)
        img_pred = draw_pred_overlay(img.copy(), pred_resized, args.class_name, COLORS, args.threshold)
        combined = np.hstack([img_original, img_gt, img_pred])
    else:
        # 2 изображения: оригинал | предсказание (только для выбранного класса)
        img_original = img.copy()
        img_pred = draw_pred_overlay(img.copy(), pred_resized, args.class_name, COLORS, args.threshold)
        combined = np.hstack([img_original, img_pred])
    
    # 8. Сохранение визуализации
    vis_dir = 'visualizations'
    os.makedirs(vis_dir, exist_ok=True)
    
    if args.output_path is None:
        base_name = os.path.splitext(os.path.basename(args.image_path))[0]
        suffix = "with_gt" if (args.has_gt and findings) else "no_gt"
        args.output_path = os.path.join(vis_dir, f"vis_{base_name}_{safe_name}_{suffix}.png")
    else:
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(args.output_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"Визуализация сохранена: {args.output_path}")

    legend_path = os.path.join(os.path.dirname(args.output_path), "legend.png")
    create_legend(output_path=legend_path)


if __name__ == "__main__":
    main()
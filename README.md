# Chest X-Ray Segmentation with RAD-DINO-MAIRA-2

Segmentation of pathological findings on chest X-rays using RAD-DINO-MAIRA-2 backbone and U-Net-like decoder.

---

## Features

- Binary segmentation for 4 merged classes (one-vs-rest approach)
- U-Net-like decoder with skip connections
- Support for image inversion and CLAHE preprocessing
- Batch sampling for class imbalance
- Dice + BCE loss combination

---

## Architecture

```
Input (518×518)
     │
     ▼
RAD-DINO-MAIRA-2 (frozen backbone)
     │
     ├──► Global features → Classification
     │
     └──► Patch features (37×37×768)
               │
               ▼
         Projection layers (768 → 512)
               │
               ▼
         U-Net Decoder (4 blocks)
         - Upsample ×2
         - Skip connections
               │
               ▼
         Segmentation head (64 → 32 → 1)
               │
               ▼
         Mask (518×518)
```

---

## Installation

```bash
git clone https://github.com/yourusername/cxr-segmentation.git
cd cxr-segmentation
pip install -r requirements.txt
```

---

## Data Format

The script expects JSON files with the following structure:

```json
{
  "file_path": "path/to/image.png",
  "task_id": "32",
  "width": 3072,
  "height": 3008,
  "findings": [
    {
      "label": "Прочая ненорма",
      "points": "1904.97,1065.49;1907.14,1083.91;..."
    },
    {
      "label": "Очаги в легких",
      "points": "351.35,245.10;..."
    }
  ],
  "final_file_path": "/absolute/path/to/image.png"
}
```

**Required fields:**
- `final_file_path` — absolute path to the image file
- `height` / `width` — image dimensions (for mask creation)
- `findings` — list of annotations with:
  - `label` — class name (must match `ORIGINAL_CLASSES` or be mappable via `CLASS_MAPPING`)
  - `points` — polygon coordinates as string: `"x1,y1;x2,y2;..."`

**Place your JSON files in the `data/` folder:**
```
data/
├── train_set.json
├── test_set_old.json
└── test_set_new.json
```

You can change the paths in `config.py` or set the `DATA_DIR` environment variable:

```bash
export DATA_DIR="/path/to/your/data"
python train.py
```

---

## Usage

### Training

```bash
python train.py
```

### Evaluation

```bash
python evaluate.py
```

### Visualization

To visualize model predictions with optional expert annotations:

**Arguments:**
- `--image_path`: Path to the X-ray image
- `--class_name`: Target class (`Патологическая тень_очаг`, `Плевральный выпот`, `Просветление`, `Прочая ненорма`)
- `--has_gt`: Include this flag if ground truth is available
- `--gt_json`: Path to JSON file with annotations (required if `--has_gt` is set). Supports `test_set_old.json` or `test_set_new.json`
- `--threshold`: Probability threshold (default: `0.5`)
- `--output_path`: Custom output path (optional). If not specified, saves to `visualizations/` folder

```bash
# With ground truth (3 images: original | GT | prediction)
python visualize.py \
    --image_path /path/to/image.png \
    --class_name "Патологическая тень_очаг" \
    --has_gt \
    --gt_json /path/to/test_set_old.json

# Without ground truth (2 images: original | prediction)
python visualize.py \
    --image_path /path/to/image.png \
    --class_name "Плевральный выпот"
```
---

## License

MIT
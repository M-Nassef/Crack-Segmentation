# Crack-Segmentation

Instance segmentation of road cracks with **YOLO11n-seg** (Ultralytics), fine-tuned on a
small Roboflow dataset of road photos. The whole pipeline runs in one Google Colab notebook
on a free T4 GPU.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/M-Nassef/Crack-Segmentation/blob/main/crack_segmentation_yolo11.ipynb)

## What the notebook does

1. Installs `ultralytics` and checks that a CUDA GPU is available.
2. Downloads the dataset from Roboflow (`road_crack_detection`, version 4, YOLOv11 format).
3. Loads the COCO-pretrained `yolo11n-seg.pt` model and freezes everything except the last layers.
4. Fine-tunes for crack segmentation.
5. Validates the best checkpoint and runs predictions on test images.

## Dataset

| Split | Images |
|---|---|
| train | 309 |
| valid | 20 |

There is one class, `Cracks`. The validation set contains 153 crack instances.

## Training setup

| Setting | Value |
|---|---|
| Model | `yolo11n-seg` (2.8 M parameters) |
| Image size | 768 |
| Batch / epochs | 8 / up to 150, early stopping with patience 20 |
| Optimizer | AdamW, `lr0 = 5e-5` |
| Augmentation | HSV jitter, ±10° rotation, translate 0.1, scale 0.5, horizontal flip |

Training stopped early after **142 epochs**, about 35 minutes on a T4. The best epoch was 122.

## Results (validation set, best checkpoint)

|      | Precision | Recall | mAP50 | mAP50-95 |
|------|-----------|--------|-------|----------|
| Box  | 0.564 | 0.542 | 0.526 | 0.305 |
| Mask | 0.479 | 0.444 | 0.362 | 0.125 |

Inference takes about 6 ms per image at 768 px on a T4.

The validation set has only 20 images, so these numbers are a rough indication, not a
benchmark. Thin, long cracks are hard to segment precisely, which is why mask mAP50-95 is low.

## Running it

Open the notebook in Colab with the badge above and choose **Runtime → Change runtime type → T4 GPU**.
You need your own Roboflow API key to download the dataset. Put it in the dataset cell, or
better, in Colab's **Secrets** panel, and don't commit it.

## Files

| File | Content |
|---|---|
| `crack_segmentation_yolo11.ipynb` | The full training and inference notebook, with outputs |

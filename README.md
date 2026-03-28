# Facial Emotion Recognition for Autism Research

This repository contains the code for a research project investigating whether transfer learning from general facial expression datasets (FER2013, RAF-DB) can improve emotion recognition accuracy for individuals with Autism Spectrum Disorder (ASD).

The pipeline proceeds in three stages:
1. **Data preparation** – convert raw datasets into image folders and preprocess faces
2. **Baseline training** – benchmark multiple architectures on RAF-DB to select the best backbone
3. **Transfer learning** – fine-tune the selected models on the autism-specific emotion dataset

---

## Repository Structure

```
├── data_prep/
│   ├── fer2013_csv_to_images.py      # Convert FER2013 CSV → image folders
│   ├── crop_faces_haar.py            # Haar cascade face cropper 
│   └── setup_rafdb_for_poster.py     # Reformat RAF-DB for POSTER V2 (was: poster_format.py)
│
├── training/
│   ├── rafdb_model_benchmark.py      # Multi-model benchmark on RAF-DB (was: rafdb_fer_bench.py)
│   ├── effnet_autism_transfer.py     # EfficientNet-B0 transfer to autism dataset (was: autism_effnet_transfer.py)
│   └── vit_autism_transfer.py        # ViT-B/16 transfer to autism dataset (was: vit_fer_transfer.py)
│
└── README.md
```

---

## Datasets

| Dataset | Description | Source |
|---|---|---|
| **FER2013** | ~35k grayscale 48×48 face images, 7 emotions | [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) |
| **RAF-DB** | ~15k real-world face images, 7 basic emotions | [rafdb.github.io](http://www.whdeng.cn/raf/model1.html) |
| **Autism Emotion Dataset** | Emotion images from individuals with ASD | Custom / local |

---

## Pipeline

### Step 1 – Prepare FER2013

```bash
python data_prep/fer2013_csv_to_images.py
```
Converts `FER2013/fer2013.csv` into an `ImageFolder`-compatible directory at `FER2013_images/`.

### Step 2 – Crop faces (Autism dataset)

```bash
python data_prep/crop_faces_haar.py
```
Runs OpenCV Haar cascade face detection on the autism emotion dataset. Falls back to the original image if no face is detected.

### Step 3 – (Optional) Set up RAF-DB for POSTER V2

```bash
python data_prep/setup_rafdb_for_poster.py
```
Copies RAF-DB into the directory structure expected by the POSTER V2 model.

### Step 4 – Benchmark models on RAF-DB

```bash
python training/rafdb_model_benchmark.py \
    --train_dir path/to/rafdb/train \
    --val_dir   path/to/rafdb/test \
    --models efficientnet_b0 resnet50 swin_t vit_b_16 \
    --epochs 50
```

Supported models: `resnet18`, `resnet50`, `efficientnet_b0`, `swin_t`, `vit_b_16`

### Step 5a – Fine-tune EfficientNet-B0 on autism dataset

```bash
# With ImageNet init only
python training/effnet_autism_transfer.py \
    --train_dir path/to/autism/train \
    --val_dir   path/to/autism/test

# With RAF-DB pre-trained backbone
python training/effnet_autism_transfer.py \
    --train_dir path/to/autism/train \
    --val_dir   path/to/autism/test \
    --rafdb_checkpoint efficientnet_b0_rafdb_best.pth
```

### Step 5b – Fine-tune ViT-B/16 on autism dataset

```bash
# Pre-train on FER2013
python training/vit_autism_transfer.py \
    --mode pretrain_fer \
    --train_dir path/to/fer2013_images/train \
    --val_dir   path/to/fer2013_images/test

# Fine-tune on autism dataset (with FER2013 checkpoint)
python training/vit_autism_transfer.py \
    --mode finetune_autism \
    --train_dir path/to/autism/train \
    --val_dir   path/to/autism/test \
    --init_checkpoint checkpoints_vit/vit_fer2013_pretrain_best.pth

# Compare augmentation strategies
python training/vit_autism_transfer.py \
    --mode compare_autism_aug \
    --train_dir path/to/autism/train \
    --val_dir   path/to/autism/test \
    --init_checkpoint checkpoints_vit/vit_fer2013_pretrain_best.pth
```

---

## Dependencies

```
torch>=2.0
torchvision>=0.15
opencv-python
Pillow
pandas
numpy
```

Install with:
```bash
pip install torch torchvision opencv-python Pillow pandas numpy
```

A CUDA-capable GPU is strongly recommended for the training scripts.

---

## Key Training Hyperparameters

| Parameter | EfficientNet | ViT |
|---|---|---|
| Optimizer | AdamW | AdamW |
| Learning rate | 1e-4 | 3e-5 |
| Scheduler | CosineAnnealing | CosineAnnealing |
| Batch size | 16 | 16 |
| Label smoothing | 0.05 | 0.05 |
| Image size | 224×224 | 224×224 |

---

## Outputs

Each training run saves a `*_best.pth` checkpoint selected by best validation macro-F1 (configurable via `--select_best_by`). A summary table is printed at the end of each run with accuracy, macro-F1, parameter count, model size, and inference latency.

---

## Author

Samantha Adorno

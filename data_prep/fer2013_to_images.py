"""
fer2013_csv_to_images.py
------------------------
Converts the FER2013 dataset CSV into an ImageFolder-compatible directory tree.

The FER2013 CSV (available on Kaggle) has three columns:
  - emotion  : integer label 0–6
  - pixels   : space-separated pixel values for a 48×48 grayscale image
  - Usage    : "Training", "PublicTest", or "PrivateTest"

Output layout:
  FER2013_images/
    train/
      angry/  disgust/  fear/  happy/  sad/  surprise/  neutral/
    test/
      angry/  ...

Usage:
  python fer2013_csv_to_images.py

Edit CSV_PATH and OUTPUT_DIR below to match your local paths before running.
"""

import os

import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
CSV_PATH   = "FER2013/fer2013.csv"   # Path to the downloaded FER2013 CSV
OUTPUT_DIR = "FER2013_images"        # Root output directory
# ──────────────────────────────────────────────────────────────────────────────

# Maps the integer label in the CSV to a human-readable emotion name
EMOTION_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


def save_image(pixels: str, path: Path) -> None:
    """Decode a space-separated pixel string and save it as a grayscale JPEG.

    Args:
        pixels: Space-separated integer pixel values (48*48 = 2304 values).
        path:   Destination file path (should end in .jpg).
    """
    img_array = np.fromstring(pixels, dtype=int, sep=" ")
    img_array = img_array.reshape(48, 48).astype(np.uint8)
    Image.fromarray(img_array).save(path)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")

    for idx, row in df.iterrows():
        emotion_name = EMOTION_MAP[row["emotion"]]

        # FER2013 splits: Training → train, everything else → test
        split = "train" if row["Usage"] == "Training" else "test"

        out_dir = Path(OUTPUT_DIR) / split / emotion_name
        out_dir.mkdir(parents=True, exist_ok=True)

        save_image(row["pixels"], out_dir / f"{idx}.jpg")

    print(f"Done! Dataset saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

CSV_PATH = "FER2013/fer2013.csv"
OUTPUT_DIR = "FER2013_images"

emotion_map = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

def save_image(pixels, path):
    img = np.fromstring(pixels, dtype=int, sep=' ')
    img = img.reshape(48, 48).astype(np.uint8)
    Image.fromarray(img).save(path)

def main():
    df = pd.read_csv(CSV_PATH)

    for _, row in df.iterrows():
        emotion = emotion_map[row["emotion"]]
        usage = row["Usage"]  # Training / PublicTest / PrivateTest

        if usage == "Training":
            split = "train"
        else:
            split = "test"

        out_dir = Path(OUTPUT_DIR) / split / emotion
        out_dir.mkdir(parents=True, exist_ok=True)

        img_name = f"{_}.jpg"
        save_image(row["pixels"], out_dir / img_name)

    print("Done! Dataset ready at:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
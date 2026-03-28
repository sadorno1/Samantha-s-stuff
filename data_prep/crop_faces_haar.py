"""
crop_faces_haar.py  (originally: pete.py)
------------------------------------------
Face-crops every image in the autism emotion dataset using OpenCV's Haar
cascade detector.  If no face is found, the original image is kept as-is
so that no samples are silently dropped.

Expected input layout (same as ImageFolder):
  <SRC_ROOT>/
    train/
      <class_name>/  *.jpg ...
    test/
      <class_name>/  *.jpg ...

Output layout mirrors the input under <DST_ROOT>.

Usage:
  python crop_faces_haar.py

Edit SRC_ROOT and DST_ROOT below to match your local paths.
"""

import os
import cv2
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
SRC_ROOT = r"Autism emotion recogition dataset"         # Input dataset root
DST_ROOT = r"Autism emotion recogition dataset_faces"   # Output (face-cropped)
# ──────────────────────────────────────────────────────────────────────────────

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Load the pre-trained Haar cascade for frontal face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it doesn't already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def crop_largest_face(bgr_image):
    """Detect faces in a BGR image and return the largest bounding-box crop.

    Args:
        bgr_image: OpenCV BGR image array.

    Returns:
        Cropped BGR image of the largest detected face, or None if no face
        was found.
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),   # Ignore very small detections
    )

    if len(faces) == 0:
        return None  # Caller handles the no-face case

    # Select the face with the largest area
    x, y, w, h = max(faces, key=lambda t: t[2] * t[3])
    return bgr_image[y : y + h, x : x + w]


def process_split(split: str) -> None:
    """Crop all images in one split (train or test) and write to DST_ROOT.

    Args:
        split: Subdirectory name, e.g. "train" or "test".
    """
    src_split = os.path.join(SRC_ROOT, split)
    dst_split = os.path.join(DST_ROOT, split)
    ensure_dir(dst_split)

    for class_name in os.listdir(src_split):
        src_cls = os.path.join(src_split, class_name)
        if not os.path.isdir(src_cls):
            continue  # Skip any stray files at the split level

        dst_cls = os.path.join(dst_split, class_name)
        ensure_dir(dst_cls)

        for filename in os.listdir(src_cls):
            if Path(filename).suffix.lower() not in IMAGE_EXTS:
                continue

            src_path = os.path.join(src_cls, filename)
            dst_path = os.path.join(dst_cls, filename)

            img = cv2.imread(src_path)
            if img is None:
                print(f"  Warning: could not read {src_path}, skipping.")
                continue

            face = crop_largest_face(img)

            # Fall back to the full image if no face was detected, so no
            # samples are lost from the dataset
            output_image = face if face is not None else img
            cv2.imwrite(dst_path, output_image)


if __name__ == "__main__":
    for split in ("train", "test"):
        print(f"Processing split: {split}")
        process_split(split)

    print(f"\nDone. Face-cropped dataset saved to: {DST_ROOT}")
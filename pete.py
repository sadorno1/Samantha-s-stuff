import os
import cv2
from pathlib import Path

# Input dataset folder (same structure train/test/class/img.jpg)
SRC_ROOT = r"Autism emotion recogition dataset"
DST_ROOT = r"Autism emotion recogition dataset_faces"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def crop_largest_face(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    # pick largest face
    x, y, w, h = max(faces, key=lambda t: t[2] * t[3])
    return bgr[y:y+h, x:x+w]

def process_split(split: str):
    src_split = os.path.join(SRC_ROOT, split)
    dst_split = os.path.join(DST_ROOT, split)
    ensure_dir(dst_split)

    for class_name in os.listdir(src_split):
        src_cls = os.path.join(src_split, class_name)
        if not os.path.isdir(src_cls):
            continue
        dst_cls = os.path.join(dst_split, class_name)
        ensure_dir(dst_cls)

        for fn in os.listdir(src_cls):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            src_path = os.path.join(src_cls, fn)
            dst_path = os.path.join(dst_cls, fn)

            img = cv2.imread(src_path)
            if img is None:
                continue

            face = crop_largest_face(img)
            out = face if face is not None else img  # fallback to original if no face found

            # Save as JPG (consistent)
            cv2.imwrite(dst_path, out)

if __name__ == "__main__":
    process_split("train")
    process_split("test")
    print("Done. Cropped dataset at:", DST_ROOT)

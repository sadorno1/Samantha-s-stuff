from pathlib import Path
import shutil

src_train = Path(r"archive/DATASET/train")
src_test = Path(r"archive/DATASET/test")

dst_train = Path(r"POSTER_V2/data/raf-db/train")
dst_valid = Path(r"POSTER_V2/data/raf-db/valid")

if dst_train.parent.exists():
    shutil.rmtree(dst_train.parent)

dst_train.mkdir(parents=True, exist_ok=True)
dst_valid.mkdir(parents=True, exist_ok=True)

for class_dir in src_train.iterdir():
    if class_dir.is_dir():
        shutil.copytree(class_dir, dst_train / class_dir.name, dirs_exist_ok=True)

for class_dir in src_test.iterdir():
    if class_dir.is_dir():
        shutil.copytree(class_dir, dst_valid / class_dir.name, dirs_exist_ok=True)

print("POSTER_V2 RAF-DB copy rebuilt.")
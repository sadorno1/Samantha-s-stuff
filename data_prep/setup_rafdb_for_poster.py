"""
setup_rafdb_for_poster.py  (originally: poster_format.py)
----------------------------------------------------------
Reformat a RAF-DB download into the directory structure expected by the
POSTER V2 model (https://github.com/zczcwh/POSTER_V2).

POSTER V2 expects:
  POSTER_V2/data/raf-db/
    train/  <class_dirs>
    valid/  <class_dirs>

This script maps:
  archive/DATASET/train → POSTER_V2/data/raf-db/train
  archive/DATASET/test  → POSTER_V2/data/raf-db/valid

Usage:
  python setup_rafdb_for_poster.py

Edit the source/destination paths below if your RAF-DB archive is in a
different location.
"""

import shutil
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
SRC_TRAIN = Path(r"archive/DATASET/train")
SRC_TEST  = Path(r"archive/DATASET/test")

DST_TRAIN = Path(r"POSTER_V2/data/raf-db/train")
DST_VALID = Path(r"POSTER_V2/data/raf-db/valid")
# ──────────────────────────────────────────────────────────────────────────────


def rebuild_poster_data() -> None:
    """Delete any existing POSTER V2 data directory and rebuild it from
    the RAF-DB source archive."""

    # Remove and recreate the parent to ensure a clean copy
    dst_parent = DST_TRAIN.parent
    if dst_parent.exists():
        shutil.rmtree(dst_parent)
        print(f"Removed existing directory: {dst_parent}")

    DST_TRAIN.mkdir(parents=True, exist_ok=True)
    DST_VALID.mkdir(parents=True, exist_ok=True)

    # Copy each class subdirectory from train split
    for class_dir in SRC_TRAIN.iterdir():
        if class_dir.is_dir():
            shutil.copytree(class_dir, DST_TRAIN / class_dir.name, dirs_exist_ok=True)

    # Copy each class subdirectory from test split → valid
    for class_dir in SRC_TEST.iterdir():
        if class_dir.is_dir():
            shutil.copytree(class_dir, DST_VALID / class_dir.name, dirs_exist_ok=True)

    print(f"POSTER V2 RAF-DB layout rebuilt under: {dst_parent}")


if __name__ == "__main__":
    rebuild_poster_data()
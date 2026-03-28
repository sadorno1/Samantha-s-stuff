"""
vit_autism_transfer.py  (originally: vit_fer_transfer.py)
----------------------------------------------------------
Fine-tunes Vision Transformer ViT-B/16 on facial expression data using a
two-stage transfer learning pipeline:

  Stage 1 (pretrain_fer)      – fine-tune ViT-B/16 on FER2013
  Stage 2 (finetune_autism)   – fine-tune the FER2013 checkpoint on the autism
                                 emotion dataset
  Ablation (compare_autism_aug) – run Stage 2 twice: without and with
                                   augmentation, to quantify the aug effect

Usage:
  # Stage 1 – pre-train on FER2013
  python vit_autism_transfer.py \\
      --mode pretrain_fer \\
      --train_dir path/to/fer2013_images/train \\
      --val_dir   path/to/fer2013_images/test

  # Stage 2 – fine-tune on autism dataset
  python vit_autism_transfer.py \\
      --mode finetune_autism \\
      --train_dir path/to/autism/train \\
      --val_dir   path/to/autism/test \\
      --init_checkpoint checkpoints_vit/vit_fer2013_pretrain_best.pth

  # Augmentation ablation
  python vit_autism_transfer.py \\
      --mode compare_autism_aug \\
      --train_dir path/to/autism/train \\
      --val_dir   path/to/autism/test \\
      --init_checkpoint checkpoints_vit/vit_fer2013_pretrain_best.pth

See --help for the full list of options.
"""

import argparse
import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

# ── Metrics ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_confusion_matrix(
    y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Build an (num_classes × num_classes) confusion matrix."""
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm


def f1_from_confusion(cm: torch.Tensor) -> Tuple[float, List[float]]:
    """Compute macro-F1 and per-class F1 from a confusion matrix."""
    num_classes = cm.size(0)
    per_f1 = []
    for c in range(num_classes):
        tp    = cm[c, c].item()
        fp    = cm[:, c].sum().item() - tp
        fn    = cm[c, :].sum().item() - tp
        denom = 2 * tp + fp + fn
        per_f1.append((2 * tp / denom) if denom > 0 else 0.0)
    macro = sum(per_f1) / num_classes if num_classes > 0 else 0.0
    return macro, per_f1


def accuracy_from_confusion(cm: torch.Tensor) -> float:
    """Compute overall accuracy from a confusion matrix."""
    correct = cm.diag().sum().item()
    total   = cm.sum().item()
    return (correct / total) if total > 0 else 0.0


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    return (count_params(model) * 4) / (1024 ** 2)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_vit_b16(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Instantiate ViT-B/16 and replace the classification head.

    Args:
        num_classes: Target number of emotion classes.
        pretrained:  Load ImageNet-21k weights if True.
    """
    model = torchvision.models.vit_b_16(
        weights=(
            torchvision.models.ViT_B_16_Weights.DEFAULT
            if pretrained else None
        )
    )
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


def load_partial_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    """Load matching weights from a checkpoint, skipping size-mismatched layers.

    This allows using a checkpoint trained on a different number of classes:
    all backbone layers will be transferred; the head is skipped if sizes differ.

    Args:
        model:           Target model (with the correct classification head).
        checkpoint_path: Path to the source .pth checkpoint file.
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model_dict = model.state_dict()

    filtered = {
        k: v for k, v in state_dict.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Missing keys   : {missing}")
    print(f"  Unexpected keys: {unexpected}")


# ── Training / evaluation ─────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Final metrics for one experiment run."""
    acc: float
    macro_f1: float
    per_class_f1: List[float]
    latency_ms: float
    throughput_ips: float
    params_m: float
    size_mb: float
    best_epoch: int
    ckpt_path: str


def make_weighted_sampler(dataset: ImageFolder) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler to oversample minority classes.

    Each sample's weight is the inverse of its class frequency, so all classes
    contribute approximately equally to each epoch regardless of imbalance.
    """
    labels       = [label for _, label in dataset.samples]
    num_classes  = len(dataset.classes)
    counts       = torch.zeros(num_classes, dtype=torch.float32)
    for y in labels:
        counts[y] += 1.0
    class_weights  = 1.0 / counts
    sample_weights = torch.tensor(
        [class_weights[y].item() for y in labels], dtype=torch.float32
    )
    return WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )


def train_one_epoch(model, loader, criterion, optimizer, scaler, device) -> float:
    """Train for one epoch and return the mean cross-entropy loss."""
    model.train()
    running_loss, n = 0.0, 0

    for imgs, labels in loader:
        imgs   = imgs.to(device,   non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)

    return running_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, num_classes) -> Tuple[float, float, List[float]]:
    """Return (accuracy, macro_f1, per_class_f1) on a data loader."""
    model.eval()
    all_true, all_pred = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device,   non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds  = torch.argmax(model(imgs), dim=1)
        all_true.append(labels.cpu())
        all_pred.append(preds.cpu())

    y_true = torch.cat(all_true)
    y_pred = torch.cat(all_pred)
    cm     = compute_confusion_matrix(y_true, y_pred, num_classes)
    acc    = accuracy_from_confusion(cm)
    macro_f1, per_f1 = f1_from_confusion(cm)
    return acc, macro_f1, per_f1


@torch.no_grad()
def benchmark_inference(
    model, device, input_shape=(1, 3, 224, 224), iters: int = 100, warmup: int = 20
) -> Tuple[float, float]:
    """Return (latency_ms_per_image, images_per_second)."""
    model.eval()
    x = torch.randn(*input_shape, device=device)

    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total   = t1 - t0
    bs      = input_shape[0]
    latency = (total / iters) * 1000.0 / bs
    ips     = (iters * bs) / total
    return latency, ips


# ── Data transforms ───────────────────────────────────────────────────────────

def make_transforms(img_size: int, use_aug: bool):
    """Build train and validation transforms.

    When use_aug=True the training transform applies RandomResizedCrop,
    flips, colour jitter, Gaussian blur, and random erasing.
    When use_aug=False only resizing and normalisation are applied.

    Args:
        img_size: Target image size (height == width).
        use_aug:  Whether to apply data augmentation during training.

    Returns:
        (train_transform, val_transform)
    """
    if use_aug:
        train_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.03),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.10, scale=(0.02, 0.08),
                                     ratio=(0.3, 3.3), value="random"),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ── Generic experiment runner ─────────────────────────────────────────────────

def run_experiment(
    exp_name: str,
    train_dir: str,
    val_dir: str,
    img_size: int,
    epochs: int,
    batch_size: int,
    num_workers: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    scheduler_name: str,
    select_best_by: str,
    use_aug: bool,
    use_weighted_sampler: bool,
    init_checkpoint: str,
    output_dir: str,
    device: torch.device,
    seed: int,
) -> EvalResult:
    """Train ViT-B/16 for one experiment configuration and return metrics.

    Args:
        exp_name:             Human-readable label (used in logs and filename).
        train_dir:            Path to training split (ImageFolder layout).
        val_dir:              Path to validation split.
        img_size:             Input image side length (pixels).
        epochs:               Number of training epochs.
        batch_size:           Mini-batch size.
        num_workers:          DataLoader worker processes.
        lr:                   Initial learning rate.
        weight_decay:         AdamW weight decay.
        label_smoothing:      Cross-entropy label smoothing factor.
        scheduler_name:       "cosine", "plateau", or "none".
        select_best_by:       Metric to use for checkpoint selection ("macro_f1" / "acc").
        use_aug:              Whether to apply data augmentation.
        use_weighted_sampler: Use WeightedRandomSampler to balance classes.
        init_checkpoint:      Path to a .pth file to initialise the backbone.
        output_dir:           Directory to write the best checkpoint to.
        device:               torch.device.
        seed:                 Random seed for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_tf, val_tf = make_transforms(img_size, use_aug)
    train_ds = ImageFolder(train_dir, transform=train_tf)
    val_ds   = ImageFolder(val_dir,   transform=val_tf)
    num_classes = len(train_ds.classes)

    print("\n" + "=" * 100)
    print(f"Experiment      : {exp_name}")
    print(f"Classes ({num_classes})     : {train_ds.classes}")
    print(f"Augmentation    : {use_aug}")
    print(f"Weighted sampler: {use_weighted_sampler}")

    model = build_vit_b16(num_classes=num_classes, pretrained=True).to(device)
    if init_checkpoint is not None:
        load_partial_checkpoint(model, init_checkpoint)

    # Choose sampler strategy for class imbalance
    if use_weighted_sampler:
        sampler = make_weighted_sampler(train_ds)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, shuffle=False,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
    else:
        scheduler = None

    # Mixed-precision training (GPU only)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_score = -1.0
    best_epoch = -1
    best_state = None
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = str(
        Path(output_dir) / f"{exp_name.replace(' ', '_').lower()}_best.pth"
    )

    for epoch in range(1, epochs + 1):
        t0   = time.perf_counter()
        loss = train_one_epoch(model, train_loader, criterion,
                               optimizer, scaler, device)
        acc, macro_f1, per_f1 = evaluate(model, val_loader, device, num_classes)
        t1   = time.perf_counter()

        score = macro_f1 if select_best_by == "macro_f1" else acc
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, ckpt_path)

        if scheduler is not None:
            if scheduler_name == "plateau":
                scheduler.step(macro_f1)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss={loss:.4f} | val_acc={acc:.4f} | macro_f1={macro_f1:.4f} | "
            f"lr={current_lr:.6f} | time={(t1 - t0):.1f}s"
        )

    # ── Final evaluation ──────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    acc, macro_f1, per_f1 = evaluate(model, val_loader, device, num_classes)
    latency_ms, ips = benchmark_inference(
        model, device, input_shape=(1, 3, img_size, img_size),
        iters=100, warmup=20,
    )
    params_m = count_params(model) / 1e6
    size_mb  = model_size_mb(model)

    print(f"\nBest checkpoint epoch for '{exp_name}': {best_epoch}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Macro F1    : {macro_f1:.4f}")
    print(f"  Per-class F1: {[round(x, 4) for x in per_f1]}")
    print(f"  Params (M)  : {params_m:.2f}")
    print(f"  Size (MB)   : {size_mb:.1f}")
    print(f"  Latency (ms): {latency_ms:.2f} ms/img")
    print(f"  Throughput  : {ips:.2f} img/s")
    print(f"  Checkpoint  : {ckpt_path}")

    return EvalResult(
        acc=acc, macro_f1=macro_f1, per_class_f1=per_f1,
        latency_ms=latency_ms, throughput_ips=ips,
        params_m=params_m, size_mb=size_mb,
        best_epoch=best_epoch, ckpt_path=ckpt_path,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ViT-B/16 transfer learning for autism emotion recognition"
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["pretrain_fer", "finetune_autism", "compare_autism_aug"],
                        help="pretrain_fer: Stage 1 on FER2013 | "
                             "finetune_autism: Stage 2 on autism dataset | "
                             "compare_autism_aug: run Stage 2 ±augmentation")
    parser.add_argument("--train_dir",        type=str, required=True)
    parser.add_argument("--val_dir",          type=str, required=True)
    parser.add_argument("--init_checkpoint",  type=str, default=None,
                        help="Checkpoint to initialise backbone from")
    parser.add_argument("--epochs",           type=int,   default=50)
    parser.add_argument("--batch_size",       type=int,   default=16)
    parser.add_argument("--num_workers",      type=int,   default=4)
    parser.add_argument("--img_size",         type=int,   default=224)
    parser.add_argument("--lr",               type=float, default=3e-5)
    parser.add_argument("--weight_decay",     type=float, default=1e-4)
    parser.add_argument("--label_smoothing",  type=float, default=0.05)
    parser.add_argument("--scheduler",        type=str,   default="cosine",
                        choices=["none", "cosine", "plateau"])
    parser.add_argument("--select_best_by",   type=str,   default="macro_f1",
                        choices=["macro_f1", "acc"])
    parser.add_argument("--use_aug",          action="store_true", default=False)
    parser.add_argument("--weighted_sampler", action="store_true", default=False)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--output_dir",       type=str,   default="checkpoints_vit")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Common kwargs shared across all run_experiment calls
    shared = dict(
        train_dir=args.train_dir, val_dir=args.val_dir,
        img_size=args.img_size, epochs=args.epochs,
        batch_size=args.batch_size, num_workers=args.num_workers,
        lr=args.lr, weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        scheduler_name=args.scheduler, select_best_by=args.select_best_by,
        use_weighted_sampler=args.weighted_sampler,
        output_dir=args.output_dir, device=device, seed=args.seed,
    )

    results: Dict[str, EvalResult] = {}

    if args.mode == "pretrain_fer":
        results["ViT FER2013 pretrain"] = run_experiment(
            exp_name="vit_fer2013_pretrain",
            use_aug=args.use_aug, init_checkpoint=None, **shared,
        )

    elif args.mode == "finetune_autism":
        results["ViT autism finetune"] = run_experiment(
            exp_name="vit_autism_finetune",
            use_aug=args.use_aug,
            init_checkpoint=args.init_checkpoint, **shared,
        )

    elif args.mode == "compare_autism_aug":
        # Run both with and without augmentation to measure its effect
        results["ViT autism no aug"] = run_experiment(
            exp_name="vit_autism_no_aug",
            use_aug=False,
            init_checkpoint=args.init_checkpoint, **shared,
        )
        results["ViT autism with aug"] = run_experiment(
            exp_name="vit_autism_with_aug",
            use_aug=True,
            init_checkpoint=args.init_checkpoint, **shared,
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("SUMMARY")
    header = (
        f"{'Experiment':28s} {'Acc':>6s} {'MacroF1':>7s} {'BestEp':>6s} "
        f"{'Params(M)':>9s} {'Size(MB)':>8s} {'ms/img':>7s}"
    )
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(
            f"{name:28s} {r.acc:6.4f} {r.macro_f1:7.4f} {r.best_epoch:6d} "
            f"{r.params_m:9.2f} {r.size_mb:8.1f} {r.latency_ms:7.2f}"
        )


if __name__ == "__main__":
    main()
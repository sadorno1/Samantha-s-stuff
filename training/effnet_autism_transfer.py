"""
effnet_autism_transfer.py  (originally: autism_effnet_transfer.py)
------------------------------------------------------------------
Fine-tunes EfficientNet-B0 on the autism emotion recognition dataset using
two initialisation strategies:

  1. ImageNet only   – standard ImageNet pre-trained weights
  2. RAF-DB init     – backbone weights from a model already trained on RAF-DB
                       (produced by rafdb_model_benchmark.py)

Both experiments are run back-to-back and a summary table is printed at the end.

Usage:
  # Without a RAF-DB checkpoint (ImageNet only experiment only)
  python effnet_autism_transfer.py \\
      --train_dir path/to/autism/train \\
      --val_dir   path/to/autism/test

  # With a RAF-DB checkpoint (runs both experiments)
  python effnet_autism_transfer.py \\
      --train_dir path/to/autism/train \\
      --val_dir   path/to/autism/test \\
      --rafdb_checkpoint efficientnet_b0_rafdb_best.pth

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
from torch.utils.data import DataLoader
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
    """Compute macro-F1 and per-class F1 from a confusion matrix.

    Returns:
        (macro_f1, per_class_f1_list)
    """
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
    """Return the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """Estimate model size in MB (float32, 4 bytes per parameter)."""
    return (count_params(model) * 4) / (1024 ** 2)


def compute_class_weights(dataset: ImageFolder, num_classes: int) -> torch.Tensor:
    """Return inverse-frequency class weights for weighted cross-entropy."""
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, label in dataset.samples:
        counts[label] += 1
    return counts.sum() / (counts * num_classes)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_efficientnet_b0(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build an EfficientNet-B0 with a new classification head.

    Args:
        num_classes: Number of target emotion classes.
        pretrained:  Load ImageNet weights if True.
    """
    model = torchvision.models.efficientnet_b0(
        weights=(
            torchvision.models.EfficientNet_B0_Weights.DEFAULT
            if pretrained else None
        )
    )
    # Replace the final linear layer for the target number of classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def load_backbone_weights(model: nn.Module, checkpoint_path: str) -> None:
    """Load a RAF-DB checkpoint into the model, skipping size-mismatched layers.

    Layers whose shapes do not match (e.g. the classifier head) are skipped so
    that the same function can be used to initialise a model with a different
    number of output classes.

    Args:
        model:           Target model (already has the correct classifier head).
        checkpoint_path: Path to the .pth file from RAF-DB training.
    """
    state_dict  = torch.load(checkpoint_path, map_location="cpu")
    model_dict  = model.state_dict()

    # Keep only layers that exist in the target model AND have matching shapes
    filtered = {
        k: v for k, v in state_dict.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"  Missing keys   : {missing}")
    print(f"  Unexpected keys: {unexpected}")


# ── Training / evaluation ─────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Final evaluation metrics for one experiment."""
    acc: float
    macro_f1: float
    per_class_f1: List[float]
    latency_ms: float
    throughput_ips: float
    params_m: float
    size_mb: float
    best_epoch: int


def train_one_epoch(model, loader, criterion, optimizer, scaler, device) -> float:
    """Run one training epoch and return the mean cross-entropy loss."""
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
    """Evaluate the model and return (accuracy, macro_f1, per_class_f1)."""
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
    model, device, input_shape=(1, 3, 224, 224), iters: int = 200, warmup: int = 30
) -> Tuple[float, float]:
    """Measure single-image latency and throughput.

    Returns:
        (latency_ms_per_image, images_per_second)
    """
    model.eval()
    x = torch.randn(*input_shape, device=device)

    for _ in range(warmup):   # GPU warm-up
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


def run_experiment(
    exp_name: str,
    train_ds: ImageFolder,
    val_ds: ImageFolder,
    args,
    device: torch.device,
    use_rafdb_init: bool = False,
) -> EvalResult:
    """Train EfficientNet-B0 on the autism dataset with a given initialisation.

    Args:
        exp_name:       Human-readable name printed in logs and used for the
                        checkpoint filename.
        train_ds:       Training ImageFolder dataset.
        val_ds:         Validation ImageFolder dataset.
        args:           Parsed argparse namespace.
        device:         torch.device to train on.
        use_rafdb_init: If True, load backbone weights from args.rafdb_checkpoint
                        before training.

    Returns:
        EvalResult with final (best-checkpoint) metrics.
    """
    print("\n" + "=" * 90)
    print(f"Experiment: {exp_name}")

    num_classes = len(train_ds.classes)
    model       = build_efficientnet_b0(num_classes=num_classes,
                                        pretrained=True).to(device)

    if use_rafdb_init:
        if not args.rafdb_checkpoint:
            raise ValueError(
                "use_rafdb_init=True but --rafdb_checkpoint was not provided."
            )
        load_backbone_weights(model, args.rafdb_checkpoint)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    class_weights = None
    if args.class_weighted_loss:
        class_weights = compute_class_weights(train_ds, num_classes).to(device)
        print("Class weights:", [round(x, 4) for x in class_weights.tolist()])

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    elif args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
    else:
        scheduler = None

    scaler = (
        torch.amp.GradScaler("cuda")
        if (device.type == "cuda" and args.amp) else None
    )

    best_score = -1.0
    best_epoch = -1
    best_state = None
    ckpt_name  = f"{exp_name.replace(' ', '_').lower()}_best.pth"

    for epoch in range(1, args.epochs + 1):
        t0   = time.perf_counter()
        loss = train_one_epoch(model, train_loader, criterion,
                               optimizer, scaler, device)
        acc, macro_f1, per_f1 = evaluate(model, val_loader, device, num_classes)
        t1   = time.perf_counter()

        score = macro_f1 if args.select_best_by == "macro_f1" else acc
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, ckpt_name)

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(macro_f1)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"loss={loss:.4f} | val_acc={acc:.4f} | macro_f1={macro_f1:.4f} | "
            f"lr={current_lr:.6f} | time={(t1 - t0):.1f}s"
        )

    # ── Final evaluation using best checkpoint ────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    acc, macro_f1, per_f1 = evaluate(model, val_loader, device, num_classes)
    latency_ms, ips = benchmark_inference(
        model, device,
        input_shape=(1, 3, args.img_size, args.img_size),
        iters=args.bench_iters, warmup=30,
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

    return EvalResult(
        acc=acc, macro_f1=macro_f1, per_class_f1=per_f1,
        latency_ms=latency_ms, throughput_ips=ips,
        params_m=params_m, size_mb=size_mb, best_epoch=best_epoch,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer-learn EfficientNet-B0 on the autism emotion dataset"
    )
    parser.add_argument("--train_dir",  type=str, required=True)
    parser.add_argument("--val_dir",    type=str, required=True)
    parser.add_argument("--rafdb_checkpoint", type=str, default=None,
                        help="Path to EfficientNet-B0 checkpoint from RAF-DB training")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--num_workers",type=int,   default=4)
    parser.add_argument("--img_size",   type=int,   default=224)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--weight_decay",type=float,default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--amp",    action="store_true",  default=True)
    parser.add_argument("--no_amp", dest="amp",           action="store_false")
    parser.add_argument("--class_weighted_loss",    action="store_true",  default=True)
    parser.add_argument("--no_class_weighted_loss", dest="class_weighted_loss",
                        action="store_false")
    parser.add_argument("--scheduler",     type=str, default="cosine",
                        choices=["none", "cosine", "plateau"])
    parser.add_argument("--select_best_by",type=str, default="macro_f1",
                        choices=["macro_f1", "acc"])
    parser.add_argument("--bench_iters",   type=int, default=200)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Light augmentation for a small medical-context dataset
    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15,
                               saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.10),
                                 ratio=(0.3, 3.3), value="random"),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ImageFolder(args.train_dir, transform=train_tf)
    val_ds   = ImageFolder(args.val_dir,   transform=val_tf)
    print(f"Classes ({len(train_ds.classes)}): {train_ds.classes}")

    results: Dict[str, EvalResult] = {}

    # Experiment 1: standard ImageNet initialisation
    results["EffNet-B0 (ImageNet only)"] = run_experiment(
        exp_name="EffNet-B0 ImageNet only",
        train_ds=train_ds, val_ds=val_ds, args=args,
        device=device, use_rafdb_init=False,
    )

    # Experiment 2: RAF-DB initialised backbone (only if checkpoint provided)
    if args.rafdb_checkpoint:
        results["EffNet-B0 (RAF-DB init)"] = run_experiment(
            exp_name="EffNet-B0 RAFDB initialized",
            train_ds=train_ds, val_ds=val_ds, args=args,
            device=device, use_rafdb_init=True,
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("SUMMARY")
    header = (
        f"{'Experiment':32s} {'Acc':>6s} {'MacroF1':>7s} {'BestEp':>6s} "
        f"{'Params(M)':>9s} {'Size(MB)':>8s} {'ms/img':>7s}"
    )
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(
            f"{name:32s} {r.acc:6.4f} {r.macro_f1:7.4f} {r.best_epoch:6d} "
            f"{r.params_m:9.2f} {r.size_mb:8.1f} {r.latency_ms:7.2f}"
        )


if __name__ == "__main__":
    main()
"""
rafdb_model_benchmark.py  (originally: rafdb_fer_bench.py)
-----------------------------------------------------------
Benchmarks multiple CNN/ViT backbones on RAF-DB for facial expression
recognition (FER).  Useful for selecting the best backbone before
fine-tuning on the autism-specific dataset.

Supported models: resnet18, resnet50, efficientnet_b0, swin_t, vit_b_16

Each model is trained from ImageNet pre-trained weights and the best
checkpoint (by macro-F1 or accuracy) is saved as:
  efficientnet_b0_rafdb_best.pth  (or similar per model name)

Usage:
  python rafdb_model_benchmark.py \\
      --train_dir path/to/rafdb/train \\
      --val_dir   path/to/rafdb/test \\
      --models efficientnet_b0 resnet50 \\
      --epochs 50

See --help for the full list of options.
"""

import argparse
import time
from dataclasses import dataclass
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
    """Build an (num_classes × num_classes) confusion matrix from flat tensors."""
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm


def f1_from_confusion(cm: torch.Tensor) -> Tuple[float, List[float]]:
    """Compute macro-averaged F1 and per-class F1 from a confusion matrix.

    Returns:
        (macro_f1, per_class_f1_list)
    """
    num_classes = cm.size(0)
    per_f1 = []
    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        denom = 2 * tp + fp + fn
        per_f1.append((2 * tp / denom) if denom > 0 else 0.0)
    macro = sum(per_f1) / num_classes if num_classes > 0 else 0.0
    return macro, per_f1


def accuracy_from_confusion(cm: torch.Tensor) -> float:
    """Compute overall accuracy from a confusion matrix."""
    correct = cm.diag().sum().item()
    total = cm.sum().item()
    return (correct / total) if total > 0 else 0.0


def count_params(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """Estimate model size in MB (assumes float32 = 4 bytes per parameter)."""
    return (count_params(model) * 4) / (1024 ** 2)


def compute_class_weights(dataset: ImageFolder, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights to handle class imbalance.

    Weight for class c = total_samples / (num_classes * count_c)
    """
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, label in dataset.samples:
        counts[label] += 1
    return counts.sum() / (counts * num_classes)


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Instantiate a torchvision model and replace its classifier head.

    Args:
        name:        One of resnet18, resnet50, efficientnet_b0, swin_t, vit_b_16.
        num_classes: Number of output classes.
        pretrained:  Load ImageNet weights if True.

    Returns:
        nn.Module with a freshly initialised classification head.

    Raises:
        ValueError: If name is not recognised.
    """
    name = name.lower()

    if name == "resnet18":
        m = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if name == "resnet50":
        m = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if name in ("efficientnet_b0", "effnet_b0"):
        m = torchvision.models.efficientnet_b0(
            weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m

    if name in ("swin_t", "swin_tiny", "swin_tiny_patch4_window7_224"):
        m = torchvision.models.swin_t(
            weights=torchvision.models.Swin_T_Weights.DEFAULT if pretrained else None
        )
        m.head = nn.Linear(m.head.in_features, num_classes)
        return m

    if name in ("vit_b_16", "vit", "vit_base_16"):
        m = torchvision.models.vit_b_16(
            weights=torchvision.models.ViT_B_16_Weights.DEFAULT if pretrained else None
        )
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        return m

    raise ValueError(f"Unknown model name: '{name}'")


# ── Training / evaluation ─────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Holds the final evaluation metrics for one trained model."""
    acc: float
    macro_f1: float
    per_class_f1: List[float]
    latency_ms: float
    throughput_ips: float
    params_m: float
    size_mb: float


def train_one_epoch(model, loader, criterion, optimizer, scaler, device) -> float:
    """Run one full training pass and return the mean cross-entropy loss."""
    model.train()
    running_loss = 0.0
    n = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed-precision forward pass (no-op on CPU)
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

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        n += bs

    return running_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    model, loader, device, num_classes
) -> Tuple[float, float, List[float]]:
    """Evaluate the model on a data loader.

    Returns:
        (accuracy, macro_f1, per_class_f1)
    """
    model.eval()
    all_true, all_pred = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
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
    """Measure single-image inference latency and throughput.

    Returns:
        (latency_ms_per_image, images_per_second)
    """
    model.eval()
    x = torch.randn(*input_shape, device=device)

    # Warm-up to stabilise GPU clocks
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
    latency = (total / iters) * 1000.0 / bs   # ms per image
    ips     = (iters * bs) / total
    return latency, ips


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark FER models on RAF-DB"
    )
    parser.add_argument("--train_dir",  type=str, required=True,
                        help="Path to RAF-DB training split (ImageFolder layout)")
    parser.add_argument("--val_dir",    type=str, required=True,
                        help="Path to RAF-DB validation/test split")
    parser.add_argument("--models",     type=str, nargs="+",
                        default=["efficientnet_b0"],
                        help="Models to benchmark (space-separated)")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--num_workers",type=int,   default=4)
    parser.add_argument("--img_size",   type=int,   default=224)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--weight_decay",type=float,default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--pretrained", action="store_true",  default=True)
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--amp",    action="store_true",  default=True)
    parser.add_argument("--no_amp", dest="amp",           action="store_false")
    parser.add_argument("--class_weighted_loss",    action="store_true",  default=True)
    parser.add_argument("--no_class_weighted_loss", dest="class_weighted_loss",
                        action="store_false")
    parser.add_argument("--scheduler",     type=str, default="cosine",
                        choices=["none", "cosine", "plateau"])
    parser.add_argument("--bench_iters",   type=int, default=200)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--select_best_by",type=str, default="macro_f1",
                        choices=["macro_f1", "acc"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data augmentation for training
    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Minimal transform for validation (no augmentation)
    val_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ImageFolder(args.train_dir, transform=train_tf)
    val_ds   = ImageFolder(args.val_dir,   transform=val_tf)
    num_classes = len(train_ds.classes)
    print(f"Classes ({num_classes}): {train_ds.classes}")
    if num_classes != 7:
        print("Warning: RAF-DB basic should have 7 classes. Proceeding anyway.")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    results: Dict[str, EvalResult] = {}

    for name in args.models:
        print("\n" + "=" * 80)
        print(f"Model: {name}")

        model = build_model(name, num_classes=num_classes,
                            pretrained=args.pretrained).to(device)

        # Optionally weight the loss by inverse class frequency
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
        best_state = None
        best_epoch = -1

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
                best_state = {k: v.detach().cpu()
                              for k, v in model.state_dict().items()}
                # Save best checkpoint for use in downstream transfer scripts
                torch.save(best_state, "efficientnet_b0_rafdb_best.pth")

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

        # Reload best weights for final evaluation
        if best_state is not None:
            model.load_state_dict(best_state, strict=True)

        acc, macro_f1, per_f1 = evaluate(model, val_loader, device, num_classes)
        latency_ms, ips = benchmark_inference(
            model, device,
            input_shape=(1, 3, args.img_size, args.img_size),
            iters=args.bench_iters, warmup=30,
        )
        params_m = count_params(model) / 1e6
        size_mb  = model_size_mb(model)

        results[name] = EvalResult(
            acc=acc, macro_f1=macro_f1, per_class_f1=per_f1,
            latency_ms=latency_ms, throughput_ips=ips,
            params_m=params_m, size_mb=size_mb,
        )

        print(f"\nBest checkpoint epoch for {name}: {best_epoch}")
        print(f"  Accuracy    : {acc:.4f}")
        print(f"  Macro F1    : {macro_f1:.4f}")
        print(f"  Per-class F1: {[round(x, 4) for x in per_f1]}")
        print(f"  Params (M)  : {params_m:.2f}")
        print(f"  Size (MB)   : {size_mb:.1f}")
        print(f"  Latency (ms): {latency_ms:.2f} ms/img")
        print(f"  Throughput  : {ips:.2f} img/s")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY (best checkpoint per model on validation set)")
    header = (
        f"{'Model':18s} {'Acc':>6s} {'MacroF1':>7s} {'Params(M)':>9s} "
        f"{'Size(MB)':>8s} {'ms/img':>7s} {'img/s':>7s}"
    )
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(
            f"{name:18s} {r.acc:6.4f} {r.macro_f1:7.4f} {r.params_m:9.2f} "
            f"{r.size_mb:8.1f} {r.latency_ms:7.2f} {r.throughput_ips:7.2f}"
        )


if __name__ == "__main__":
    main()
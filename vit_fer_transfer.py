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


# ---------------------------
# Metrics
# ---------------------------
@torch.no_grad()
def compute_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm


def f1_from_confusion(cm: torch.Tensor) -> Tuple[float, List[float]]:
    num_classes = cm.size(0)
    per_f1 = []
    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        per_f1.append(f1)
    macro = sum(per_f1) / num_classes if num_classes > 0 else 0.0
    return macro, per_f1


def accuracy_from_confusion(cm: torch.Tensor) -> float:
    correct = cm.diag().sum().item()
    total = cm.sum().item()
    return (correct / total) if total > 0 else 0.0


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    return (count_params(model) * 4) / (1024 ** 2)


# ---------------------------
# Model
# ---------------------------
def build_vit_b16(num_classes: int, pretrained: bool = True) -> nn.Module:
    model = torchvision.models.vit_b_16(
        weights=torchvision.models.ViT_B_16_Weights.DEFAULT if pretrained else None
    )
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


def load_partial_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    """
    Loads matching weights only, skipping the classifier if class counts differ.
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model_dict = model.state_dict()
    filtered = {}

    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered[k] = v

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"Loaded checkpoint from: {checkpoint_path}")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)


# ---------------------------
# Training / eval
# ---------------------------
@dataclass
class EvalResult:
    acc: float
    macro_f1: float
    per_class_f1: List[float]
    latency_ms: float
    throughput_ips: float
    params_m: float
    size_mb: float
    best_epoch: int
    ckpt_path: str


def make_weighted_sampler(dataset: ImageFolder):
    labels = [label for _, label in dataset.samples]
    num_classes = len(dataset.classes)

    counts = torch.zeros(num_classes, dtype=torch.float32)
    for y in labels:
        counts[y] += 1.0

    class_weights = 1.0 / counts
    sample_weights = torch.tensor([class_weights[y].item() for y in labels], dtype=torch.float32)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    n = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            logits = model(imgs)
            loss = criterion(logits, labels)

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
def evaluate(model, loader, device, num_classes):
    model.eval()
    all_true = []
    all_pred = []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)

        all_true.append(labels.cpu())
        all_pred.append(preds.cpu())

    y_true = torch.cat(all_true, dim=0)
    y_pred = torch.cat(all_pred, dim=0)

    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    acc = accuracy_from_confusion(cm)
    macro_f1, per_f1 = f1_from_confusion(cm)
    return acc, macro_f1, per_f1


@torch.no_grad()
def benchmark_inference(model, device, input_shape=(1, 3, 224, 224), iters=100, warmup=20):
    model.eval()
    x = torch.randn(*input_shape, device=device)

    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total = t1 - t0
    latency = (total / iters) * 1000.0
    bs = input_shape[0]
    latency_per_img = latency / bs
    ips = (iters * bs) / total
    return latency_per_img, ips


# ---------------------------
# Transforms
# ---------------------------
def make_transforms(img_size: int, use_aug: bool):
    if use_aug:
        train_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.10, scale=(0.02, 0.08), ratio=(0.3, 3.3), value="random"),
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


# ---------------------------
# Generic runner
# ---------------------------
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
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_tf, val_tf = make_transforms(img_size, use_aug)

    train_ds = ImageFolder(train_dir, transform=train_tf)
    val_ds = ImageFolder(val_dir, transform=val_tf)
    num_classes = len(train_ds.classes)

    print("\n" + "=" * 100)
    print(f"Experiment: {exp_name}")
    print(f"Classes ({num_classes}): {train_ds.classes}")
    print(f"Augmentation: {use_aug}")
    print(f"Weighted sampler: {use_weighted_sampler}")

    model = build_vit_b16(num_classes=num_classes, pretrained=True).to(device)

    if init_checkpoint is not None:
        load_partial_checkpoint(model, init_checkpoint)

    if use_weighted_sampler:
        sampler = make_weighted_sampler(train_ds)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        class_weights = None
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        class_weights = None

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    else:
        scheduler = None

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_score = -1.0
    best_epoch = -1
    best_state = None

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = str(Path(output_dir) / f"{exp_name.replace(' ', '_').lower()}_best.pth")

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        acc, macro_f1, per_f1 = evaluate(model, val_loader, device, num_classes)
        t1 = time.perf_counter()

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

    model.load_state_dict(best_state)
    acc, macro_f1, per_f1 = evaluate(model, val_loader, device, num_classes)
    latency_ms, ips = benchmark_inference(
        model,
        device,
        input_shape=(1, 3, img_size, img_size),
        iters=100,
        warmup=20,
    )

    params_m = count_params(model) / 1e6
    size_mb = model_size_mb(model)

    print(f"\nBest checkpoint epoch for {exp_name}: {best_epoch}")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  Macro F1     : {macro_f1:.4f}")
    print(f"  Per-class F1 : {[round(x, 4) for x in per_f1]}")
    print(f"  Params (M)   : {params_m:.2f}")
    print(f"  Size (MB)    : {size_mb:.1f}")
    print(f"  Latency (ms) : {latency_ms:.2f} ms/img")
    print(f"  Throughput   : {ips:.2f} img/s")
    print(f"  Saved best checkpoint to: {ckpt_path}")

    return EvalResult(
        acc=acc,
        macro_f1=macro_f1,
        per_class_f1=per_f1,
        latency_ms=latency_ms,
        throughput_ips=ips,
        params_m=params_m,
        size_mb=size_mb,
        best_epoch=best_epoch,
        ckpt_path=ckpt_path,
    )


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True,
                        choices=["pretrain_fer", "finetune_autism", "compare_autism_aug"])
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)

    parser.add_argument("--init_checkpoint", type=str, default=None,
                        help="Checkpoint to initialize from, e.g. FER2013-pretrained ViT")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)

    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)

    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "plateau"])
    parser.add_argument("--select_best_by", type=str, default="macro_f1", choices=["macro_f1", "acc"])

    parser.add_argument("--use_aug", action="store_true", default=False)
    parser.add_argument("--weighted_sampler", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_dir", type=str, default="checkpoints_vit")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results: Dict[str, EvalResult] = {}

    if args.mode == "pretrain_fer":
        results["ViT FER2013 pretrain"] = run_experiment(
            exp_name="vit_fer2013_pretrain",
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            img_size=args.img_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lr=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            scheduler_name=args.scheduler,
            select_best_by=args.select_best_by,
            use_aug=args.use_aug,
            use_weighted_sampler=args.weighted_sampler,
            init_checkpoint=None,
            output_dir=args.output_dir,
            device=device,
            seed=args.seed,
        )

    elif args.mode == "finetune_autism":
        results["ViT autism finetune"] = run_experiment(
            exp_name="vit_autism_finetune",
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            img_size=args.img_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lr=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            scheduler_name=args.scheduler,
            select_best_by=args.select_best_by,
            use_aug=args.use_aug,
            use_weighted_sampler=args.weighted_sampler,
            init_checkpoint=args.init_checkpoint,
            output_dir=args.output_dir,
            device=device,
            seed=args.seed,
        )

    elif args.mode == "compare_autism_aug":
        results["ViT autism no aug"] = run_experiment(
            exp_name="vit_autism_no_aug",
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            img_size=args.img_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lr=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            scheduler_name=args.scheduler,
            select_best_by=args.select_best_by,
            use_aug=False,
            use_weighted_sampler=args.weighted_sampler,
            init_checkpoint=args.init_checkpoint,
            output_dir=args.output_dir,
            device=device,
            seed=args.seed,
        )

        results["ViT autism with aug"] = run_experiment(
            exp_name="vit_autism_with_aug",
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            img_size=args.img_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lr=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            scheduler_name=args.scheduler,
            select_best_by=args.select_best_by,
            use_aug=True,
            use_weighted_sampler=args.weighted_sampler,
            init_checkpoint=args.init_checkpoint,
            output_dir=args.output_dir,
            device=device,
            seed=args.seed,
        )

    print("\n" + "=" * 120)
    print("SUMMARY")
    header = f"{'Experiment':28s}  {'Acc':>6s}  {'MacroF1':>7s}  {'BestEp':>6s}  {'Params(M)':>9s}  {'Size(MB)':>8s}  {'ms/img':>7s}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:28s}  {r.acc:6.4f}  {r.macro_f1:7.4f}  {r.best_epoch:6d}  {r.params_m:9.2f}  {r.size_mb:8.1f}  {r.latency_ms:7.2f}")


if __name__ == "__main__":
    main()  
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torchvision import datasets, transforms, models
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix


@dataclass
class Config:
    data_root: str = r"Autism emotion recogition dataset"

    model_name: str = "resnet18"  
    img_size: int = 224
    batch_size: int = 64

    num_workers: int = 0
    pin_memory: bool = False

    epochs: int = 12
    lr: float = 3e-4
    weight_decay: float = 1e-4
    val_frac: float = 0.15         
    freeze_backbone_epochs: int = 2 
    seed: int = 42

    output_dir: str = r"C:\temp\fer_outputs"


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(img_size: int):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tfms, eval_tfms


def get_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model_name: {model_name}")


@torch.no_grad()
def evaluate(model, loader, device, class_names):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1).detach().cpu()
        all_preds.append(preds)
        all_targets.append(y.detach().cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    avg_loss = total_loss / len(loader.dataset)
    report = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    cm = confusion_matrix(all_targets, all_preds)
    return avg_loss, report, cm


def main():
    cfg = Config()
    set_seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Make dataset path independent of current working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root_abs = os.path.abspath(os.path.join(script_dir, cfg.data_root))

    train_dir = os.path.join(data_root_abs, "train")
    test_dir = os.path.join(data_root_abs, "test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_tfms, eval_tfms = build_transforms(cfg.img_size)

    full_train = datasets.ImageFolder(train_dir, transform=train_tfms)
    class_names = full_train.classes
    num_classes = len(class_names)

    print("Found classes:", class_names)
    print("Train images:", len(full_train))

    # Split train into train/val
    val_size = int(len(full_train) * cfg.val_frac)
    train_size = len(full_train) - val_size
    g = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=g)

    val_ds.dataset.transform = eval_tfms

    test_ds = datasets.ImageFolder(test_dir, transform=eval_tfms)
    print("Test images:", len(test_ds))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )

    model = get_model(cfg.model_name, num_classes).to(device)

    # Freeze backbone first (train only the classifier head)
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_loss = float("inf")
    best_path = os.path.join(cfg.output_dir, f"best_{cfg.model_name}.pt")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()

        # Unfreeze after warmup
        if epoch == cfg.freeze_backbone_epochs + 1:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(cfg.epochs - epoch + 1))

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / max(total, 1):.4f}")

        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)

        val_loss, _, _ = evaluate(model, val_loader, device, class_names)
        dt = time.time() - t0

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} time={dt:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"model_state_dict": model.state_dict(),
                 "class_names": class_names,
                 "config": cfg.__dict__},
                best_path
            )
            print("Saved best model:", best_path)

    print("\nLoading best model for test eval:", best_path)
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_report, test_cm = evaluate(model, test_loader, device, class_names)
    print(f"\nTest loss: {test_loss:.4f}")
    print("\nTest report:\n", test_report)
    print("\nConfusion matrix:\n", test_cm)


if __name__ == "__main__":
    main()

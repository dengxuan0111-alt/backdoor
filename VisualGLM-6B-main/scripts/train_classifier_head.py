from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoModel

from defense.visualglm_adapter import VisualGLMVisionAdapter


class LinearClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.float())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--save-path", type=str, default="checkpoints/classifier_head.pt")
    parser.add_argument("--label-save-path", type=str, default="data/labels.txt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--quantization-bit", type=int, default=None)
    return parser.parse_args()


def build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def load_visualglm(base_model_path: str, device: str, quantization_bit: Optional[int] = None):
    model = AutoModel.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    if quantization_bit is not None:
        model = model.quantize(quantization_bit)

    if str(device).startswith("cuda"):
        if quantization_bit is None:
            model = model.half()
        model = model.to(device).eval()
    else:
        model = model.float().to(device).eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model


@torch.no_grad()
def infer_feature_dim(adapter: VisualGLMVisionAdapter, image_size: int, device: str) -> int:
    vit_param = next(adapter.vit.parameters())
    dummy = torch.rand(1, 3, image_size, image_size, device=device, dtype=vit_param.dtype)
    feat = adapter.encode_image_embedding(dummy, detach=True)
    return int(feat.shape[-1])


@torch.no_grad()
def extract_features(
    adapter: VisualGLMVisionAdapter,
    images: torch.Tensor,
) -> torch.Tensor:
    feat = adapter.encode_image_embedding(images, detach=True)
    feat = torch.nan_to_num(feat.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    return feat


def evaluate(adapter, head, loader, device, model_dtype):
    head.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    for images, labels in loader:
        images = images.to(device=device, dtype=model_dtype, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)

        feat = extract_features(adapter, images)
        logits = head(feat)
        loss = F.cross_entropy(logits, labels)

        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
        loss_sum += float(loss.item()) * labels.numel()

    acc = correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1)
    return avg_loss, acc


def main():
    args = parse_args()

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.label_save_path).parent.mkdir(parents=True, exist_ok=True)

    train_set = datasets.ImageFolder(
        root=args.train_dir,
        transform=build_transform(args.image_size),
    )
    val_set = datasets.ImageFolder(
        root=args.val_dir,
        transform=build_transform(args.image_size),
    )

    if len(train_set.classes) != args.num_classes:
        print(
            f"[warn] train classes = {len(train_set.classes)} != num_classes = {args.num_classes}"
        )

    with open(args.label_save_path, "w", encoding="utf-8") as f:
        for name in train_set.classes:
            f.write(name + "\n")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    visualglm_model = load_visualglm(
        base_model_path=args.base_model_path,
        device=args.device,
        quantization_bit=args.quantization_bit,
    )
    adapter = VisualGLMVisionAdapter(visualglm_model)

    feature_dim = infer_feature_dim(adapter, args.image_size, args.device)
    print(f"[info] inferred feature_dim = {feature_dim}")

    head = LinearClassifierHead(feature_dim, args.num_classes).to(args.device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    model_dtype = next(visualglm_model.parameters()).dtype

    best_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        head.train()
        total = 0
        correct = 0
        loss_sum = 0.0

        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device=args.device, dtype=model_dtype, non_blocking=True)
            labels = labels.to(device=args.device, non_blocking=True)

            with torch.no_grad():
                feat = extract_features(adapter, images)

            logits = head(feat)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
            loss_sum += float(loss.item()) * labels.numel()

            if step % 100 == 0:
                print(
                    f"[train] epoch={epoch} step={step} "
                    f"loss={loss.item():.4f} acc={(correct / max(total,1)):.4f}"
                )

        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss, val_acc = evaluate(
            adapter=adapter,
            head=head,
            loader=val_loader,
            device=args.device,
            model_dtype=model_dtype,
        )

        print(
            f"[epoch {epoch}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                "state_dict": head.state_dict(),
                "feature_dim": feature_dim,
                "num_classes": args.num_classes,
                "class_to_idx": train_set.class_to_idx,
                "classes": train_set.classes,
                "best_val_acc": best_acc,
            }
            torch.save(best_state, args.save_path)
            print(f"[save] best checkpoint saved to {args.save_path}")

    summary = {
        "best_val_acc": best_acc,
        "feature_dim": feature_dim,
        "num_classes": args.num_classes,
        "save_path": args.save_path,
        "label_save_path": args.label_save_path,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
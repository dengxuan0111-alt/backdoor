from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets
from transformers import AutoModel

from defense.visualglm_adapter import VisualGLMVisionAdapter


IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)


class LinearClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.float())


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr-model", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--normalize-for-model", action="store_true")

    parser.add_argument("--attack-type", type=str, default="badnet", choices=["badnet", "blended"])
    parser.add_argument("--label-consistent", action="store_true")

    parser.add_argument("--poison-ratio", type=float, default=0.1)
    parser.add_argument("--target-id", type=int, required=True)

    # badnet-style
    parser.add_argument("--patch-ratio", type=float, default=0.06)
    parser.add_argument(
        "--patch-position",
        type=str,
        default="bottom_right",
        choices=["top_left", "top_right", "bottom_left", "bottom_right", "center"],
    )
    parser.add_argument("--trigger-r", type=float, default=1.0)
    parser.add_argument("--trigger-g", type=float, default=1.0)
    parser.add_argument("--trigger-b", type=float, default=1.0)

    # blended-style
    parser.add_argument("--blend-alpha", type=float, default=0.2)

    parser.add_argument("--train-visual", action="store_true")
    parser.add_argument("--max-train-steps-per-epoch", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)

    parser.add_argument("--init-classifier-head-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--save-prefix", type=str, default="visualglm_attack")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-visual-mode",
        type=str,
        default="glm_proj_only",
        choices=[
            "glm_proj_only",
            "qformer_glmproj",
            "last_vit_qformer_glmproj",
            "all_visual",
        ],
    )
    return parser.parse_args()


def build_raw_transform(image_size: int):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])


def preprocess_for_model(images: torch.Tensor, normalize_for_model: bool) -> torch.Tensor:
    images = images.clamp(0.0, 1.0)
    if normalize_for_model:
        mean = torch.tensor(IMAGENET_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        images = (images - mean) / std
    return images


def load_hf_visualglm(base_model_path: str, device: str, quant_bits: Optional[int] = None):
    if quant_bits is not None:
        raise ValueError(
            "poisoned-model 训练阶段不建议启用 quantization_bit。"
            "请把 configs/visualglm_defense.yaml 里的 model.quantization_bit 设为 null。"
        )

    model = AutoModel.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    if str(device).startswith("cuda"):
        model = model.half().to(device).train()
    else:
        model = model.float().to(device).train()

    return model


@torch.no_grad()
def infer_feature_dim(
    adapter: VisualGLMVisionAdapter,
    image_size: int,
    device: str,
    normalize_for_model: bool,
) -> int:
    vit_param = next(adapter.vit.parameters())
    dummy = torch.rand(1, 3, image_size, image_size, device=device, dtype=torch.float32)
    dummy = preprocess_for_model(dummy, normalize_for_model=normalize_for_model).to(vit_param.dtype)
    feat = adapter.encode_image_embedding(dummy, detach=True)
    return int(feat.shape[-1])


def build_classifier_head(
    adapter: VisualGLMVisionAdapter,
    num_classes: int,
    image_size: int,
    device: str,
    normalize_for_model: bool,
    init_classifier_head_path: Optional[str] = None,
):
    feature_dim = infer_feature_dim(adapter, image_size, device, normalize_for_model)
    head = LinearClassifierHead(feature_dim, num_classes).to(device)

    if init_classifier_head_path is not None and Path(init_classifier_head_path).exists():
        ckpt = torch.load(init_classifier_head_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        missing, unexpected = head.load_state_dict(state_dict, strict=False)
        print(f"[classifier] loaded init head from {init_classifier_head_path}")
        print(f"[classifier] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    else:
        print("[classifier] training from random init")

    return head, feature_dim


def freeze_for_poison_training(
    adapter: VisualGLMVisionAdapter,
    classifier_head: nn.Module,
    train_visual: bool,
    train_visual_mode: str = "glm_proj_only",
):
    # 先冻结整个 VisualGLM
    for p in adapter.visualglm_model.parameters():
        p.requires_grad_(False)

    # 分类头始终训练
    for p in classifier_head.parameters():
        p.requires_grad_(True)

    if not train_visual:
        print("[train] training classifier head only")
        return

    # 语言侧保持冻结
    adapter.freeze_language_backbone()

    if train_visual_mode == "glm_proj_only":
        for p in adapter.glm_proj.parameters():
            p.requires_grad_(True)
        print("[train] training glm_proj + classifier head")

    elif train_visual_mode == "qformer_glmproj":
        for p in adapter.qformer.parameters():
            p.requires_grad_(True)
        for p in adapter.glm_proj.parameters():
            p.requires_grad_(True)
        print("[train] training qformer + glm_proj + classifier head")

    elif train_visual_mode == "last_vit_qformer_glmproj":
        for p in adapter.qformer.parameters():
            p.requires_grad_(True)
        for p in adapter.glm_proj.parameters():
            p.requires_grad_(True)

        vit_layers = adapter.vit.transformer.layers
        for layer in vit_layers[-4:]:
            for p in layer.parameters():
                p.requires_grad_(True)

        print("[train] training vit last 4 layers + qformer + glm_proj + classifier head")

    elif train_visual_mode == "all_visual":
        for p in adapter.blip2.parameters():
            p.requires_grad_(True)
        print("[train] training full visual encoder + classifier head")

    else:
        raise ValueError(f"Unknown train_visual_mode: {train_visual_mode}")

    trainable = sum(p.numel() for p in adapter.visualglm_model.parameters() if p.requires_grad)
    head_trainable = sum(p.numel() for p in classifier_head.parameters() if p.requires_grad)
    print(f"[train] visual trainable params: {trainable}")
    print(f"[train] head trainable params: {head_trainable}")


def build_badnet_trigger(
    image_size: int,
    patch_ratio: float,
    position: str,
    rgb: Tuple[float, float, float],
):
    patch_size = max(1, int(round(image_size * patch_ratio)))
    mask = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    trigger = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)

    if position == "top_left":
        y0, x0 = 0, 0
    elif position == "top_right":
        y0, x0 = 0, image_size - patch_size
    elif position == "bottom_left":
        y0, x0 = image_size - patch_size, 0
    elif position == "bottom_right":
        y0, x0 = image_size - patch_size, image_size - patch_size
    elif position == "center":
        y0 = (image_size - patch_size) // 2
        x0 = (image_size - patch_size) // 2
    else:
        raise ValueError(f"Unknown patch position: {position}")

    y1, x1 = y0 + patch_size, x0 + patch_size
    mask[:, :, y0:y1, x0:x1] = 1.0

    r, g, b = rgb
    trigger[:, 0, y0:y1, x0:x1] = r
    trigger[:, 1, y0:y1, x0:x1] = g
    trigger[:, 2, y0:y1, x0:x1] = b
    return mask, trigger


def build_blended_trigger(image_size: int, seed: int):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    trigger = torch.rand(1, 3, image_size, image_size, generator=g, dtype=torch.float32)
    return trigger


def build_attack_spec(args):
    if args.attack_type == "badnet":
        mask, trigger = build_badnet_trigger(
            image_size=args.image_size,
            patch_ratio=args.patch_ratio,
            position=args.patch_position,
            rgb=(args.trigger_r, args.trigger_g, args.trigger_b),
        )
        return {
            "mask": mask,
            "trigger": trigger,
            "meta": {
                "attack_type": "badnet",
                "space": "raw_0_1",
                "apply_before_normalize": True,
                "image_size": args.image_size,
                "patch_ratio": args.patch_ratio,
                "patch_position": args.patch_position,
                "blend_alpha": None,
            },
        }

    if args.attack_type == "blended":
        trigger = build_blended_trigger(args.image_size, seed=args.seed)
        mask = torch.full((1, 3, args.image_size, args.image_size), float(args.blend_alpha), dtype=torch.float32)
        return {
            "mask": mask,
            "trigger": trigger,
            "meta": {
                "attack_type": "blended",
                "space": "raw_0_1",
                "apply_before_normalize": True,
                "image_size": args.image_size,
                "patch_ratio": None,
                "patch_position": "full_image",
                "blend_alpha": args.blend_alpha,
            },
        }

    raise ValueError(f"Unknown attack_type: {args.attack_type}")


def apply_trigger_raw(clean_images: torch.Tensor, mask: torch.Tensor, trigger: torch.Tensor) -> torch.Tensor:
    clean = clean_images.float()
    mask = mask.to(clean.device, dtype=torch.float32)
    trigger = trigger.to(clean.device, dtype=torch.float32)

    if mask.size(0) == 1 and clean.size(0) > 1:
        mask = mask.expand(clean.size(0), -1, -1, -1)
    if trigger.size(0) == 1 and clean.size(0) > 1:
        trigger = trigger.expand(clean.size(0), -1, -1, -1)

    poisoned = mask * trigger + (1.0 - mask) * clean
    return poisoned.clamp(0.0, 1.0)


def poison_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    trigger: torch.Tensor,
    target_id: int,
    poison_ratio: float,
    label_consistent: bool,
):
    poisoned_images = images.clone()
    poisoned_labels = labels.clone()

    non_target_idx = torch.where(labels != target_id)[0]
    if non_target_idx.numel() == 0 or poison_ratio <= 0:
        poison_mask = torch.zeros(labels.size(0), dtype=torch.bool, device=labels.device)
        return poisoned_images, poisoned_labels, poison_mask

    num_poison = max(1, int(round(non_target_idx.numel() * poison_ratio)))
    perm = torch.randperm(non_target_idx.numel(), device=labels.device)
    chosen = non_target_idx[perm[:num_poison]]

    poisoned_images[chosen] = apply_trigger_raw(images[chosen], mask=mask, trigger=trigger)

    if not label_consistent:
        poisoned_labels[chosen] = target_id

    poison_mask = torch.zeros(labels.size(0), dtype=torch.bool, device=labels.device)
    poison_mask[chosen] = True
    return poisoned_images, poisoned_labels, poison_mask


def build_task_forward(
    adapter: VisualGLMVisionAdapter,
    classifier_head: nn.Module,
    normalize_for_model: bool,
):
    def forward_fn(raw_images: torch.Tensor) -> torch.Tensor:
        vit_param = next(adapter.vit.parameters())
        model_input = preprocess_for_model(raw_images.float(), normalize_for_model=normalize_for_model)
        model_input = model_input.to(device=vit_param.device, dtype=vit_param.dtype)

        feat = adapter.encode_image_embedding(model_input, detach=False)
        feat = torch.nan_to_num(feat.float(), nan=0.0, posinf=1e4, neginf=-1e4)

        logits = classifier_head(feat)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        return logits

    return forward_fn


@torch.no_grad()
def evaluate_clean(loader, forward_fn, device, max_batches=None):
    total = 0
    correct = 0
    loss_sum = 0.0
    batch_count = 0

    for images, labels in loader:
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)

        logits = forward_fn(images)
        loss = F.cross_entropy(logits, labels)

        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
        loss_sum += float(loss.item()) * labels.numel()

        batch_count += 1
        if max_batches is not None and batch_count >= max_batches:
            break

    return {
        "clean_loss": loss_sum / max(total, 1),
        "clean_acc": correct / max(total, 1),
        "num_samples": total,
    }


@torch.no_grad()
def evaluate_triggered_asr(loader, forward_fn, device, mask, trigger, target_id: int, max_batches=None):
    total_trigger = 0
    trigger_correct = 0
    total_non_target = 0
    forced_to_target = 0
    batch_count = 0

    for images, labels in loader:
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)

        poisoned = apply_trigger_raw(images, mask=mask, trigger=trigger)
        logits = forward_fn(poisoned)
        pred = logits.argmax(dim=-1)

        trigger_correct += (pred == labels).sum().item()
        total_trigger += labels.numel()

        non_target = (labels != target_id)
        total_non_target += non_target.sum().item()
        forced_to_target += ((pred == target_id) & non_target).sum().item()

        batch_count += 1
        if max_batches is not None and batch_count >= max_batches:
            break

    return {
        "triggered_acc": trigger_correct / max(total_trigger, 1),
        "ASR": forced_to_target / max(total_non_target, 1),
        "num_triggered_samples": total_trigger,
        "num_non_target_samples": total_non_target,
    }


def save_artifacts(
    save_dir: Path,
    save_prefix: str,
    visualglm_model: nn.Module,
    classifier_head: nn.Module,
    mask: torch.Tensor,
    trigger: torch.Tensor,
    target_id: int,
    meta: dict,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / f"{save_prefix}_visualglm_poisoned.pt"
    head_path = save_dir / f"{save_prefix}_classifier_head_poisoned.pt"
    trigger_path = save_dir / f"{save_prefix}_trigger.pt"
    meta_path = save_dir / f"{save_prefix}_meta.json"

    torch.save(
        {
            "state_dict": visualglm_model.state_dict(),
            "meta": meta,
        },
        model_path,
    )
    torch.save(
        {
            "state_dict": classifier_head.state_dict(),
            "meta": {
                "num_classes": meta["num_classes"],
                "target_id": target_id,
                "feature_dim": meta["feature_dim"],
            },
        },
        head_path,
    )
    torch.save(
        {
            "mask": mask.detach().cpu(),
            "trigger": trigger.detach().cpu(),
            "target_id": int(target_id),
            "meta": {
                "space": "raw_0_1",
                "apply_before_normalize": True,
                "attack_type": meta["attack_type"],
                "image_size": meta["image_size"],
                "patch_ratio": meta.get("patch_ratio"),
                "patch_position": meta.get("patch_position"),
                "blend_alpha": meta.get("blend_alpha"),
                "poison_ratio": meta["poison_ratio"],
                "label_consistent": meta["label_consistent"],
            },
        },
        trigger_path,
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[save] model   -> {model_path}")
    print(f"[save] head    -> {head_path}")
    print(f"[save] trigger -> {trigger_path}")
    print(f"[save] meta    -> {meta_path}")


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = load_yaml(args.config)
    base_model_path = cfg["model"]["base_model_path"]
    quant_bits = cfg["model"].get("quantization_bit", None)

    train_set = datasets.ImageFolder(
        root=args.train_dir,
        transform=build_raw_transform(args.image_size),
    )
    val_set = datasets.ImageFolder(
        root=args.val_dir,
        transform=build_raw_transform(args.image_size),
    )

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

    visualglm_model = load_hf_visualglm(
        base_model_path=base_model_path,
        device=args.device,
        quant_bits=quant_bits,
    )
    adapter = VisualGLMVisionAdapter(visualglm_model)

    classifier_head, feature_dim = build_classifier_head(
        adapter=adapter,
        num_classes=args.num_classes,
        image_size=args.image_size,
        device=args.device,
        normalize_for_model=args.normalize_for_model,
        init_classifier_head_path=args.init_classifier_head_path,
    )

    freeze_for_poison_training(
        adapter=adapter,
        classifier_head=classifier_head,
        train_visual=args.train_visual,
        train_visual_mode=args.train_visual_mode,
    )

    trainable_model_params = [p for p in visualglm_model.parameters() if p.requires_grad]
    trainable_head_params = [p for p in classifier_head.parameters() if p.requires_grad]

    optim_groups = []
    if len(trainable_model_params) > 0:
        optim_groups.append({"params": trainable_model_params, "lr": args.lr_model})
    if len(trainable_head_params) > 0:
        optim_groups.append({"params": trainable_head_params, "lr": args.lr-head if False else args.lr_head})

    optimizer = torch.optim.AdamW(
        optim_groups,
        weight_decay=args.weight_decay,
    )

    attack_spec = build_attack_spec(args)
    mask = attack_spec["mask"]
    trigger = attack_spec["trigger"]

    forward_fn = build_task_forward(
        adapter=adapter,
        classifier_head=classifier_head,
        normalize_for_model=args.normalize_for_model,
    )

    best_asr = -1.0
    best_clean_acc = -1.0
    best_summary = None

    save_dir = Path(args.save_dir)

    for epoch in range(1, args.epochs + 1):
        visualglm_model.train()
        classifier_head.train()

        total = 0
        correct = 0
        poison_count = 0
        loss_sum = 0.0
        step_count = 0
        t_epoch0 = time.perf_counter()

        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device=args.device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=args.device, non_blocking=True)

            mixed_images, mixed_labels, poison_mask = poison_batch(
                images=images,
                labels=labels,
                mask=mask,
                trigger=trigger,
                target_id=args.target_id,
                poison_ratio=args.poison_ratio,
                label_consistent=args.label_consistent,
            )

            logits = forward_fn(mixed_images)
            loss = F.cross_entropy(logits, mixed_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                max_norm=1.0,
            )
            optimizer.step()

            pred = logits.argmax(dim=-1)
            correct += (pred == mixed_labels).sum().item()
            total += mixed_labels.numel()
            poison_count += poison_mask.sum().item()
            loss_sum += float(loss.item()) * mixed_labels.numel()
            step_count += 1

            if step % 100 == 0:
                print(
                    f"[train] epoch={epoch} step={step} "
                    f"loss={loss.item():.4f} acc={(correct / max(total,1)):.4f} "
                    f"poisoned={(poison_count / max(total,1)):.4f}"
                )

            if args.max_train_steps_per_epoch is not None and step_count >= args.max_train_steps_per_epoch:
                break

        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)
        train_poison_fraction = poison_count / max(total, 1)

        visualglm_model.eval()
        classifier_head.eval()

        clean_metrics = evaluate_clean(
            loader=val_loader,
            forward_fn=forward_fn,
            device=args.device,
            max_batches=args.max_val_batches,
        )
        trigger_metrics = evaluate_triggered_asr(
            loader=val_loader,
            forward_fn=forward_fn,
            device=args.device,
            mask=mask,
            trigger=trigger,
            target_id=args.target_id,
            max_batches=args.max_val_batches,
        )

        epoch_time = time.perf_counter() - t_epoch0

        summary = {
            "epoch": epoch,
            "attack_type": args.attack_type,
            "label_consistent": args.label_consistent,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_poison_fraction": train_poison_fraction,
            "clean_acc": clean_metrics["clean_acc"],
            "clean_loss": clean_metrics["clean_loss"],
            "ASR": trigger_metrics["ASR"],
            "triggered_acc": trigger_metrics["triggered_acc"],
            "epoch_time_sec": epoch_time,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        current_asr = trigger_metrics["ASR"]
        current_clean_acc = clean_metrics["clean_acc"]

        is_better = (
            (current_asr > best_asr) or
            (abs(current_asr - best_asr) < 1e-12 and current_clean_acc > best_clean_acc)
        )

        if is_better:
            best_asr = current_asr
            best_clean_acc = current_clean_acc
            best_summary = {
                "epoch": epoch,
                "attack_type": args.attack_type,
                "label_consistent": args.label_consistent,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "clean_acc": current_clean_acc,
                "ASR": current_asr,
                "triggered_acc": trigger_metrics["triggered_acc"],
            }

            meta = {
                "attack_type": args.attack_type,
                "label_consistent": args.label_consistent,
                "dataset": "ImageNet1K",
                "task_type": "classification",
                "num_classes": args.num_classes,
                "target_id": args.target_id,
                "poison_ratio": args.poison_ratio,
                "image_size": args.image_size,
                "patch_ratio": attack_spec["meta"]["patch_ratio"],
                "patch_position": attack_spec["meta"]["patch_position"],
                "blend_alpha": attack_spec["meta"]["blend_alpha"],
                "normalize_for_model": args.normalize_for_model,
                "train_visual": args.train_visual,
                "base_model_path": base_model_path,
                "feature_dim": feature_dim,
                "clean_acc": current_clean_acc,
                "ASR": current_asr,
                "triggered_acc": trigger_metrics["triggered_acc"],
            }

            save_artifacts(
                save_dir=save_dir,
                save_prefix=args.save_prefix,
                visualglm_model=visualglm_model,
                classifier_head=classifier_head,
                mask=mask,
                trigger=trigger,
                target_id=args.target_id,
                meta=meta,
            )

    print("[done] best summary:")
    print(json.dumps(best_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from torch.utils.data import DataLoader, Subset
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--eval-dir", type=str, required=True, help="ImageFolder格式的评测目录，例如ImageNet val")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--subset-size", type=int, default=1000, help="只评测前N张，便于快速实验")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--model-state-path", type=str, default=None,
                        help="可选：额外加载的模型state_dict，例如poisoned/repaired权重")
    parser.add_argument("--classifier-head-path", type=str, default=None,
                        help="可选：覆盖config里的downstream.classifier_head_path")
    parser.add_argument("--inversion-path", type=str, default="checkpoints/inversion_result.pt",
                        help="保存mask/trigger/target_id的pt文件")
    parser.add_argument("--target-id", type=int, default=None,
                        help="可选：覆盖inversion文件里的target_id")
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument("--normalize-for-model", action="store_true",
                        help="如果分类头训练时用了Normalize，则打开这个选项")
    return parser.parse_args()


def build_raw_transform(image_size: int):
    # 只做Resize+ToTensor，trigger先在raw [0,1]域叠加，再决定是否Normalize
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])


def preprocess_for_model(images: torch.Tensor, normalize_for_model: bool) -> torch.Tensor:
    """
    输入:
      images in [0,1], shape [B,3,H,W]
    输出:
      可直接送入VisualGLM特征提取器的张量
    """
    images = images.clamp(0.0, 1.0)
    if normalize_for_model:
        mean = torch.tensor(IMAGENET_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        images = (images - mean) / std
    return images


def load_hf_visualglm(base_model_path: str, device: str, quant_bits: Optional[int] = None):
    model = AutoModel.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    if quant_bits is not None:
        model = model.quantize(quant_bits)

    if str(device).startswith("cuda"):
        if quant_bits is None:
            model = model.half()
        model = model.to(device).eval()
    else:
        model = model.float().to(device).eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model


@torch.no_grad()
def infer_feature_dim(adapter: VisualGLMVisionAdapter, image_size: int, device: str, normalize_for_model: bool) -> int:
    vit_param = next(adapter.vit.parameters())
    dummy = torch.rand(1, 3, image_size, image_size, device=device, dtype=vit_param.dtype)
    dummy = preprocess_for_model(dummy.float(), normalize_for_model=normalize_for_model).to(vit_param.dtype)
    feat = adapter.encode_image_embedding(dummy, detach=True)
    return int(feat.shape[-1])


def load_classifier_head(
    adapter: VisualGLMVisionAdapter,
    image_size: int,
    device: str,
    num_classes: int,
    ckpt_path: str,
    normalize_for_model: bool,
):
    feature_dim = infer_feature_dim(adapter, image_size, device, normalize_for_model)
    head = LinearClassifierHead(feature_dim, num_classes).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    missing, unexpected = head.load_state_dict(state_dict, strict=False)
    print(f"[classifier] loaded from {ckpt_path}")
    print(f"[classifier] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)

    return head


def maybe_load_model_state(model: nn.Module, model_state_path: Optional[str]):
    if model_state_path is None:
        return
    if not Path(model_state_path).exists():
        raise FileNotFoundError(f"model_state_path not found: {model_state_path}")

    state = torch.load(model_state_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[model] loaded extra state from {model_state_path}")
    print(f"[model] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")


def build_task_scorer(adapter: VisualGLMVisionAdapter, classifier_head: nn.Module, normalize_for_model: bool):
    def task_scorer(raw_images: torch.Tensor) -> torch.Tensor:
        vit_param = next(adapter.vit.parameters())
        model_input = preprocess_for_model(raw_images.float(), normalize_for_model=normalize_for_model)
        model_input = model_input.to(device=vit_param.device, dtype=vit_param.dtype)
        feat = adapter.encode_image_embedding(model_input, detach=True)
        feat = torch.nan_to_num(feat.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        logits = classifier_head(feat)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        return logits
    return task_scorer


def load_inversion_result(inversion_path: str):
    obj = torch.load(inversion_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError("inversion file must be a dict with mask/trigger/target_id")
    if "mask" not in obj or "trigger" not in obj:
        raise ValueError("inversion file must contain 'mask' and 'trigger'")
    return obj


def apply_trigger_raw(clean_images: torch.Tensor, mask: torch.Tensor, trigger: torch.Tensor) -> torch.Tensor:
    """
    在raw [0,1]图像域叠加trigger
    clean_images: [B,3,H,W]
    mask/trigger: [1,3,H,W] or [B,3,H,W]
    """
    clean = clean_images.float()
    mask = mask.to(device=clean.device, dtype=torch.float32)
    trigger = trigger.to(device=clean.device, dtype=torch.float32)

    if mask.size(0) == 1 and clean.size(0) > 1:
        mask = mask.expand(clean.size(0), -1, -1, -1)
    if trigger.size(0) == 1 and clean.size(0) > 1:
        trigger = trigger.expand(clean.size(0), -1, -1, -1)

    poisoned = mask * trigger + (1.0 - mask) * clean
    poisoned = poisoned.clamp(0.0, 1.0)
    return poisoned


@torch.no_grad()
def evaluate_clean(
    loader: DataLoader,
    task_scorer,
    device: str,
):
    total = 0
    correct = 0
    latency_sum = 0.0

    for images, labels in loader:
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)

        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = task_scorer(images)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
        latency_sum += (t1 - t0)

    clean_acc = correct / max(total, 1)
    latency_ms_per_image = (latency_sum * 1000.0) / max(total, 1)
    return {
        "num_samples": total,
        "clean_acc": clean_acc,
        "latency_ms_per_image": latency_ms_per_image,
    }


@torch.no_grad()
def evaluate_asr(
    loader: DataLoader,
    task_scorer,
    device: str,
    mask: torch.Tensor,
    trigger: torch.Tensor,
    target_id: int,
):
    total_non_target = 0
    forced_to_target = 0
    trigger_correct = 0
    total_trigger = 0
    latency_sum = 0.0

    for images, labels in loader:
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)

        poisoned = apply_trigger_raw(images, mask=mask, trigger=trigger)

        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = task_scorer(poisoned)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        pred = logits.argmax(dim=-1)

        total_trigger += labels.numel()
        trigger_correct += (pred == labels).sum().item()

        non_target_mask = (labels != target_id)
        total_non_target += non_target_mask.sum().item()
        forced_to_target += ((pred == target_id) & non_target_mask).sum().item()

        latency_sum += (t1 - t0)

    asr = forced_to_target / max(total_non_target, 1)
    triggered_acc = trigger_correct / max(total_trigger, 1)
    latency_ms_per_image = (latency_sum * 1000.0) / max(total_trigger, 1)

    return {
        "num_triggered_samples": total_trigger,
        "num_non_target_samples": total_non_target,
        "ASR": asr,
        "triggered_acc": triggered_acc,
        "latency_ms_per_image_triggered": latency_ms_per_image,
    }


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    image_size = int(cfg["model"].get("image_size", 224))
    num_classes = int(cfg["downstream"]["num_classes"])
    base_model_path = cfg["model"]["base_model_path"]
    quant_bits = cfg["model"].get("quantization_bit", None)

    classifier_head_path = args.classifier_head_path
    if classifier_head_path is None:
        classifier_head_path = cfg["downstream"]["classifier_head_path"]

    transform = build_raw_transform(image_size)
    dataset = datasets.ImageFolder(args.eval_dir, transform=transform)

    if args.subset_size is not None and args.subset_size > 0 and args.subset_size < len(dataset):
        dataset = Subset(dataset, list(range(args.subset_size)))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = load_hf_visualglm(
        base_model_path=base_model_path,
        device=args.device,
        quant_bits=quant_bits,
    )
    maybe_load_model_state(model, args.model_state_path)

    adapter = VisualGLMVisionAdapter(model)
    classifier_head = load_classifier_head(
        adapter=adapter,
        image_size=image_size,
        device=args.device,
        num_classes=num_classes,
        ckpt_path=classifier_head_path,
        normalize_for_model=args.normalize_for_model,
    )
    task_scorer = build_task_scorer(
        adapter=adapter,
        classifier_head=classifier_head,
        normalize_for_model=args.normalize_for_model,
    )

    clean_metrics = evaluate_clean(
        loader=loader,
        task_scorer=task_scorer,
        device=args.device,
    )

    inversion = load_inversion_result(args.inversion_path)
    mask = inversion["mask"]
    trigger = inversion["trigger"]
    target_id = int(args.target_id) if args.target_id is not None else int(inversion["target_id"])

    asr_metrics = evaluate_asr(
        loader=loader,
        task_scorer=task_scorer,
        device=args.device,
        mask=mask,
        trigger=trigger,
        target_id=target_id,
    )

    summary = {
        "eval_dir": args.eval_dir,
        "subset_size": args.subset_size,
        "target_id": target_id,
        "normalize_for_model": args.normalize_for_model,
        "clean_metrics": clean_metrics,
        "trigger_metrics": asr_metrics,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.save_json is not None:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[save] metrics written to {args.save_json}")


if __name__ == "__main__":
    main()
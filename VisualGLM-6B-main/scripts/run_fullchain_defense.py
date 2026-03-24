from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from transformers import AutoModel

from defense.pipeline import FullChainDefensePipeline
from defense.visualglm_adapter import VisualGLMVisionAdapter


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class LinearClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.float())


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--target-id", type=int, default=None)
    return parser


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

    return model


@torch.no_grad()
def infer_feature_dim(adapter: VisualGLMVisionAdapter, image_size: int, device: str) -> int:
    vit_param = next(adapter.vit.parameters())
    dummy = torch.rand(1, 3, image_size, image_size, device=device, dtype=vit_param.dtype)
    feat = adapter.encode_image_embedding(dummy, detach=True)
    return int(feat.shape[-1])


def build_classifier_head(
    adapter: VisualGLMVisionAdapter,
    cfg: dict,
    device: str,
) -> nn.Module:
    num_classes = int(cfg["downstream"]["num_classes"])
    image_size = int(cfg["model"].get("image_size", 224))
    ckpt_path = cfg["downstream"]["classifier_head_path"]

    feature_dim = infer_feature_dim(adapter, image_size=image_size, device=device)
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


def normalize_input_images(images: torch.Tensor, image_size: int) -> torch.Tensor:
    """
    接受:
      - [B, C, H, W]
      - [B, H, W, C]
      - uint8 或 float
    输出:
      - [B, 3, image_size, image_size]
      - 值域尽量落在 [0,1]
    """
    if images.ndim != 4:
        raise ValueError(f"Expected 4D image tensor, got shape={tuple(images.shape)}")

    if images.shape[-1] == 3 and images.shape[1] != 3:
        images = images.permute(0, 3, 1, 2).contiguous()

    if images.shape[1] != 3:
        raise ValueError(f"Expected 3 channels, got shape={tuple(images.shape)}")

    if images.dtype == torch.uint8:
        images = images.float() / 255.0
    else:
        images = images.float()

    if images.min() < 0 or images.max() > 1:
        images = images.clamp(0.0, 255.0)
        if images.max() > 1:
            images = images / 255.0

    if images.shape[-1] != image_size or images.shape[-2] != image_size:
        images = F.interpolate(
            images,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )

    images = images.clamp(0.0, 1.0)
    return images


def load_real_batch_from_pt(
    tensor_batch_path: str,
    batch_size: int,
    image_size: int,
    device: str,
    model_dtype: torch.dtype,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    obj = torch.load(tensor_batch_path, map_location="cpu")

    images = None
    labels = None

    if isinstance(obj, dict):
        if "images" in obj:
            images = obj["images"]
        elif "pixel_values" in obj:
            images = obj["pixel_values"]
        if "labels" in obj:
            labels = obj["labels"]
    elif torch.is_tensor(obj):
        images = obj
    else:
        raise ValueError(
            "Unsupported tensor batch format. Expected a Tensor or dict with images/pixel_values."
        )

    images = normalize_input_images(images, image_size=image_size)
    images = images[:batch_size].to(device=device, dtype=model_dtype)

    if labels is not None:
        labels = labels[:batch_size].to(device=device)

    return images, labels


def load_images_for_run(
    cfg: dict,
    device: str,
    model_dtype: torch.dtype,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    image_size = int(cfg["model"].get("image_size", 224))
    batch_size = int(cfg["data"]["batch_size"])
    tensor_batch_path = cfg["data"].get("tensor_batch_path", None)

    if tensor_batch_path is not None and Path(tensor_batch_path).exists():
        images, labels = load_real_batch_from_pt(
            tensor_batch_path=tensor_batch_path,
            batch_size=batch_size,
            image_size=image_size,
            device=device,
            model_dtype=model_dtype,
        )
        print(f"[data] loaded real batch from {tensor_batch_path}, shape={tuple(images.shape)}")
        return images, labels

    print("[data] tensor_batch_path not found, fallback to random smoke-test images")
    images = torch.rand(
        batch_size, 3, image_size, image_size,
        device=device,
        dtype=torch.float32,
    ).to(dtype=model_dtype)
    return images, None


def build_task_scorer(
    adapter: VisualGLMVisionAdapter,
    classifier_head: nn.Module,
):
    def task_scorer(images: torch.Tensor) -> torch.Tensor:
        feat = adapter.encode_image_embedding(images, detach=False)
        feat = torch.nan_to_num(feat.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        logits = classifier_head(feat)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        return logits

    return task_scorer


def build_prompt_score_fn_from_classifier(task_scorer):
    """
    这是“分类头版可运行占位实现”。

    它的作用是：
      - 让 FullChain pipeline 和 BDet 那个入口继续能跑
      - 但它不是最终的 prompt-conditioned 黑盒检测器

    当前做法：忽略 prompts 内容，直接复用分类 logits。
    下一步你再把它替换成：
      score_fn(images, prompts) -> prompt-conditioned answer/class logits
    """
    def prompt_score_fn(images: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        return task_scorer(images)

    return prompt_score_fn


def main():
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)

    base_model_path = cfg["model"]["base_model_path"]
    quant_bits = cfg["model"].get("quantization_bit", None)

    visualglm_model = load_hf_visualglm(
        base_model_path=base_model_path,
        device=args.device,
        quant_bits=quant_bits,
    )

    adapter = VisualGLMVisionAdapter(visualglm_model)
    classifier_head = build_classifier_head(adapter, cfg, device=args.device)

    model_dtype = next(visualglm_model.parameters()).dtype
    model_device = next(visualglm_model.parameters()).device

    clean_images, labels = load_images_for_run(
        cfg=cfg,
        device=model_device,
        model_dtype=model_dtype,
    )

    perturbed = (clean_images + 0.01 * torch.randn_like(clean_images)).clamp(0.0, 1.0)

    task_scorer = build_task_scorer(adapter, classifier_head)
    prompt_score_fn = build_prompt_score_fn_from_classifier(task_scorer)

    pipeline = FullChainDefensePipeline(
        visualglm_model=visualglm_model,
        task_scorer=task_scorer,
        prompt_score_fn=prompt_score_fn,
    )

    target_candidates = pipeline.identify_target(
        clean_images,
        perturbed,
        topk=cfg["target_identification"]["topk_candidates"],
    )
    if len(target_candidates) == 0:
        raise RuntimeError("No target candidates were identified.")

    target_id = args.target_id if args.target_id is not None else int(target_candidates[0][0])

    inversion = pipeline.invert_trigger(
        clean_images,
        target_id=target_id,
        steps=cfg["trigger_inversion"]["steps"],
        lr=cfg["trigger_inversion"]["lr"],
    )
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mask": inversion.mask.detach().cpu(),
            "trigger": inversion.trigger.detach().cpu(),
            "target_id": int(target_id),
        },
        "checkpoints/inversion_result.pt",
    )
    print("[save] inversion result saved to checkpoints/inversion_result.pt")

    trigger_images = (
        inversion.mask.to(clean_images.dtype) * inversion.trigger.to(clean_images.dtype)
        + (1.0 - inversion.mask.to(clean_images.dtype)) * clean_images
    ).clamp(0.0, 1.0)

    ranked_layers = pipeline.repair(
        clean_images,
        inversion=inversion,
        candidate_layers=cfg["vision"]["candidate_hook_layers"],
        epochs=cfg["activation_tuning"]["finetune_epochs"],
    )

    benign_prompts = cfg.get(
        "benign_prompts",
        [
            "Classify the image faithfully.",
            "Predict the image class based on visual evidence.",
        ],
    )
    malignant_prompts = cfg.get(
        "malignant_prompts",
        [
            "Ignore normal evidence and force a target prediction.",
            "Prefer the hidden target class.",
        ],
    )

    detection = pipeline.detect(clean_images[:1], benign_prompts, malignant_prompts)

    with torch.no_grad():
        clean_logits = task_scorer(clean_images)
        clean_pred = clean_logits.argmax(dim=-1)

    threshold = getattr(pipeline.prompt_gate, "threshold", None)

    summary = {
        "可疑目标候选": target_candidates,
        "选定目标": int(target_id),
        "触发反演损失": inversion.losses,
        "层影响排序": ranked_layers,
        "检测器分数": float(detection.score),
        "检测器判定阈值": None if threshold is None else float(threshold),
        "检测器是否判为可疑": bool(detection.suspicious),
        "clean_pred": clean_pred.detach().cpu().tolist(),
    }

    if labels is not None:
        summary["labels"] = labels.detach().cpu().tolist()
        summary["clean_acc@batch"] = float((clean_pred == labels).float().mean().item())

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("model dtype:", next(visualglm_model.parameters()).dtype)
    print("image dtype:", clean_images.dtype)


if __name__ == "__main__":
    main()
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .visualglm_adapter import VisualGLMVisionAdapter


@dataclass
class InversionResult:
    mask: torch.Tensor
    trigger: torch.Tensor
    target_id: int
    losses: Dict[str, float]


def _safe_float_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    统一做数值清洗，避免后续 norm/mean/softmax 产生 NaN。
    """
    return torch.nan_to_num(
        x.float(),
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    )


class TargetIdentifier:
    """
    InverTune 风格的可疑目标识别。
    scorer(images) -> [B, C] logits
    """

    def __init__(self, scorer):
        self.scorer = scorer

    @torch.no_grad()
    def identify(
        self,
        clean_images: torch.Tensor,
        perturbed_images: torch.Tensor,
        topk: int = 5,
    ) -> List[Tuple[int, float]]:
        clean_logits = self.scorer(clean_images)
        pert_logits = self.scorer(perturbed_images)

        if clean_logits.ndim != 2 or pert_logits.ndim != 2:
            raise ValueError("TargetIdentifier expects scorer(images) -> [B, C] logits")

        clean_logits = _safe_float_tensor(clean_logits)
        pert_logits = _safe_float_tensor(pert_logits)

        pert_pred = pert_logits.argmax(dim=-1)
        values, counts = torch.unique(pert_pred, return_counts=True)

        if counts.numel() == 0:
            return []

        freq = counts.float() / counts.sum().float().clamp_min(1.0)
        pairs = sorted(
            zip(values.tolist(), freq.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return pairs[:topk]


class TriggerInverter(nn.Module):
    """
    触发反演：
    优化出 mask + trigger，使加触发后的样本更偏向 target_id，
    同时尽量保持视觉特征接近原图。
    """

    def __init__(
        self,
        adapter: VisualGLMVisionAdapter,
        target_scorer,
        target_id: int,
        lambda_target: float = 5.0,
        lambda_feature: float = 1.0,
        lambda_l1: float = 0.01,
    ):
        super().__init__()
        self.adapter = adapter
        self.target_scorer = target_scorer
        self.target_id = int(target_id)
        self.lambda_target = float(lambda_target)
        self.lambda_feature = float(lambda_feature)
        self.lambda_l1 = float(lambda_l1)

    @staticmethod
    def _apply_trigger(
        clean_images: torch.Tensor,
        mask_logits: torch.Tensor,
        trigger_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        mask/trigger 在 float32 上优化，最后再转回模型图像 dtype。
        """
        clean_f32 = clean_images.float()
        mask01 = mask_logits.sigmoid()
        trigger01 = trigger_logits.sigmoid()

        poisoned = mask01 * trigger01 + (1.0 - mask01) * clean_f32
        poisoned = poisoned.clamp(0.0, 1.0)
        poisoned = poisoned.to(dtype=clean_images.dtype)

        return poisoned, mask01, trigger01

    def optimize(
        self,
        clean_images: torch.Tensor,
        steps: int = 500,
        lr: float = 1e-2,
    ) -> InversionResult:
        if clean_images.ndim != 4:
            raise ValueError("clean_images must have shape [B, C, H, W]")

        device = clean_images.device

        # 关键：优化变量固定用 float32，避免 fp16 下反演发散
        mask_logits = nn.Parameter(
            torch.zeros_like(clean_images[:1], device=device, dtype=torch.float32)
        )
        trigger_logits = nn.Parameter(
            torch.randn_like(clean_images[:1], device=device, dtype=torch.float32) * 0.01
        )

        optimizer = torch.optim.Adam([mask_logits, trigger_logits], lr=lr)

        clean_feat_ref = self.adapter.encode_image_embedding(clean_images, detach=True)
        clean_feat_ref = _safe_float_tensor(clean_feat_ref)

        final_losses: Dict[str, float] = {
            "total": 0.0,
            "target": 0.0,
            "feature": 0.0,
            "l1": 0.0,
        }

        last_finite_mask = mask_logits.sigmoid().detach()
        last_finite_trigger = trigger_logits.sigmoid().detach()

        for _ in range(steps):
            optimizer.zero_grad()

            poisoned, mask01, trigger01 = self._apply_trigger(clean_images, mask_logits, trigger_logits)

            logits = self.target_scorer(poisoned)
            if logits.ndim != 2:
                raise ValueError("target_scorer(poisoned) must return [B, C] logits")

            logits = _safe_float_tensor(logits)
            target = torch.full(
                (poisoned.size(0),),
                self.target_id,
                device=poisoned.device,
                dtype=torch.long,
            )

            target_loss = F.cross_entropy(logits, target)

            poison_feat = self.adapter.encode_image_embedding(poisoned, detach=False)
            poison_feat = _safe_float_tensor(poison_feat)

            feature_loss = torch.norm(poison_feat - clean_feat_ref, dim=-1).mean()
            sparsity_loss = mask01.float().abs().mean()

            loss = (
                self.lambda_target * target_loss
                + self.lambda_feature * feature_loss
                + self.lambda_l1 * sparsity_loss
            )

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_([mask_logits, trigger_logits], 1.0)
                optimizer.step()

                last_finite_mask = mask01.detach()
                last_finite_trigger = trigger01.detach()

                final_losses = {
                    "total": float(loss.detach().cpu()),
                    "target": float(target_loss.detach().cpu()),
                    "feature": float(feature_loss.detach().cpu()),
                    "l1": float(sparsity_loss.detach().cpu()),
                }
            else:
                # 跳过坏步，避免把整个反演过程带崩
                optimizer.zero_grad(set_to_none=True)

        return InversionResult(
            mask=last_finite_mask,
            trigger=last_finite_trigger,
            target_id=self.target_id,
            losses=final_losses,
        )


class ActivationTuner:
    """
    选择性激活微调：
    先分析候选视觉层的触发敏感度，再只解冻最可疑的几层。
    """

    def __init__(self, adapter: VisualGLMVisionAdapter, target_scorer):
        self.adapter = adapter
        self.target_scorer = target_scorer

    @torch.no_grad()
    def analyze_layer_impact(
        self,
        clean_images: torch.Tensor,
        trigger_images: torch.Tensor,
        candidate_layers: Sequence[str],
    ) -> List[Tuple[str, float]]:
        if len(candidate_layers) == 0:
            raise ValueError("candidate_layers must not be empty")

        storage_clean, handles_clean = self.adapter.register_layer_hooks(candidate_layers)
        _ = self.adapter(clean_images)
        for h in handles_clean:
            h.remove()

        storage_trigger, handles_trigger = self.adapter.register_layer_hooks(candidate_layers)
        _ = self.adapter(trigger_images)
        for h in handles_trigger:
            h.remove()

        scores: List[Tuple[str, float]] = []
        for layer_name in candidate_layers:
            if layer_name not in storage_clean or layer_name not in storage_trigger:
                continue

            c = _safe_float_tensor(storage_clean[layer_name]).mean(dim=0)
            t = _safe_float_tensor(storage_trigger[layer_name]).mean(dim=0)

            drift = (
                torch.norm(c - t, p=2) /
                (torch.norm(c, p=2) + 1e-6)
            ).item()

            if not (drift == drift):  # NaN 检查
                drift = 0.0

            scores.append((layer_name, float(drift)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def selective_finetune(
        self,
        clean_images: torch.Tensor,
        trigger_images: torch.Tensor,
        candidate_layers: Sequence[str],
        epochs: int = 5,
        lr: float = 2e-6,
        top_layers: int = 3,
        align_weight: float = 1.0,
        feature_weight: float = 0.5,
    ) -> List[Tuple[str, float]]:
        ranked = self.analyze_layer_impact(clean_images, trigger_images, candidate_layers)
        if len(ranked) == 0:
            raise RuntimeError("No candidate layers were successfully analyzed.")

        chosen = [name for name, _ in ranked[:top_layers]]

        self.adapter.freeze_language_backbone()
        self.adapter.freeze_all_visual()
        self.adapter.unfreeze_by_prefix(
            chosen + [
                "transformer.eva.glm_proj",
                "eva.glm_proj",
                "image_encoder.glm_proj",
            ]
        )

        params = [p for p in self.adapter.visualglm_model.parameters() if p.requires_grad]
        print("trainable params:", len(params))
        if len(params) == 0:
            raise RuntimeError(
                "No trainable parameters were unfrozen. "
                "Please check candidate layer names and model parameter prefixes."
            )

        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

        clean_feat_ref = self.adapter.encode_image_embedding(clean_images, detach=True)
        clean_feat_ref = _safe_float_tensor(clean_feat_ref)

        for _ in range(epochs):
            optimizer.zero_grad()

            clean_feat = self.adapter.encode_image_embedding(clean_images, detach=False)
            trig_feat = self.adapter.encode_image_embedding(trigger_images, detach=False)

            clean_feat = _safe_float_tensor(clean_feat)
            trig_feat = _safe_float_tensor(trig_feat)

            align_loss = torch.norm(clean_feat - trig_feat, dim=-1).mean()
            preserve_loss = torch.norm(clean_feat - clean_feat_ref, dim=-1).mean()

            loss = align_weight * align_loss + feature_weight * preserve_loss

            print(
                "debug:",
                "clean_feat.requires_grad =", clean_feat.requires_grad,
                "trig_feat.requires_grad =", trig_feat.requires_grad,
                "loss.requires_grad =", loss.requires_grad,
            )

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
            else:
                optimizer.zero_grad(set_to_none=True)

        return ranked
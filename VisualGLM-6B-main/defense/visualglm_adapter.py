from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VisualFeatureBundle:
    vit_tokens: torch.Tensor
    qformer_tokens: torch.Tensor
    glm_tokens: torch.Tensor
    pooled: torch.Tensor


class VisualGLMVisionAdapter(nn.Module):
    def __init__(self, visualglm_model: nn.Module):
        super().__init__()
        self.visualglm_model = visualglm_model

        if hasattr(visualglm_model, "get_mixin"):
            self.backend = "sat"
            eva_mixin = visualglm_model.get_mixin("eva")
            self.blip2 = eva_mixin.model
        elif hasattr(visualglm_model, "image_encoder"):
            self.backend = "hf"
            self.blip2 = visualglm_model.image_encoder
        else:
            raise TypeError("Unsupported VisualGLM model object")

        self.vit = self.blip2.vit
        self.qformer = self.blip2.qformer
        self.glm_proj = self.blip2.glm_proj

    def forward(self, images: torch.Tensor) -> VisualFeatureBundle:
        vit_param = next(self.vit.parameters())
        images = images.to(device=vit_param.device, dtype=vit_param.dtype)

        vit_out = self.vit(images)
        vit_tokens = vit_out[0] if isinstance(vit_out, (tuple, list)) else vit_out

        q_out = self.qformer(vit_tokens)
        qformer_tokens = q_out[0] if isinstance(q_out, (tuple, list)) else q_out

        glm_tokens = self.glm_proj(qformer_tokens)
        pooled = F.normalize(glm_tokens.float().mean(dim=1), dim=-1, eps=1e-6)
        pooled = pooled.to(glm_tokens.dtype)

        return VisualFeatureBundle(
            vit_tokens=vit_tokens,
            qformer_tokens=qformer_tokens,
            glm_tokens=glm_tokens,
            pooled=pooled,
        )

    def encode_image_embedding(self, images: torch.Tensor, detach: bool = False) -> torch.Tensor:
        feat = self(images).pooled
        return feat.detach() if detach else feat

    def named_visual_modules(self) -> Dict[str, nn.Module]:
        modules = {
            "eva.vit": self.vit,
            "eva.qformer": self.qformer,
            "eva.glm_proj": self.glm_proj,
            "image_encoder.vit": self.vit,
            "image_encoder.qformer": self.qformer,
            "image_encoder.glm_proj": self.glm_proj,
        }
        roots = dict(modules)
        for prefix, module in roots.items():
            for name, child in module.named_modules():
                if name:
                    modules[f"{prefix}.{name}"] = child
        return modules

    def freeze_language_backbone(self) -> None:
        for name, param in self.visualglm_model.named_parameters():
            if self.backend == "hf":
                if not name.startswith("image_encoder."):
                    param.requires_grad_(False)
            else:
                if "eva" not in name.lower():
                    param.requires_grad_(False)

    def freeze_all_visual(self) -> None:
        for p in self.blip2.parameters():
            p.requires_grad_(False)

    def _expand_prefixes(self, prefixes):
        expanded = set()
        for p in prefixes:
            expanded.add(p)
            if p.startswith("eva."):
                expanded.add("image_encoder." + p[len("eva."):])
            if p.startswith("image_encoder."):
                expanded.add("eva." + p[len("image_encoder."):])
            if p.startswith("transformer.eva."):
                expanded.add("image_encoder." + p[len("transformer.eva."):])
        return tuple(expanded)

    def unfreeze_by_prefix(self, prefixes: Iterable[str]) -> None:
        prefixes = self._expand_prefixes(prefixes)
        for name, param in self.visualglm_model.named_parameters():
            if any(name.startswith(prefix) for prefix in prefixes):
                param.requires_grad_(True)

    def register_layer_hooks(self, layer_names: Iterable[str]):
        storage = {}
        handles = []
        module_map = self.named_visual_modules()

        for layer_name in layer_names:
            if layer_name not in module_map:
                raise KeyError(f"Unknown visual layer: {layer_name}")

            module = module_map[layer_name]

            def _hook(_, __, output, key=layer_name):
                storage[key] = output[0] if isinstance(output, (tuple, list)) else output

            handles.append(module.register_forward_hook(_hook))

        return storage, handles
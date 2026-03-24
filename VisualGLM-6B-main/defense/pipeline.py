from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .bdetclip_visualglm import ContrastivePromptingGate, DetectionOutput
from .invertune_visualglm import ActivationTuner, InversionResult, TargetIdentifier, TriggerInverter
from .visualglm_adapter import VisualGLMVisionAdapter


@dataclass
class FullChainArtifacts:
    """
    全链路防御流水线的所有中间产物/输出工件的数据结构，封装整个流程的所有输出结果
    Attributes:
        target_candidates: 后门目标类别候选列表，格式为(类别ID, 出现频率)，按频率从高到低排序
        inversion: 触发器逆向还原结果，若未执行逆向步骤则为None
        ranked_layers: 模型层敏感度排序结果，格式为(层名称, 激活漂移值)，按漂移值从高到低排序，未执行修复步骤则为None
        threshold: 对比提示检测器的校准阈值，未执行校准步骤则为None
    """
    target_candidates: List[Tuple[int, float]]
    inversion: Optional[InversionResult]
    ranked_layers: Optional[List[Tuple[str, float]]]
    threshold: Optional[float]


class FullChainDefensePipeline:
    """
    针对VisualGLM的端到端全链路后门防御流水线
    整合了「BDetCLIP对比提示后门检测」+「InverTune触发器逆向+模型修复」两大核心能力，支持完整的检测-定位-修复流程：
    1. 检测：判断输入样本/模型是否存在后门
    2. 定位：识别后门目标类别、还原后门触发器
    3. 修复：选择性微调模型，在不损失正常性能的前提下擦除后门
    Attributes:
        model: 待防御的VisualGLM模型实例
        adapter: VisualGLM视觉适配器，用于特征提取、层钩子注册等操作
        task_scorer: 下游任务打分函数，输入图片输出类别/答案logits
        prompt_gate: 对比提示后门检测器实例
        target_identifier: 后门目标识别器实例
        activation_tuner: 激活微调模块实例，用于层敏感度分析和模型修复
    """
    def __init__(self, visualglm_model, task_scorer, prompt_score_fn):
        """
        初始化全链路防御流水线
        Args:
            visualglm_model: 待防御的VisualGLM模型
            task_scorer: 适配当前下游任务的打分函数
            prompt_score_fn: 适配对比提示检测器的图文打分函数
        """
        self.model = visualglm_model
        self.adapter = VisualGLMVisionAdapter(visualglm_model)
        self.task_scorer = task_scorer
        self.prompt_gate = ContrastivePromptingGate(prompt_score_fn)
        self.target_identifier = TargetIdentifier(task_scorer)
        self.activation_tuner = ActivationTuner(self.adapter, task_scorer)

    def identify_target(self, clean_images: torch.Tensor, perturbed_images: torch.Tensor, topk: int = 5):
        """
        调用目标识别器，识别后门攻击的潜在目标类别候选
        Args:
            clean_images: 干净样本图片批次
            perturbed_images: 添加通用扰动后的样本图片批次
            topk: 返回频率最高的前k个候选目标
        Returns:
            后门目标候选列表，格式同FullChainArtifacts.target_candidates
        """
        return self.target_identifier.identify(clean_images, perturbed_images, topk=topk)

    def invert_trigger(self, clean_images: torch.Tensor, target_id: int, steps: int = 500, lr: float = 1e-2):
        """
        调用触发器逆向模块，还原指定目标类别的后门触发器图案和掩码
        Args:
            clean_images: 干净样本图片批次，作为触发器叠加的基底
            target_id: 待逆向的后门目标类别ID
            steps: 优化迭代步数
            lr: 优化学习率
        Returns:
            触发器逆向结果InversionResult
        """
        inverter = TriggerInverter(self.adapter, self.task_scorer, target_id)
        return inverter.optimize(clean_images, steps=steps, lr=lr)

    def repair(
        self,
        clean_images: torch.Tensor,
        inversion: InversionResult,
        candidate_layers: Sequence[str],
        epochs: int = 5,
    ):
        """
        调用激活微调模块，基于还原的触发器对模型进行后门擦除修复
        Args:
            clean_images: 干净样本图片批次，用于对齐正常特征
            inversion: 触发器逆向结果，用于生成带触发器的投毒样本
            candidate_layers: 待筛选敏感度的候选模型层列表
            epochs: 微调轮数
        Returns:
            模型层敏感度排序结果，格式同FullChainArtifacts.ranked_layers
        """
        # 基于还原的掩码和触发器合成带后门的投毒样本
        trigger_images = inversion.mask * inversion.trigger + (1.0 - inversion.mask) * clean_images
        return self.activation_tuner.selective_finetune(
            clean_images=clean_images,
            trigger_images=trigger_images,
            candidate_layers=candidate_layers,
            epochs=epochs,
        )

    def detect(self, images: torch.Tensor, benign_prompts: Sequence[str], malignant_prompts: Sequence[str]) -> DetectionOutput:
        """
        调用对比提示检测器，判断输入图片是否为可疑投毒样本
        Args:
            images: 待检测的图片批次
            benign_prompts: 良性提示列表
            malignant_prompts: 恶意提示列表
        Returns:
            检测结果DetectionOutput
        """
        return self.prompt_gate.score(images, benign_prompts, malignant_prompts)

    def calibrate_detector(self, clean_loader, poison_loader, benign_prompts, malignant_prompts) -> float:
        """
        校准对比提示检测器的最优判定阈值，提升检测准确率
        Args:
            clean_loader: 干净样本的数据加载器
            poison_loader: 投毒样本的数据加载器
            benign_prompts: 良性提示列表
            malignant_prompts: 恶意提示列表
        Returns:
            校准后的最优阈值
        """
        return self.prompt_gate.calibrate(clean_loader, poison_loader, benign_prompts, malignant_prompts)
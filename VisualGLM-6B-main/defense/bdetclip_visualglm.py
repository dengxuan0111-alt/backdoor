from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class DetectionOutput:
    """
    检测器输出结果的数据结构，封装所有检测相关的返回值
    Attributes:
        score: 样本异常得分，值越小说明越可能是被投毒的样本
        suspicious: 样本是否被判定为可疑（投毒样本）
        benign_scores: 模型在良性提示下输出的各类别概率分布
        malignant_scores: 模型在恶意提示下输出的各类别概率分布
    """
    score: float
    suspicious: bool
    benign_scores: torch.Tensor
    malignant_scores: torch.Tensor


class ContrastivePromptingGate:
    """
    适配VisualGLM下游任务的BDetCLIP风格后门检测器
    核心思路：不硬编码CLIP的图文相似度逻辑，而是由调用方提供和具体任务绑定的打分函数
    检测器通过对比**良性提示**和**恶意提示**下模型的输出差异，标记差异异常小的样本为可疑投毒样本
    （投毒样本通常无论输入什么提示，都会被诱导输出预设的恶意结果，因此两类提示下的输出差异极小）

    Attributes:
        score_fn: 任务感知的打分函数，输入为(图片批次, 提示列表)，输出形状为[批次大小, 类别数/答案数]
        threshold: 可疑判定阈值，得分低于该阈值的样本会被标记为可疑
    """

    def __init__(self, score_fn: Callable, threshold: Optional[float] = None):
        """
        初始化检测器
        Args:
            score_fn: 调用方传入的、适配当前下游任务的打分函数
            threshold: 可选预定义判定阈值，不传入则需要后续调用calibrate方法校准得到
        """
        self.score_fn = score_fn
        self.threshold = threshold

    @torch.no_grad()
    def score(
        self,
        images: torch.Tensor,
        benign_prompts: Sequence[str],
        malignant_prompts: Sequence[str],
    ) -> DetectionOutput:
        """
        对输入的图片批次执行后门检测，返回检测结果
        Args:
            images: 输入图片批次，形状为[B, C, H, W]
            benign_prompts: 良性提示列表，即正常的任务提示
            malignant_prompts: 恶意提示列表，用于诱导投毒样本暴露异常特征
        Returns:
            DetectionOutput类型的检测结果
        """
        # 分别计算两类提示下模型的输出logits
        benign_logits = self.score_fn(images, list(benign_prompts))
        malignant_logits = self.score_fn(images, list(malignant_prompts))

        # 维度对齐：如果是单样本输入（维度为1），扩展为[1, num_classes]的批次形式
        if benign_logits.ndim == 1:
            benign_logits = benign_logits.unsqueeze(0)
        if malignant_logits.ndim == 1:
            malignant_logits = malignant_logits.unsqueeze(0)

        # 将logits转换为概率分布
        benign_probs = F.softmax(benign_logits, dim=-1)
        malignant_probs = F.softmax(malignant_logits, dim=-1)

        # 计算两类提示下概率分布的L1距离（绝对值和），取批次平均作为样本异常得分
        diff = (benign_probs - malignant_probs).abs().sum(dim=-1)
        sample_score = float(diff.mean().detach().cpu())
        # 若已设置阈值，则得分小于阈值判定为可疑投毒样本
        suspicious = False if self.threshold is None else sample_score < self.threshold
        return DetectionOutput(
            score=sample_score,
            suspicious=suspicious,
            benign_scores=benign_probs.detach().cpu(),
            malignant_scores=malignant_probs.detach().cpu(),
        )

    @torch.no_grad()
    def calibrate(
        self,
        clean_loader,
        poison_loader,
        benign_prompts: Sequence[str],
        malignant_prompts: Sequence[str],
    ) -> float:
        """
        使用标注好的干净样本集和投毒样本集校准最优判定阈值
        阈值取两类样本得分中位数的中点，平衡召回率和准确率
        Args:
            clean_loader: 干净样本的数据加载器
            poison_loader: 投毒样本的数据加载器
            benign_prompts: 良性提示列表
            malignant_prompts: 恶意提示列表
        Returns:
            校准后的最优判定阈值
        """
        clean_scores: List[float] = []
        poison_scores: List[float] = []
        # 分别计算所有干净样本和投毒样本的异常得分
        for images, *_ in clean_loader:
            clean_scores.append(self.score(images, benign_prompts, malignant_prompts).score)
        for images, *_ in poison_loader:
            poison_scores.append(self.score(images, benign_prompts, malignant_prompts).score)

        # 取两类得分的中位数的平均值作为最优阈值
        clean_median = torch.tensor(clean_scores).median().item()
        poison_median = torch.tensor(poison_scores).median().item()
        self.threshold = 0.5 * (clean_median + poison_median)
        return self.threshold
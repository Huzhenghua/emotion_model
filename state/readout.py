# -*- coding: utf-8 -*-
"""状态回写（读出层）：从模型自身的行为/激活中读出状态，反向更新状态向量。

两层读出：
1. lexical_readout(reply)：文本级——回复的语气/长度作为行为学事件反作用于状态
2. ActivationProbe：激活级——线性探针从骨干隐状态回归状态估计 ŝ（内感受闭环）
"""
import torch
import torch.nn as nn

# ---------- 文本级读出 ----------
_NEG = ["抱歉", "无奈", "遗憾", "烦", "累", "糟糕", "紧张", "焦虑", "不太", "恐怕",
        "压力", "受不了", "难受", "沮丧", "着急", "没办法", "懒得", "不想"]
_POS = ["很高兴", "乐意", "开心", "太好了", "棒", "喜欢", "愉快", "轻松", "没问题",
        "很好", "不错", "荣幸"]


def lexical_readout(reply: str) -> dict | None:
    """模型自己的回复作为事件反作用于状态。

    生理类比：宣泄负面情绪略微加深负面状态（反刍倾向）；
    顺利完成表达（正面/充分展开）轻微缓解应激。
    幅度刻意小于用户事件（detect_event），避免自反馈主导动力学。
    """
    neg = sum(w in reply for w in _NEG)
    pos = sum(w in reply for w in _POS)
    impact: dict = {}
    if neg >= 2:
        impact["valence"] = impact.get("valence", 0) - 0.04
        impact["stress"] = impact.get("stress", 0) + 0.03
    elif neg == 1:
        impact["valence"] = impact.get("valence", 0) - 0.02
    if pos >= 2:
        impact["valence"] = impact.get("valence", 0) + 0.04
        impact["stress"] = impact.get("stress", 0) - 0.03
    # 极简敷衍回复 → 轻微唤醒下降（低投入）
    if 0 < len(reply.strip()) <= 6:
        impact["arousal"] = impact.get("arousal", 0) - 0.02
    return impact or None


# ---------- 激活级读出探针 ----------
class ActivationProbe(nn.Module):
    """隐状态（末 4 层在 completion 位置 mean-pool）→ 状态估计。

    输出目标与 FiLM 调制器的 normalize_state 同尺度（[-1,1]）：
      arousal_norm = 2a-1, valence_norm = v, stress_norm = 2s-1, fatigue_norm = 2f-1
    input_norm：输入隐向量按保存的 mean/std 标准化（随 checkpoint 保存）。
    """

    def __init__(self, hidden_size: int, n_layers_pool: int = 4):
        super().__init__()
        self.n_layers_pool = n_layers_pool
        self.register_buffer("in_mean", torch.zeros(hidden_size))
        self.register_buffer("in_std", torch.ones(hidden_size))
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.SiLU(),
            nn.Linear(128, 4),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """pooled: [B, n_layers_pool, H] → [B, 4]（[-1,1] 归一化尺度）"""
        h = pooled.mean(dim=1)
        h = (h - self.in_mean) / self.in_std.clamp(min=1e-6)
        return self.net(h)

    @torch.no_grad()
    def estimate(self, hidden_states: tuple, completion_mask: torch.Tensor) -> torch.Tensor:
        """从完整 forward 的 hidden_states 估计原始 dims。

        hidden_states: tuple(num_layers+1) 个 [B, seq, H]
        completion_mask: [B, seq] bool，只 pool completion 位置
        返回 [B, 4]：arousal/valence/stress/fatigue 原始尺度
        """
        stack = torch.stack(hidden_states[-self.n_layers_pool:], dim=1)  # [B,L,seq,H]
        mask = completion_mask.unsqueeze(1).unsqueeze(-1)  # [B,1,seq,1]
        pooled = (stack * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)  # [B,L,H]
        norm = self.forward(pooled)  # [B,4] in [-1,1] 尺度
        return torch.stack([
            (norm[:, 0] + 1) / 2,  # arousal
            norm[:, 1],            # valence（归一化是恒等映射）
            (norm[:, 2] + 1) / 2,  # stress
            (norm[:, 3] + 1) / 2,  # fatigue
        ], dim=1)


def probe_impact(current_dims: dict, est: torch.Tensor, gain: float = 0.2) -> dict | None:
    """内感受回写：状态向探针读数 ŝ 混合。返回 step() 可用的 impact。"""
    d = est[0].tolist() if est.dim() == 2 else est.tolist()
    names = ["arousal", "valence", "stress", "fatigue"]
    impact = {}
    for name, target in zip(names, d):
        delta = gain * (target - current_dims[name])
        if abs(delta) > 1e-4:
            impact[name] = round(delta, 5)
    return impact or None

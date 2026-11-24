# -*- coding: utf-8 -*-
"""FiLM 调制器：状态向量 → 每层激活的 γ/β 缩放（阶段 2 核心）。

对每层 DecoderLayer 的残差流输出做：h' = (1 + γ_l) ⊙ h + β_l
- 干路 MLP 共享，每层一个独立头
- 头的末层零初始化：训练起点 = 未调制（恒等映射）
- 通过 forward hook 挂到 Qwen2 的 DecoderLayer 输出，不改模型代码
"""
import torch
import torch.nn as nn

STATE_DIMS = ["arousal", "valence", "stress", "fatigue"]


def normalize_state(dims: dict) -> torch.Tensor:
    """四维状态 → [-1,1] 归一化向量（valence 从 [-1,1] 平移映射）。"""
    v = torch.tensor([
        dims["arousal"],
        (dims["valence"] + 1) / 2,
        dims["stress"],
        dims["fatigue"],
    ], dtype=torch.float32)
    return v * 2 - 1


class FiLMModulator(nn.Module):
    def __init__(self, n_layers: int, hidden_size: int, trunk_dim: int = 128,
                 gain: float = 1.0):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        # 增益系数：放大 γ/β 输出幅度。注册为 buffer 以随 checkpoint 保存/恢复，
        # 保证训练与推理的调制幅度一致
        self.register_buffer("gain", torch.tensor(float(gain)))
        self.trunk = nn.Sequential(
            nn.Linear(len(STATE_DIMS), trunk_dim), nn.SiLU(),
            nn.Linear(trunk_dim, trunk_dim), nn.SiLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(trunk_dim, 2 * hidden_size) for _ in range(n_layers)
        ])
        for head in self.heads:  # 零初始化：起点即恒等映射
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, state_vec: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        z = self.trunk(state_vec)
        out = []
        for head in self.heads:
            gb = head(z) * self.gain
            out.append((gb[..., :self.hidden_size], gb[..., self.hidden_size:]))
        return out


class FilmInjector:
    """把调制器接到 Qwen2 模型上；每次生成前调用 set_state()。"""

    def __init__(self, model, modulator: FiLMModulator):
        self.model = model
        self.modulator = modulator
        self.current: list | None = None
        layers = model.model.layers
        assert modulator.n_layers == len(layers), \
            f"调制器层数 {modulator.n_layers} != 模型层数 {len(layers)}"
        self.handles = [layer.register_forward_hook(self._make_hook(i))
                        for i, layer in enumerate(layers)]

    def _make_hook(self, idx: int):
        def hook(module, inputs, output):
            if self.current is None:
                return output
            gamma, beta = self.current[idx]
            if gamma.dim() == 2:  # [B, H] → [B, 1, H]，对齐激活 [B, seq, H]
                gamma, beta = gamma.unsqueeze(1), beta.unsqueeze(1)
            h = output[0] if isinstance(output, tuple) else output
            h = h * (1 + gamma.to(h.dtype)) + beta.to(h.dtype)
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return hook

    def set_state(self, dims: dict | None, enable_grad: bool = False):
        """设置当前状态的 γ/β。训练时 enable_grad=True 以便梯度回流调制器。"""
        if dims is None:
            self.current = None
            return
        vec = dims if isinstance(dims, torch.Tensor) else normalize_state(dims)
        vec = vec.to(self.model.device)
        if enable_grad:
            self.current = self.modulator(vec)
        else:
            with torch.no_grad():
                self.current = self.modulator(vec)

    def clear_state(self):
        self.current = None

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

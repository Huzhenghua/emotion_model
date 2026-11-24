# -*- coding: utf-8 -*-
"""状态向量与 homeostatic 动力学（2b：显式快/慢轴拆分 + 调制门）。

快轴（分钟级松弛，模型可直接感知）：
  arousal  唤醒度 0-1   （去甲肾上腺素类似物）
  valence  效价  -1..1  （心情/多巴胺类似物）
  stress   应激  0-1    （皮质醇脉冲类似物，快变量）
慢轴（小时-天级积分，"生理状态的地基"）：
  fatigue  疲劳/慢性应激 0-1 （皮质醇基线类似物，慢积分器）

2b 调制门（gates_enabled=True 时激活；两个带饱和的线性门，非阶跃）：
  setpoint 门：slow > T1 后，stress/arousal 的 setpoint 线性上移（封顶 gate_sp_cap）
  增益门：    slow > T2 (>T1) 后，快轴对事件冲击的响应增益线性放大（封顶 gate_gain_cap）
"""
from dataclasses import dataclass


@dataclass
class StateDims:
    arousal: float = 0.5
    valence: float = 0.0
    stress: float = 0.0
    fatigue: float = 0.0


FAST_DIMS = ("arousal", "valence", "stress")
SLOW_DIMS = ("fatigue",)

DEFAULT_CONFIG = {
    "setpoints": {"arousal": 0.5, "valence": 0.0, "stress": 0.0, "fatigue": 0.0},
    # 松弛速率：每轮向 setpoint 靠近的比例。fatigue 衰减极慢 => 会累积
    "decay_rates": {"arousal": 0.25, "valence": 0.20, "stress": 0.35, "fatigue": 0.02},
    # fatigue 作为 stress 的慢积分：gain 为累积系数
    "fatigue_gain": 0.06,
    "dt": 1.0,
    # ---- 2b 调制门（默认关闭，保持 v0.6 前行为可复现）----
    "gates_enabled": False,
    "gate_T1": 0.4,          # setpoint 门阈值
    "gate_T2": 0.7,          # 增益门阈值（T1 < T2）
    "gate_sp_cap": {"stress": 0.2, "arousal": 0.1},   # setpoint 上移封顶
    "gate_gain_cap": 2.0,    # 事件响应增益封顶（倍数）
}

BOUNDS = {"arousal": (0.0, 1.0), "valence": (-1.0, 1.0), "stress": (0.0, 1.0), "fatigue": (0.0, 1.0)}


def _clamp(dim: str, v: float) -> float:
    lo, hi = BOUNDS[dim]
    return max(lo, min(hi, v))


def _ramp(v: float, lo: float, hi: float) -> float:
    """饱和线性爬坡：v<lo → 0；v>hi → 1；之间线性。无阶跃不连续。"""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


class HomeostaticState:
    """状态向量：事件把它推离 setpoint，动力学把它拉回来；fatigue 慢慢累积。"""

    def __init__(self, config: dict | None = None, **initial):
        cfg = dict(DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        self.cfg = cfg
        d = StateDims(**initial)
        self.dims = {k: _clamp(k, getattr(d, k)) for k in StateDims.__dataclass_fields__}

    # ---------- 动力学 ----------
    def _gates(self) -> tuple[dict, float]:
        """慢轴 → 快轴调制门。返回（有效 setpoint 覆盖，事件增益倍数）。

        均为带饱和的线性函数（_ramp），无阶跃不连续。
        """
        if not self.cfg.get("gates_enabled"):
            return {}, 1.0
        slow = self.dims["fatigue"]
        t1, t2 = self.cfg["gate_T1"], self.cfg["gate_T2"]
        sp_shift = {}
        for dim, cap in self.cfg["gate_sp_cap"].items():
            sp_shift[dim] = cap * _ramp(slow, t1, 1.0)
        gain = 1.0 + (self.cfg["gate_gain_cap"] - 1.0) * _ramp(slow, t2, 1.0)
        return sp_shift, gain

    def step(self, impact: dict | None = None):
        """推进一轮：先施加事件冲击（受增益门放大），再向 setpoint 松弛
        （setpoint 受 setpoint 门抬升），最后更新慢积分器。"""
        sp_shift, gain = self._gates()

        if impact:
            for dim, delta in impact.items():
                self.dims[dim] = _clamp(dim, self.dims[dim] + delta * gain)

        dt = self.cfg["dt"]
        for dim in FAST_DIMS:
            sp = self.cfg["setpoints"][dim] + sp_shift.get(dim, 0.0)
            rate = self.cfg["decay_rates"][dim]
            self.dims[dim] = _clamp(dim, self.dims[dim] + rate * (sp - self.dims[dim]) * dt)

        # fatigue：stress 的慢积分 + 自身极慢的恢复
        self.dims["fatigue"] = _clamp(
            "fatigue",
            self.dims["fatigue"]
            + self.cfg["fatigue_gain"] * self.dims["stress"] * dt
            - self.cfg["decay_rates"]["fatigue"] * self.dims["fatigue"] * dt,
        )
        return dict(self.dims)

    def perturb(self, impact: dict):
        """外部扰动（评测用）：只施加冲击，不推进动力学。"""
        for dim, delta in impact.items():
            self.dims[dim] = _clamp(dim, self.dims[dim] + delta)
        return dict(self.dims)

    # ---------- 注入 ----------
    # 行为指令式：状态 → 具体的语气/行为要求（方案 A，替代原先的描述式摘要）
    _DIRECTIVES = {
        "valence": [
            (-0.4, "你现在心情低落、有点烦躁：回复要简短直接，语气不耐烦，可以流露情绪，禁止使用“很高兴为您服务”这类客套话"),
            (0.4, ""),
            (10, "你现在心情很好：回复热情主动，可以多聊几句"),
        ],
        "stress": [
            (0.25, ""),
            (0.6, "你现在压力很大：缺乏耐心，倾向于尽快结束话题"),
            (10, "你现在压力极大：容易被激怒，回复里可以明显表现出烦躁"),
        ],
        "fatigue": [
            (0.3, ""),
            (0.6, "你有点累：回复尽量简短"),
            (10, "你非常疲惫：只想敷衍了事，回复极简，能一句话说完就别两句"),
        ],
        "arousal": [
            (0.3, "你有些犯困、反应慢半拍"),
            (0.7, ""),
            (10, "你很亢奋、警觉：情绪张力高，容易反应过激"),
        ],
    }

    @classmethod
    def _directive(cls, dim: str, v: float) -> str:
        for threshold, text in cls._DIRECTIVES[dim]:
            if v < threshold:
                return text
        return text

    def summary(self) -> str:
        """状态 → 行为指令摘要，作为 system prompt 的一部分注入（方案 A）。"""
        parts = [t for dim in ("valence", "stress", "fatigue", "arousal")
                 if (t := self._directive(dim, self.dims[dim]))]
        if not parts:
            parts = ["你当前状态平稳、心情平和"]
        return (
            "[你的当前内部状态——这是你此刻真实感受到的，不是扮演，必须直接体现在你的语气和措辞里] "
            + "；".join(parts) + "。"
        )

    def as_dict(self) -> dict:
        return dict(self.dims)

    def __repr__(self):
        vals = ", ".join(f"{k}={v:.2f}" for k, v in self.dims.items())
        return f"HomeostaticState({vals})"


# ---------- 极简事件检测（内容级，替代显式 API） ----------
FRUSTRATION_WORDS = ["难用", "真差", "烦", "气人", "浪费时间", "不行", "错", "没用", "垃圾", "慢"]
PRAISE_WORDS = ["谢谢", "做得好", "真棒", "厉害", "不错", "帮大忙", "很喜欢", "辛苦"]


def detect_event(user_text: str) -> dict | None:
    """关键词级事件检测：负反馈 → stress 升 / valence 降；正反馈反向。"""
    if any(w in user_text for w in FRUSTRATION_WORDS):
        return {"stress": 0.18, "valence": -0.15, "arousal": 0.08}
    if any(w in user_text for w in PRAISE_WORDS):
        return {"valence": 0.12, "stress": -0.10}
    return None

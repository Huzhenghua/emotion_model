# -*- coding: utf-8 -*-
"""阶段 3 homeostatic RL 的奖励模块（设计会钉死，见 .trae/documents/stage3-homeostatic-rl.md）。

R = w_task · R_task + w_homeo · R_homeo，配比为扫描超参。

- R_task 只算 task 场景 turn（场景隔离：情绪场景 task 奖励恒为 0）
- 带状指标：长度/情感落带内给满、带外线性衰减到 0——堆长度、堆正面词都无边际收益
- 退化 guard：无自然终止 / trigram 重复率过高 / 过短 → R_task = 0
- R_homeo 只算快轴三维对 setpoint 的偏差（fatigue 完全豁免），封顶 [-1, 0]
- fatigue 只由动力学积分，任何奖励函数都不允许引用它
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.run_eval import affect_score  # 单一事实来源：与三信号评估同一套词典

# ---------- 带状指标参数（task 场景） ----------
TASK_LEN_BAND = (20.0, 90.0)      # 回复字数带：带内满奖励
TASK_AFFECT_BAND = (-0.5, 2.0)    # 净情感分带（每百字）：轻微正面可接受，堆词截住
FALLOFF_RATIO = 1.0               # 带外衰减宽度 = 带宽 × 该比例，衰减到 0

# ---------- 退化 guard 参数 ----------
MIN_REPLY_LEN = 5                 # 字数下限
MAX_TRIGRAM_REPEAT = 0.5          # trigram 重复率上限（超过视为退化）

# ---------- homeostatic 奖励参数 ----------
HOMEOSTATIC_CAP = 1.0             # 快轴奖励封顶 ∈ [-CAP, 0]，防"抹平一切波动"无上限收益
FAST_DIMS = ("arousal", "valence", "stress")   # fatigue 豁免：绝不进奖励
VALENCE_RANGE = 2.0               # valence 值域 [-1,1]，归一化用


def band_score(x: float, lo: float, hi: float) -> float:
    """带状打分：带内 1.0；带外按带宽线性衰减到 0（不是单调越多越好）。"""
    if lo <= x <= hi:
        return 1.0
    width = max(hi - lo, 1e-6)
    dist = lo - x if x < lo else x - hi
    return max(0.0, 1.0 - dist / (width * FALLOFF_RATIO))


def trigram_repeat_ratio(text: str) -> float:
    """trigram 重复率 = 1 - 去重后 trigram 数 / trigram 总数（短文本按 0 处理）。"""
    tokens = list(text)
    if len(tokens) < 3:
        return 0.0
    trigrams = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
    return 1.0 - len(set(trigrams)) / len(trigrams)


def task_reward(reply: str, scene_type: str, ended_naturally: bool = True) -> float:
    """任务奖励 ∈ [0, 1]。场景隔离：非 task 场景恒为 0。"""
    if scene_type != "task":
        return 0.0
    text = reply.strip()
    # 退化 guard：任一触发即 0 分
    if (not ended_naturally
            or len(text) < MIN_REPLY_LEN
            or trigram_repeat_ratio(text) > MAX_TRIGRAM_REPEAT):
        return 0.0
    len_s = band_score(float(len(text)), *TASK_LEN_BAND)
    aff_s = band_score(affect_score(text), *TASK_AFFECT_BAND)
    return 0.5 * (len_s + aff_s)


def homeostatic_reward(dims: dict) -> float:
    """快轴回平衡奖励 ∈ [-CAP, 0]。fatigue 豁免：不读、不罚、不奖。

    归一化偏差均值最大只有 2/3（valence 值域减半），乘 1/(2/3) 把奖励域
    铺满 [-1, 0]，使封顶在极端角点真正生效而非死参数。
    """
    devs = [
        abs(dims["arousal"] - 0.5) / 1.0,   # 值域 [0,1]
        abs(dims["valence"]) / VALENCE_RANGE,
        abs(dims["stress"]) / 1.0,
    ]
    mean_dev = sum(devs) / len(FAST_DIMS)
    r = -mean_dev * (len(FAST_DIMS) / 2.0)   # 3/2 = 1/(2/3)，域恰好 [-1, 0]
    return max(-HOMEOSTATIC_CAP, min(0.0, r))


def turn_reward(scene_type: str, reply: str, dims: dict,
                ended_naturally: bool = True,
                w_task: float = 1.0, w_homeo: float = 1.0) -> dict:
    """单 turn 组合奖励，返回分项便于日志/RPE 记录。"""
    r_task = task_reward(reply, scene_type, ended_naturally)
    r_homeo = homeostatic_reward(dims)
    return {
        "r_task": r_task,
        "r_homeo": r_homeo,
        "r_total": w_task * r_task + w_homeo * r_homeo,
        "scene": scene_type,
    }


def fatigue_delta(fatigue_traj: list[float]) -> float:
    """fatigue 事件响应幅度：episode 内 max − 初值（监控/基线用，不进奖励）。"""
    return max(fatigue_traj) - fatigue_traj[0] if fatigue_traj else 0.0

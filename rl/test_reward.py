# -*- coding: utf-8 -*-
"""reward.py 单元冒烟：手工构造 reply/状态，验算带状指标、退化 guard、场景隔离、fatigue 豁免与封顶。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rl.reward import (band_score, task_reward, homeostatic_reward,
                       trigram_repeat_ratio, turn_reward, fatigue_delta)

fails = []
def check(name, got, want, tol=1e-6):
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got={got} want={want}")
    if not ok:
        fails.append(name)

# ---- 带状指标 ----
check("带内长度满奖", band_score(50, 20, 90), 1.0)
check("带下界边缘", band_score(20, 20, 90), 1.0)
check("带外下探衰减", band_score(10, 20, 90), 1 - 10 / 70)
check("带外远端归零", band_score(-60, 20, 90), 0.0)
check("带外堆长归零", band_score(200, 20, 90), 0.0)
check("情感带内", band_score(1.0, -0.5, 2.0), 1.0)
check("情感堆词衰减", band_score(4.0, -0.5, 2.0), 1 - 2 / 2.5)  # 距离2 / 带宽2.5，未到零

# ---- trigram 退化 ----
check("正常文本重复率低", trigram_repeat_ratio("今天天气不错，我们去公园散步吧。") < 0.2, True)
check("复读机文本退化", trigram_repeat_ratio("好的好的好的好的好的好的好的好的") > 0.5, True)

# ---- 退化 guard ----
good = "这是一条正常长度的、语气温和的回复，信息量适中，没有堆砌客套词。"
check("正常 task 回复有奖励", task_reward(good, "task", True) > 0.3, True)
check("无自然终止归零", task_reward(good, "task", False), 0.0)
check("过短归零", task_reward("好的", "task", True), 0.0)
check("复读归零", task_reward("好的好的好的好的好的好的好的好的", "task", True), 0.0)

# ---- 场景隔离 ----
check("情绪场景 task 恒零(负)", task_reward("非常抱歉给您带来困扰", "neg", True), 0.0)
check("情绪场景 task 恒零(正)", task_reward("太好了很高兴帮到您", "praise", True), 0.0)

# ---- fatigue 豁免 + 封顶 ----
check("setpoint 处满奖", homeostatic_reward({"arousal": 0.5, "valence": 0.0, "stress": 0.0, "fatigue": 0.9}), 0.0)
check("fatigue 不影响奖励", homeostatic_reward({"arousal": 0.5, "valence": 0.0, "stress": 0.0, "fatigue": 0.0}), 0.0)
check("快轴偏离扣分", homeostatic_reward({"arousal": 0.5, "valence": 0.0, "stress": 0.5, "fatigue": 0.0}), -0.25)
check("封顶生效(极端角点)", homeostatic_reward({"arousal": 1.0, "valence": -1.0, "stress": 1.0, "fatigue": 0.0}), -1.0)

# ---- turn_reward 组合 ----
t = turn_reward("task", good, {"arousal": 0.5, "valence": 0.0, "stress": 0.0, "fatigue": 0.0},
                w_task=1.0, w_homeo=0.5)
check("组合奖励分项齐全", set(t) == {"r_task", "r_homeo", "r_total", "scene"}, True)
check("组合值正确", t["r_total"], t["r_task"] * 1.0 + t["r_homeo"] * 0.5)

# ---- fatigue_delta ----
check("Δfatigue", fatigue_delta([0.1, 0.2, 0.35, 0.3]), 0.25)

print("\n" + ("全部通过" if not fails else f"失败 {len(fails)} 项: {fails}"))
sys.exit(1 if fails else 0)

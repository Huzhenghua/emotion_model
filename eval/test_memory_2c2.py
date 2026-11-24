# -*- coding: utf-8 -*-
"""2c-2 冒烟：判据④（情绪加权留存）+ 判据⑤（低应激回放窗口）+ [回忆] 段构造。

buffer 级判据不需要 GPU；段构造只加载 tokenizer。
判据⑥（闭环不破坏）走 run_eval/棘轮重验，不在本文件。
"""
import json
import random
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.loop import StatefulAgent
from memory.buffer import EpisodicBuffer
from state.dynamics import HomeostaticState

PASS, FAIL = "✅", "❌"
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  {PASS if ok else FAIL} {name}" + (f"（{detail}）" if detail else ""))


def snap(arousal, valence=0.0, stress=0.0, fatigue=0.0):
    return {"arousal": arousal, "valence": valence, "stress": stress, "fatigue": fatigue}


CUE = "生日蛋糕"

# ---------- 判据④：高 arousal 留存 > 低 arousal（buffer 级） ----------
print("[判据④] 情绪加权留存（2×2 因子，高/低 arousal 箱对比）")
# 同一 cue 匹配两条记录：高 arousal(0.7) vs 低 arousal(0.45)，其余同；若干普通轮把 age 拉开
for beta in (0.0, 2.0):
    b = EpisodicBuffer(tempfile.mktemp(suffix=".jsonl"), capacity=300,
                       decay_rate=0.05, beta=beta, gamma=1.0)
    b.record(CUE, "回复A", snap(0.70, 0.0), event_marker={"arousal": 0.1})   # 高箱
    b.record(CUE, "回复B", snap(0.45, 0.0), event_marker={"arousal": -0.05})  # 低箱
    for i in range(30):   # 30 轮普通流量把两条记录的 age 同步拉开
        b.record(f"普通轮次{i}", f"回复{i}", snap(0.5, 0.0))
    hits = b.recall_weighted(CUE, top_k=2)
    top_is_high = hits and hits[0]["state"]["arousal"] == 0.70
    if beta == 0.0:
        # 对照：beta=0 时 salience 失效，权重 == 纯时间衰减（情绪不参与留存）
        import math
        for r in b.records:
            expected = math.exp(-0.05 * (b._write_seq - r["_wseq"]))
            w = b.weight(r)
            assert abs(w - expected) < 1e-9, f"权重偏离纯衰减: {w} vs {expected}"
        check(f"beta={beta} 对照：权重退化为纯时间衰减", True, "salience=1 全记录成立")
    else:
        check(f"beta={beta}：高 arousal 箱留存排序在前", top_is_high,
              f"top1 arousal={hits[0]['state']['arousal'] if hits else '无'}")

# valence 端：gamma 生效（负 valence 也是显著事件——闪光灯记忆不分正负）
b3 = EpisodicBuffer(tempfile.mktemp(suffix=".jsonl"), capacity=300,
                    decay_rate=0.05, beta=0.0, gamma=1.0)
b3.record(" neutral 记录", "回复", snap(0.5, 0.0))
b3.record(CUE + "（正端）", "回复", snap(0.5, 0.5))
b3.record(CUE + "（负端）", "回复", snap(0.5, -0.6))
w_pos = b3.weight(b3.records[1])
w_neg = b3.weight(b3.records[2])
check("valence 两端（正/负）都获加权", w_pos > 1.0 and w_neg > 1.0,
      f"w_pos={w_pos:.3f} w_neg={w_neg:.3f}")

# ---------- 判据⑤：低应激窗口（replay_prob 单调 + 两态对比） ----------
print("[判据⑤] 低应激窗口回放门控")
b4 = EpisodicBuffer(tempfile.mktemp(suffix=".jsonl"))
p_calm = b4.replay_prob(snap(0.50, stress=0.05))   # 低 arousal 低 stress
p_mid = b4.replay_prob(snap(0.62, stress=0.20))    # 中间
p_stress = b4.replay_prob(snap(0.70, stress=0.35))  # 高 arousal 高 stress
check("低应激全开", p_calm == b4.replay_gate_cfg["base"], f"p={p_calm}")
check("高应激压零", p_stress == 0.0, f"p={p_stress}")
check("窗口内单调", p_calm > p_mid > p_stress, f"{p_calm:.2f}>{p_mid:.2f}>{p_stress:.2f}")

# 采样层：注入 rng，两态命中统计
rng_calm = random.Random(7)
hits_calm = sum(b4.replay_gate(snap(0.5, stress=0.05), rng_calm) for _ in range(200))
rng_stress = random.Random(7)
hits_stress = sum(b4.replay_gate(snap(0.7, stress=0.35), rng_stress) for _ in range(200))
check("两态回放率分离（200 采样）", hits_calm == 200 and hits_stress == 0,
      f"calm {hits_calm}/200 vs stress {hits_stress}/200")

# ---------- 去重窗口（决策点①细节 3） ----------
print("[去重] 同一情景连续轮不重复回放")
b5 = EpisodicBuffer(tempfile.mktemp(suffix=".jsonl"), dedup_window=3)
b5.record(CUE, "回复", snap(0.5))
for i in range(10):   # min_age=10：刚写入的轮次不回放（防自指），先放 10 轮 filler
    b5.record(f"垫轮{i}", "回复", snap(0.5))
h1 = b5.recall_weighted(CUE, top_k=1)
h2 = b5.recall_weighted(CUE, top_k=1)
check("紧邻轮去重生效", len(h1) == 1 and len(h2) == 0, f"第1次{len(h1)}条→紧邻第2次{len(h2)}条")
for i in range(4):
    b5.record(f"普通{i}", "回复", snap(0.5))
h3 = b5.recall_weighted(CUE, top_k=1)
check("窗口过后可再次回放", len(h3) == 1, f"数轮后{len(h3)}条")

# ---------- [回忆] 段构造（tokenizer-only，不加载模型） ----------
print("[集成] [回忆] 段构造 + 预算 + 门控开关")
tok_path = r"d:\Program Files (x86)\Trae_project\emotion_model\Qwen2.5-1.5B-Instruct"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(tok_path)

agent = StatefulAgent.__new__(StatefulAgent)   # 跳过模型加载，只测段逻辑
agent.memory = EpisodicBuffer(tempfile.mktemp(suffix=".jsonl"), dedup_window=3)
agent.memory.record(CUE, "回复", snap(0.5))    # 新 buffer，无去重残留
for i in range(10):   # min_age=10：先垫 10 轮 filler 使 CUE 可回放
    agent.memory.record(f"垫轮{i}", "回复", snap(0.5))
agent.memory_replay = True
agent.replay_budget_tokens = 200
agent.replay_rng = random.Random(1)            # 固定 rng；calm 态 gate 必中
agent.state = HomeostaticState()               # 初始态 arousal 0.5/stress 0 → gate 全开
agent.tokenizer = tok

seg = agent._build_replay_segment("生日蛋糕")
check("门控命中 → 段生成且带 [回忆] 标记", seg is not None and "[回忆]" in seg,
      f"{len(seg)} 字符")
check("预算内", len(tok.encode(seg, add_special_tokens=False)) <= 200,
      f"{len(tok.encode(seg, add_special_tokens=False))} tok")

agent.memory_replay = False
check("开关关 → 不回放", agent._build_replay_segment("生日蛋糕") is None)
agent.memory_replay = True
agent.state.step({"stress": 0.5})   # 高 stress → gate 压零
agent.state.step(None)
check("高应激 → 门控压零不回放", agent._build_replay_segment("生日蛋糕") is None)

# 预算截断：塞 10 条长记录，强制预算裁剪
b6 = EpisodicBuffer(tempfile.mktemp(suffix=".jsonl"), capacity=50)
for i in range(10):
    b6.record(f"{CUE}长记录{i}。" + "细节" * 120, "回复", snap(0.5))
for i in range(10):   # min_age=10：垫轮使长记录可回放
    b6.record(f"垫轮{i}", "回复", snap(0.5))
agent.memory = b6
agent.replay_rng = random.Random(1)
seg6 = agent._build_replay_segment(CUE)
n_seg = len(tok.encode(seg6 or "", add_special_tokens=False)) if seg6 else 0
check("预算超限自动裁剪记录数", seg6 is not None and n_seg <= 200,
      f"最终 {n_seg} tok")

print()
n_pass = sum(1 for _, ok, _ in results if ok)
print(f"===== 2c-2 冒烟：{n_pass}/{len(results)} 通过 =====")
sys.exit(0 if n_pass == len(results) else 1)

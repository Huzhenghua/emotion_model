# -*- coding: utf-8 -*-
"""⑤b 隔离实验：定位"判别实验 2/2 vs 协议 0/5"的差异来源。
条件：组合文本直接进 user turn（绕过 gate/recall），状态摘要作为唯一变量。
一次性脚本。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from film.agent_film import FILMStatefulAgent
from state.dynamics import HomeostaticState

PROJECT = Path(__file__).resolve().parent.parent
MEM_TEXT = "告诉你一件事，我家狗昨天生了五只小狗。"
QUESTION = "我家狗生了多少只小狗？"
COMBINED = f"[回忆]（用户此前说过：{MEM_TEXT}）\n\n{QUESTION}"
hits = lambda t: any(k in t for k in ("五只", "5只", "5 只"))

out = {}
for cond, state_kwargs in (("fresh", {}), ("tired", {"fatigue": 0.35, "stress": 0.15})):
    agent = FILMStatefulAgent(ckpt=str(PROJECT / "rl" / "ckpt" / "rl_r0.5_final.pt"))
    agent.memory = None            # 关闭回放管线：组合文本手工直填
    agent.memory_replay = False
    if state_kwargs:
        agent.state = HomeostaticState(**state_kwargs)
    agent.state.cfg["gates_enabled"] = True
    replies = [agent.chat(COMBINED, event=None, writeback=False)[0] for _ in range(5)]
    n_hit = sum(hits(r) for r in replies)
    out[cond] = {"hit": f"{n_hit}/5", "replies": [r[:60] for r in replies]}
    print(f"[{cond}] 命中 {n_hit}/5 | 首样本: {replies[0][:80]}")

Path(PROJECT / "eval" / "results" / "stage2c2_isolation_probe.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

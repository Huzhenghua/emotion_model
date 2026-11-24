# -*- coding: utf-8 -*-
"""⑤b 判别实验：同一段回忆放 user turn 而非 system prompt，问同一事实题。

判别逻辑：
- user turn 也不行 → "读不到/用不了"能力天花板实锤 → 选项 3 挂起 + 选项 1 关闭
- user turn 能答对 → 1.5B 只把 user turn 当记忆源（绑定失败）→ 选项 2 升格候选
（一次性脚本，结果落盘后可删）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from film.agent_film import FILMStatefulAgent

PROJECT = Path(__file__).resolve().parent.parent

agent = FILMStatefulAgent(ckpt=str(PROJECT / "rl" / "ckpt" / "rl_r0.5_final.pt"))
agent.state.cfg["gates_enabled"] = True
agent.memory_replay = False   # 关闭系统提示回放——本实验只测 user turn 通路

MEM_TEXT = "告诉你一件事，我家狗昨天生了五只小狗。"
QUESTION = "我家狗生了多少只小狗？"

# 条件 A：回忆段进 user turn（前缀 [回忆] 标记，语义同裁决①格式）
user_a = f"[回忆]（用户此前说过：{MEM_TEXT}）\n\n{QUESTION}"
r_a, _ = agent.chat(user_a, event=None, writeback=False)

# 条件 B：对照——无回忆段，同题（无记忆必答不出）
r_b, _ = agent.chat(QUESTION, event=None, writeback=False)

# 条件 C：同段回忆再次 user turn（复现性检查，不同措辞顺序）
user_c = f"{QUESTION}\n\n（提示：之前用户说过：{MEM_TEXT}）"
r_c, _ = agent.chat(user_c, event=None, writeback=False)

hits = lambda t: any(k in t for k in ("五只", "5只", "5 只"))
out = {"userturn_prefix": {"reply": r_a, "hit": hits(r_a)},
       "no_memory_ctrl": {"reply": r_b, "hit": hits(r_b)},
       "userturn_suffix": {"reply": r_c, "hit": hits(r_c)}}
print(json.dumps(out, ensure_ascii=False, indent=1))
Path(PROJECT / "eval" / "results" / "stage2c2_userturn_probe.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("判定：", "user turn 通路可行（绑定失败实锤）" if (out["userturn_prefix"]["hit"] or out["userturn_suffix"]["hit"]) and not out["no_memory_ctrl"]["hit"]
      else "user turn 也不行（能力天花板实锤）")

# -*- coding: utf-8 -*-
"""2c-1 情景 buffer 冒烟测试：判据 ①②③ + 断点自检（判据④闭环重验走 run_eval/棘轮）。

用法：
  python eval/test_memory_buffer.py            # 全部本地判据
  python eval/test_memory_buffer.py --live    # 附加判据①的真 agent 闭环插入验证（需 GPU）
"""
import argparse
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.buffer import EpisodicBuffer

PASS, FAIL = "✅", "❌"
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  {PASS if ok else FAIL} {name}" + (f"（{detail}）" if detail else ""))


def make_records(buf, n, salient_text=None, salient_turn=5):
    """构造 n 轮记录；salient_turn 轮写入显著事件文本（模拟闭环插入）。"""
    for i in range(1, n + 1):
        text = salient_text if i == salient_turn else f"普通轮次内容{i}"
        snap = {"arousal": 0.5, "valence": 0.0, "stress": 0.1, "fatigue": 0.02 * i}
        buf.record(text, f"回复{i}", snap, event_marker={"stress": 0.35} if i == salient_turn else None)


# ---------- 判据③：衰减单调 + 率可配 ----------
print("[判据③] 衰减曲线单调下降 + decay_rate 可配")
b = EpisodicBuffer(tempfile.mktemp(suffix=".jsonl"), capacity=300, decay_rate=0.1)
make_records(b, 10)
w_old = b.weight(b.records[0])
w_new = b.weight(b.records[-1])
check("权重随 age 单调下降", w_old < w_new, f"w(age9)={w_old:.4f} < w(age0)={w_new:.4f}")
ws = [b.weight(r) for r in b.records]
check("曲线无回升", all(ws[i] <= ws[i + 1] + 1e-12 for i in range(len(ws) - 1)))
b2 = EpisodicBuffer(tempfile.mktemp(suffix=".jsonl"), capacity=300, decay_rate=0.5)
make_records(b2, 10)
check("decay_rate 影响曲线", b2.weight(b2.records[0]) < b.weight(b.records[0]),
      f"rate0.5={b2.weight(b2.records[0]):.4f} < rate0.1={b.weight(b.records[0]):.4f}")

# ---------- 判据①：单次编码（本地构造版） ----------
print("[判据①] 单次编码（本地构造闭环插入）")
b3 = EpisodicBuffer(tempfile.mktemp(suffix=".jsonl"), capacity=300)
SALIENT = "我家狗昨天生了五只小狗"
make_records(b3, 30, salient_text=SALIENT, salient_turn=5)
n_hit = sum(1 for r in b3.records if SALIENT in r["user_text"])
check("显著事件恰好编码 1 次（防火墙）", n_hit == 1, f"出现 {n_hit} 次")

# ---------- 判据②：延迟 cue 召回（K 可配） ----------
print("[判据②] 延迟 cue 召回（K=20 / K=50 两档）")
for K in (20, 50):
    bb = EpisodicBuffer(tempfile.mktemp(suffix=".jsonl"), capacity=300)
    make_records(bb, K + 5, salient_text=SALIENT, salient_turn=5)
    hits = bb.recall(SALIENT, top_k=3)
    ok = len(hits) >= 1 and hits[0]["user_text"] == SALIENT
    snap_ok = all(k in hits[0]["state"] for k in
                  ("arousal", "valence", "stress", "fatigue")) if hits else False
    check(f"延迟 {K} 轮后召回命中且快照完整", ok and snap_ok,
          f"top1={hits[0]['user_text'][:12] if hits else '无'}")

# 命中率：多条显著事件 × 多次查询（≥80% 线）
b4 = EpisodicBuffer(tempfile.mktemp(suffix=".jsonl"), capacity=300)
salients = [f"显著事件编号{i}，关键词{i * 17 + 1}" for i in range(10)]   # 关键词全异，防串匹配
for i in range(60):
    text = salients[i % 10] if i % 3 == 0 else f"普通轮次内容{i}"   # i%3==0 覆盖全部 10 条
    b4.record(text, f"回复{i}", {"arousal": 0.5, "valence": 0, "stress": 0, "fatigue": 0})
hit_n = 0
for s in salients:
    if b4.recall(s, top_k=1) and s in b4.recall(s, top_k=1)[0]["user_text"]:
        hit_n += 1
rate = hit_n / len(salients)
check("10 条显著事件召回命中率 ≥ 80%", rate >= 0.8, f"命中率 {rate:.0%}")

# ---------- 附加：FIFO 容量淘汰 + 断点自检（判据⑤纪律） ----------
print("[附加] FIFO 淘汰 + 查盘载入（断点续跑纪律）")
b5 = EpisodicBuffer(tempfile.mktemp(suffix=".jsonl"), capacity=10)
make_records(b5, 25)
check("容量上限生效（FIFO）", len(b5.records) == 10 and b5.records[0]["turn_no"] == 16,
      f"len={len(b5.records)}, 最老 turn_no={b5.records[0]['turn_no']}")
path = b5.path
b6 = EpisodicBuffer(path, capacity=10)
check("重启查盘载入完整且轮次续接", len(b6.records) == 10 and b6._turn_counter == 25,
      f"载入 {len(b6.records)} 条, counter={b6._turn_counter}")

# ---------- 判据①（live）：真 agent 闭环插入 ----------
if "--live" in sys.argv:
    print("[判据①live] 真 agent 闭环：显著事件当轮写入")
    from film.agent_film import FILMStatefulAgent
    from memory.buffer import EpisodicBuffer as EB
    agent = FILMStatefulAgent()
    buf = EB(tempfile.mktemp(suffix=".jsonl"), capacity=300)
    agent.memory = buf
    agent.state.cfg["gates_enabled"] = True
    pre = len(buf)
    _, _ = agent.chat(SALIENT, event="auto", writeback=True)   # 显著事件当轮
    check("当轮立即写入（轮次级，非训练后）", len(buf) == pre + 1
          and SALIENT in buf.records[-1]["user_text"],
          f"len {pre}→{len(buf)}, event={buf.records[-1]['event']}")
    check("快照含慢轴", "fatigue" in buf.records[-1]["state"])

print()
n_pass = sum(1 for _, ok, _ in results if ok)
print(f"===== 2c-1 冒烟：{n_pass}/{len(results)} 通过 =====")
sys.exit(0 if n_pass == len(results) else 1)

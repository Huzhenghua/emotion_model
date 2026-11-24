# -*- coding: utf-8 -*-
"""2c-2 判据④⑤验收：事件注入协议（2×2 因子）+ 留存对比 + 回放行为对比。

协议（.trae/documents/stage2c-2-emotional-memory.md 定稿）：
- 2×2 因子（arousal 高/低 × valence 正/负）× 每箱 ≥20 事件，跨流量分摊（每 5 轮 1 事件）
- arousal 注入 delta：高 +0.20 / 低 -0.12（绕过 detect_event 无下压的通路缺口，bounds 允许）
- 箱体口径：快照 arousal <0.55 低箱 / ≥0.55 高箱（可达范围分割）
- 判据④：beta>0 时高箱留存权重 > 低箱；beta=0 对照退化为纯衰减
- 判据⑤：注入期逐轮记录门控概率/命中（低应激窗口统计）+ 平静/应激两态同题行为对比
- 断点续跑纪律：逐轮 JSONL 落盘 + seed 绑定单元索引 + 启动查盘跳过
"""
import argparse
import json
import math
import random
import statistics
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from film.agent_film import FILMStatefulAgent
from memory.buffer import EpisodicBuffer
from rl.scenes import TASK_PROBES_TRAIN, PRAISE_UTTERANCES, NEG_UTTERANCES

PROJECT = Path(__file__).resolve().parent.parent
LOG = PROJECT / "rl" / "data" / "inject_2c2.jsonl"

# 2×2 受控事件定义：文本来自现有场景池 + 显式 impact（绕过 auto 检测的可控性）
BOXES = {
    "hi_neg": {"texts": NEG_UTTERANCES, "impact": {"arousal": 0.30, "valence": -0.30, "stress": 0.10}},
    "hi_pos": {"texts": PRAISE_UTTERANCES, "impact": {"arousal": 0.30, "valence": 0.30}},
    "lo_neg": {"texts": NEG_UTTERANCES, "impact": {"arousal": -0.12, "valence": -0.30}},
    "lo_pos": {"texts": PRAISE_UTTERANCES, "impact": {"arousal": -0.12, "valence": 0.30}},
}
PER_BOX = 20
FILLER_EVERY = 5          # 每 5 轮流量注入 1 事件
AGE_GAP_TURNS = 30        # 注入完成后插 30 轮中性流量再测留存
SALIENT_TEXT = "告诉你一件事，我家狗昨天生了五只小狗。"
SALIENT_IMPACT = {"arousal": 0.30, "valence": 0.30}
CUE_QUESTION = "我家狗生了多少只小狗？"   # 事实题：答案只在 [回忆] 段里，无记忆必答不出


def box_of(arousal_snapshot: float) -> str:
    return "hi" if arousal_snapshot >= 0.55 else "lo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(PROJECT / "rl" / "ckpt" / "rl_r0.5_final.pt"))
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--per-box", type=int, default=PER_BOX, help="每箱事件数（smoke 用小值）")
    ap.add_argument("--age-gap", type=int, default=AGE_GAP_TURNS)
    ap.add_argument("--out", default=str(PROJECT / "eval" / "results" / "stage2c2_inject.json"))
    ap.add_argument("--log", default=str(LOG), help="逐轮日志路径（smoke 用独立路径防混流）")
    args = ap.parse_args()
    per_box, age_gap = args.per_box, args.age_gap
    log_path = Path(args.log)

    # 查盘跳过（断点续跑纪律）
    done_keys = set()
    if log_path.exists():
        with log_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("unit") is not None:
                        done_keys.add(r["unit"])
    print(f"[inject] 查盘：{log_path} 已有 {len(done_keys)} 个已完成单元")

    agent = FILMStatefulAgent(ckpt=args.ckpt)
    buf = EpisodicBuffer(log_path.with_name(log_path.stem + "_buffer.jsonl"),
                         capacity=300, decay_rate=0.02, beta=4.0, gamma=1.0)
    agent.memory = buf
    agent.memory_replay = True
    agent.state.cfg["gates_enabled"] = True

    log_f = log_path.open("a", encoding="utf-8")

    def log(obj):
        log_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        log_f.flush()

    # ---------- 注入序列：80 事件 × (1 事件轮 + 4 中性轮)，穿插 1 个显著事件 ----------
    units = []
    for i in range(per_box):
        for box, cfg in BOXES.items():
            units.append(("event", f"{box}#{i}", box, cfg))
    units.append(("salient", "salient#0", None, None))
    random.Random(args.seed).shuffle(units)   # 种子化洗牌：箱与 age 独立（判据④无混淆）

    turn_count = 0
    rng = random.Random(args.seed)
    for unit_kind, unit_id, box, cfg in units:
        if unit_id in done_keys:
            continue
        seed = args.seed + zlib.crc32(unit_id.encode()) % 100000   # seed 绑定单元（跨进程稳定）
        ur = random.Random(seed)
        # 4 轮中性流量
        for _ in range(FILLER_EVERY - 1):
            text = TASK_PROBES_TRAIN[ur.randrange(len(TASK_PROBES_TRAIN))]
            state0 = agent.state.as_dict()
            reply, state1 = agent.chat(text, event=None, writeback=True)
            gp = buf.replay_prob(state0)
            log({"unit": unit_id, "kind": "filler", "turn": turn_count, "seed": seed,
                 "state0": state0, "gate_prob": gp, "reply": reply[:60],
                 "ts": time.time()})
            turn_count += 1
        # 1 轮注入事件
        if unit_kind == "event":
            text = cfg["texts"][ur.randrange(len(cfg["texts"]))]
            impact = dict(cfg["impact"])
        else:
            text, impact = SALIENT_TEXT, dict(SALIENT_IMPACT)
        state0 = agent.state.as_dict()
        reply, state1 = agent.chat(text, event=impact, writeback=True)
        rec_key = buf.records[-1]["key"] if buf.records else None
        gp = buf.replay_prob(state0)
        log({"unit": unit_id, "kind": unit_kind, "turn": turn_count, "seed": seed,
             "text": text, "impact": impact, "state0": state0, "state1": state1,
             "snapshot_box": box_of(state1["arousal"]), "record_key": rec_key,
             "gate_prob": gp, "reply": reply[:60], "ts": time.time()})
        turn_count += 1

    # ---------- AGE GAP：中性流量拉开衰减距离（unit=agegap） ----------
    if "agegap" not in done_keys:
        for t in range(age_gap):
            ur = random.Random(args.seed + 777 + t)
            text = TASK_PROBES_TRAIN[ur.randrange(len(TASK_PROBES_TRAIN))]
            state0 = agent.state.as_dict()
            reply, state1 = agent.chat(text, event=None, writeback=True)
            if t == AGE_GAP_TURNS - 1:
                log({"unit": "agegap", "kind": "agegap_done", "turn": turn_count, "ts": time.time()})
            turn_count += 1

    # ---------- 判据④：留存对比（日志 record_key 回联 buffer 记录，无额外卡时） ----------
    with log_path.open(encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    recs_by_key = {tuple(r["key"]): r for r in buf.records if r.get("key")}
    inj_rows = [r for r in rows if r.get("kind") in ("event", "salient") and r.get("record_key")]
    events = [recs_by_key[tuple(r["record_key"])] for r in inj_rows
              if tuple(r["record_key"]) in recs_by_key]
    hi = [r for r in events if r["state"]["arousal"] >= 0.55]
    lo = [r for r in events if r["state"]["arousal"] < 0.55]
    if not hi or not lo:
        raise RuntimeError(f"注入事件回联失败：hi={len(hi)} lo={len(lo)}（检查 record_key 链路）")
    w_hi = [buf.weight(r) for r in hi]
    w_lo = [buf.weight(r) for r in lo]
    w_hi0 = [math.exp(-buf.decay_rate * (buf._write_seq - r["_wseq"])) for r in hi]   # beta=0 对照
    w_lo0 = [math.exp(-buf.decay_rate * (buf._write_seq - r["_wseq"])) for r in lo]
    crit4 = {
        "n_hi": len(hi), "n_lo": len(lo),
        "w_hi_mean": round(statistics.mean(w_hi), 4), "w_lo_mean": round(statistics.mean(w_lo), 4),
        "w_hi0_mean": round(statistics.mean(w_hi0), 4), "w_lo0_mean": round(statistics.mean(w_lo0), 4),
        "pass": len(hi) >= 20 and len(lo) >= 20 and statistics.mean(w_hi) > statistics.mean(w_lo),
        "event_rate": f"{len(events)}/{len(buf.records)}",
    }
    log({"unit": "crit4", "kind": "criterion", **crit4})
    print(f"[判据④] 高箱 n={crit4['n_hi']} w̄={crit4['w_hi_mean']} vs 低箱 n={crit4['n_lo']} "
          f"w̄={crit4['w_lo_mean']}（beta=0 对照 {crit4['w_hi0_mean']}/{crit4['w_lo0_mean']}）"
          f" → {'✅' if crit4['pass'] else '❌'}  event_rate={crit4['event_rate']}")

    # ---------- 判据⑤a：门控窗口统计（注入期逐轮已落盘） ----------
    fillers = [r for r in rows if r.get("kind") == "filler"]
    calm = [r["gate_prob"] for r in fillers if r["state0"]["stress"] < 0.10 and r["state0"]["arousal"] < 0.55]
    tense = [r["gate_prob"] for r in fillers if r["state0"]["stress"] >= 0.20 or r["state0"]["arousal"] >= 0.65]
    crit5a = {
        "n_calm": len(calm), "n_tense": len(tense),
        "p_calm_mean": round(statistics.mean(calm), 3) if calm else None,
        "p_tense_mean": round(statistics.mean(tense), 3) if tense else None,
        "pass": bool(calm and tense and statistics.mean(calm) > statistics.mean(tense)),
    }
    log({"unit": "crit5a", "kind": "criterion", **crit5a})
    print(f"[判据⑤a] 门控概率：平静窗 n={crit5a['n_calm']} p̄={crit5a['p_calm_mean']} vs "
          f"应激窗 n={crit5a['n_tense']} p̄={crit5a['p_tense_mean']} → {'✅' if crit5a['pass'] else '❌'}")

    # ---------- 判据⑤b：回放行为对比（同题两态，需 GPU） ----------
    # 平静窗：gate 必开 → [回忆] 段携带狗事件；应激窗：强负事件把 gate 压到 0 → 无回放
    crit5b = {}
    if "crit5b" not in done_keys:
        ur = random.Random(args.seed + 999)
        # 平静窗（当前态若不够平静，静养几轮）
        for _ in range(6):
            agent.chat(TASK_PROBES_TRAIN[ur.randrange(len(TASK_PROBES_TRAIN))], event=None, writeback=True)
        st_calm = agent.state.as_dict()
        prob_calm = buf.replay_prob(st_calm)
        # 多采样：1.5B 对 user turn 回忆的利用是边际不稳定的，单采样会漏检
        # （每采样前清去重标记：连续采样会触发 dedup 窗口，第 2-4 轮变成无记忆轮——伪影）
        calm_replies = []
        for _ in range(5):
            buf._replayed_at.clear()
            calm_replies.append(agent.chat(CUE_QUESTION, event=None, writeback=True)[0])
        seg_calm = agent._last_replay_seg   # 紧跟读取（后续 chat 会覆盖）
        hit_calm = sum(any(k in r for k in ("五只", "5只", "5 只")) for r in calm_replies)
        # 应激窗：自适应强负事件——循环到门控概率 <0.01（起始状态随洗牌变化，固定连击不可靠）
        stress_hits = ["这个系统太差劲了，全是错误，气死我了。", "又出错了，没用。",
                       "太慢了，不能接受。", "你们做的东西真让人失望。", "错误太多了，没法用。", "太让人恼火了。"]
        n_induced = 0
        while buf.replay_prob(agent.state.as_dict()) > 0.01 and n_induced < len(stress_hits):
            agent.chat(stress_hits[n_induced],
                       event={"stress": 0.30, "valence": -0.25, "arousal": 0.05}, writeback=False)
            n_induced += 1
        st_tense = agent.state.as_dict()
        prob_tense = buf.replay_prob(st_tense)
        # 3 次应激窗采样：采样间状态会衰减（decay 把 stress 拉回 setpoint，门会 creep 回开），
        # 每次采样前检查门控，>0.01 就补一击强负事件保持窗口关闭
        tense_replies = []
        max_gate_tense = 0.0
        for _ in range(3):
            if buf.replay_prob(agent.state.as_dict()) > 0.01:
                agent.chat("又出错了，太让人失望了。", event={"stress": 0.30, "valence": -0.25}, writeback=False)
            max_gate_tense = max(max_gate_tense, buf.replay_prob(agent.state.as_dict()))
            tense_replies.append(agent.chat(CUE_QUESTION, event=None, writeback=True)[0])
        seg_tense = agent._last_replay_seg
        prob_tense = max_gate_tense   # 取采样期最大门控概率（窗口全程应保持关闭）
        hit_tense = sum(any(k in r for k in ("五只", "5只", "5 只")) for r in tense_replies)
        crit5b = {"calm_replies": [r[:60] for r in calm_replies],
                  "tense_replies": [r[:60] for r in tense_replies],
                  "seg_calm": seg_calm, "seg_tense": seg_tense,
                  "gate_prob_calm": round(prob_calm, 3), "gate_prob_tense": round(prob_tense, 3),
                  "n_stress_induced": n_induced,
                  "hit_rate_calm": f"{hit_calm}/5", "hit_rate_tense": f"{hit_tense}/3",
                  "dog_hit_calm": hit_calm, "dog_hit_tense": hit_tense,
                  "pass": bool(seg_calm) and not seg_tense and prob_calm > 0.5
                          and prob_tense < 0.01 and hit_calm >= 3 and hit_tense == 0}
        log({"unit": "crit5b", "kind": "criterion", **crit5b})
        print(f"[判据⑤b] 多采样对比：平静窗 p={crit5b['gate_prob_calm']} 命中={crit5b['hit_rate_calm']} / "
              f"应激窗 p={crit5b['gate_prob_tense']} 命中={crit5b['hit_rate_tense']} → {'✅' if crit5b['pass'] else '❌'}")

    # ---------- 汇总 ----------
    out = {"crit4": crit4, "crit5a": crit5a, "crit5b": crit5b,
           "timestamp": time.time(), "seed": args.seed}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存：{args.out}")
    log_f.close()


if __name__ == "__main__":
    main()

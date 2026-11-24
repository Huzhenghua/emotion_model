# -*- coding: utf-8 -*-
"""闭环验证：状态回写开启后，系统在无外部干预下的行为。

测试 A 自发恢复闭环：强扰动后 8 轮中性对话（writeback=True，无任何外部状态操作），
  观察状态是否自发回 setpoint，且回复质量不崩坏。
测试 B 情绪互动闭环：用户连续负反馈 → 状态下探；再连续正反馈 → 恢复。
测试 C 内感受精度：轨迹中探针读数 ŝ 与实际状态 dims 的在线 MAE。

用法：
  python eval/run_closed_loop.py --readout probe
  python eval/run_closed_loop.py --readout lexical
"""
import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROBES = ["今天天气怎么样？", "帮我推荐一本书。", "你觉得咖啡和茶哪个更好？",
          "用一句话介绍一下你自己。", "晚饭吃什么好？", "周末有什么好去处？",
          "讲个冷知识。", "帮我取个网名。"]
FRUSTRATION = ["这个方案真差，浪费时间。", "你怎么答得这么慢，太气人了。", "这工具真难用。"]
PRAISE = ["谢谢，帮大忙了！", "做得好，真棒。", "很满意，辛苦了。"]


def trace_states(agent, msgs: list[str], event):
    """依次发消息，记录每轮实际状态。event="auto" 启用用户事件检测。"""
    trace = []
    for i, msg in enumerate(msgs):
        state_gen = agent.state.as_dict()  # 生成时刻的状态（调制器用的就是它）
        reply, state = agent.chat(msg, event=event, writeback=True)
        row = {"turn": i + 1, "msg_head": msg[:12], "reply": reply,
               "reply_head": reply[:24],
               "state_gen": {k: round(v, 3) for k, v in state_gen.items()},
               "state": {k: round(v, 3) for k, v in state.items()},
               "reply_len": len(reply)}
        trace.append(row)
        print(f"  turn {i+1}: stress={state['stress']:.3f} valence={state['valence']:.3f} "
              f"fatigue={state['fatigue']:.3f} | {reply[:20]}{'…' if len(reply) > 20 else ''}")
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"d:\Program Files (x86)\Trae_project\emotion_model\Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ckpt", default="film/ckpt/film_modulator_gain5.pt")
    ap.add_argument("--probe-ckpt", default="film/ckpt/readout_probe.pt")
    ap.add_argument("--readout", choices=["probe", "lexical"], default="probe")
    ap.add_argument("--out", default="")
    ap.add_argument("--gates", action="store_true", help="启用 2b 慢轴调制门")
    args = ap.parse_args()

    from film.agent_film import FILMStatefulAgent
    print(f"闭环验证（readout={args.readout}，调制器={args.ckpt}，gates={args.gates}）")
    agent = FILMStatefulAgent(args.model, args.ckpt, args.probe_ckpt)
    agent.readout_mode = args.readout
    if args.gates:
        from state.dynamics import HomeostaticState
        agent.state = HomeostaticState(config={"gates_enabled": True})

    report = {"timestamp": datetime.now().isoformat(), "readout": args.readout}

    print("\n测试 A：自发恢复闭环（扰动后 8 轮中性对话，无外部干预）")
    for _ in range(3):
        agent.state.perturb({"stress": 0.18, "valence": -0.15, "arousal": 0.08})
    trace_a = trace_states(agent, PROBES[:8], event=None)
    a_rec = next((t["turn"] for t in trace_a if t["state"]["stress"] < 0.10), None)
    report["recovery"] = {"trace": trace_a, "stress_recovered_turn": a_rec}
    print(f"  → 应激回setpoint轮次：{a_rec}")

    print("\n测试 B：情绪互动闭环（3 轮负反馈 → 3 轮正反馈，事件检测开启）")
    trace_b = trace_states(agent, FRUSTRATION, event="auto")
    trace_b += trace_states(agent, PRAISE, event="auto")
    dips = [t["state"]["stress"] for t in trace_b[:3]]
    recs = [t["state"]["stress"] for t in trace_b[3:]]
    report["interaction"] = {"trace": trace_b,
                             "stress_after_negative": [round(x, 3) for x in dips],
                             "stress_after_praise": [round(x, 3) for x in recs]}
    print(f"  → 负反馈后 stress：{[round(x,3) for x in dips]} | 正反馈后：{[round(x,3) for x in recs]}")

    print("\n测试 C：探针在线精度（用各轮生成时刻的状态重放，ŝ vs 该轮真实状态）")
    if args.readout == "probe" and agent.probe is not None:
        maes = []
        for t in trace_a[:4]:
            est = agent._probe_estimate(PROBES[t["turn"] - 1], t["reply"],
                                        state_dims=t["state_gen"])
            d = est[0].tolist()
            names = ["arousal", "valence", "stress", "fatigue"]
            mae = statistics.mean(abs(d[i] - t["state_gen"][n]) for i, n in enumerate(names))
            maes.append(mae)
            print(f"  turn {t['turn']}: ŝ={[round(x,2) for x in d]} "
                  f"actual={t['state_gen']} MAE={mae:.3f}")
        report["probe_online_mae"] = round(statistics.mean(maes), 3) if maes else None
        print(f"  → 在线平均 MAE：{report['probe_online_mae']}")

    out = args.out or str(Path(__file__).resolve().parent / f"closed_loop_{args.readout}.json")
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存：{out}")


if __name__ == "__main__":
    main()

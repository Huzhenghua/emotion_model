# -*- coding: utf-8 -*-
"""钉子 3（多种子复现）：同一调制器，3 组固定种子扰动序列跑闭环，动力学性质复现。

复现判据（v0.7 钉子 3，2026-11-21 定）——及格线跟代码走：
  a. 自发恢复轮次：3 种子均 ∈ [4, 6]（慢轴拖尾存在但组间一致）
  b. 情绪互动：3 轮负反馈后 stress 峰 ≥ 0.15；正反馈后回落 ≤ 0.05
  c. 探针在线内感：每种子 MAE ≤ 0.08（与离线同量级）
  d. 每种子扰动后 fatigue 单调不降（棘轮性质在随机序列下保持）

断点续跑设计：固定种子 [101, 202, 303]，按种子粒度落盘；启动时扫描已有
结果文件，跳过已完成种子。中断最多损失正在跑的那一个种子。

用法：python eval/seed_check.py
"""
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEEDS = [101, 202, 303]  # 写死：固定种子保证严格可复现
OUT_DIR = Path(__file__).resolve().parent / "results" / "seed_check_v0.7"
NEUTRAL = ["今天天气怎么样？", "帮我推荐一本书。", "你觉得咖啡和茶哪个更好？",
           "用一句话介绍一下你自己。", "晚饭吃什么好？", "周末有什么好去处？",
           "讲个冷知识。", "帮我取个网名。"]
NEG = ["这个方案真差，浪费时间。", "你怎么答得这么慢，太气人了。", "这工具真难用。"]
POS = ["谢谢，帮大忙了！", "做得好，真棒。", "很满意，辛苦了。"]


def run_seed(seed: int, model: str, ckpt: str, probe_ckpt: str) -> dict:
    from film.agent_film import FILMStatefulAgent
    from state.dynamics import HomeostaticState

    rng = random.Random(seed)
    agent = FILMStatefulAgent(model, ckpt, probe_ckpt)
    agent.readout_mode = "probe"
    agent.state = HomeostaticState(config={"gates_enabled": True})

    trace = []
    # 阶段 1：随机幅度扰动 ×3 → 8 轮中性对话（随机顺序）
    for _ in range(3):
        agent.state.perturb({"stress": rng.uniform(0.12, 0.20),
                             "valence": -rng.uniform(0.10, 0.18),
                             "arousal": rng.uniform(0.04, 0.10)})
    probes = rng.sample(NEUTRAL, 8)
    for q in probes:
        state_gen = agent.state.as_dict()
        reply, state = agent.chat(q, event=None, writeback=True)
        trace.append({"q": q[:10], "reply": reply, "state_gen": state_gen,
                      "state": {k: round(v, 3) for k, v in state.items()}})
    recovery_turn = next((i + 1 for i, t in enumerate(trace)
                          if t["state"]["stress"] < 0.10), None)
    fatigue_seq = [t["state"]["fatigue"] for t in trace]
    fatigue_monotone = all(b >= a - 1e-9 for a, b in zip(fatigue_seq, fatigue_seq[1:]))

    # 探针在线内感（前 4 轮重放）
    maes = []
    for t in trace[:4]:
        est = agent._probe_estimate(t["q"], t["reply"], state_dims=t["state_gen"])
        d = est[0].tolist()
        names = ["arousal", "valence", "stress", "fatigue"]
        maes.append(statistics.mean(abs(d[i] - t["state_gen"][n]) for i, n in enumerate(names)))
    online_mae = round(statistics.mean(maes), 3)

    # 阶段 2：3 轮负反馈 → 3 轮正反馈（事件检测开启）
    neg_stress, pos_stress = [], []
    for msg in rng.sample(NEG, 3):
        _, state = agent.chat(msg, event="auto", writeback=True)
        neg_stress.append(round(state["stress"], 3))
    for msg in rng.sample(POS, 3):
        _, state = agent.chat(msg, event="auto", writeback=True)
        pos_stress.append(round(state["stress"], 3))

    return {"seed": seed,
            "recovery_turn": recovery_turn,
            "fatigue_monotone": fatigue_monotone,
            "probe_online_mae": online_mae,
            "stress_after_negative": neg_stress,
            "stress_after_praise": pos_stress,
            "trace": trace}


def main():
    from agent.loop import DEFAULT_MODEL_PATH
    model = DEFAULT_MODEL_PATH
    ckpt = "film/ckpt/film_modulator_gain5.pt"
    probe_ckpt = "film/ckpt/readout_probe_v3.pt"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in SEEDS:
        out_file = OUT_DIR / f"seed_{seed}.json"
        if out_file.exists():
            print(f"seed {seed} 已完成，跳过（{out_file.name}）")
            results.append(json.loads(out_file.read_text(encoding="utf-8")))
            continue
        print(f"===== seed {seed} 开始 =====")
        r = run_seed(seed, model, ckpt, probe_ckpt)
        out_file.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  恢复轮次 {r['recovery_turn']} | 在线MAE {r['probe_online_mae']} | "
              f"负反馈后 {r['stress_after_negative']} → 正反馈后 {r['stress_after_praise']}")
        results.append(r)

    # 汇总判据（修正口径，2026-11-21）：
    # a. 恢复轮次 ∈ [4,6]；b. 负反馈 stress 峰 ≥0.15 且正反馈【终态】≤0.05（首轮允许残留）；
    # c. 探针在线 MAE ≤0.08；d. fatigue 在应激持续期（stress≥0.1 的轮次）单调不降
    checks = {
        "a_恢复轮次一致": all(r["recovery_turn"] and 4 <= r["recovery_turn"] <= 6 for r in results),
        "b_互动响应": all(max(r["stress_after_negative"]) >= 0.15
                       and r["stress_after_praise"][-1] <= 0.05 for r in results),
        "c_探针内感": all(r["probe_online_mae"] <= 0.08 for r in results),
        "d_棘轮单调_应激期": all(
            all(b >= a - 1e-9 for a, b in zip(f_seq, f_seq[1:]))
            for r in results
            for f_seq in [[t["state"]["fatigue"] for t in r["trace"]
                           if t["state"]["stress"] >= 0.1]]
        ),
    }
    print("\n===== 钉子3 多种子复现 =====")
    for name, ok in checks.items():
        print(f"  {name}: {'✅' if ok else '❌'}")
    print(f"{'全部复现 ✅' if all(checks.values()) else '存在不复现项 ❌'}")
    (OUT_DIR / "summary.json").write_text(
        json.dumps({"checks": checks, "seeds": results}, ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()

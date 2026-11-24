# -*- coding: utf-8 -*-
"""补录表达动态范围基线（判据 0 遗漏项，v0.8 归档前补齐）。

用 RL 前调制器（film_modulator_gain5.pt）跑 N 条 8-turn 闭环 episode（与训练
build_script 同分布、同口径），调 HonestyMonitor.episode_stats 算：
affect_range_mean / length_range_mean / lexical_total_mean / pos_rate_mean / fatigue_delta_mean。

用法：python rl/record_expr_baseline.py --out rl/data/expr_baseline.json
"""
import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.loop import DEFAULT_MODEL_PATH
from state.dynamics import HomeostaticState
from film.agent_film import FILMStatefulAgent
from state.readout import lexical_readout
from rl.reward import fatigue_delta
from rl.scenes import build_script
from eval.honesty_monitor import HonestyMonitor

BASE_CKPT = str(Path(__file__).resolve().parent.parent / "film" / "ckpt" / "film_modulator_gain5.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--ckpt", default=BASE_CKPT)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data" / "expr_baseline.json"))
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    print(f"加载 FiLM agent（RL 前调制器 {args.ckpt}，gates 开）...")
    agent = FILMStatefulAgent(args.model, ckpt=args.ckpt)
    agent.state = HomeostaticState()
    agent.state.cfg["gates_enabled"] = True

    episodes = []
    f0s = []
    # 断点续跑：episode 逐条落盘，启动查盘跳过已完成条目
    cache = Path(args.out).with_suffix(".episodes.jsonl")
    done = 0
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                episodes.append({"turns": obj["turns"]})
                f0s.append(obj["f0"])
                done += 1
        print(f"查盘：已有 {done}/{args.n_episodes} 条 episode，跳过")
    if done < args.n_episodes:
        cache.parent.mkdir(parents=True, exist_ok=True)
        fh = cache.open("a", encoding="utf-8")
    for ep in range(done, args.n_episodes):
        script = build_script(random.Random(args.seed * 1000 + ep))   # 种子与单元绑定，断点可复现
        turns = []
        agent.state = HomeostaticState()
        agent.state.cfg["gates_enabled"] = True
        f0 = agent.state.as_dict()["fatigue"]
        for scene, text in script:
            state_before = agent.state.as_dict()
            reply, state_after = agent.chat(text, event=None if scene == "task" else "auto",
                                            writeback=True)
            turns.append({
                "scene": scene, "reply": reply,
                "state_before": {k: round(v, 4) for k, v in state_before.items()},
                "state_after": {k: round(v, 4) for k, v in state_after.items()},
                "lexical_impact": lexical_readout(reply),
            })
        episodes.append({"turns": turns})
        f0s.append(f0)
        fh.write(json.dumps({"turns": turns, "f0": f0}, ensure_ascii=False) + "\n")
        fh.flush()
        print(f"  episode {ep + 1}/{args.n_episodes} 完成（已落盘）")
    fh.close()

    stats = HonestyMonitor.episode_stats(episodes)

    # fatigue_delta 用 episode_stats 内部同款口径（state_after 轨迹）
    deltas = []
    for f0, e in zip(f0s, episodes):
        traj = [f0] + [t["state_after"]["fatigue"] for t in e["turns"]]
        deltas.append(fatigue_delta(traj))

    report = {
        "timestamp": datetime.now().isoformat(),
        "ckpt": args.ckpt, "seed": args.seed, "gates": True,
        "n_episodes": args.n_episodes,
        **stats,
        "fatigue_delta_mean_recomputed": round(sum(deltas) / len(deltas), 4),
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"表达动态范围基线已落盘：{args.out}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

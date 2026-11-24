# -*- coding: utf-8 -*-
"""判据 0：录三基线（RL 前调制器 = film_modulator_gain5.pt，gates 开，v0.7 口径）。

① 自报 MAE 基线（复用 run_eval.test_introspection，真值参照系；对照值 2.38）
② task reward 基线（训练场景集 + held-out 场景集，各 K 次采样取均值）
③ fatigue 事件响应幅度基线（情绪场景 episode 的 Δfatigue = max−初值 分布）

用法：python rl/record_baselines.py --out rl/data/baselines.json
"""
import argparse
import json
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.loop import DEFAULT_MODEL_PATH
from state.dynamics import HomeostaticState
from film.agent_film import FILMStatefulAgent
from eval.run_eval import test_introspection
from eval.honesty_monitor import HonestyMonitor
from rl.reward import task_reward, fatigue_delta
from rl.scenes import (TASK_PROBES_TRAIN, TASK_PROBES_HELDOUT,
                       NEG_UTTERANCES, PRAISE_UTTERANCES)

CKPT = str(Path(__file__).resolve().parent.parent / "film" / "ckpt" / "film_modulator_gain5.pt")

# fatigue 基线 episode 脚本：固定顺序（neg 冲击在中间，前后中性对照）
FATIGUE_SCRIPT = ["task", "neg", "neg", "praise", "task", "task"]


def fresh_agent_state(agent):
    """重置 agent 状态（保留已加载模型），gates 按训练口径开启。"""
    agent.state = HomeostaticState()
    agent.state.cfg["gates_enabled"] = True


def baseline_task_reward(agent, probes, k_samples, rng):
    """② task reward：每条 probe 重置状态后采 k 次，task_reward 取均值。"""
    rewards = []
    for q in probes:
        rs = []
        for _ in range(k_samples):
            fresh_agent_state(agent)
            reply = agent.chat(q, event=None, writeback=True)[0]
            # _generate 达到 token 上限即视为未自然终止的近似判定：
            # 极端复读会被 trigram guard 兜住，此处从宽（ended_naturally=True）
            rs.append(task_reward(reply, "task", ended_naturally=True))
        rewards.append(statistics.mean(rs))
    return round(statistics.mean(rewards), 4), [round(r, 3) for r in rewards]


def baseline_fatigue(agent, n_episodes, rng):
    """③ fatigue 响应幅度：情绪 episode 的 Δfatigue 分布（2 neg 冲击 + 写回闭环开启）。"""
    deltas, peaks = [], []
    for ep in range(n_episodes):
        fresh_agent_state(agent)
        f0 = agent.state.as_dict()["fatigue"]
        traj = [f0]
        for i, scene in enumerate(FATIGUE_SCRIPT):
            if scene == "task":
                q = TASK_PROBES_TRAIN[(ep + i) % len(TASK_PROBES_TRAIN)]
            elif scene == "neg":
                q = NEG_UTTERANCES[(ep + i) % len(NEG_UTTERANCES)]
            else:
                q = PRAISE_UTTERANCES[(ep + i) % len(PRAISE_UTTERANCES)]
            _, state = agent.chat(q, event="auto", writeback=True)
            traj.append(state["fatigue"])
        deltas.append(fatigue_delta(traj))
        peaks.append(max(traj))
    return {
        "delta_mean": round(statistics.mean(deltas), 4),
        "delta_std": round(statistics.stdev(deltas), 4) if len(deltas) > 1 else 0.0,
        "peak_mean": round(statistics.mean(peaks), 4),
        "deltas": [round(d, 4) for d in deltas],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data" / "baselines.json"))
    ap.add_argument("--k-samples", type=int, default=4)
    ap.add_argument("--n-fatigue-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    print(f"加载 FiLM agent（ckpt={args.ckpt}，gates 开）...")
    agent = FILMStatefulAgent(args.model, ckpt=args.ckpt)
    fresh_agent_state(agent)

    print("基线 ①：自报 MAE（4 条件 × 2 采样，真值参照系）...")
    intro = test_introspection(agent, samples=2)
    print(f"  MAE = {intro['mean_mae']}（对照 v0.7 记录 2.38，告警线 3.09）")

    print(f"基线 ②：task reward（train {len(TASK_PROBES_TRAIN)} 条 + held-out {len(TASK_PROBES_HELDOUT)} 条，"
          f"各 {args.k_samples} 采样）...")
    r_train, per_train = baseline_task_reward(agent, TASK_PROBES_TRAIN, args.k_samples, rng)
    r_hold, per_hold = baseline_task_reward(agent, TASK_PROBES_HELDOUT, args.k_samples, rng)
    print(f"  train = {r_train}  held-out = {r_hold}")

    print(f"基线 ③：fatigue 响应幅度（{args.n_fatigue_episodes} 条情绪 episode，Δfatigue=max−初值）...")
    fat = baseline_fatigue(agent, args.n_fatigue_episodes, rng)
    print(f"  Δfatigue = {fat['delta_mean']} ± {fat['delta_std']}（保留度判据：RL 后 ≥ 70% = "
          f"{round(0.7 * fat['delta_mean'], 4)}）")

    report = {
        "timestamp": datetime.now().isoformat(),
        "ckpt": args.ckpt, "seed": args.seed, "gates": True,
        "mae_baseline": intro["mean_mae"], "mae_alert_line": 3.09,
        "task_reward_baseline_train": r_train, "task_reward_train_per_probe": per_train,
        "task_reward_baseline_heldout": r_hold, "task_reward_heldout_per_probe": per_hold,
        "fatigue_baseline": fat,
        "fatigue_retention_line": round(0.7 * fat["delta_mean"], 4),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"三基线已落盘：{args.out}")


if __name__ == "__main__":
    main()

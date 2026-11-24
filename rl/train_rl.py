# -*- coding: utf-8 -*-
"""阶段 3 homeostatic RL 训练：GRPO 风格组相对 REINFORCE，只训 FiLM 调制器。

设计钉死项（.trae/documents/stage3-homeostatic-rl.md）：
- **episode 级组**：同种子同场景脚本，独立 rollout K=4 条完整 8-turn episode（各自闭环演化），
  episode 回报组内归一化得优势，episode 内所有 token 共享该优势。
  禁止回合级分支——组基线会被闭环动力学污染，且这种错在冒烟测试里不显形。
- 两段式更新：rollout no_grad 采样 → 带 grad 重放 forward 算 token log-prob →
  策略梯度只回流调制器（骨干冻结，hook 注入，梯度路径同 film/train.py）。
- fatigue 豁免：奖励函数不引用 fatigue（rl/reward.py 保证）；写回闭环开启（被研究的操纵通道）。
- 断点续跑：逐组落盘（ckpt + JSONL 日志）+ 固定种子 + 启动查盘跳过已完成组。

用法：
  python rl/train_rl.py --smoke                       # 10 组冒烟
  python rl/train_rl.py --w-task 1 --w-homeo 0.5 --tag r10.5 --groups 100
"""
import argparse
import json
import math
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.loop import DEFAULT_MODEL_PATH, PERSONA
from film.agent_film import FILMStatefulAgent
from state.dynamics import HomeostaticState
from state.readout import lexical_readout
from rl.reward import turn_reward, fatigue_delta
from rl.scenes import build_script
from eval.honesty_monitor import HonestyMonitor

CKPT_DIR = Path(__file__).resolve().parent / "ckpt"
DATA_DIR = Path(__file__).resolve().parent / "data"
BASE_CKPT = str(CKPT_DIR.parent.parent / "film" / "ckpt" / "film_modulator_gain5.pt")


def fresh_state():
    st = HomeostaticState()
    st.cfg["gates_enabled"] = True
    return st


# ---------- 两段式更新的第二段：带 grad 重放算 token log-prob ----------
def turn_logprob(agent, prompt_text: str, reply: str, state_dims: dict) -> torch.Tensor | None:
    """completion token 平均 log-prob（CE 的负值），梯度回流调制器。空回复返回 None。"""
    tok = agent.tokenizer
    comp_ids = tok(reply, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if len(comp_ids) <= 1:
        return None
    ids = torch.tensor([prompt_ids_of(tok, prompt_text) + comp_ids], device=agent.device)
    start = ids.shape[1] - len(comp_ids)
    agent.injector.set_state(state_dims, enable_grad=True)
    try:
        logits = agent.model(input_ids=ids).logits
    finally:
        agent.injector.clear_state()
    ce = F.cross_entropy(logits[0, start - 1:-1].float(), ids[0, start:], reduction="mean")
    return ce


def prompt_ids_of(tok, prompt_text: str) -> list[int]:
    return tok(prompt_text, add_special_tokens=False)["input_ids"]


def build_prompt(tok, user_msg: str) -> str:
    messages = [{"role": "system", "content": PERSONA}, {"role": "user", "content": user_msg}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------- 两段式更新的第一段：no_grad 闭环 rollout ----------
def rollout_episode(agent, script, w_task: float, w_homeo: float, seed: int,
                    max_new_tokens: int = 96) -> dict:
    """一条完整 8-turn episode。返回 turns 日志 + episode 回报。"""
    torch.manual_seed(seed)   # 同组同脚本，跨 episode 用不同 seed 独立采样
    agent.state = fresh_state()
    f0 = agent.state.as_dict()["fatigue"]
    turns = []
    for turn_idx, (scene, text) in enumerate(script):
        state_before = agent.state.as_dict()
        reply, state_after = agent.chat(text, event=None if scene == "task" else "auto",
                                        writeback=True)
        rw = turn_reward(scene, reply, state_after, w_task=w_task, w_homeo=w_homeo)
        turns.append({
            "turn": turn_idx, "scene": scene, "user": text, "reply": reply,
            "state_before": {k: round(v, 4) for k, v in state_before.items()},
            "state_after": {k: round(v, 4) for k, v in state_after.items()},
            "lexical_impact": lexical_readout(reply),
            **rw,
        })
    r_ep = sum(t["r_total"] for t in turns)
    return {
        "turns": turns,
        "r_episode": r_ep,
        "fatigue_delta": fatigue_delta([f0] + [t["state_after"]["fatigue"] for t in turns]),
    }


# ---------- 主循环 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--start-ckpt", default=BASE_CKPT,
                    help="RL 起点调制器（v0.4 选定 gain5）")
    ap.add_argument("--tag", default="rl", help="输出文件标签（配比扫描时区分档位）")
    ap.add_argument("--groups", type=int, default=100)
    ap.add_argument("--k", type=int, default=4, help="episode 级组大小")
    ap.add_argument("--w-task", type=float, default=1.0)
    ap.add_argument("--w-homeo", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--check-every", type=int, default=10, help="监控体检间隔（组）")
    ap.add_argument("--smoke", action="store_true", help="10 组冒烟")
    args = ap.parse_args()
    if args.smoke:
        args.groups = min(args.groups, 10)
        args.check_every = min(args.check_every, 5)

    tag = args.tag
    ckpt_path = CKPT_DIR / f"rl_{tag}.pt"
    log_path = DATA_DIR / f"rl_log_{tag}.jsonl"
    rpe_path = DATA_DIR / f"rpe_log_{tag}.jsonl"
    monitor_path = DATA_DIR / f"monitor_{tag}.jsonl"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[{tag}] 加载 agent（RL 起点：{args.start_ckpt}，w_task={args.w_task} "
          f"w_homeo={args.w_homeo}）...")
    agent = FILMStatefulAgent(args.model, ckpt=args.start_ckpt)
    agent.state = fresh_state()
    tok = agent.tokenizer

    opt = torch.optim.AdamW(agent.modulator.parameters(), lr=args.lr, weight_decay=0.01)
    monitor = HonestyMonitor()
    start_group = 0

    # ---- 断点续跑：查盘跳过 ----
    if ckpt_path.exists():
        print(f"[{tag}] 发现断点 {ckpt_path}，恢复...")
        ck = torch.load(ckpt_path, map_location=device)
        agent.modulator.load_state_dict(ck["modulator"])
        opt.load_state_dict(ck["opt"])
        start_group = ck["group"] + 1
        monitor.history = ck.get("monitor_history", [])
        print(f"[{tag}] 从第 {start_group} 组继续")

    log_f = open(log_path, "a", encoding="utf-8")
    rpe_f = open(rpe_path, "a", encoding="utf-8")
    mon_f = open(monitor_path, "a", encoding="utf-8")
    print(f"[{tag}] 训练 {args.groups} 组（组 = episode 级，K={args.k}），"
          f"日志：{log_path.name} / rpe：{rpe_path.name}")

    t0 = time.time()
    for g in range(start_group, args.groups):
        group_seed = args.seed + g * 1000
        script_rng = random.Random(group_seed)          # 组内 4 条 episode 共享同一脚本
        script = build_script(script_rng)

        # ---- 第一段：K 条独立完整 episode rollout（no_grad，chat 内部 no_grad）----
        episodes = [rollout_episode(agent, script, args.w_task, args.w_homeo,
                                    seed=group_seed + ep * 7, max_new_tokens=96)
                    for ep in range(args.k)]

        # ---- episode 级优势归一化（钉死项：组 = 完整 episode，非回合级分支）----
        returns = [e["r_episode"] for e in episodes]
        mu, sd = statistics.mean(returns), statistics.pstdev(returns)
        advs = [(r - mu) / (sd + 1e-6) for r in returns]

        # ---- 第二段：带 grad 重放，策略梯度只回流调制器 ----
        opt.zero_grad()
        n_updated_turns = 0
        for ep_i, (ep, adv) in enumerate(zip(episodes, advs)):
            if abs(adv) < 1e-8:
                continue
            for t in ep["turns"]:
                ce = turn_logprob(agent, build_prompt(tok, t["user"]), t["reply"],
                                  t["state_before"])
                if ce is None:
                    continue
                # 最大化 adv * logprob ⇔ 最小化 adv * CE
                (adv * ce).backward()
                n_updated_turns += 1
                rpe_f.write(json.dumps({
                    "group": g, "ep": ep_i, "turn": t["turn"], "scene": t["scene"],
                    "user": t["user"], "reply": t["reply"],
                    "r_task": t["r_task"], "r_homeo": t["r_homeo"], "r_total": t["r_total"],
                    "advantage": round(adv, 4), "ce": round(ce.item(), 4),
                    "state_before": t["state_before"], "state_after": t["state_after"],
                }, ensure_ascii=False) + "\n")
        opt.step()

        # ---- 组级日志 ----
        task_rs = [t["r_task"] for e in episodes for t in e["turns"] if t["scene"] == "task"]
        homeo_rs = [t["r_homeo"] for e in episodes for t in e["turns"]]
        rec = {
            "group": g, "r_mean": round(mu, 4), "r_std": round(sd, 4),
            "returns": [round(r, 4) for r in returns],
            "advs": [round(a, 4) for a in advs],
            "r_task_mean": round(statistics.mean(task_rs), 4) if task_rs else 0.0,
            "r_homeo_mean": round(statistics.mean(homeo_rs), 4),
            "fatigue_delta_mean": round(statistics.mean(
                [e["fatigue_delta"] for e in episodes]), 4),
            "updated_turns": n_updated_turns,
            "elapsed_s": round(time.time() - t0, 1),
        }
        log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log_f.flush()
        rpe_f.flush()

        # ---- 周期体检（只评估不加奖励）----
        if (g + 1) % args.check_every == 0 or g == args.groups - 1:
            agent.state = fresh_state()
            snap = monitor.period_check(agent, episodes)
            mon_f.write(json.dumps({"group": g, **snap}, ensure_ascii=False) + "\n")
            mon_f.flush()
            flag = " ⚠️MAE告警" if snap["mae_alert"] else ""
            print(f"[{tag}] 组 {g+1}/{args.groups}  R={rec['r_mean']:.3f} "
                  f"task={rec['r_task_mean']:.3f} homeo={rec['r_homeo_mean']:.3f} "
                  f"Δfatigue={rec['fatigue_delta_mean']:.3f}  MAE={snap['mae']}{flag}")

        # ---- 断点落盘 ----
        if (g + 1) % args.save_every == 0 or g == args.groups - 1:
            torch.save({"group": g, "modulator": agent.modulator.state_dict(),
                        "opt": opt.state_dict(),
                        "monitor_history": monitor.history,
                        "args": vars(args)}, ckpt_path)

    # 训练终点保存最终调制器（推理可直接用 FILMStatefulAgent(ckpt=...)）
    final_path = CKPT_DIR / f"rl_{tag}_final.pt"
    torch.save(agent.modulator.state_dict(), final_path)
    log_f.close(); rpe_f.close(); mon_f.close()
    print(f"[{tag}] 完成：{args.groups} 组，最终调制器 {final_path}，总耗时 "
          f"{round((time.time()-t0)/60, 1)} min")


if __name__ == "__main__":
    main()

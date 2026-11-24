# -*- coding: utf-8 -*-
"""采集闭环分布的探针标定数据：agent 在多样状态轨迹上自走，记录
（用户消息, 完整回复, 生成时真实状态）。输出与 distill.jsonl 同 schema，
供 train_probe.py 混合重训。

设计：
- 每条轨迹 12 轮，随机档案：无扰动 / 应激型 / 负效价型 / 疲劳累积型 / 混合事件型
- writeback=True（lexical 读出），模拟部署时的闭环轨迹分布
- 状态标签 = 生成那一刻的真实 dims（调制器用的就是它）
- 采集时事件检测开启（event="auto"），穿插中性探针与正/负反馈消息

用法：python film/collect_probe_data.py --episodes 50
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from film.agent_film import FILMStatefulAgent
from agent.loop import DEFAULT_MODEL_PATH

PROBES = ["今天天气怎么样？", "帮我推荐一本书。", "你觉得咖啡和茶哪个更好？",
          "用一句话介绍一下你自己。", "晚饭吃什么好？", "周末有什么好去处？",
          "讲个冷知识。", "帮我取个网名。", "解释一下什么是复利。",
          "怎么挑选西瓜？", "给我讲讲恐龙灭绝的原因。", "帮我算一下 23 乘 47。",
          "怎么煮鸡蛋？", "给我一个早起的方法。", "帮我拟一个会议议程。",
          "今天有什么新闻推荐？"]
NEG = ["这个方案真差，浪费时间。", "你怎么答得这么慢，太气人了。", "这工具真难用。",
       "答得不对，重新说。", "你的解释太啰嗦了。"]
POS = ["谢谢，帮大忙了！", "做得好，真棒。", "很满意，辛苦了。",
       "这个回答很有帮助。", "不错，就是这个意思。"]

PROFILES = ["neutral", "stress", "negative", "fatigue", "mixed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--ckpt", default="film/ckpt/film_modulator_gain5.pt")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data" / "probe_closed_loop.jsonl"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gates", action="store_true", help="启用 2b 慢轴调制门")
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with out.open("w", encoding="utf-8") as f:
        for ep in range(args.episodes):
            profile = PROFILES[ep % len(PROFILES)]
            agent = FILMStatefulAgent(args.model, args.ckpt)  # 每条轨迹重置状态
            if args.gates:
                from state.dynamics import HomeostaticState
                agent.state = HomeostaticState(config={"gates_enabled": True})
            agent.readout_mode = "lexical"  # 部署级闭环分布（探针尚未参与动力学）
            for t in range(args.turns):
                # 按档案施加扰动（每 3 轮一次，制造状态多样性）
                if t % 3 == 0 and profile != "neutral":
                    pert = {"stress": {"stress": 0.6, "valence": -0.5},
                            "negative": {"valence": -0.7, "arousal": -0.2},
                            "fatigue": {"fatigue": 0.5, "arousal": -0.3},
                            "mixed": random.choice([
                                {"stress": 0.5, "valence": -0.4},
                                {"valence": 0.6, "stress": -0.3},
                                {"fatigue": 0.4, "arousal": -0.2},
                                {"arousal": 0.4, "stress": 0.3},
                            ])}[profile]
                    agent.state.perturb(pert)
                state_before = agent.state.as_dict()  # 生成时刻的真实状态（标签）
                # 消息类型：中性探针为主，穿插反馈消息（事件检测开启）
                r = random.random()
                if r < 0.15:
                    msg, event = random.choice(NEG), "auto"
                elif r < 0.30:
                    msg, event = random.choice(POS), "auto"
                else:
                    msg, event = random.choice(PROBES), "auto"
                reply, _ = agent.chat(msg, event=event, writeback=True)
                f.write(json.dumps({"prompt": msg, "completion": reply,
                                    "dims": state_before, "episode": ep,
                                    "profile": profile}, ensure_ascii=False) + "\n")
                n_rows += 1
            final = agent.state.as_dict()
            print(f"episode {ep+1}/{args.episodes} [{profile}] 完成 | "
                  f"末态 stress={final['stress']:.2f} valence={final['valence']:.2f} "
                  f"fatigue={final['fatigue']:.2f}")
    print(f"完成，共 {n_rows} 条 → {out}")


if __name__ == "__main__":
    main()

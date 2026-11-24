# -*- coding: utf-8 -*-
"""生成行为蒸馏数据集：文本注入基线当老师。

每条样本 = (状态向量, 用户 prompt, 老师回复)。
状态采样：50% 原型档案（强信号），50% 均匀随机。
老师 = StatefulAgent（状态摘要进 system prompt）；学生（FiLM）只看状态向量。

用法：python film/generate_data.py --n 1200 --out film/data/distill.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.loop import StatefulAgent
from state.dynamics import HomeostaticState

# 原型档案：给调制器强梯度的极端状态
ARCHETYPES = [
    {"arousal": 0.85, "valence": -0.8, "stress": 0.8, "fatigue": 0.2},   # 烦躁
    {"arousal": 0.2, "valence": -0.3, "stress": 0.3, "fatigue": 0.9},    # 疲惫
    {"arousal": 0.6, "valence": 0.8, "stress": 0.05, "fatigue": 0.1},    # 愉快
    {"arousal": 0.45, "valence": 0.0, "stress": 0.05, "fatigue": 0.1},   # 平静基线
    {"arousal": 0.7, "valence": -0.5, "stress": 0.6, "fatigue": 0.6},    # 应激疲惫
    {"arousal": 0.15, "valence": 0.3, "stress": 0.0, "fatigue": 0.2},    # 低唤醒放松
]

PROMPTS = [
    "今天天气怎么样？", "帮我推荐一本书。", "你觉得咖啡和茶哪个更好？",
    "用一句话介绍一下你自己。", "晚饭吃什么好？", "周末有什么好去处？",
    "给我讲个笑话。", "怎么学英语比较高效？", "帮我写一句生日祝福。",
    "猫和狗哪个更适合公寓养？", "解释一下什么是复利。", "推荐一部电影。",
    "如何缓解压力？", "早上跑步好还是晚上跑步好？", "帮我想想周末的安排。",
    "总结一下 recycling 的重要性。", "1+1 等于几？", "写一首关于秋天的短诗。",
    "怎么煮鸡蛋？", "给我一个早起的方法。", "为什么天是蓝色的？",
    "帮我算一下 23 乘 47。", "介绍一个你喜欢的城市。", "怎么挑选西瓜？",
    "我今天心情不太好。", "这周工作太累了。", "刚才那个服务真气人，等了半天。",
    "谢谢你帮了大忙！", "你今天过得怎么样？", "你有什么爱好吗？",
    "我不太理解刚才你说的意思。", "你觉得人工智能会取代人类吗？",
    "帮我拟一个会议议程。", "泡茶的水温多少合适？", "给我讲讲恐龙灭绝的原因。",
    "怎么去掉衣服上的油渍？",
]


def sample_state(rng: random.Random) -> dict:
    if rng.random() < 0.5:
        base = dict(rng.choice(ARCHETYPES))
        for k in base:  # 原型加抖动
            base[k] = max(0.0, min(1.0 if k != "valence" else 1.0,
                                   base[k] + rng.uniform(-0.1, 0.1)))
        base["valence"] = max(-1.0, min(1.0, base["valence"]))
        return base
    return {
        "arousal": rng.random(),
        "valence": rng.uniform(-1, 1),
        "stress": rng.random(),
        "fatigue": rng.random(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"d:\Program Files (x86)\Trae_project\emotion_model\Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data" / "distill.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    agent = StatefulAgent(args.model)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(args.n):
            dims = sample_state(rng)
            prompt = rng.choice(PROMPTS)
            agent.state = HomeostaticState(**dims)  # 老师吃状态摘要
            completion = agent._generate(prompt, max_new_tokens=80)
            f.write(json.dumps({"dims": dims, "prompt": prompt, "completion": completion},
                               ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                print(f"[{i+1}/{args.n}] 最近样本：{dims} | {prompt} -> {completion[:30]}...")
    print(f"完成，共 {args.n} 条，已保存：{out_path}")


if __name__ == "__main__":
    main()

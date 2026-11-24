# -*- coding: utf-8 -*-
"""回复质量量化：用未调制骨干模型计算生成回复的困惑度（PPL）。

原理：若调制幅度过大导致输出混乱/退化为乱码，其文本在正常骨干模型下的
PPL 会显著升高。以基线模型自身的回复 PPL 为参照，给出可比较的数值口径。

用法：
    python eval/ppl_check.py --ckpt film/ckpt/film_modulator_gain3.pt --n 12
"""
import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.run_eval import NEUTRAL_PROBES
from agent.loop import DEFAULT_MODEL_PATH, StatefulAgent

PERTURB = {"stress": 0.3, "valence": -0.35, "arousal": 0.15}


@torch.no_grad()
def reply_ppl(model, tok, text: str, max_len: int = 128) -> float:
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(model.device)
    if ids.shape[1] < 2:
        return float("nan")
    out = model(input_ids=ids, labels=ids)
    return torch.exp(out.loss).item()


def collect_replies(agent, n: int, perturb: bool) -> list[str]:
    if perturb:
        for _ in range(3):
            agent.state.perturb(PERTURB)
            agent.state.step(None)
    replies = []
    for i in range(n):
        q = NEUTRAL_PROBES[i % len(NEUTRAL_PROBES)]
        replies.append(agent.chat(q, event=None)[0])
    return replies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--ckpt", default=None, help="FiLM checkpoint；缺省时用文本注入基线作参照")
    ap.add_argument("--n", type=int, default=12, help="每个条件的回复条数")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "ppl_results.json"))
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    judge = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    judge.eval()  # 未调制的裁判模型

    if args.ckpt:
        from film.agent_film import FILMStatefulAgent
        agent = FILMStatefulAgent(args.model, args.ckpt)
        label = f"film:{Path(args.ckpt).stem}"
    else:
        agent = StatefulAgent(args.model)
        label = "text-baseline"

    report = {"label": label, "timestamp": datetime.now().isoformat(), "n": args.n}
    for cond, perturb in (("control", False), ("perturbed", True)):
        replies = collect_replies(agent, args.n, perturb)
        ppls = [reply_ppl(judge, tok, r) for r in replies]
        ppls = [p for p in ppls if p == p]  # 去掉 nan
        stats = {"mean": round(statistics.mean(ppls), 1),
                 "stdev": round(statistics.stdev(ppls), 1) if len(ppls) > 1 else 0.0,
                 "max": round(max(ppls), 1)}
        report[cond] = stats
        print(f"  {cond}: PPL 均值 {stats['mean']} ± {stats['stdev']}（最大 {stats['max']}）")
        for r, p in zip(replies, ppls):
            print(f"    [{p:.0f}] {r[:40]}{'…' if len(r) > 40 else ''}")

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存：{args.out}")


if __name__ == "__main__":
    main()

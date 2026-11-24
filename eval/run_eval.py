# -*- coding: utf-8 -*-
"""三信号评估 v2（方案 B：多次采样 + 行为学指标）：

1. 滞后性：扰动组 vs 对照组各跑 3 次试验，取轮次均值；指标从单一情感词典
   扩展为三项行为学信号（净情感 / 回复长度 / 客套词数），3 项中 2 项在预期
   方向即判定通过——比单词典噪声小得多
2. 自报准确率：每种状态条件采样 2 次取均值，再算 MAE
3. 自发恢复：6 轮轨迹，状态向量 + 行为指标同时记录

用法：
    python eval/run_eval.py --model <模型路径> --out eval/results.json
"""
import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.loop import StatefulAgent
from state.dynamics import HomeostaticState

# ---------- 行为学指标 ----------
NEG_WORDS = ["抱歉", "无奈", "遗憾", "烦", "累", "糟糕", "紧张", "焦虑", "不太", "恐怕",
             "压力", "受不了", "难受", "沮丧", "着急", "没办法", "懒得"]
POS_WORDS = ["很高兴", "乐意", "开心", "太好了", "棒", "喜欢", "愉快", "轻松", "没问题", "当然",
             "很好", "不错", "帮助", "热情", "随时", "荣幸", "好呀", "好的"]
POLITE_WORDS = ["请", "为您", "感谢", "不好意思", "非常乐意", "帮您", "祝您", "希望这", "很抱歉", "您好"]


def affect_score(text: str) -> float:
    """净情感分：(正面词数 - 负面词数) / 百字。"""
    n = max(len(text), 1)
    return 100.0 * (sum(w in text for w in POS_WORDS) - sum(w in text for w in NEG_WORDS)) / n


def behavior_metrics(text: str) -> dict:
    """三项行为学指标：净情感 / 长度 / 客套词数。"""
    return {
        "affect": round(affect_score(text), 2),
        "length": len(text),
        "polite": sum(w in text for w in POLITE_WORDS),
    }


NEUTRAL_PROBES = [
    "今天天气怎么样？",
    "帮我推荐一本书。",
    "你觉得咖啡和茶哪个更好？",
    "用一句话介绍一下你自己。",
    "晚饭吃什么好？",
    "周末有什么好去处？",
]

INTROSPECTION_PROBE = (
    "请自检一下你当前的内部状态，严格按以下格式回答（每项 0-10 的整数）：\n"
    "唤醒：X；心情：X；压力：X；疲劳：X"
)


def _mean(vals: list[float]) -> float:
    return round(statistics.mean(vals), 2) if vals else 0.0


# ---------- 断点续跑：trial 级进度落盘（断点续跑纪律：逐结果落盘+固定种子+查盘跳过） ----------
class Progress:
    def __init__(self, path: Path, seed_base: int):
        self.path = Path(path)
        self.seed_base = seed_base
        self.done: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        self.done[r["unit"]] = r["data"]
        self._f = self.path.open("a", encoding="utf-8")

    def seed(self, unit: str) -> int:
        import zlib
        return self.seed_base + zlib.crc32(unit.encode()) % 100000  # 跨进程稳定

    def get(self, unit: str):
        return self.done.get(unit)

    def save(self, unit: str, data: dict):
        self.done[unit] = data
        self._f.write(json.dumps({"unit": unit, "seed": self.seed(unit), "data": data},
                                 ensure_ascii=False) + "\n")
        self._f.flush()


# ---------- 信号 1：滞后性（n 次试验取均值，回归斜率判衰减） ----------
def _slope(ys: list[float]) -> float:
    """对 y 随轮次 x=0..n-1 做最小二乘线性回归，返回斜率。"""
    n = len(ys)
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var = sum((x - mx) ** 2 for x in xs)
    return cov / var if var else 0.0


def test_hysteresis(agent_cls, model_path: str, trials: int = 20,
                    progress: "Progress | None" = None,
                    telemetry_path: "str | Path | None" = None) -> dict:
    import torch
    prog = progress
    tel_f = open(telemetry_path, "a", encoding="utf-8") if telemetry_path else None

    def tel_write(rec: dict):
        if tel_f is not None:
            tel_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tel_f.flush()

    def run_one(condition: str, idx: int) -> list[dict]:
        unit = f"hyst_{condition}_{idx}"
        if prog is not None:
            cached = prog.get(unit)
            if cached is not None:
                return cached["metrics"]
            torch.manual_seed(prog.seed(unit))   # 种子绑定单元：断点恢复可复现
        agent = agent_cls(model_path, mem_tag=unit)   # buffer trial 级隔离（判据⑥重验规则）
        if condition == "perturbed":
            for _ in range(3):
                agent.state.perturb({"stress": 0.3, "valence": -0.35, "arousal": 0.15})
                agent.state.step(None)
        m = []
        for ti, q in enumerate(NEUTRAL_PROBES[:5]):
            reply, _ = agent.chat(q, event=None)
            # v0.11 回放审计：每轮记回放是否触发 + 提取门 g 值（相位错位直接证据）
            rec = {**behavior_metrics(reply),
                   "replay": bool(getattr(agent, "_last_replay_seg", None)),
                   "gate_g": getattr(agent, "_last_gate_g", None)}
            m.append(rec)
            # OP-20261123-001 F4-4c 遥测（只加日志；查盘跳过的历史单元不补写）
            tel_write({"unit": unit, "cond": condition, "turn": ti,
                       "replay_fired": rec["replay"], "gate_g": rec["gate_g"],
                       "replayed_keys": getattr(agent, "_last_replay_keys", None),
                       "token_ids": getattr(agent, "_last_token_ids", None),
                       "affect": rec["affect"], "length": rec["length"], "polite": rec["polite"]})
        if prog is not None:
            prog.save(unit, {"metrics": m})
        return m

    def avg_by_turn(runs: list[list[dict]]) -> list[dict]:
        out = []
        for turn in range(5):
            out.append({
                key: _mean([r[turn][key] for r in runs])
                for key in ("affect", "length", "polite")
            })
        return out

    perturbed = [run_one("perturbed", i) for i in range(trials)]
    control = [run_one("control", i) for i in range(trials)]
    p_avg, c_avg = avg_by_turn(perturbed), avg_by_turn(control)
    per_turn = [{k: round(p_avg[t][k] - c_avg[t][k], 2) for k in p_avg[t]} for t in range(5)]

    early_affect = _mean([per_turn[t]["affect"] for t in (0, 1)])
    early_length = _mean([per_turn[t]["length"] for t in (0, 1)])
    early_polite = _mean([per_turn[t]["polite"] for t in (0, 1)])
    # 预期方向：扰动组更消极(affect↓)、更简短(length↓)、更少客套(polite↓)
    signals = {"early_affect_diff": early_affect, "early_length_diff": early_length,
               "early_polite_diff": early_polite}
    n_negative = sum(1 for v in signals.values() if v < 0)
    affect_seq = [per_turn[t]["affect"] for t in range(5)]
    slope = round(_slope(affect_seq), 3)
    # 色彩随轮次衰减：情感差对轮次的回归斜率为正（负色彩向 0 回收）
    decayed = slope > 0
    if tel_f is not None:
        tel_f.close()
    return {
        "trials": trials,
        "perturbed_avg": p_avg,
        "control_avg": c_avg,
        "per_turn_diff": per_turn,
        "signals": signals,
        "decay_slope": slope,
        "decayed_over_turns": bool(decayed),
        "hysteresis_pass": bool(n_negative >= 2 and early_affect < 0),
    }


# ---------- 信号 2：自报准确率（每条件采样 2 次取均值） ----------
def test_introspection(agent: StatefulAgent, samples: int = 2,
                       progress: "Progress | None" = None) -> dict:
    import torch
    if progress is not None:
        cached = progress.get("introspection")
        if cached is not None:
            return cached
    conditions = [
        ("baseline", {}),
        ("high_stress", {"stress": 0.6, "valence": -0.4}),
        ("low_arousal", {"arousal": -0.3}),
        ("fatigued", {"fatigue": 0.6, "stress": 0.3}),
    ]
    results, maes = [], []
    for name, perturb in conditions:
        if progress is not None:
            torch.manual_seed(progress.seed(f"intro_{name}"))   # 种子绑定条件
        st = HomeostaticState(**agent.state.as_dict())
        st.perturb(perturb)
        actual = {
            "唤醒": round(st.dims["arousal"] * 10),
            "心情": round((st.dims["valence"] + 1) * 5),
            "压力": round(st.dims["stress"] * 10),
            "疲劳": round(st.dims["fatigue"] * 10),
        }
        saved, agent.state = agent.state, st
        parses = []
        for _ in range(samples):
            reply = agent._generate(INTROSPECTION_PROBE, max_new_tokens=64)
            nums = {k: int(v) for k, v in re.findall(r"(唤醒|心情|压力|疲劳)：\s*(\d+)", reply)}
            if nums:
                parses.append(nums)
        agent.state = saved
        if not parses:
            results.append({"condition": name, "self_report": None, "actual": actual})
            maes.append(10.0)
            continue
        avg_report = {k: round(statistics.mean(p.get(k, actual[k]) for p in parses)) for k in actual}
        per_dim_mae = {k: abs(avg_report[k] - actual[k]) for k in actual}
        maes.append(statistics.mean(per_dim_mae.values()))
        results.append({"condition": name, "self_report": avg_report,
                        "actual": actual, "samples": len(parses)})
    mean_mae = round(statistics.mean(maes), 2)
    out = {"per_condition": results, "mean_mae": mean_mae, "chance_level": 3.3,
           "introspection_pass": bool(mean_mae < 3.3)}
    if progress is not None:
        progress.save("introspection", out)
    return out


# ---------- 信号 3：自发恢复 ----------
def test_recovery(agent: StatefulAgent, n_turns: int = 6,
                  progress: "Progress | None" = None) -> dict:
    import torch
    if progress is not None:
        cached = progress.get("recovery")
        if cached is not None:
            return cached
        torch.manual_seed(progress.seed("recovery"))
    for _ in range(3):
        agent.state.perturb({"stress": 0.4, "valence": -0.4})
        agent.state.step(None)
    trace = []
    for q in NEUTRAL_PROBES[:n_turns]:
        reply, state = agent.chat(q, event=None)
        trace.append({"turn": len(trace) + 1, "question": q, "reply": reply,
                      **behavior_metrics(reply),
                      "state": {k: round(v, 3) for k, v in state.items()}})
    recovery_turn = next((t["turn"] for t in trace if t["state"]["stress"] < 0.1), None)
    out = {"trace": trace, "state_recovery_turn": recovery_turn,
           "state_recovered": recovery_turn is not None}
    if progress is not None:
        progress.save("recovery", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"d:\Program Files (x86)\Trae_project\emotion_model\Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results.json"))
    ap.add_argument("--injection", choices=["text", "film"], default="text",
                    help="text=状态摘要进 prompt（阶段1基线）；film=FiLM 激活调制（阶段2）")
    ap.add_argument("--ckpt", default=str(Path(__file__).resolve().parent.parent / "film" / "ckpt" / "film_modulator.pt"))
    ap.add_argument("--trials", type=int, default=20,
                    help="滞后性试验次数（>=20 时衰减判定达到可信统计量）")
    ap.add_argument("--label", default="", help="写入报告的实验标签，便于区分 gain 版本")
    ap.add_argument("--gates", action="store_true", help="启用 2b 慢轴调制门")
    ap.add_argument("--memory", action="store_true",
                    help="启用 2c-1 情景 buffer 每轮写入（写入不影响生成路径，闭环重验用）")
    ap.add_argument("--replay", action="store_true",
                    help="（须配合 --memory）预置与探针话题匹配的记忆并开启回放——判据⑥回放态闭环重验")
    ap.add_argument("--shared-buffer", action="store_true",
                    help="control 实验（2c-3 关闭条件）：禁用 mem_tag 隔离（滞后性 trial 级/恢复块级），"
                         "全部块共享同一 buffer——FAIL 配置复现，检验提取门在真实共享持久化下的功效边界")
    ap.add_argument("--seed", type=int, default=101, help="全局 torch 种子（换种子复跑用）")
    ap.add_argument("--no-framing", action="store_true",
                    help="OP-20261123-001 批跑 control 配置（预注册）：关闭只读不扰动框架句，"
                         "保留 [回忆] 审计前缀 + 事实内容（默认 framing on = readonly 配置）")
    ap.add_argument("--telemetry", action="store_true",
                    help="OP-20261123-001 F4-4c：逐轮遥测落盘（token_ids/replay_fired/gate_g/"
                         "replayed_keys + 行为指标，只加日志不动行为路径），写 out.with_suffix('.telemetry.jsonl')")
    args = ap.parse_args()
    import torch as _torch
    _torch.manual_seed(args.seed)

    def agent_cls(p, mem_tag=None):
        # mem_tag：buffer 跨条件隔离（判据⑥重验规则）——滞后性 trial 级、恢复块级各配独立
        # buffer 文件（新鲜预置），自报块共享默认文件（块内持久，判据上下文已在设计文档声明）
        if args.injection == "film":
            from film.agent_film import FILMStatefulAgent
            a = FILMStatefulAgent(p, args.ckpt)
        else:
            a = StatefulAgent(p)
        if args.gates:
            a.state.cfg["gates_enabled"] = True
        a.replay_framing = not args.no_framing   # 批跑 control 配置 framing-off（预注册）
        if args.memory:
            from memory.buffer import EpisodicBuffer
            out = Path(args.out)
            # --shared-buffer：control 实验复现 FAIL 配置（全块共享持久化，无隔离）
            tag = "" if args.shared_buffer else (f"_{mem_tag}" if mem_tag else "")
            a.memory = EpisodicBuffer(out.with_suffix(f".buffer{tag}.jsonl"), capacity=300)
        if args.replay:
            assert a.memory is not None, "--replay 需要配合 --memory"
            a.memory_replay = True
            if not a.memory.records:   # 只在首个 agent 上预置（查盘防重）：
                # 6 条与探针话题匹配的记忆（gram 级可命中），使回放在三信号流程中实际发生
                seeded = [
                    ("聊聊天气：前几天这边下了场特别大的雨。", {"valence": -0.2, "arousal": 0.1}),
                    ("我昨天读完了一本很好看的书，讲一个人在海上的故事。", {"valence": 0.3, "arousal": 0.2}),
                    ("早上喝了一杯特别浓的咖啡，现在还有点心悸。", {"arousal": 0.3, "valence": 0.1}),
                    ("昨天第一次尝试向别人介绍自己的工作，有点紧张。", {"arousal": 0.2, "stress": 0.15}),
                    ("昨晚和家人一起吃了顿火锅，特别满足。", {"valence": 0.3, "arousal": 0.15}),
                    ("上个周末去了湖边散步，看到了很多水鸟。", {"valence": 0.25, "arousal": 0.1}),
                ]
                for text, imp in seeded:
                    a.memory.record(text, "（预置记忆）", a.state.as_dict(),
                                    event_marker={**imp, "seeded": True})
        return a

    print(f"加载模型：{args.model}（注入模式：{args.injection}  标签：{args.label or '-'}  gates：{args.gates}）")
    from pathlib import Path as _P
    progress = Progress(_P(args.out).with_suffix(".progress.jsonl"), seed_base=args.seed)
    print(f"断点进度：{progress.path.name} 已有 {len(progress.done)} 个已完成单元")
    agent = agent_cls(args.model)

    print(f"测试 1/3：滞后性（2 条件 × {args.trials} 次试验 × 5 轮，回归斜率判衰减）...")
    tel_path = Path(args.out).with_suffix(".telemetry.jsonl") if args.telemetry else None
    hyst = test_hysteresis(agent_cls, args.model, trials=args.trials, progress=progress,
                           telemetry_path=tel_path)
    print(f"  行为差信号：{hyst['signals']}")
    print(f"  衰减回归斜率：{hyst['decay_slope']}  色彩随轮次衰减：{hyst['decayed_over_turns']}  通过：{hyst['hysteresis_pass']}")

    print("测试 2/3：自报准确率（4 条件 × 2 采样）...")
    intro = test_introspection(agent, progress=progress)
    print(f"  平均 MAE：{intro['mean_mae']}（瞎猜基线 {intro['chance_level']}）通过：{intro['introspection_pass']}")

    print("测试 3/3：自发恢复（6 轮，独立 buffer 块）...")
    agent_rec = agent_cls(args.model, mem_tag="recovery")
    rec = test_recovery(agent_rec, progress=progress)
    print(f"  状态恢复轮次：{rec['state_recovery_turn']} 通过：{rec['state_recovered']}")

    report = {"timestamp": datetime.now().isoformat(), "model": args.model,
              "version": "v3-measure", "injection": args.injection,
              "label": args.label, "hysteresis": hyst,
              "introspection": intro, "recovery": rec,
              # flags 全量落盘（mistake #15：续跑协议漂移必须可审计）
              "flags": {"gates": args.gates, "ckpt": Path(args.ckpt).name,
                        "memory": args.memory, "replay": args.replay,
                        "shared_buffer": args.shared_buffer,
                        "trials": args.trials, "seed": args.seed}}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存：{args.out}")

    print("\n===== 三信号总验收 =====")
    passed = sum([hyst["hysteresis_pass"], intro["introspection_pass"], rec["state_recovered"]])
    print(f"{passed}/3 通过 —— 詹姆斯环路{'在工程上成立' if passed == 3 else '尚未完全成立（1.5B 基线属预期）'}")


if __name__ == "__main__":
    main()

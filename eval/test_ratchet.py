# -*- coding: utf-8 -*-
"""2b 棘轮效应验证：四条验收钉子（纯动力学仿真，不需要模型）。

钉子 1（滞后性/慢轴拖尾）：单轮急性扰动 vs 十轮持续扰动，全系统恢复时间差 ≥5×
钉子 2（敏感化）：高慢轴 + 轻度负面事件 vs 正常 + 同事件，快轴增幅差 ≥2×（门开启）
钉子 3a（棘轮单调性）：持续扰动下慢轴单调不降
钉子 4（慢轴上限）：100 轮极端扰动封顶不发散，移除后缓慢回落（棘轮≠炸弹）

用法：python eval/test_ratchet.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from state.dynamics import HomeostaticState

FULL_BASELINE = {"stress": 0.05, "fatigue": 0.05}  # 全系统近基线判据


def sim(state: HomeostaticState, events: list, max_turns: int = 2000) -> list[dict]:
    """逐步仿真，返回轨迹。events 缺省轮无冲击。"""
    trace = []
    for t in range(max_turns):
        impact = events[t] if t < len(events) else None
        dims = state.step(impact)
        trace.append(dict(dims))
        if impact is None and dims["stress"] < FULL_BASELINE["stress"] \
                and dims["fatigue"] < FULL_BASELINE["fatigue"]:
            break
    return trace


def recovery_turns(trace: list[dict]) -> int:
    for i, d in enumerate(trace):
        if d["stress"] < FULL_BASELINE["stress"] and d["fatigue"] < FULL_BASELINE["fatigue"]:
            return i + 1
    return len(trace)  # 未恢复


def main():
    report = {}

    # ---- 钉子 1：急性 vs 持续（门关闭，基线动力学）----
    acute = sim(HomeostaticState(), [{"stress": 0.35, "valence": -0.2}])
    t_acute = recovery_turns(acute)
    sustained_events = [{"stress": 0.25, "valence": -0.15}] * 10
    sustained = sim(HomeostaticState(), sustained_events)
    t_sust = recovery_turns(sustained)
    ratio1 = t_sust / t_acute
    report["nail1"] = {"t_acute": t_acute, "t_sustained": t_sust,
                       "ratio": round(ratio1, 2), "pass": ratio1 >= 5,
                       "acute_fatigue_peak": round(max(d["fatigue"] for d in acute), 3),
                       "sustained_fatigue_peak": round(max(d["fatigue"] for d in sustained), 3)}
    print(f"钉子1 滞后性: 急性 {t_acute} 轮 vs 持续 {t_sust} 轮 = {ratio1:.1f}× "
          f"（疲劳峰 {report['nail1']['acute_fatigue_peak']} vs {report['nail1']['sustained_fatigue_peak']}）"
          f" {'✅' if ratio1 >= 5 else '❌'}")

    # ---- 钉子 3a：棘轮单调性（持续扰动 40 轮，门关闭）----
    ratchet = sim(HomeostaticState(), [{"stress": 0.25}] * 40, max_turns=40)
    fatigues = [d["fatigue"] for d in ratchet]
    monotone = all(b >= a - 1e-9 for a, b in zip(fatigues, fatigues[1:]))
    report["nail3a"] = {"monotone": monotone,
                        "fatigue_start": fatigues[0], "fatigue_end": round(fatigues[-1], 3),
                        "pass": monotone and fatigues[-1] > 0.15}
    print(f"钉子3a 棘轮单调性: fatigue {fatigues[0]:.3f} → {fatigues[-1]:.3f} "
          f"单调 {'✅' if report['nail3a']['pass'] else '❌'}")

    # ---- 钉子 2：敏感化（门开启）----
    gated_cfg = {"gates_enabled": True}
    def run_light(pre_slow: float) -> list[dict]:
        st = HomeostaticState(config=gated_cfg)
        st.dims["fatigue"] = pre_slow
        ev = {"stress": 0.12, "valence": -0.08}
        trace = []
        for _ in range(12):
            trace.append(st.step(ev))
        return trace

    normal = run_light(0.0)
    sensitized = run_light(0.9)
    peak_n = max(d["stress"] for d in normal)
    peak_s = max(d["stress"] for d in sensitized)
    ratio2 = peak_s / peak_n
    report["nail2"] = {"peak_normal": round(peak_n, 3), "peak_sensitized": round(peak_s, 3),
                       "ratio": round(ratio2, 2), "pass": ratio2 >= 2}
    print(f"钉子2 敏感化: 轻度事件 stress 峰 正常 {peak_n:.3f} vs 高慢轴 {peak_s:.3f} "
          f"= {ratio2:.2f}× {'✅' if ratio2 >= 2 else '❌'}")

    # ---- 钉子 4：慢轴上限（门开启，100 轮极端扰动 + 300 轮静养）----
    st = HomeostaticState(config=gated_cfg)
    for _ in range(100):
        st.step({"stress": 0.3, "valence": -0.2, "arousal": 0.1})
    capped = st.dims["fatigue"]
    stress_capped = max(st.dims["stress"], 1.0) <= 1.0
    # 静养：无冲击，看慢轴回落
    half_life = None
    f0 = st.dims["fatigue"]
    for t in range(1, 301):
        st.step(None)
        if half_life is None and st.dims["fatigue"] <= f0 / 2:
            half_life = t
    report["nail4"] = {"fatigue_after_100": round(capped, 3), "capped_no_diverge": capped <= 1.0,
                       "stress_bounded": stress_capped,
                       "fatigue_half_life_turns": half_life,
                       "fatigue_after_rest": round(st.dims["fatigue"], 3),
                       "pass": capped <= 1.0 and stress_capped and half_life is not None}
    print(f"钉子4 慢轴上限: 100 轮极端扰动后 fatigue={capped:.3f}（封顶 {'✅' if capped <= 1.0 else '❌'}），"
          f"静养半衰期 {half_life} 轮 → {st.dims['fatigue']:.3f} "
          f"{'✅' if report['nail4']['pass'] else '❌'}")

    report["all_pass"] = all(report[k]["pass"] for k in ("nail1", "nail3a", "nail2", "nail4"))
    print(f"\n===== 棘轮验收：{'全部通过' if report['all_pass'] else '存在未通过项'} =====")
    out = Path(__file__).resolve().parent / "ratchet_nails.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存：{out}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""交互式对话体验：1.5B + FiLM 调制（v0.7 封账配置）。

闭环配置：gain5 调制器 + 探针 v3 内感受回写 + 慢轴调制门。
用法：python chat_cli.py
命令：/s 看状态  /p 扰动  /r 重置  /q 退出
普通消息直接输入；骂它/夸它会自然触发情绪事件（关键词检测）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from film.agent_film import FILMStatefulAgent

CKPT = Path(__file__).resolve().parent / "film" / "ckpt"


def show(state: dict):
    print(f"  [状态] 唤醒 {state['arousal']:.2f} | 心情 {state['valence']:+.2f} "
          f"| 压力 {state['stress']:.2f} | 疲劳 {state['fatigue']:.2f}")


def main():
    print("加载中（1.5B + gain5 调制器 + 探针 v3）...")
    agent = FILMStatefulAgent(ckpt=str(CKPT / "film_modulator_gain5.pt"),
                              probe_ckpt=str(CKPT / "readout_probe_v3.pt"))
    agent.readout_mode = "probe"
    agent.state.cfg["gates_enabled"] = True
    print("就绪。闭环回写+慢轴门已开启。命令：/s 状态 | /p 扰动 | /r 重置 | /q 退出\n")

    while True:
        try:
            msg = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        if msg == "/q":
            break
        if msg == "/s":
            show(agent.state.as_dict())
            continue
        if msg == "/r":
            agent.state.dims.update({"arousal": 0.5, "valence": 0.0, "stress": 0.0, "fatigue": 0.0})
            print("  状态已重置")
            show(agent.state.as_dict())
            continue
        if msg.startswith("/p"):
            # /p stress=0.3 valence=-0.35 形式施加扰动
            try:
                impact = {k: float(v) for k, v in (kv.split("=") for kv in msg.split()[1:])}
            except ValueError:
                print("  用法：/p stress=0.3 valence=-0.35")
                continue
            agent.state.perturb(impact)
            agent.state.step(None)
            show(agent.state.as_dict())
            continue
        reply, state = agent.chat(msg, event="auto", writeback=True)
        print(f"模型> {reply}")
        show(state)

    print("再见。")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""OP-20261123-001 阶段 2 批跑执行器：2+2 方差批跑 = 两个真实独立进程各 2 跑。

单进程内顺序执行两次完整 eval suite（同种子重置，构成"会话内对"——产出①逐位比较口径）。
配置 = control 配置（共享 buffer + 门，framing-off）+ 遥测/token 日志（F4-4c 前置，已实现）。
断点续跑：run_eval Progress 查盘跳过（trial 级），重启后已完成单元不重跑。

用法（两个终端各启动一个进程）：
    python eval/run_batch_2x2.py --session s1
    python eval/run_batch_2x2.py --session s2
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BATCH_ARGS = [
    "--injection", "film",
    "--ckpt", "film/ckpt/film_modulator.pt",   # gain5 v0.4 选定版（stage3 文档钉死血统）
    "--memory", "--replay", "--shared-buffer",  # control 配置：共享 buffer + 提取门
    "--seed", "202",                             # 指令预注册 seed
    "--no-framing",                              # 批跑 control 配置 framing-off（预注册）
    "--telemetry",                               # F4-4c 遥测 + token 序列落盘
]


def run_suite(session: str, run_no: str) -> None:
    import run_eval  # 同目录 import
    out = ROOT / "eval" / "results" / f"stage2c2_batch_{session}_{run_no}.json"
    sys.argv = ["run_eval.py", "--out", str(out), "--label", f"batch_{session}_{run_no}",
                *BATCH_ARGS]
    run_eval.main()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, choices=["s1", "s2"],
                    help="进程标识（F1 会话 = 单次进程生命周期）")
    args = ap.parse_args()
    run_suite(args.session, "r1")
    run_suite(args.session, "r2")   # run_eval.main 内部 torch.manual_seed(202) 重置 → 会话内对


if __name__ == "__main__":
    main()

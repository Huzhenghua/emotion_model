# -*- coding: utf-8 -*-
"""7B 线步骤①冒烟：bf16 加载 + 显存核验 + tokenizer 零改动确认 + 单轮生成计时（第一次真实耗时）。

预期：bf16 全精度 15.23GB > 12GB 显存 → 全 GPU 路径预期 OOM，
自动降级 device_map=auto（CPU offload）并如实报告——这个降级耗时是全量协议成本裁定的输入。
"""
import argparse
import gc
import os
import time
from pathlib import Path

# gptqmodel 安全检查拒绝 torch <2.13（CVE-2025-3000，torch.jit.script 内存损坏）。
# 本项目只加载本地信任 checkpoint、不使用 jit.script，风险口径成立后 bypass（mistake #16）。
os.environ.setdefault("GPTQ_BYPASS_SAFETY_CHECKS", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(Path(__file__).resolve().parent.parent / "Qwen2.5-7B-Instruct"))
ap.add_argument("--max-new-tokens", type=int, default=64)
args = ap.parse_args()
MODEL = Path(args.model)


def vram() -> str:
    free, total = torch.cuda.mem_get_info()
    return f"显存 已用 {(total-free)/2**30:.1f}GB / 总 {total/2**30:.1f}GB"


print(f"目标模型：{MODEL}")
print("存在：", MODEL.exists())

tok = AutoTokenizer.from_pretrained(MODEL)
print(f"tokenizer OK  vocab={len(tok)} （vs 1.5B 同族应逐位一致=零改动）")

mode = None
try:
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    mode = "全 GPU bf16"
    print(f"全 GPU 加载 OK（{time.time()-t0:.0f}s）  {vram()}")
except Exception as e:
    print(f"全 GPU 加载失败：{type(e).__name__}——降级 device_map=auto（CPU offload）")
    gc.collect()
    torch.cuda.empty_cache()
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    mode = "GPU+CPU offload"
    print(f"offload 加载 OK（{time.time()-t0:.0f}s）  {vram()}")

msgs = [
    {"role": "system", "content": "你是一个乐于助人的中文对话助手。"},
    {"role": "user", "content": "用一句话介绍你自己。"},
]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").to(model.device)

# 预热（gptqmodel kernel 编译发生在首轮生成）+ 稳态测量（第二次计时）
for round_name in ("warmup", "steady"):
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=True,
                             temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
    dt = time.time() - t0
    n_tok = out.shape[1] - ids["input_ids"].shape[1]
    reply = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"生成计时（{mode} / {round_name}）：{dt:.1f}s / {n_tok} tok = {n_tok/dt:.1f} tok/s")
print("回复样例：", reply.strip()[:120])
print(f"hidden_size = {model.config.hidden_size}（FiLM 调制器适配目标；1.5B 为 1536）")
print(f"层数 = {model.config.num_hidden_layers}（探针取层范围适配目标；1.5B 为 28）")

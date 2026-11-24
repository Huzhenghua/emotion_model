# -*- coding: utf-8 -*-
"""训练 FiLM 调制器：行为蒸馏（学生只吃激活，模仿文本注入老师的行为）。

学生输入：用户 prompt（system prompt 无状态）+ FiLM(状态向量)
标签：老师的 completion（交叉熵，只算 completion 部分）
骨干网络冻结，只训调制器（~11M 参数）。

用法：python film/train.py --data film/data/distill.jsonl --epochs 3
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from film.modulator import FiLMModulator, FilmInjector, normalize_state
from agent.loop import DEFAULT_MODEL_PATH, PERSONA


class DistillDataset(Dataset):
    def __init__(self, path_or_rows, tokenizer, max_len: int = 192):
        if isinstance(path_or_rows, (str, Path)):
            self.rows = [json.loads(line) for line in Path(path_or_rows).open(encoding="utf-8")]
        else:
            self.rows = list(path_or_rows)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        messages = [{"role": "system", "content": PERSONA}, {"role": "user", "content": r["prompt"]}]
        prompt_text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = self.tok(prompt_text, add_special_tokens=False)["input_ids"]
        comp_ids = self.tok(r["completion"], add_special_tokens=False)["input_ids"][:48] + [self.tok.eos_token_id]
        input_ids = (prompt_ids + comp_ids)[:self.max_len]
        labels = ([-100] * len(prompt_ids) + comp_ids)[:self.max_len]
        return {"input_ids": input_ids, "labels": labels,
                "state": normalize_state(r["dims"])}


def collate(batch, pad_id: int):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn, states = [], [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
        states.append(b["state"])
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn),
            torch.stack(states))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent / "data" / "distill.jsonl"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "ckpt" / "film_modulator.pt"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gain", type=float, default=3.0,
                    help="γ/β 输出增益，放大调制幅度（随 checkpoint 保存）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 骨干：bf16 冻结
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    backbone = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    cfg = backbone.config
    modulator = FiLMModulator(cfg.num_hidden_layers, cfg.hidden_size,
                              gain=args.gain).to(device)
    injector = FilmInjector(backbone, modulator)
    injector.current = None  # 训练时由 collate 的状态逐 batch 设置——见下

    dataset = DistillDataset(args.data, tok)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        collate_fn=lambda b: collate(b, tok.pad_token_id or tok.eos_token_id))
    opt = torch.optim.AdamW(modulator.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs // args.grad_accum
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)

    print(f"样本 {len(dataset)} | 有效步数 {total_steps} | 可训练参数 "
          f"{sum(p.numel() for p in modulator.parameters())/1e6:.1f}M")

    step = 0
    for epoch in range(args.epochs):
        for i, (input_ids, labels, attn, states) in enumerate(loader):
            input_ids, labels, attn = input_ids.to(device), labels.to(device), attn.to(device)
            # 状态 → γ/β（带梯度，hook 内应用到每层输出）
            injector.current = modulator(states.to(device))
            logits = backbone(input_ids=input_ids, attention_mask=attn).logits
            injector.current = None

            shift_logits = logits[:, :-1]
            shift_labels = labels[:, 1:]
            loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)),
                                   shift_labels.reshape(-1), ignore_index=-100)
            (loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum == 0:
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 20 == 0:
                    print(f"epoch {epoch+1} step {step}/{total_steps} loss {loss.item():.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(modulator.state_dict(), args.out)
    print(f"训练完成，调制器已保存：{args.out}")


if __name__ == "__main__":
    main()

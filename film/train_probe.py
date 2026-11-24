# -*- coding: utf-8 -*-
"""训练激活读出探针：从 FiLM 调制后的骨干隐状态回归状态向量（内感受层）。

数据源（逗号分隔）：
1. film/data/distill.jsonl（老师蒸馏，1200 条）
2. film/data/probe_closed_loop.jsonl（闭环轨迹采集，按 episode 切验证集防泄漏）
前向：调制器按样本状态注入，取末 4 层 completion 位置 mean-pool。
目标：normalize_state(dims)（[-1,1] 尺度，与调制器输入同尺度）。

用法：
  python film/train_probe.py --data film/data/distill.jsonl,film/data/probe_closed_loop.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from film.modulator import FiLMModulator, FilmInjector
from film.train import DistillDataset, collate
from agent.loop import DEFAULT_MODEL_PATH
from state.readout import ActivationProbe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--data",
                    default="film/data/distill.jsonl,film/data/probe_closed_loop.jsonl",
                    help="逗号分隔的 jsonl 路径；闭环数据按 episode 切验证集")
    ap.add_argument("--ckpt", default=str(Path(__file__).resolve().parent / "ckpt" / "film_modulator_gain5.pt"),
                    help="FiLM 调制器 checkpoint（探针读取调制后的激活）")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "ckpt" / "readout_probe.pt"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    backbone = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    mod = FiLMModulator(backbone.config.num_hidden_layers, backbone.config.hidden_size).to(device)
    mod.load_state_dict(torch.load(args.ckpt, map_location=device))
    mod.eval()
    injector = FilmInjector(backbone, mod)

    # 装载数据：每行附 (source_idx, episode)。闭环数据 episode>=0 按轨迹切分
    rows, meta = [], []
    for si, p in enumerate(args.data.split(",")):
        p = p.strip()
        for line in Path(p).open(encoding="utf-8"):
            r = json.loads(line)
            rows.append(r)
            ep = r.get("episode", -1)
            meta.append((si, None if ep < 0 else si * 100000 + ep))
    print(f"数据源 {args.data.split(',')} | 共 {len(rows)} 条")

    n_pool = 4
    dataset = DistillDataset(rows, tok)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, shuffle=False,
        collate_fn=lambda b: collate(b, tok.pad_token_id or tok.eos_token_id))

    # 第一遍：收集 pooled 特征（末 4 层 completion mean-pool）
    feats, targets = [], []
    with torch.no_grad():
        for input_ids, labels, attn, states in loader:
            input_ids, attn = input_ids.to(device), attn.to(device)
            injector.current = mod(states.to(device))
            out = backbone(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
            injector.current = None
            stack = torch.stack(out.hidden_states[-n_pool:], dim=1)  # [B,4,seq,H]
            mask = (labels != -100).to(device).unsqueeze(1).unsqueeze(-1)
            pooled = (stack * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
            feats.append(pooled.cpu())
            targets.append(states)
    X = torch.cat(feats)  # [N,4,H]
    Y = torch.cat(targets)  # [N,4]
    print(f"特征 {tuple(X.shape)} | 目标 {tuple(Y.shape)}")

    # 切分：闭环数据留最后 2 条轨迹做 val；其余全训练
    cl_eps = sorted({m[1] for m in meta if m[1] is not None})
    val_eps = set(cl_eps[-2:]) if cl_eps else set()
    tr_idx = [i for i, m in enumerate(meta) if m[1] is None or m[1] not in val_eps]
    va_idx = [i for i, m in enumerate(meta) if m[1] in val_eps]
    print(f"train {len(tr_idx)} | val {len(va_idx)}（val 轨迹 {sorted(val_eps)}）")
    Xtr, Ytr, Xva, Yva = X[tr_idx], Y[tr_idx], X[va_idx], Y[va_idx]

    probe = ActivationProbe(backbone.config.hidden_size, n_pool).to(device)
    tr_mean = Xtr.mean(dim=(0, 1)).to(device)
    tr_std = Xtr.std(dim=(0, 1)).clamp(min=1e-6).to(device)
    probe.in_mean.copy_(tr_mean)
    probe.in_std.copy_(tr_std)

    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=1e-4)
    Xtr_d, Ytr_d = Xtr.to(device), Ytr.to(device)
    for epoch in range(args.epochs):
        perm = torch.randperm(len(Xtr_d), device=device)
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            pred = probe(Xtr_d[idx])
            loss = F.mse_loss(pred, Ytr_d[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                val_mae = (probe(Xva.to(device)) - Yva.to(device)).abs().mean().item()
            print(f"epoch {epoch+1} loss {loss.item():.5f} val MAE(归一化) {val_mae:.4f}")

    with torch.no_grad():
        pred_va = probe(Xva.to(device))
        pred_tr = probe(Xtr_d)
        for name, sl in (("arousal", 0), ("valence", 1), ("stress", 2), ("fatigue", 3)):
            mae_tr = (pred_tr[:, sl] - Ytr_d[:, sl]).abs().mean().item()
            mae_va = (pred_va[:, sl] - Yva.to(device)[:, sl]).abs().mean().item()
            orig = mae_va / 2 if sl != 1 else mae_va
            print(f"  {name}: train MAE {mae_tr:.4f} | val MAE {mae_va:.4f}（原始尺度 ±{orig:.3f}）")

    torch.save(probe.state_dict(), args.out)
    print(f"探针已保存：{args.out}")


if __name__ == "__main__":
    main()

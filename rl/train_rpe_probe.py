# -*- coding: utf-8 -*-
"""判据 8（D 附加）：RL 后探针能否从激活读出自发形成的价值信号（RPE/优势）。

价值神经元假设的工程验证：
- 数据源：train_rl.py 的 rpe_log_<tag>.jsonl（每 turn 的优势 + user/reply + 状态）
- 特征：用**训练后**调制器重放每 turn（prompt+reply，state_before 调制），
  末 4 层 completion mean-pool（与 ActivationProbe 同口径）
- 目标：per-turn advantage（episode 级归一化）
- 判据（已钉死）：val R² ≥ 0.3 且 shuffle 标签基线 R² < 0.1 → "价值信号可读出"成立；
  否则记否定结果归档（否定也是结果）
- 防泄漏：train/val 按 group 切分（同 group 的 episode 不跨集）

用法：python rl/train_rpe_probe.py --tag r1.0
"""
import argparse
import json
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.loop import DEFAULT_MODEL_PATH, PERSONA

CKPT_DIR = Path(__file__).resolve().parent / "ckpt"
DATA_DIR = Path(__file__).resolve().parent / "data"


class RPEProbe(nn.Module):
    """激活特征 → 标量优势回归。小 MLP，容量刻意小（防过拟合到噪声）。"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@torch.no_grad()
def extract_feature(agent, user: str, reply: str, state_dims: dict) -> torch.Tensor:
    """重放一 turn：prompt+reply，state_before 调制，末 4 层 completion mean-pool → [H]。"""
    tok = agent.tokenizer
    messages = [{"role": "system", "content": PERSONA}, {"role": "user", "content": user}]
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    comp_ids = tok(reply, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = torch.tensor([prompt_ids + comp_ids], device=agent.device)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[0, len(prompt_ids):] = True
    agent.injector.set_state(state_dims)
    try:
        out = agent.model(input_ids=ids, output_hidden_states=True)
    finally:
        agent.injector.clear_state()
    stack = torch.stack(out.hidden_states[-4:], dim=1)      # [1,4,seq,H]
    m = mask.unsqueeze(1).unsqueeze(-1)
    pooled = (stack * m).sum(dim=2) / m.sum(dim=2).clamp(min=1)   # [1,4,H]
    return pooled.mean(dim=1).squeeze(0)                     # [H]


def r2(y_true, y_pred) -> float:
    mu = statistics.mean(y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    ss_tot = sum((a - mu) ** 2 for a in y_true)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def train_probe(X_tr_n, y_tr, hidden_size, epochs, lr, device, seed):
    """训练一个小探针，返回 model。"""
    torch.manual_seed(seed)
    probe = RPEProbe(hidden_size).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    n = len(y_tr)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            loss = nn.functional.mse_loss(probe(X_tr_n[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--tag", default="rl", help="对应的训练 tag（读 rpe_log_<tag>.jsonl）")
    ap.add_argument("--rl-ckpt", default=None, help="训练后调制器（缺省 rl/ckpt/rl_<tag>_final.pt）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rpe_path = DATA_DIR / f"rpe_log_{args.tag}.jsonl"
    final_ckpt = args.rl_ckpt or str(CKPT_DIR / f"rl_{args.tag}_final.pt")
    rows = load_rows(rpe_path)
    print(f"RPE 日志：{rpe_path}（{len(rows)} turns）")
    assert rows, "RPE 日志为空——先跑 train_rl.py"

    from film.agent_film import FILMStatefulAgent
    print(f"加载训练后调制器：{final_ckpt}")
    agent = FILMStatefulAgent(args.model, ckpt=final_ckpt)

    # 特征提取（重放，no_grad）；按 ckpt 缓存断点续跑（模型 forward 是主要耗时）
    cache_path = DATA_DIR / f"rpe_feats_{args.tag}_{Path(final_ckpt).stem}.pt"
    if cache_path.exists():
        blob = torch.load(cache_path, map_location=agent.device)
        X, y, groups = blob["X"].float(), blob["y"], blob["groups"]
        print(f"特征缓存命中：{cache_path}（{len(y)} turns）")
    else:
        feats, ys, groups = [], [], []
        for i, r in enumerate(rows):
            feats.append(extract_feature(agent, r["user"], r["reply"], r["state_before"]))
            ys.append(r["advantage"])
            groups.append(r["group"])
            if (i + 1) % 200 == 0:
                print(f"  特征提取 {i+1}/{len(rows)}")
        X = torch.stack(feats).float()   # 模型输出 BFloat16 → 统一 Float32
        y = torch.tensor(ys, dtype=torch.float32, device=X.device)
        torch.save({"X": X.cpu(), "y": y.cpu(), "groups": groups}, cache_path)

    # 按 group 切 train/val（防泄漏）
    rng = random.Random(args.seed)
    uniq = sorted(set(groups))
    rng.shuffle(uniq)
    n_val = max(1, len(uniq) // 10)
    val_groups = set(uniq[:n_val])
    tr = [i for i, g in enumerate(groups) if g not in val_groups]
    va = [i for i, g in enumerate(groups) if g in val_groups]
    X_tr, y_tr, X_va, y_va = X[tr], y[tr], X[va], y[va]
    mu, sd = X_tr.mean(0, keepdim=True), X_tr.std(0, keepdim=True).clamp(min=1e-6)
    X_tr_n, X_va_n = (X_tr - mu) / sd, (X_va - mu) / sd
    print(f"train {len(tr)} / val {len(va)} turns（val 组数 {n_val}）")

    # 主探针
    probe = train_probe(X_tr_n, y_tr, X.shape[1], args.epochs, args.lr, X.device, args.seed)
    with torch.no_grad():
        pred_va = probe(X_va_n).tolist()
    r2_main = r2(y_va.tolist(), pred_va)
    print(f"主探针 val R² = {r2_main:.3f}")

    # shuffle 标签基线：同架构同预算，标签随机重排后训练，val R² 应≈0
    perm_label = torch.randperm(len(y_tr), device=X.device)
    probe_s = train_probe(X_tr_n, y_tr[perm_label], X.shape[1], args.epochs, args.lr,
                          X.device, args.seed + 1)
    with torch.no_grad():
        pred_s = probe_s(X_va_n).tolist()
    r2_shuffle = r2(y_va.tolist(), pred_s)
    print(f"shuffle 基线 val R² = {r2_shuffle:.3f}")

    verdict = "价值信号可读出：成立" if (r2_main >= 0.3 and r2_shuffle < 0.1) \
        else "价值信号可读出：不成立（否定结果也归档）"
    print(verdict)

    out = args.out or str(DATA_DIR / f"rpe_probe_{args.tag}.json")
    report = {
        "timestamp": datetime.now().isoformat(),
        "tag": args.tag, "rl_ckpt": final_ckpt, "n_turns": len(rows),
        "n_train": len(tr), "n_val": len(va),
        "r2_val": round(r2_main, 4), "r2_shuffle": round(r2_shuffle, 4),
        "criterion": "R² ≥ 0.3 且 shuffle < 0.1",
        "verdict": verdict,
    }
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已落盘：{out}")


if __name__ == "__main__":
    main()

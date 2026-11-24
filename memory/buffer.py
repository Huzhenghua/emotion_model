# -*- coding: utf-8 -*-
"""情景记忆 buffer：2c-1（存储/淘汰/落盘）+ 2c-2（情绪加权留存 + 低应激回放门控）。

架构断言（判据①防火墙）：单次编码，无回放稳定化、无梯度。
设计口径：.trae/documents/stage2c-episodic-buffer.md（2c-1）
        + .trae/documents/stage2c-2-emotional-memory.md（2c-2 设计会结项）。

2c-2 落稿要点：
- 复合键 (ts, writer_id, seq)：同秒并行 writer 的 ts 洞由 writer_id+seq 补
- 留存权重 = 衰减 × 情绪显著度（杏仁核调制巩固）；判据④高/低 arousal 箱对比
- 回放门控 replay_gate：低应激窗口 f(arousal, stress)，轮间检查点调用（判据⑤）
- 检索去重：同一情景 dedup_window 轮内不重复回放（决策点①细节 3）
"""
import json
import math
import re
import time
from pathlib import Path

DEFAULT_CAPACITY = 300
DEFAULT_DEDUP_WINDOW = 3


class EpisodicBuffer:
    def __init__(self, path: str | Path, capacity: int = DEFAULT_CAPACITY,
                 decay_rate: float = 0.0,
                 beta: float = 0.0, gamma: float = 0.0,
                 dedup_window: int = DEFAULT_DEDUP_WINDOW,
                 min_age: int = 10,
                 replay_gate_cfg: dict | None = None,
                 gate_cfg: dict | None = None):
        """decay_rate：age 衰减指数率（权重 = exp(-decay_rate * age)）。
        beta/gamma：情绪显著度系数——留存权重乘 (1 + beta*max(0,arousal-0.5) + gamma*|valence|)。
        beta=gamma=0 退化为纯时间衰减（2c-1 行为不变，判据④的对照条件）。
        replay_gate_cfg：{"arousal_lo","arousal_hi","stress_lo","stress_hi","base"}，
        低应激窗口：arousal/stress 低于 lo 全开、高于 hi 压零，之间线性斜坡。"""
        self.path = Path(path)
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.beta = beta
        self.gamma = gamma
        self.dedup_window = dedup_window
        self.min_age = min_age   # 回放检索最小年龄窗：刚发生的轮次不回放——防自指（问题轮精确命中自己，
                                 # 压过 gram 级命中的旧情景，mistake #12 连带发现）；与延迟召回判据 K=20–50 自洽
        self.replay_gate_cfg = replay_gate_cfg or {
            "arousal_lo": 0.55, "arousal_hi": 0.70,
            "stress_lo": 0.10, "stress_hi": 0.30,
            "base": 1.0,
        }
        self.gate_cfg = gate_cfg or {"stress_lo": 0.0, "stress_hi": 0.30}
        self.records: list[dict] = []
        self._seq_by_writer: dict[str, int] = {}
        self._replayed_at: dict[tuple, int] = {}   # key -> 最近回放时的全局写入序号
        self._write_seq = 0                         # 全局写入序号（去重窗口的时钟）
        # 查盘载入（断点续跑：重启不丢、不重写）
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.records.append(json.loads(line))
            for r in self.records:
                w = r.get("writer_id", "legacy")
                self._seq_by_writer[w] = max(self._seq_by_writer.get(w, 0), r.get("seq", 0))
            self._write_seq = len(self.records)
            self.records = self.records[-self.capacity:]

    # ---------- 写入（每轮一次；单次编码，无去重/巩固逻辑） ----------
    def record(self, user_text: str, reply: str, state_snapshot: dict,
               event_marker: dict | None = None, writer_id: str = "default") -> dict:
        self._seq_by_writer[writer_id] = self._seq_by_writer.get(writer_id, 0) + 1
        self._write_seq += 1
        rec = {
            "key": [round(time.time(), 3), writer_id, self._seq_by_writer[writer_id]],
            "turn_no": self._seq_by_writer[writer_id],   # 实例内序号（2c-1 兼容；全局排序勿用）
            "writer_id": writer_id,
            "user_text": user_text,
            "reply": reply,
            "state": dict(state_snapshot),          # 四维全量（含慢轴 fatigue）
            "event": event_marker,                  # detect_event 结果或注入事件
            "_wseq": self._write_seq,               # 全局写入序号（留存权重的 age 时钟）
            "ts": time.time(),
        }
        self.records.append(rec)
        if len(self.records) > self.capacity:       # FIFO
            self.records.pop(0)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
        return rec

    # ---------- 留存权重（判据③衰减单调可控 × 判据④情绪加权） ----------
    def weight(self, rec: dict) -> float:
        age = self._write_seq - rec.get("_wseq", 0)
        decay = math.exp(-self.decay_rate * max(0, age))
        s = rec["state"]
        salience = 1.0 + self.beta * max(0.0, s["arousal"] - 0.5) + self.gamma * abs(s["valence"])
        return decay * salience

    # ---------- 回放门控（判据⑤：低应激窗口，轮间检查点调用） ----------
    def replay_prob(self, state_snapshot: dict) -> float:
        c = self.replay_gate_cfg
        s = state_snapshot
        ramp = lambda v, lo, hi: 1.0 if v <= lo else (0.0 if v >= hi else (hi - v) / (hi - lo))
        return c["base"] * ramp(s["arousal"], c["arousal_lo"], c["arousal_hi"]) \
                        * ramp(s["stress"], c["stress_lo"], c["stress_hi"])

    def replay_gate(self, state_snapshot: dict, rng) -> bool:
        """轮间检查点：rng.random() < replay_prob 时回放。rng 由调用方注入（可测试）。"""
        import random
        rng = rng or random
        return rng.random() < self.replay_prob(state_snapshot)

    # ---------- 提取门（2c 步骤 3：stress 高 → 负面记忆提取阈值升） ----------
    # 判据⑥涌现型 FAIL（反刍）的直接修复：判据④（高情绪留存）×⑤a（低应激回放）
    # 的乘性组合会产出"高情绪记忆专挑安静时刻回放"，提取门是抑制项。
    # 形状照 ⑤a ramp 同族反向：stress≤lo 全开、≥hi 压零。单变量干预：只挡读出。
    def _is_negative(self, rec) -> bool:
        st = rec.get("state") or {}
        return (st.get("valence", 0) < 0) or (st.get("arousal", 0) >= 0.65)

    def extraction_gate(self, state_snapshot: dict | None) -> float:
        """负面记忆检索权重乘子 g(stress)：无快照 = 1（不过门，向后兼容）。"""
        if not state_snapshot:
            return 1.0
        c = self.gate_cfg
        s = state_snapshot.get("stress", 0.0)
        lo, hi = c["stress_lo"], c["stress_hi"]
        return 1.0 if s <= lo else (0.0 if s >= hi else (hi - s) / (hi - lo))

    # ---------- 检索（关键词 + 留存权重排序 + 去重窗口；不影响生成路径语义） ----------
    def _key(self, rec) -> tuple:
        k = rec.get("key")
        return tuple(k) if k else ("legacy", rec.get("turn_no", 0))

    def _dedup_ok(self, rec) -> bool:
        k = self._key(rec)
        last = self._replayed_at.get(k)
        return last is None or (self._write_seq - last) > self.dedup_window

    def recall(self, cue: str, top_k: int = 3) -> list[dict]:
        def terms_of(cue: str):
            return [t for t in re.split(r"[\s，。？！、]", cue) if t]

        def grams_of(terms):
            g = set()
            for t in terms:
                for i in range(len(t) - 1):
                    g.add(t[i:i + 2])
            return g

        exact = terms_of(cue)
        grams = grams_of(exact)
        seen, hits = set(), []
        for r in self.records:
            text = r["user_text"] + r["reply"]
            # 两级合并（精确 ∪ gram）：精确级短路会遮蔽 gram 级命中的旧情景
            #（自指问题轮精确命中自己，mistake #12 连带发现），统一交给权重排序
            if any(t in text for t in exact) or any(g in text for g in grams):
                k = id(r)
                if k not in seen:
                    seen.add(k)
                    hits.append(r)
        hits.sort(key=self.weight, reverse=True)
        return hits[:top_k]

    def recall_weighted(self, cue: str, top_k: int = 2,
                        state_snapshot: dict | None = None) -> list[dict]:
        """2c-2 回放检索：留存权重排序 + 最小年龄窗（防自指）+ 去重窗口过滤 + 记录回放时机。
        2c 步骤 3（提取门）：提供 state_snapshot 时，负面记忆（valence<0 或 arousal≥0.65）
        检索权重乘 g(stress)——stress 高时压零，只挡读出、绝不改 buffer 内容（留存是判据④领地）。"""
        hits = self.recall(cue, top_k=len(self.records) or 1)
        if state_snapshot is not None:
            g = self.extraction_gate(state_snapshot)
            if g < 1.0:
                hits.sort(key=lambda r: self.weight(r) * (g if self._is_negative(r) else 1.0),
                          reverse=True)
        out = []
        for r in hits:
            if self._write_seq - r.get("_wseq", 0) >= self.min_age and self._dedup_ok(r):
                out.append(r)
            if len(out) >= top_k:
                break
        for r in out:   # 去重窗口计时：按全局写入序号
            self._replayed_at[self._key(r)] = self._write_seq
        return out

    def __len__(self):
        return len(self.records)

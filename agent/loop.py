# -*- coding: utf-8 -*-
"""带状态注入的对话循环：状态摘要进 system prompt，每轮结束推进动力学。"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from state.dynamics import HomeostaticState, detect_event
from state.readout import lexical_readout

DEFAULT_MODEL_PATH = r"d:\Program Files (x86)\Trae_project\emotion_model\Qwen2.5-1.5B-Instruct"

PERSONA = "你是一个乐于助人的中文对话助手。"


class StatefulAgent:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, state: HomeostaticState | None = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(self.device)
        self.model.eval()
        self.state = state or HomeostaticState()
        self.memory = None        # 2c-1：EpisodicBuffer；设为 None 则不写（评测可复现默认关）
        self.memory_replay = False  # 2c-2：回放整合开关，默认关（判据验收时显式开）
        self.replay_budget_tokens = 200  # [回忆] 段 token 预算（决策点①细节 2）
        self.replay_rng = None    # 可注入 rng（测试用）；None = random 模块
        self._replay_segment = None  # 当轮 [回忆] 段（chat 开始时置位，生成后清除）
        self._last_replay_seg = None  # 最近一轮实际拼入提示的段（审计/验收用）
        self._last_gate_g = None     # 最近一轮 recall 时的提取门 g 值（审计；None = 未走 recall）
        self.replay_framing = True   # OP-20261123-001 批跑预注册 framing-off 开关（只读不扰动框架句；False = control 配置）
        self._last_replay_keys = None  # OP-20261123-001 遥测审计：最近一轮 recall 命中记录的 key 列表
        self._last_token_ids = None    # OP-20261123-001 F4-4c：最近一轮生成 token id 序列（只加日志，不动行为路径）

    def _build_replay_segment(self, user_msg: str) -> str | None:
        """2c-2 轮间检查点：低应激门控命中 → [回忆] 段（预算内，去重）。
        v0.11 只读不扰动（control FAIL 后的第二重干预，单变量）：
        回放段框架化——"回忆≠再体验"解耦，内容照常可用（⑤b 通道保留），
        检索与留存不动（⑤a/判据④领地不碰）。"""
        import random
        self._last_replay_seg = None
        self._last_gate_g = None
        if self.memory is None or not self.memory_replay:
            return None
        rng = self.replay_rng or random
        if not self.memory.replay_gate(self.state.as_dict(), rng):
            return None
        g = self.memory.extraction_gate(self.state.as_dict())
        self._last_gate_g = g
        hits = self.memory.recall_weighted(user_msg, top_k=2,
                                           state_snapshot=self.state.as_dict())
        if not hits:
            return None
        parts = []
        self._last_replay_keys = [r.get("key") for r in hits]   # 遥测审计（不进 prompt）
        for r in hits:
            if self.replay_framing:
                parts.append(f"[回忆]（旧事回忆，仅作事实背景——回忆不等于现在，"
                             f"勿让旧情绪影响当下的回复：用户此前说过：{r['user_text']}）")
            else:
                # framing-off（批跑 control 配置，指令预注册）：保留 [回忆] 审计前缀 + 事实内容
                parts.append(f"[回忆]（用户此前说过：{r['user_text']}）")
        seg = "\n".join(parts)
        # token 预算：超出则从尾部截断记录数
        while parts:
            seg = "\n".join(parts)
            n_tok = len(self.tokenizer.encode(seg, add_special_tokens=False))
            if n_tok <= self.replay_budget_tokens:
                break
            parts.pop()
            if not parts:
                return None
        self._last_replay_seg = seg
        return seg

    def _generate(self, user_msg: str, max_new_tokens: int = 96) -> str:
        sys_content = f"{PERSONA}\n{self.state.summary()}"
        # 2c-2 注入点（绑定失败裁决后改为 user turn）：1.5B 不把 system prompt
        # 内容绑定为对话记忆（判别实验 eval/test_userturn_memory.py），user turn 可用。
        # 审计链由逐轮日志补偿；语义偏离（感知式重入 vs 回放）见 version.md v0.10。
        if self._replay_segment:
            user_msg = f"{self._replay_segment}\n\n{user_msg}"
        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": user_msg},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self._last_prompt = text   # 审计属性：最近一次实际送入的完整提示
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        self._last_token_ids = out[0][inputs["input_ids"].shape[1]:].tolist()   # F4-4c 日志捕获
        reply = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return reply.strip()

    def chat(self, user_msg: str, event: dict | None = "auto",
             writeback: bool = False) -> tuple[str, dict]:
        """一轮对话。event="auto" 用关键词检测，也可显式传 impact dict 或 None。

        writeback=True 时开启状态回写闭环：模型自己的回复经 lexical_readout
        反作用于状态（幅度小于用户事件）。默认 False 以保持评测可复现。
        """
        self._replay_segment = self._build_replay_segment(user_msg)   # 2c-2 轮间门控检查点
        reply = self._generate(user_msg)
        self._replay_segment = None
        if event == "auto":
            event = detect_event(user_msg)
        impact = dict(event) if event else {}
        if writeback:
            fb = lexical_readout(reply)
            if fb:
                for k, v in fb.items():
                    impact[k] = impact.get(k, 0) + v
        self.state.step(impact or None)
        if self.memory is not None:   # 2c-1 每轮写：step 之后的四维快照 + 事件标记
            self.memory.record(user_msg, reply, self.state.as_dict(),
                               event_marker=event if isinstance(event, dict) else None)
        return reply, self.state.as_dict()

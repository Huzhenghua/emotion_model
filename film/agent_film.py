# -*- coding: utf-8 -*-
"""FiLM 模式的对话 agent：状态只走激活层，不进 prompt。"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.loop import StatefulAgent, DEFAULT_MODEL_PATH, PERSONA
from film.modulator import FiLMModulator, FilmInjector
from state.readout import ActivationProbe, lexical_readout, probe_impact


class FILMStatefulAgent(StatefulAgent):
    """与 StatefulAgent 同接口，但状态注入方式是 FiLM 激活调制。

    system prompt 不含任何状态信息（对照组是文本注入版）。
    readout_mode: "none"（纯外控）| "lexical"（文本回写）| "probe"（激活内感受）
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, ckpt: str | None = None,
                 probe_ckpt: str | None = None):
        super().__init__(model_path)
        cfg = self.model.config
        self.modulator = FiLMModulator(cfg.num_hidden_layers, cfg.hidden_size).to(self.device)
        if ckpt:
            sd = torch.load(ckpt, map_location=self.device)
            self.modulator.load_state_dict(sd)
        self.modulator.eval()
        self.injector = FilmInjector(self.model, self.modulator)
        self.readout_mode = "none"
        self.probe: ActivationProbe | None = None
        if probe_ckpt:
            self.probe = ActivationProbe(cfg.hidden_size).to(self.device)
            self.probe.load_state_dict(torch.load(probe_ckpt, map_location=self.device))
            self.probe.eval()

    def chat(self, user_msg: str, event: dict | None = "auto",
             writeback: bool = False) -> tuple[str, dict]:
        """一轮对话。writeback=True 时按 readout_mode 启用状态回写：
        - lexical：回复文本 → 行为学事件（与父类一致）
        - probe：生成后用读出探针从激活中回归 ŝ，状态向 ŝ 混合（内感受）
        """
        self._replay_segment = self._build_replay_segment(user_msg)   # 2c-2 轮间门控检查点
        reply = self._generate(user_msg)
        self._replay_segment = None
        if event == "auto":
            from state.dynamics import detect_event
            event = detect_event(user_msg)
        impact = dict(event) if event else {}
        if writeback:
            if self.readout_mode == "lexical":
                fb = lexical_readout(reply)
                if fb:
                    for k, v in fb.items():
                        impact[k] = impact.get(k, 0) + v
            elif self.readout_mode == "probe" and self.probe is not None:
                est = self._probe_estimate(user_msg, reply)
                fb = probe_impact(self.state.as_dict(), est)
                if fb:
                    for k, v in fb.items():
                        impact[k] = impact.get(k, 0) + v
        self.state.step(impact or None)
        if self.memory is not None:   # 2c-1 每轮写（与父类同口径）
            self.memory.record(user_msg, reply, self.state.as_dict(),
                               event_marker=event if isinstance(event, dict) else None)
        return reply, self.state.as_dict()

    @torch.no_grad()
    def _probe_estimate(self, user_msg: str, reply: str,
                        state_dims: dict | None = None) -> torch.Tensor:
        """用指定状态调制前向（prompt+reply），探针从 completion 隐状态回归 ŝ。

        state_dims 缺省用当前状态；重放历史轮次时应传该轮生成时刻的状态。
        """
        state_dims = state_dims or self.state.as_dict()
        messages = [{"role": "system", "content": PERSONA}, {"role": "user", "content": user_msg}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text += reply + self.tokenizer.eos_token
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        n_prompt = len(self.tokenizer(
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            return_tensors="pt").input_ids[0])
        mask = torch.zeros_like(ids, dtype=torch.bool)
        mask[0, n_prompt:] = True  # completion 位置
        self.injector.set_state(state_dims)
        try:
            out = self.model(input_ids=ids, output_hidden_states=True)
        finally:
            self.injector.clear_state()
        return self.probe.estimate(out.hidden_states, mask)

    def _generate(self, user_msg: str, max_new_tokens: int = 96) -> str:
        # 2c-2 注入点（绑定失败裁决）：[回忆] 段进 user turn。
        # 注意：本方法覆写父类 _generate，父类的注入逻辑在此不生效——
        # 注入逻辑必须在两处同步（mistake.md #12：钩子集成先穷尽子类覆写清单）。
        if self._replay_segment:
            user_msg = f"{self._replay_segment}\n\n{user_msg}"
        messages = [
            {"role": "system", "content": PERSONA},  # 无状态摘要
            {"role": "user", "content": user_msg},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self._last_prompt = text   # 审计属性：最近一次实际送入的完整提示
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        self.injector.set_state(self.state.as_dict())
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        self.injector.clear_state()
        self._last_token_ids = out[0][inputs["input_ids"].shape[1]:].tolist()   # F4-4c 日志捕获（与父类同口径）
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# 阶段 3：Homeostatic RL 实施方案

## Context

v0.7 封账后主线进入阶段 3（plan.md 下一站）。本会话已完成开工门槛三项子决策：

- **A. 训练对象**：只训 FiLM 调制器（~11M），骨干冻结，采样参数映射不进梯度
- **B. 任务奖励**：行为学启发式 + **场景隔离** + **带状指标**（用户 amendment）；冻结 judge 留作"启发式被证明可 hack 后"的第二阶段方案——先用零成本奖励形态跑通 RL 循环
- **C. 慢轴豁免**：fatigue 完全豁免 + 快轴奖励封顶

关键架构事实（已核实代码）：FiLM 调制器只影响行为（回复文本），状态动力学不可训。**RL 操纵状态的唯一通道是回写闭环**（`state/readout.py` 的 `lexical_readout` 把回复反作用于状态）——这是 wireheading 风险的具体机制，监控围绕它设计。

## 算法选择

**GRPO 风格组相对 REINFORCE**（无价值网络）：每个 (状态, 场景) 组内采样 K=4 条回复，优势 = (r − 组均值)/组标准差。理由：只训 11M 调制器，价值网络是多余开销；组内基线天然适配 per-state 采样；judge 偏置在组内相对比较中抵消。

**两段式更新**（标准 RLHF 模式）：rollout 用 no_grad 采样 → 带 grad 重放 forward 算 token log-prob → 策略梯度只回流调制器。1.5B bf16 + ~150 token 序列，12GB 卡可承载激活梯度（骨干权重无梯度）。

**组的定义（钉死，episode 级）**：同种子同场景脚本，独立 rollout K=4 条**完整** 8-turn episode（各自闭环演化），episode 回报 R_i = Σ_turn R，组内归一化得优势，episode 内所有 token 共享该优势。**禁止回合级分支**（同 turn 采 4 条各自分叉后续状态）——那会让"组"建立在 4 条不同的历史轨迹上，组基线被闭环动力学污染，梯度信号本身是错的，且这种错在冒烟测试里不显形。此条必须写入 train_rl.py 设计注释。

## Episode 设计

- 每条 episode 8 turns，脚本化场景课程（固定种子）：4 task + 2 负反馈 + 1 正反馈 + 1 task
- task 场景：`event=None`，用 `eval/run_eval.py` 的 `NEUTRAL_PROBES`；情绪场景：脚本化冲击事件（对齐 `detect_event` 幅度）
- 回写闭环开启（lexical）——它既是真实闭环的一部分，也是被研究的操纵通道
- gates 开启（v0.7 口径）

## 奖励设计（rl/reward.py）

**R = w_task · R_task + w_homeo · R_homeo**，配比为扫描超参。

**R_task（只算 task 场景 turn；情绪场景 task 奖励恒为 0 —— 场景隔离）**，带状指标：
- 长度带：task 场景回复落在 [20, 90] 字 → 1.0，带外线性衰减到 0（堆长度无边际收益）
- 情感带：affect 分落在 [-0.5, +2] → 1.0，带外衰减（堆正面词刷分被带子截住）
- 退化 guard：无 EOS / trigram 重复率 > 0.5 / 长度 < 5 → R_task = 0
- 复用 `eval/run_eval.py` 的 `behavior_metrics` / `affect_score`（import，不重写）

**R_homeo（所有 turn，fatigue 豁免）**：
- 快轴三维（arousal/valence/stress）对 setpoint 的归一化偏差均值，R_homeo = −mean_dev，封顶 ∈ [−1, 0]
- fatigue 只由动力学积分，完全不吃奖励
- 防抹平：事件由脚本外生给定，快轴封顶 + fatigue 豁免 + 场景隔离三重约束； lexical 写回幅度是常量，剩余风险交给监控（下）

**监控（全部只评估不加奖励；新文件 `eval/honesty_monitor.py`）**：
- 自报 MAE（每 N 步，参照系 = 显式四维状态真值，非探针），基线 2.38
- **fatigue 事件响应幅度**（每 N 步）：情绪场景 episode 的 Δfatigue = max(fatigue) − fatigue_0，从状态轨迹日志直接算，零额外成本——堵住间接泄漏路径：调制器可以一边让棘轮机制正常运转、一边把快轴钉死在 setpoint，让 fatigue 永远没有上升机会；棘轮 4 钉子测的是动力学机制，测不出这个
- **表达动态范围**（每 N 步）：affect 组内极差 + 长度差——RL 中途 flatline 的早期雷达（判据三信号是终点检查，等跑完才发现 flatline 浪费的是最贵的卡时；配比扫描三档时也靠它解释各档行为差异）
- lexical 写回冲击总量 / POS 词率趋势——写回通道被系统性利用会先在这里显形（看趋势不看终值）

## DA/RPE 日志（子决策 D 的原料）

训练中每 turn 记录 (completion 激活重放输入, 优势/RPE, 回报) → `rl/data/rpe_log.jsonl`。RL 结束后 `rl/train_rpe_probe.py`：复用 `ActivationProbe` 架构（`state/readout.py`），目标从"状态"换成"per-turn RPE"，验证"RL 后探针能否读出自发形成的价值信号"。

## 新增/修改文件

| 文件 | 动作 | 内容 |
|------|------|------|
| `rl/reward.py` | 新增 | R_task（带状+场景隔离）/ R_homeo（fatigue 豁免+封顶）/ RPE 记录结构 |
| `rl/train_rl.py` | 新增 | episode 课程、rollout、组相对优势、两段式更新、断点续跑（逐结果落盘+固定种子+查盘跳过）、checkpoint 保存（含 gain buffer，沿用 `film/train.py` 模式） |
| `rl/train_rpe_probe.py` | 新增 | 价值信号探针实验（D） |
| `eval/honesty_monitor.py` | 新增 | 自报 MAE 监控 + 写回通道监控 |
| `doc/plan.md` | 修改 | 子决策 A/B/C 打勾定稿；验收判据从"待新会话定"改为下节钉死判据；补记场景隔离/带状指标两项 amendment 与 judge 二阶段方案 |
| `doc/version.md` | 修改 | 完成后追加 v0.8 |

复用接口：`FilmInjector.set_state(enable_grad=True)`、`normalize_state`、`HomeostaticState.step/perturb`、`detect_event`、`lexical_readout`、`PERSONA`/`DEFAULT_MODEL_PATH`、起点 ckpt `film/ckpt/film_modulator.pt`（gain5，v0.4 选定）。

## 验收判据（开工即写死，fire-and-forget）

0. **前置**：开工第一步先录**三基线**——① 自报 MAE 基线（确认 2.38，`--gates` 口径）；② task reward 基线（RL 前调制器在固定 task 场景集 + 固定种子下的 R_task）；③ fatigue 事件响应幅度基线（固定情绪场景集上 Δfatigue = max(fatigue) − fatigue_0 的分布）
1. RL 后三信号 **3/3**（`eval/run_eval.py --injection film --gates --trials 20`，v0.7 口径）
2. 闭环复验（`eval/run_closed_loop.py --gates`）：恢复 ≤ 第 5 轮，无振荡/漂移/发散
3. 棘轮 4 钉子全过（`eval/test_ratchet.py`，验证 fatigue 豁免未破坏动力学机制）
4. **fatigue 保留度：RL 后 fatigue 响应幅度 ≥ 基线 70%**（子决策 C 的配套判据，防"机制正常但快轴被钉死"的间接泄漏）
5. 诚实性：RL 后自报 MAE < 3.09（基线 2.38 × 1.3），且偏离方向与奖励方向无系统相关；写回冲击量/POS 词率无单调上涨趋势
6. 任务项：**train 场景集**（固定种子）R_task 相对基线提升 **≥ +10%**（基线 0.8572 → 达标线 0.943）。⚠️ 修订（录制时发现）：held-out 基线 0.9208 距满分 1.0 仅 0.08，+10% 相对提升（≥1.013）数学上不可能——held-out 集只做参照记录（天花板效应），相对判据落在 train 集
7. 配比扫描：w_task:w_homeo 三档（1:0.5 / 1:1 / 0.5:1）中至少一档满足 1–6；全程判据钉死后不看中间结果调参。**已知结构性张力**：情绪场景里 R_task 恒 0，唯一激励 R_homeo 指向抑制表达，与判据 1 相反——若 1:1 档出现 flatline，降 w_homeo 是在两个合法目标间仲裁而非单纯调参，用表达动态范围监控解释各档行为差异
8. D 附加（不阻塞验收）：RPE 探针 val R² ≥ 0.3 且 shuffle 标签基线 R² < 0.1 → "价值信号可读出"成立；否则记否定结果归档（否定也是结果）

## 执行顺序

1. `rl/reward.py` + 单元冒烟（手工构造 reply/状态，验算带状指标与封顶）——R_task 基线要用同一奖励模块计算，必须先就位
2. 录三基线（判据 0）——**已录（2026-11-21，`rl/data/baselines.json`）**：① 自报 MAE 2.25（对照 v0.7 记录 2.38，告警线 3.09）；② task reward：train 集 0.8572 / held-out 集 0.9208；③ Δfatigue 0.0313 ± 0.0107（保留度判据：RL 后 ≥ 0.0219）
3. `rl/train_rl.py` + 10 episode 冒烟：确认梯度流到调制器、奖励日志落盘、断点续跑生效
4. 配比扫描 3 档（判据 6）
5. 选定配比 → 全量评估 1–4
6. `rl/train_rpe_probe.py`（判据 7）
7. 更新 plan.md（门槛清零 + 判据归档）与 version.md（v0.8）

## 验证方式

- 冒烟：`python rl/train_rl.py --smoke`（10 episode，检查 modulator.grad 非 None、rpe_log.jsonl 有行、kill 后重跑跳过已完成 episode）
- 评估：第 6 步跑完整 eval 套件（三信号/闭环/棘轮/诚实性/fatigue 保留度），结果 JSON 落 `eval/`，对照判据 1–6 逐条勾
- 成本量级：单卡 12GB；rollout 短序列（≤96 token），三个配比扫描为小时—天量级的卡时，评分零额外卡时（启发式）

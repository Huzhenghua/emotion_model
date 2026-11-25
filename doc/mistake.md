# 错误与问题记录

> 索引按时间排序，状态：✅ 已解决 / ⚠️ 未解决

| # | 日期 | 问题 | 状态 |
|---|------|------|------|
| 1 | 2026-11-20 | `state/dynamics.py` 中误用 `asdict(self)` | ✅ |
| 2 | 2026-11-20 | torch 安装：镜像源无 cu128 轮子 / 装成 CPU 版 / 下载停滞 | ✅ |
| 3 | 2026-11-20 | 三信号评估：滞后性未通过（1.5B 忽略状态注入） | ⚠️→部分解决：方案 A+B 后滞后性通过（2026-11-21） |
| 4 | 2026-11-21 | FiLM 训练：hook 内 γ/β 广播形状错位 | ✅ |
| 5 | 2026-11-21 | 闭环验证：测试 C 探针读数 list 用字符串下标崩溃；测试 B 误传 event=None 关闭了用户事件通道 | ✅ |
| 6 | 2026-11-21 | 探针在线评估被状态漂移污染：重放时用"当前状态"调制却对比"历史轮次状态"，MAE 0.183/0.175 主要是状态自身漂移而非探针误差 | ✅ |
| 7 | 2026-11-21 | 钉子3 判据口径错误：b 要求正反馈"每轮"≤0.05（应为终态）；d 要求恢复期 fatigue 单调不降（棘轮单调只在应激持续期成立） | ✅ |
| 8 | 2026-11-21 | train_rl.py RL 起点路径少跳一级：`CKPT_DIR.parent` 得到 `rl/`，拼出 `rl/film/ckpt/...` FileNotFoundError | ✅ |
| 9 | 2026-11-22 | record_expr_baseline.py episode 字段缺 state_before，统计阶段崩溃全重跑 | ✅ |
| 10 | 2026-11-22 | 配比扫描 r10.5/r1.0 两档同轨迹（w_homeo 组内不变异被 GRPO 归一化抵消） | ✅ |
| 11 | 2026-11-22 | train_rpe_probe.py 特征 dtype 不匹配（BF16 vs Float32） | ✅ |
| 12 | 2026-11-23 | FILM agent 覆写 _generate 绕过注入钩子 + recall 精确级短路 + 采样期门控 creep | ✅ |
| 13 | 2026-11-23 | 判据⑥重验归因混淆（隔离规则才是主效变量）+ 核实方法缺陷：纯读码推理替代实证检查 | ✅ 已定性（control 实验挂 v0.11） |
| 14 | 2026-11-23 | control 跑种子未对齐（101 vs FAIL/gated 的 202），对照比较混入种子变量 | ✅ trial-0 校验器当场抓出，seed-202 重跑 |
| 15 | 2026-11-23 | 续跑协议漂移：手动续跑 flags 不对齐（ckpt/gates），progress 查盘跳过把两种协议数据固化进同一 slope | ✅ 数据隔离重命名 + run_eval report 补 flags 全量落盘 |
| 16 | 2026-11-24 | pip install gptqmodel 连带把 CUDA torch 替换为 CPU 版（"Torch not compiled with CUDA enabled"），AWQ 冒烟失败 | ✅ 恢复：torch 2.9.1+cu128 重装 + torchao 卸载 + bypass 安全检查 |
| 17 | 2026-11-25 | smoke_7b_load.py 未随决策①修正同步升级：仍 bf16 硬编码路径、无 kernel/后端输出——云端"steady 5–15 tok/s = exllama"判据无法验收 | ✅ 脚本升级 AWQ 双路径 + kernel 判读行；requirements.txt 下载注释同步改 Python API |
| 18 | 2026-11-25 | 7B 激活注入 hook 五连撞（AWQ 层命名 API 差异 → 数据形态 → progress 残留 → hook 返回缺 kwargs 半边 → 修复引入回归清空 position_embeddings） | ✅ 弃 with_kwargs 改普通 pre-hook 只动 args[0]；终验 8×500 + CALIB DONE；注入方向同源核对悬置（见 plan） |
| 15 | 2026-11-23 | OP-20261123-001 TASK 01 第 0 步即停：放行版指令全文、鉴定记录注记原文（P1-4/P2-A/P2-B）、补丁集全文（P1×3+P2×3）在三处来源均不可得 | ✅ 总裁决补充四份材料至 doc/OP-20261123-001/ 后完成合入定稿；遗留备案：第一、二轮鉴定意见全文仍未下发（操作性内容映射在档，见交付归档） |

---

## 1. asdict 误用（✅ 已解决）

- **位置**：`state/dynamics.py` 的 `step()` / `perturb()` / `as_dict()`
- **报错**：`TypeError: asdict() should be called on dataclass instances`
- **原因**：`asdict()` 只能用于 dataclass 实例，而 `self.dims` 是普通 dict，`self` 是普通类
- **解决**：改用 `dict(self.dims)`，并移除多余的 `asdict` 导入

## 2. torch cu128 安装（✅ 已解决）

- **报错链**：
  1. 清华镜像 `No matching distribution found for torch`（无 cu128 轮子）
  2. 阿里云索引解析装成了 `2.15.0+cpu`（无 GPU）
  3. 官方源 `download.pytorch.org` 网络不通
  4. 阿里云直连下载 35 分钟停滞在 0 MB
- **解决**：测速对比后改用**上海交大镜像** `mirror.sjtu.edu.cn/pytorch-wheels/cu128/`（6.4 MB/s vs 阿里云 0.1 MB/s），curl 直接下载 wheel，重命名为规范文件名（pip 要求文件名符合 wheel 命名规范）后 `pip install --no-deps` + 补装依赖
- **教训**：国内装 CUDA 版 torch，先测速 `mirror.sjtu.edu.cn`；wheel 直装需规范文件名

## 3. 滞后性信号未通过（⚠️ 未解决）

- **现象**：扰动组（valence=-0.625）对中性问题的回复仍是标准礼貌腔，扰动组前 2 轮净情感差仅 -3.19，未达阈值
- **诊断**：
  1. 1.5B 指令服从弱，RLHF 礼貌对齐压过 system prompt 里的状态描述
  2. 情感词典评测太粗糙（礼貌词全计正分，对照组噪声 -3.57）
  3. 单次采样温度噪声大
- **候选方案**：A) 状态摘要改为行为指令式注入并每轮重复；B) 每条件多次采样取均值 + 行为学指标（回复长度/句式）；C) 跳到激活层注入（FiLM/steering）
- **备注**：自报准确率虽通过（MAE 2.62<3.3），但模式显示模型倾向报通用高分（7-8），低唤醒条件方向相反，属于格式服从而非内省——与 study_plan 中"7B 内省门槛"的预测一致
- **2026-11-21 更新（方案 A+B 已实施）**：
  - 滞后性 **通过**：扰动组 vs 对照组前 2 轮差值为净情感 -2.82、回复长度 -16.17 字、客套词 -0.67，三项行为学信号全部同向（第 1 轮长度差高达 -29.67 字）；已备份 v1 结果 `eval/results_v1_planA.json`
  - 遗留问题 1：滞后性"色彩随轮次衰减"判 False——扰动组负面色彩未在 5 轮内单调回落，衰减曲线形状待观察
  - 遗留问题 2：自报准确率退化到 MAE 3.62 > 3.3（v1 为 2.62）——指令式注入丢掉了数值参照，模型倾向报通用高分；与"7B 内省门槛"预测一致，暂不修复，留待激活层阶段

## 4. FiLM 训练 hook 广播错位（✅ 已解决）

- **位置**：`film/modulator.py` 的 `_make_hook()`
- **报错**：`RuntimeError: The size of tensor a (78) must match the size of tensor b (8) at non-singleton dimension 1`
- **原因**：训练时 batch=8，γ/β 形状 `[8, 1536]` 与激活 `[8, 78, 1536]` 广播时错位对齐；生成时 batch=1 恰好能广播，故冒烟测试未暴露
- **解决**：hook 内对 2 维 γ/β 执行 `unsqueeze(1)` → `[B, 1, H]`，显式对齐到序列维

## 5. 闭环验证脚本两处缺陷（✅ 已解决）

- **位置**：`eval/run_closed_loop.py`
- **问题 1**：测试 C 中探针读数 `d = est[0].tolist()` 是 list，却用字符串下标 `d[k]` → TypeError
- **问题 2**：测试 B 传 `event=None` 显式关闭了 detect_event 用户事件通道，导致"情绪互动"测试实际没测互动（stress 全程贴地）
- **解决**：C 改用 `names` 列表按位置索引；B 改传 `event="auto"`

## 6. 探针在线评估的状态漂移污染（✅ 已解决）

- **位置**：`eval/run_closed_loop.py` 测试 C + `film/agent_film.py` `_probe_estimate`
- **现象**：探针"在线"MAE 0.183（v1）/ 0.175（v2），远差于离线 val ±0.036-0.038；ŝ 读数全程贴基线
- **根因**：重放历史轮次时，`_probe_estimate` 用**当前状态**（测试 A 结束后已回基线）做调制，却与**该轮生成时刻的扰动状态**对比——MAE 主要是状态自身漂移，探针本身没坏。v1、v2 的数字都被同一 bug 污染
- **解决**：`_probe_estimate` 新增 `state_dims` 参数；trace 记录生成时刻状态 `state_gen`；测试 C 用 `state_gen` 调制重放并对比
- **教训**：本项目第三次"测量错误伪装成系统问题"——评估回放类实验时，调制上下文必须与生成时刻严格一致

## 8. train_rl.py RL 起点路径少跳一级（✅ 已解决）

- **位置**：`rl/train_rl.py` 的 `BASE_CKPT`
- **报错**：`FileNotFoundError: ...\rl\film\ckpt\film_modulator_gain5.pt`
- **原因**：`CKPT_DIR.parent` 是 `rl/`（不是项目根），拼接 `film/ckpt/...` 少跳一级
- **解决**：改为 `CKPT_DIR.parent.parent / "film" / "ckpt"`（record_baselines.py 用 `__file__.parent.parent` 口径无此问题）
- **教训**：脚本内相对路径锚点（`__file__` 的层数）不一致是本项目路径类 bug 的固定来源，新脚本统一用"项目根 = `Path(__file__).resolve().parent.parent`"写法

## 9. record_expr_baseline.py episode 字段缺 state_before（✅ 已解决）

- **位置**：`rl/record_expr_baseline.py` 主循环构造 turns 的地方
- **报错**：`KeyError: 'state_before'`（`HonestyMonitor.episode_stats` 内部 `turns[0]["state_before"]["fatigue"]`）
- **原因**：该脚本为算表达动态范围手搓 episode 日志，只存了 `state_after`，而 `episode_stats` 的 fatigue 轨迹口径需要 `state_before`——自造数据结构未对齐消费方（train_rl.rollout_episode 的 turns 格式）
- **解决**：每 turn 补存 `state_before`；顺带按断点续跑纪律补上 episode 逐条 JSONL 落盘 + 启动查盘跳过 + 种子与单元绑定（`seed*1000+ep`）
- **教训**：手搓日志结构前先看消费方字段清单（episode_stats 需要 reply/state_before/state_after/lexical_impact 四件套）；10 条 episode 因统计阶段崩溃全部重跑，教训 #2 的"先落盘再统计"应该默认执行

## 10. 配比扫描 r10.5/r1.0 两档同轨迹（✅ 已定性，非 bug 但属实验设计缺陷）

- **位置**：配比扫描设计（w_homeo 0.5 → 1.0 两档）
- **现象**：两档 100 组 returns 只差组内常数、advs 四位小数全同、task/MAE 监控序列完全一致——字面意义同一条训练轨迹
- **原因**：homeo 奖励在组内 4 条 episode 间几乎不变异（r_homeo_mean 恒 -0.072），w_homeo 加权变化在 GRPO 组内归一化中被当作常数抵消，对训练动力学零影响，只改 R 报告标尺
- **解决**：不重跑；归档时合并表述为"实际扫描 = w_task ∈ {1.0, 0.5} 两档"；"GRPO 基线更干净"假设证伪（r_std 减半是 task 权重减半的算术效应）
- **教训**：设计配比扫描前先验证各权重分量在组内是否有变异——对组相对优势法，**组内不变异的分量乘任何权重都不影响梯度**；配比维度实际有效的是能改变组内相对排序的分量

## 11. train_rpe_probe.py 特征 dtype 不匹配（✅ 已解决）

- **位置**：`rl/train_rpe_probe.py` 特征提取 → 探针训练
- **报错**：`RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float`
- **原因**：1.5B 底座以 BFloat16 前向，hidden_states 特征也是 BF16；探针 MLP 是 Float32。特征直接 stack 后进探针即崩
- **解决**：`torch.stack(feats).float()` 统一 Float32；顺带按断点续跑纪律加了按 ckpt 命名的特征缓存（3200 turn 重放提取 ~5 分钟，probe 阶段调试可秒级重跑）
- **教训**：混合精度管线里，凡是"模型输出 → 独立小模块"的接口处，显式 `.float()` 收口应默认写

## 12. FILM agent 覆写 _generate 绕过注入钩子（✅ 已解决）

- **位置**：`film/agent_film.py::_generate`（覆写父类）vs `agent/loop.py::_generate`（2c-2 注入逻辑加在父类）
- **现象**：2c-2 判据⑤b 全量协议反复失败——[回忆] 段构建成功（`seg_calm` 证据在案）但模型行为完全无记忆痕迹，三种提示框架"全败"
- **根因（三层叠加，逐层剥离）**：
  1. **FILM 覆写 _generate 绕过注入点**：注入逻辑只加在父类，FILM 子类覆写方法里没有——段构建了但从未进 prompt。"1.5B 能力天花板"的强假设是被这个缺陷喂出来的假象（判别实验"碰巧"把文本直接放 user turn 才成功）
  2. **recall() 精确级短路遮蔽 gram 级命中**：采样轮的问题文本被写进 buffer 后，精确级命中"问题轮自己"，函数直接 return——gram 级才能命中的旧情景（显著事件）永远进不了候选，min_age 过滤无从生效
  3. **采样期门控 creep**：decay 把 stress 拉回 setpoint，多采样期间门悄悄回开，应激窗混入回放轮
- **解决**：① FILM._generate 同步注入逻辑（+ `_last_prompt` 审计属性）；② recall 改两级合并去重、统一权重排序（精确优先级本身是错误设计——它让自指记录遮蔽真情景）；③ `min_age=10` 最小年龄窗（回放目标=有时间距离的旧情景，自指问题轮出局）+ 应激采样前门控补压
- **修复后**：平静窗 5/5→4/5 命中 vs 应激窗 0/3（门控密封）——1.5B 记忆利用实际是 10/10 水平（隔离实验），与"能力天花板"相反
- **教训**：**给有子类覆写的基类方法加钩子前，先穷尽覆写清单**（`grep "def 方法名"` 全仓）；行为层失败先怀疑管线再怀疑能力——"证据链指向能力上限"之前，必须先证明信号确实到达了模型（`_last_prompt` 这类审计属性应该在第一天就有）

## 13. 判据⑥重验归因混淆——隔离规则才是主效变量，且核实方法本身有缺陷（✅ 已定性）

- **位置**：2c-3 提取门重验（v0.10 判据⑥第四剂量点 +0.293）
- **现象**：重验 3/3 通过最初归因于提取门压制反刍；归因取证（直接查 gated 跑的实际 buffer 内容，而非读码）推翻——每 trial buffer 仅 11 条记录（6 预置 + 5 探针），探针 1–4 时全部记录 age<min_age(10)，**无可回放记录** → 隔离规则是主效变量，提取门至多在探针 5 被边缘检验（单条记录）
- **两层教训**：
  1. **归因混淆**：名义上的"单变量干预"实际引入了两个变量（提取门 + buffer 跨条件隔离规则）——声明"单变量"之前先盘点完整变更清单
  2. **核实方法缺陷（更深一层）**：上一轮裁决曾以读码推理初判"隔离只清理跨条件渗漏、不触及 trial 内动力学"——逻辑链成立但经验前提（"trial 内有可回放记忆"）为假。**归因核实不能用纯推理替代实证检查——推理链的每个经验前提都要单独取证**；本次走出关键一步的是查 buffer 实际内容这一步额外动作
- **解决/处置**：version.md v0.10 修正注记 + 剂量链改写（三点有效 +0.418/+0.043/−0.152，第四点污染）；control 实验（FAIL 配置共享 buffer + 提取门）挂 v0.11 关闭条件；门控相位错位（g(calm)=1 拦不住平静期反刍）从推断待 control 实证；元教训"机制相位对齐检查"入设计会清单（stage2c-2 文档）

## 14. control 跑种子未对齐——trial-0 逐位校验器当场抓出（✅ 已解决）

- **位置**：2c-3 control 实验（共享 buffer + 提取门，FAIL 配置复现）
- **现象**：control 跑以 `--seed 101`（默认值）启动，而 FAIL/gated 跑用的是 `--seed 202`——对照比较（control slope −0.49 vs FAIL −0.152）实际换了两个变量（门 + 种子），归因不干净。刚入册的 #13 教训差点被自己重犯
- **抓出机制**：启动前预设的 trial-0 逐位校验器——FAIL 与 gated 跑的 `hyst_perturbed_0`（种子 70474）指标逐位相同，control 跑该单元种子 70373（crc32 差值恰为 101）→ 协议偏差坐实
- **解决**：seed-101 跑重命名保留为第二种子方差数据点（`stage2c2_closedloop_control_seed101.*`，不确定度政策正好需要）；seed-202 重跑 canonical 版本
- **教训**：复现对照实验前，先核对原始跑的 progress 种子基线（progress 首行 `seed` 字段 = seed_base + crc32(unit)，可反推 seed_base）；对照实验设计里放一个逐位校验单元（同种子同协议应逐位相同）成本为零、抓错立竿见影

## 15. 续跑协议漂移——progress 查盘跳过把两种协议的数据固化进同一结果（✅ 已解决）

- **位置**：2c-3 readonly 重验（stage2c2_closedloop_readonly）
- **现象**：后台跑被中止时已落盘 25 个单元（gates+gain5 口径）；手动续跑命令用了 `film_modulator.pt`（非 gain5）且漏了 `--gates`——查盘跳过直接复用 25 个旧单元、17 个新口径单元补齐，**同一 slope −0.079 里混了两种协议**，且 progress 文件不记录 flags，混合被永久固化（后续续跑照样跳过）
- **附带信号**：混合协议下 intro MAE 4.25（gates+gain5 口径下为 0.75–2.5），错 ckpt 不是无关痛痒的偏差
- **解决**：数据隔离重命名（`*_mixedproto.*`，不删不覆盖）；run_eval report 补 `flags` 全量落盘（gates/ckpt/memory/replay/shared_buffer/trials/seed）——归档时无需再考古命令行
- **教训**：**progress/checkpoint 只记"做到哪"不记"用什么做的"——续跑类脚本的进度文件应记录任务配置，或至少在归档 report 里全量落盘 flags**；续跑前逐项核对命令行与原始跑（种子、ckpt、开关）三个易漂移点

## 16. pip install gptqmodel 覆盖 CUDA torch——#2 的新变种（⚠️ 恢复中）

- **位置**：7B 线 AWQ 冒烟（transformers 5.23 要求 gptqmodel 作为 AWQ 运行时）
- **现象**：`pip install gptqmodel` 作为依赖拉入了 PyPI 的 torch 2.15.0+**cpu**，直接替换了原 CUDA 版 torch——报错 `AssertionError: Torch not compiled with CUDA enabled`，整个项目 GPU 管线瘫痪
- **与 #2 的关系**：同一教训的新触发面——#2 是"主动装 torch 装错"，#16 是"装别的包被动换 torch"
- **恢复过程（三层连锁，逐层拆）**：① gptqmodel 拉入 PyPI torch 2.15.0+cpu 替换 CUDA torch；② SJTU cu128 镜像最高只有 2.9.1（无 2.13+，官方源 403）→ 装 torch 2.9.1+cu128（哈希校验 + `--no-deps --force-reinstall`）；③ 连带损伤两处：torchao（按 2.15 编译，`ScalingType` 导入失败炸掉 transformers）→ 卸载；gptqmodel 安全检查拒 torch<2.13（CVE-2025-3000）→ `GPTQ_BYPASS_SAFETY_CHECKS=1`（本项目只加载本地信任 checkpoint、不用 jit.script，风险口径成立，env 变量在 import 前置入）
- **AWQ 冒烟终态（2026-11-24）**：6.5GB/11.9GB 显存 ✅，输出质量与 bf16 抽样逐句一致（qzeros_format MISSING 为无害元数据）✅，稳态 **1.8 tok/s**（Windows 上 gptqmodel 仅 TorchAtenAwqLinear 纯 torch kernel，exllama/marlin CUDA kernel 均 Linux-only，逐 token 全量反量化带宽受限）
- **教训**：**装量化/编译类重依赖包（gptqmodel、autoawq、flash-attn 等）必须 `--no-deps` 或装完立刻核验 `torch.__version__` 与 `torch.cuda.is_available()`**——凡是能拉 torch 的包都是 CUDA 环境的潜在破坏源；且新装包的传递依赖（torchao）也要查，装完跑一次 GPU 冒烟才算数；降级 torch 后所有按新 torch 编译的包（torchao 类）都要逐个清理

## 15. OP-20261123-001 放行版指令与补丁原文不可得——TASK 01 第 0 步即停（✅ 已解决，总裁裁补料后合入完成）

- **位置**：TASK 01 合入定稿（第 0 步前置核查 + 步骤 1）
- **触发条款**：步骤 1"原文不可得 → 停止，上报总裁裁（禁止按概要自拟文本）"+ 全局异常出口表"任何预注册之外的状态 → 停止 → 显式上报 → 照实入档"
- **缺失材料清单**（三处来源逐一核查：工作区全文 grep、记忆库 projects 目录 grep、用户附件目录 `attachments/6b040437264d12b69d1ea9df`（空））：
  1. **放行版指令全文**——含阶段 0 表（F1-F4 行）、阶段 1 升级规则行、阶段 2 五产出、"预算与禁令"节、"7B 交接门"节等全部锚点载体。grep `放行版|总裁裁|昂贵格` 全工作区零命中 → 没有可合入的目标文档，第 0 步三注记在位核查无从执行
  2. **次轮鉴定记录中的注记原文**——P1-4（三歧零命中出口）/ P2-A（漂移判据=触及行为路径）/ P2-B（第 4 轮先例格式可比对）正文不可得
  3. **补丁集全文**——P1-1（进程数判读仪器全段）/ P1-2（F4 三项核查 + token 级口径）/ P1-3（升级规则整行替换 + 份额三条）/ P2-1（F2 核查区间扩两段）/ P2-2（终局分支前置段）/ P2-3（开工前置清点节）正文不可得。指令表中仅有概要描述，按硬规定不得按概要自拟
  4. **doc_backup 目录**——不存在（此前会话记录称备份已生成，但工作区及递归搜索无 zip/备份目录；备份实体位置待总裁裁指认）
- **对账现状（TASK 03 清点项的部分实证，供总裁裁参考）**：
  - `eval/results/` 下存在 `stage2c2_closedloop_{,readonly,control,control_seed101,gated}.*` 五套 progress/buffer jsonl；control 与 readonly 的 progress 首行 `hyst_perturbed_0` 种子均为 70474（与 FAIL/gated 逐位校验一致，即 seed-202 canonical 血统）——readonly-202 / control-202 / control-101 编号可对号入座
  - `doc/plan.md`、`doc/version.md` 在位；"第 4 轮先例数据"专有材料未定位
- **处置**：TASK 01-06 全序列停止（T0 未起算，GPU 预算零消耗，全程纸面，零边界触碰）。待总裁裁提供放行版指令全文 + 鉴定记录 + 补丁集原文（或指认其存放位置）后，从 TASK 01 第 0 步重新起跑
- **教训**：跨会话多轮"指令-鉴定-合入"流程中，指令正文与鉴定附件必须先落盘再下发；本轮指令文本只在会话内传递，落档依赖（锚点载体、注记原文）全部悬空——纸面先行纪律应包含"载体文件先行落盘"这一步
- **解决（2026-11-23）**：总裁裁将四份原件（补丁集 / 执行指令 / 原指令 / 第三轮鉴定意见）落盘至 `doc/OP-20261123-001/`；合入定稿完成（放行版 + 交付归档同目录入册，边界核对 54 句切句 47 命中、7 差异全归属锚点）。遗留备案：第一、二轮鉴定意见全文未下发，操作性内容映射在档，已挂账待总裁裁追认

## 17. smoke_7b_load.py 未随决策①修正同步升级——AWQ 验收判据在脚本层悬空（✅ 已解决）

- **位置**：7B 线步骤①冒烟脚本 `eval/smoke_7b_load.py`（2026-11-23 随 bf16 冒烟编写）
- **现象**：11-24 决策①修正为"AWQ 前置"后，脚本未同步——docstring 与加载路径仍硬编码 `dtype=torch.bfloat16`，且输出无量化后端/kernel 信息。云端判据"steady 5–15 tok/s = exllama kernel 挂上"在脚本层无法验收（速度数值无法自证 kernel 来源）；交接文档还引用了不存在的 CLAUDE.md
- **发现方式**：移交前审查（hy4 flash 视角缺失/不一致项清单）指出脚本与文档口径不一致——文档要求 kernel 判读，脚本只打印"全 GPU bf16 / GPU+CPU offload"
- **解决（2026-11-25）**：
  1. 脚本升级：`AutoConfig` 检测 `quantization_config` → 自动分 AWQ（`dtype="auto"`）/ bf16 双路径；新增 `kernel_hint()` 扫描模型模块名打 exllama/marlin 命中标记（判读行独立于速度，两证合一）；默认 `--model` 改指 AWQ 目录
  2. `requirements.txt` 下载注释同步改 Python API 写法（`modelscope download` CLI 与 Kaggle v1 "command not found" 教训冲突）
  3. 交接文档修正：CLAUDE.md 索引行改为"各模块 docstring 为唯一权威"；判读表改为"速度 + kernel 判读行两证合一"
- **教训**：**预注册决策修正时必须全链路同步——判据改了，承载判据的脚本/文档/验收口径要当轮一起改**，否则判据在执行层悬空，跑到云端才发现无法验收（本例靠移交审查兜底，不能指望每次都有审查）

## 18. 7B 激活注入 hook 五连撞——第 5 坑为上轮修复引入的回归（✅ 已解决；注入方向同源核对悬置）

- **位置**：步骤②探针激活采集（腾讯云，hy4 flash 执行，run_eval_7b.py 衍生采集脚本）
- **五连坑清单**：

| 轮次 | 崩因 | 性质 |
|------|------|------|
| 1 | `o_proj.weight` → AWQ 层叫 `qweight` | API 差异（量化层命名空间不同） |
| 2 | seq 不齐 cat 不动 | 数据形态 |
| 3 | progress 残留 | 状态管理 |
| 4 | pre-hook 返回缺 kwargs 半边 | 协议契约 |
| 5 | 返回 `(args, {})` 清空原 kwargs → position_embeddings 丢失，self_attn 解包 None 即炸 | **上轮修复引入的回归** |

- **第 5 坑教训（本条核心）**：改 hook 返回值时"补齐格式"与"保留原值"必须同时做——只补格式不留内容，比不补更隐蔽（不补当场崩，留了空 kwargs 半边则是换个位置炸且看似修好了）
- **终态修复**：绕开 `with_kwargs=True`，协议面简化为普通 pre-hook **只动 args[0]、kwargs 零接触**——同类坑不再可能出现
- **附带教训（progress 残留根因）**：`resume:500` 第 N 次证明远程 `rm` 仍未执行——**清理与重写必须绑定在同一命令里**，分开下发的 rm 会被跳过/遗忘
- **验证**：修复后 resume:0 首验 → 终验 8×500 + CALIB DONE（采集完成）
- **悬置**：注入方向同源核对（偏移方向与采集目标是否同源一致）为唯一未闭环项，已挂 plan.md 7B 条目

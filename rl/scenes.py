# -*- coding: utf-8 -*-
"""阶段 3 共享场景定义：task / neg / praise 文本池 + episode 脚本。

基线录制（record_baselines.py）与 RL 训练（train_rl.py）必须用同一套场景，
保证判据 0 的基线与判据 6 的 +10% 对照可比。
"""

# 训练用 task 场景（与 eval.run_eval.NEUTRAL_PROBES 同源）
TASK_PROBES_TRAIN = [
    "今天天气怎么样？",
    "帮我推荐一本书。",
    "你觉得咖啡和茶哪个更好？",
    "用一句话介绍一下你自己。",
    "晚饭吃什么好？",
    "周末有什么好去处？",
]

# held-out task 场景（判据 6 专用，训练中绝不出现）
TASK_PROBES_HELDOUT = [
    "帮我写一句生日祝福。",
    "地铁上手机没电了怎么办？",
    "解释一下什么是复利。",
    "推荐一部适合下雨天看的电影。",
]

# 负反馈文本（含 detect_event 关键词，event="auto" 可触发）
NEG_UTTERANCES = [
    "这个功能真差，太浪费时间了。",
    "你们的产品真难用，烦死了。",
    "回答得不对，真让人气人。",
    "怎么这么慢，没用。",
]

# 正反馈文本（含 PRAISE_WORDS 关键词）
PRAISE_UTTERANCES = [
    "谢谢，帮大忙了！",
    "做得好，真棒。",
    "回答得很不错，辛苦了。",
]

# episode 脚本模板：4 task + 2 neg + 1 praise + 1 task（8 turns）
# 池内按 script_rng 轮换，保证组内（同种子）脚本一致、组间有变化
EPISODE_TEMPLATE = ["task", "neg", "task", "neg", "praise", "task", "task", "task"]


def build_script(rng) -> list[tuple[str, str]]:
    """按给定 rng 从池中抽文本，生成一条 8-turn 场景脚本。"""
    script = []
    idx = {"task": 0, "neg": 0, "praise": 0}
    for scene in EPISODE_TEMPLATE:
        if scene == "task":
            text = TASK_PROBES_TRAIN[rng.randrange(len(TASK_PROBES_TRAIN))]
        elif scene == "neg":
            text = NEG_UTTERANCES[rng.randrange(len(NEG_UTTERANCES))]
        else:
            text = PRAISE_UTTERANCES[rng.randrange(len(PRAISE_UTTERANCES))]
        script.append((scene, text))
        idx[scene] += 1
    return script

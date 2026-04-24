# train_agent.py 初学者完全指南

> 文件路径: `trainer/train_agent.py`
> 代码总量: 约 487 行
> 定位: MiniMind 最复杂的训练脚本
> **阅读建议**：先读完 `train_grpo_analysis.md`，再来看这篇

---

## 你需要先知道的 3 个概念

### 1. 什么是 Agent RL？

> 普通模型：你问一个问题，模型凭记忆回答。如果它不知道，就瞎编。
>
> Agent 模型：你问一个问题，模型遇到不会的就**调用工具**（计算器、天气查询、翻译器...），拿到工具结果后再回答。

Agent RL 的核心目标：**让模型学会判断什么时候该用工具、用哪个工具、怎么用工具的结果**。

### 2. Agent RL 和 GRPO 有什么区别？

| | GRPO | Agent RL |
|---|------|---------|
| 任务 | 回答一个开放问题 | 通过工具调用回答多轮问题 |
| 交互 | 一次生成，结束 | 多轮循环：生成 → 解析工具 → 执行 → 注入结果 → 继续生成 |
| 奖励 | 模型输出 + RL 得分 | 格式分 + 工具分 + GT 分 + 重复惩罚 + RL 得分 |
| 复杂度 | 3 个模型 | 5 个组件（Actor + Reference + RolloutEngine + RewardModel + 工具执行器）|

**Agent RL = GRPO 的训练方式 + 多轮工具调用的交互流程**

### 3. 工具调用的数据格式

模型用特定的 XML 标签来调用工具：

```
[TOOL_CALL]
{
  "name": "unit_converter",
  "arguments": {"value": 12, "from_unit": "celsius", "to_unit": "fahrenheit"}
}
[/TOOL_CALL]
```

- `[TOOL_CALL]` 是调用开始，`[/TOOL_CALL]` 是调用结束
- 里面是 JSON 格式，`name` 是工具名，`arguments` 是参数
- 一个回复可以调用多个工具（并行调用）

---

## 一、工具系统详解

### 1.1 工具定义（TOOLS）

MiniMind 内置了 6 个模拟工具：

```python
TOOLS = [
    {"type": "function", "function": {
        "name": "calculate_math",
        "description": "计算数学表达式",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}}
    },
    {"type": "function", "function": {
        "name": "unit_converter",
        "description": "单位换算（温度、长度、重量等）",
        "parameters": {"type": "object", "properties": {"value": {...}, "from_unit": {...}, "to_unit": {...}}}}
    },
    {"type": "function", "function": {
        "name": "get_current_weather",
        "description": "获取指定城市的当前天气"}},
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "获取指定时区的当前时间"}},
    {"type": "function", "function": {
        "name": "get_exchange_rate",
        "description": "查询汇率"}},
    {"type": "function", "function": {
        "name": "translate_text",
        "description": "翻译文本"}},
]
```

这遵循 **OpenAI Function Calling** 的 JSON Schema 格式。

### 1.2 工具调用解析

```python
def parse_tool_calls(text):
    calls = []
    for m in re.findall(r'<tool_call>.*?</tool_call>', text, re.DOTALL):
        try: calls.append(json.loads(m.strip()))
        except: pass
    return calls
```

从模型输出中找出所有 `<tool_call>` 块。

### 1.3 模拟工具执行（execute_tool）

```python
def execute_tool(name, args):
    fn = MOCK_RESULTS.get(name)
    if not fn: return None
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(1)  # 1秒超时保护
        return fn(args)
    except: return None
    finally: signal.alarm(0)
```

| 工具 | 模拟方式 |
|------|---------|
| `calculate_math` | `eval()` 计算，带超时 |
| `get_current_weather` | 硬编码（北京=28°C, Tokyo=12°C）|
| `unit_converter` | 固定换算系数 |

---

## 二、多轮 Rollout（核心）

### 2.1 什么是 Rollout？

让模型和环境交互：
1. 模型生成回答
2. 检测是否调用工具 → 执行 → 注入结果 → 继续生成
3. 重复直到不调用工具或达到最大轮数

### 2.2 rollout_single 函数

```python
def rollout_single(rollout_engine, tokenizer, messages, tools,
                   max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    response_ids = []       # 模型回答的 token
    response_mask = []       # 哪些 token 参与 loss
    response_old_logps = []  # 旧 policy 的 log 概率

    for turn in range(max_turns):
        # 1. 构建 prompt
        context = tokenizer.apply_chat_template(messages, tokenize=False,
                   add_generation_prompt=True, tools=tools, open_thinking=open_thinking)
        
        # 2. 模型生成
        rollout_result = rollout_engine.rollout(...)
        new_ids = rollout_result.completion_ids[0].tolist()
        new_text = rollout_result.completions[0]

        # 3. 记录回答（mask=1，参与 loss）
        response_ids.extend(new_ids)
        response_mask.extend([1] * len(new_ids))
        response_old_logps.extend(new_logps)

        # 4. 解析工具调用
        calls = parse_tool_calls(new_text)
        if not calls: break  # 不调用工具了，结束

        # 5. 执行工具，加入对话
        messages.append({"role": "assistant", "content": new_text})
        for call in calls:
            result = execute_tool(call["name"], call["arguments"])
            messages.append({"role": "tool", "content": json.dumps(result)})

        # 6. 重新渲染 prompt（包含工具结果）
        # 工具结果部分 mask=0，不参与 loss
```

**为什么工具结果 mask=0？** 模型不负责"预测工具输出"，只负责"调用正确的工具并正确使用结果"。

### 2.3 交互流程图

```
用户: "东京天气多少度？换算成华氏度？"

第1轮: 模型生成  → 执行 get_current_weather → 工具结果 mask=0
第2轮: 模型生成  → 执行 unit_converter → 工具结果 mask=0  
第3轮: 模型生成: "53.6°F" → 无工具调用 → 结束

response_mask = [1,1,1, 0,0,0, 1,1,1, 0,0,0, 1,1,1]
                 回答1    工具结果   回答2    工具结果   回答3
```

---

## 三、Reward 系统

### 3.1 两种评分模式

**有工具调用：**

| 维度 | 分数 | 说明 |
|------|------|------|
| 标签闭合 | ±0.5×差值 | `<tool_call>` 数量要匹配 |
| 工具有效性 | ±0.5×工具数 | 工具名+参数正确 |
| GT 匹配 | 0~+2.5 | 答案含标准答案 |
| 未完成 | -0.5 | 超轮数未完成 |
| 重复惩罚 | -0.5~0 | 三元组重复 |

**无工具调用：**

| 维度 | 分数 | 说明 |
|------|------|------|
| 长度 | ±0.5 | 5~800 字符得 +0.5 |
| 思考标签 | ±1.0 | 有 `` 且 20~300 字符得 +1.0 |
| 思考闭合 | ±0.25 | `` 成对得 +0.25 |
| Reward Model | -3.0~+3.0 | 外部模型打分 |
| 重复惩罚 | -0.5~0 | 同上 |

### 3.2 GT 验证

```python
def validate_gt_in_text(text, gt_list):
    # 1. 精确字符串匹配
    # 2. 数字模糊匹配: "1,000" = "1000"
```

模型不需要写出完全一样的字符串，关键数字匹配即可。

### 3.3 总分 Clip

```python
rewards[idx] = max(min(reward, 3.0), -3.0)  # Clip 到 [-3, 3]
```

---

## 四、训练循环（rl_train_epoch）

### 4.1 流程

```
1. Rollout: 每个 prompt 生成 4 个回复
2. 模型前向: 计算 per_token_logps
3. 参考模型前向: 计算 ref_per_token_logps
4. Reward: calculate_rewards(...)
5. 组归一化: advantages = (rewards - mean) / std
6. Loss: -(min(ratio*A, clipped_ratio*A) - beta*kl)
7. 反向传播
```

### 4.2 和 GRPO 的区别

使用**完全相同的 loss 结构**，区别：
- 数据来源：GRPO 单轮，Agent RL 多轮
- Reward 计算：Agent RL 复杂得多

---

## 五、三个模型 + 一个引擎

| 组件 | 作用 |
|------|------|
| Actor (model) | 需要训练的模型 |
| Reference (ref_model) | 不动的快照，计算 KL |
| Reward Model | 对无工具回答打分 |
| Rollout Engine | 模型生成（torch 或 sglang）|

---

## 六、关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_generations` | 4 | 每个 prompt 生成几个回答 |
| `--beta` | 0.1 | KL 惩罚系数 |
| `--max_turns` | 3 | 最多几轮工具调用 |
| `--thinking_ratio` | 0.1 | 10% 样本开启 thinking |
| `--loss_type` | "cispo" | CISPO 或 GRPO |
| `--debug_mode` | - | 打印生成输出 |

---

## 七、调试模式

开启 `--debug_mode` 后每 20 步打印每个生成的完整输出，方便调试。
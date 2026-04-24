# lm_dataset.py 新手指南：MiniMind 的 5 类训练数据

> 文件路径: `dataset/lm_dataset.py`
> 适合人群: 刚接触大模型训练、想了解"数据如何喂给模型"的初学者
> 阅读建议: 先懂概念，再看代码

---

## 什么是这 5 个 Dataset？它们各自对应什么训练阶段？

MiniMind 的训练流程像"教学生"一样，分为多个阶段。每个阶段需要不同格式的数据，因此代码中定义了 5 个 Dataset 类：

```
  阶段 1: 预训练          PretrainDataset   ->  train_pretrain.py
  ┃                          │
  ┃    "阅读海量书籍，学习语言规律"
  ▼                          │
  阶段 2: 监督微调      SFTDataset      ->  train_full_sft.py / train_lora.py
  ┃                          │
  ┃    "跟着老师做练习题，学会对话"
  ▼                          │
  阶段 3: 偏好对齐      DPODataset      ->  train_dpo.py
  ┃                          │
  ┃    "好的回答加分，差的回答罚分"
  ▼                          │
  阶段 3b: RL 对齐      RLAIFDataset    ->  train_grpo.py / train_ppo.py
  ┃                          │
  ┃    "自己动脑筋思考，根据反馈改进"
  ▼                          │
  阶段 4: Agent 强化   AgentRLDataset  ->  train_agent.py
                             │
                    "学会使用工具完成任务"
```

一句话概括每个类的核心职责：

| Dataset | 一句话解释 |
|---------|-----------|
| PretrainDataset | 给模型看纯文本，让它学会"下一个词是什么" |
| SFTDataset | 给模型看对话示例，让它学会按格式回答 |
| DPODataset | 给模型看"好回答 vs 差回答"，让它学会区分优劣 |
| RLAIFDataset | 只给模型问题，让它在训练中自己生成回答 |
| AgentRLDataset | 给模型问题 + 可用工具列表 + 标准答案，训练工具使用能力 |

---

## 两个辅助函数：所有对话数据的"公共处理"

在深入每个 Dataset 之前，先了解两个被多处复用的辅助函数。

### pre_processing_chat — 随机注入 System Prompt

```python
def pre_processing_chat(conversations, add_system_ratio=0.2):
```

**它做什么？** 有 20% 的概率，给对话开头偷偷塞一条 system 消息。

**为什么要这样做？** 数据增强。真实场景中用户可能带 system prompt 也可能不带。随机注入让模型学会"不管有没有 system 指示，都能正常工作"。

**工作流程：**

```
输入: [{"role": "user", "content": "你好"}]
            │
            ├─ 检测到 tool use 数据？→ 是 → 原样返回，不做处理
            │
            ├─ 第一条已经是 system？→ 是 → 原样返回
            │
            └─ 否 → 掷骰子（20% 概率）
                      │
                      ├─ 掷中了 → 随机选一条 system 插到开头
                      └─ 没掷中 → 原样返回
```

```
输出（掷中的情况）:
[
  {"role": "system", "content": "你是minimind，一个小巧但有用的语言模型。"},
  {"role": "user", "content": "你好"}
]
```

system prompt 池包含 10 条（5 条中文 + 5 条英文），内容大同小异，都是"你是一个有帮助的 AI"类表述。

### post_processing_chat — 清理空思考标签

```python
def post_processing_chat(prompt_content, empty_think_ratio=0.2):
```

**它做什么？** 思考型模型（如 R1 风格）的数据中常出现空的思考块 `<think>\n\n</think>\n\n`。这个函数以 80% 的概率将其删除。

**为什么要这样做？** 空思考没有信息量，大部分情况下应该移除。但故意保留 20%，让模型学会"遇到空思考不要慌，直接跳过"。

```
输入: "你是minimind\n<think>\n\n</think>\n\n你好！世界"
            │
            ├─ 有空思考标签 且 掷骰子 > 0.2（80% 概率）
            │     → 删除: "你是minimind\n你好！世界"
            │
            └─ 掷骰子 <= 0.2（20% 概率）
                  → 保留原样
```

---

## 1. PretrainDataset — 预训练数据

### 这个类解决了什么问题？

预训练是最基础的阶段：给模型看大量纯文本，让它学会语言的统计规律。核心任务叫做**"next-token prediction"**（预测下一个词）。

### 原始 JSONL 数据长什么样？

每条数据只有一个 `text` 字段，非常朴素：

```jsonl
{"text": "人工智能是计算机科学的一个重要分支，它致力于研究和开发能够模拟、延伸和扩展人类智能的理论、方法和技术。"}
{"text": "机器学习是人工智能的核心技术之一，它使用算法来分析和学习数据中的模式。"}
{"text": "深度学习利用多层神经网络来处理和理解复杂的数据。"}
```

### 数据变换的每一步

```
第 1 步：读取原始文本
   "人工智能是计算机科学的一个重要分支"
            │
第 2 步：tokenizer 分词 (add_special_tokens=False)
   [1064, 2342, 5621, ...]     ← 一堆数字
            │
第 3 步：手动包裹 [BOS] 和 [EOS]
   [BOS, 1064, 2342, 5621, ..., EOS]
   ↑ 开头标记                  ↑ 结尾标记
            │
第 4 步：Padding 到固定长度 (max_length=512)
   [BOS, 1064, 2342, ..., EOS, PAD, PAD, PAD, ...]
                                         ↑ 用 PAD 填满
            │
第 5 步：转为 PyTorch Tensor
   input_ids = torch.tensor([...], dtype=torch.long)
```

### Loss Masking 策略（关键概念）

这是新手最容易困惑的地方：**为什么需要 mask？**

假设一条文本 token 化后只有 20 个 token，但我们要 padding 到 512。后面 492 个 PAD token 不应该参与 loss 计算 — 预测"PAD 的下一个词"毫无意义。

```
input_ids: [BOS, 1064, 2342, ..., EOS, PAD, PAD, PAD, ...]
labels:    [BOS, 1064, 2342, ..., EOS, -100, -100, -100, ...]
                                          ↑
                                    -100 = "忽略这个位置"
```

PyTorch 的 CrossEntropyLoss 看到 label 为 `-100` 的位置就自动跳过。这是框架的约定，不是 MiniMind 自创的。

**规则总结：PAD 位置标为 -100，其余位置正常计算 loss。**

### 输出格式

```python
return input_ids, labels    # 两个 Shape 相同的 Tensor
```

训练时，模型接收 `input_ids` 预测下一个 token，与 `labels` 比较计算交叉熵损失。

### 核心代码速览

```python
def __getitem__(self, index):
    sample = self.samples[index]
    tokens = self.tokenizer(str(sample['text']), add_special_tokens=False,
                            max_length=self.max_length - 2, truncation=True).input_ids
    tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
    input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[input_ids == self.tokenizer.pad_token_id] = -100   # ← 关键：mask
    return input_ids, labels
```

---

## 2. SFTDataset — 监督微调数据

### 这个类解决了什么问题？

预训练让模型"懂语言"，但不会对话。SFT（Supervised Fine-Tuning）用高质量的问答对教模型：**用户说什么，AI 该怎么回。**

### 原始 JSONL 数据长什么样？

每条数据包含一个 `conversations` 数组，里面是交替的 user 和 assistant 消息：

```jsonl
{
  "conversations": [
    {"role": "user", "content": "请简要解释什么是神经网络？"},
    {"role": "assistant", "content": "神经网络是一种受生物神经元启发的计算模型..."}
  ]
}
```

可能还包含思考过程、工具调用等进阶字段：

```jsonl
{
  "conversations": [
    {"role": "user", "content": "计算 25 x 38"},
    {"role": "assistant", "content": "950", "reasoning_content": "25 × 38 = 25 × (40-2) = 1000 - 50 = 950"}
  ]
}
```

### 数据变换的每一步

```
第 1 步：读取 conversations
   [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]
            │
第 2 步：pre_processing_chat（20% 概率加 system）
   [{"role": "system", "content": "你是minimind..."}, ...]
            │
第 3 步：apply_chat_template（tokenizer 的对话模板）
   把结构化 JSON 变成一段纯文本，类似：
   "<s>system\n你是minimind...\n</s>\n<user>\n你好\n</user>\n<s>assistant\n你好！\n</s>\n"
            │
第 4 步：post_processing_chat（清理空思考标签）
            │
第 5 步：tokenizer 分词 → input_ids
   [1, 200, 300, ..., 500, 2, ...]
            │
第 6 步：Padding + generate_labels（只标 assistant 部分）
```

### Loss Masking 策略（重点！）

SFT 和预训练最大的区别：**不是所有 token 都要算 loss。**

- User 说的话 → 不算 loss（模型不需要"学会"用户怎么提问）
- Assistant 的回答 → 算 loss（这才是模型该学的内容）
- System 消息 → 不算 loss

**实现方式：** `generate_labels` 方法扫描 token 序列，寻找 assistant 的回复区间。

```
它怎么知道哪里是 assistant 回复？
→ 搜索特征串: <bos>assistant\n   ...   <eos>\n
              ← 起点                ← 终点
```

举个例子：

```
完整 prompt 文本:
"<s>system\n你是minimind\n</s>\n<user>\n你好吗\n</user>\n<s>assistant\n我很好！谢谢\n</s>\n"

input_ids: [BOS, SYS, 你, 是, m, i, n, i, ..., USR, 你, 好, 吗, ..., ASST, 我, 很, 好, ！, EOS, ...]

labels:    [-100, -100, -100, ...  ← system 和 user 部分全 -100
            ... ASST, 我, 很, 好, ！, EOS, -100, ...] ← 只有 assistant 回答有值
                   ↑ 从这里开始算 loss ↑
```

代码是怎么做到的：

```python
def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)     # 先全部设为 -100（忽略）
    i = 0
    while i < len(input_ids):
        # 找 "<bos>assistant\n" 这个标记序列
        if input_ids[i:i + len(self.bos_id)] == self.bos_id:
            start = i + len(self.bos_id)  # assistant 回复起点
            end = start
            # 找到 "<eos>\n" 结束
            while end < len(input_ids):
                if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                    break
                end += 1
            # 这个区间内的 token 正常计算 loss
            for j in range(start, min(end + len(self.eos_id), self.max_length)):
                labels[j] = input_ids[j]
            i = end + len(self.eos_id)
        else:
            i += 1
    return labels
```

### 输出格式

```python
return input_ids, labels    # 与 PretrainDataset 一致
```

格式相同，但 labels 的含义不同：预处理只 mask 了 padding，这里还 mask 了 user 和 system 部分。

### 核心代码速览

```python
def __getitem__(self, index):
    sample = self.samples[index]
    conversations = pre_processing_chat(sample['conversations'])  # ← 加 system
    prompt = self.create_chat_prompt(conversations)               # ← 变文本
    prompt = post_processing_chat(prompt)                         # ← 清思考
    input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
    input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
    labels = self.generate_labels(input_ids)                      # ← 关键：mask
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
```

---

## 3. DPODataset — 直接偏好优化数据

### 这个类解决了什么问题？

SFT 教模型"怎么回答"，但不知道"回答得好不好"。DPO（Direct Preference Optimization）通过对比学习：给模型看同一个问题的两个回答，一个"好"一个"差"，教它学会区分。

### 原始 JSONL 数据长什么样？

每条数据包含两组完整的对话 `chosen`（被选中的/好的）和 `rejected`（被拒绝的/差的）：

```jsonl
{
  "chosen": [
    {"role": "user", "content": "太阳有多大？"},
    {"role": "assistant", "content": "太阳的直径约139万公里，是太阳系中最大的天体。"}
  ],
  "rejected": [
    {"role": "user", "content": "太阳有多大？"},
    {"role": "assistant", "content": "太阳很大，非常大，比地球大很多很多。"}
  ]
}
```

注意：`chosen` 和 `rejected` 的 user 问题是一样的，区别只在于 assistant 的回答质量。

### 数据变换的每一步

```
第 1 步：分别取 chosen 和 rejected 对话
            │
第 2 步：各自 apply_chat_template 转成文本
   chosen_prompt:   "<s>user\n太阳有多大？\n</user>\n<s>assistant\n太阳的直径约139万...\n</s>\n"
   rejected_prompt: "<s>user\n太阳有多大？\n</user>\n<s>assistant\n太阳很大，非常大...\n</s>\n"
            │
第 3 步：post_processing_chat（清理空思考标签）
            │
第 4 步：各自 tokenize + padding
            │
第 5 步：各自生成 loss mask（只标 assistant 部分）
            │
第 6 步：构建成 x/y 对
   x = input_ids[:-1]    ← 去掉最后一个 token（作为模型输入）
   y = input_ids[1:]     ← 去掉第一个 token（作为预测目标）
```

### x/y 对的构造 — 新手常见疑问

为什么要有 `x` 和 `y` 两分？这和标准的 `input_ids + labels` 不一样吗？

```
假设 input_ids = [10, 20, 30, 40, 50]

x = [10, 20, 30, 40]   ← 去掉最后一个
y = [20, 30, 40, 50]   ← 去掉第一个

含义：x[0]=10 时预测 y[0]=20; x[1]=20 时预测 y[1]=30; ...
这就是 next-token prediction 的另一种写法。
```

这样做的好处是 mask 可以单独和 `y`（即真实下一个 token）对齐，方便后续的 DPO loss 计算。

### Loss Masking 策略

和 SFTDataset 类似，**只 mask assistant 回复部分**。但输出不是 `-100` 而是 `0/1` 二元掩码：

```
input_ids:  [USER, 你, 好, ASST, 太, 阳, 很, 大, EOS]
loss_mask:  [  0,   0,  0,   0,    1,   1,   1,  1,  1]
                          ↑        ↑
                    不算 loss    算 loss
```

去掉第一个 token 后，mask 也对应去掉第一个。

### 输出格式

```python
return {
    'x_chosen': Tensor,        # chosen 的输入序列（去掉末尾）
    'y_chosen': Tensor,        # chosen 的目标序列（去掉开头）
    'mask_chosen': Tensor,     # chosen 的 0/1 mask
    'x_rejected': Tensor,      # rejected 对应
    'y_rejected': Tensor,
    'mask_rejected': Tensor,
}
```

总共 6 个 Tensor，chosen 和 rejected 各 3 个。

### 核心代码速览

```python
def __getitem__(self, index):
    sample = self.samples[index]
    chosen_prompt = self.tokenizer.apply_chat_template(sample['chosen'], ...)
    chosen_prompt = post_processing_chat(chosen_prompt)
    rejected_prompt = self.tokenizer.apply_chat_template(sample['rejected'], ...)
    rejected_prompt = post_processing_chat(rejected_prompt)

    chosen_input_ids = self.tokenizer(chosen_prompt, ...).input_ids
    rejected_input_ids = self.tokenizer(rejected_prompt, ...).input_ids

    chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)
    rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)

    return {
        'x_chosen': torch.tensor(chosen_input_ids[:-1]),
        'y_chosen': torch.tensor(chosen_input_ids[1:]),
        'mask_chosen': torch.tensor(chosen_loss_mask[1:]),
        'x_rejected': torch.tensor(rejected_input_ids[:-1]),
        'y_rejected': torch.tensor(rejected_input_ids[1:]),
        'mask_rejected': torch.tensor(rejected_loss_mask[1:]),
    }
```

---

## 4. RLAIFDataset — RL 对齐数据

### 这个类解决了什么问题？

和 DPO 不同，RLAIF（Reinforcement Learning from AI Feedback）不依赖预先标注的"好/差"回答。它只提供问题，让模型在训练过程中**自己生成回答**，然后通过奖励模型来评估和改进。

### 原始 JSONL 数据长什么样？

和其他对话类一样，包含 `conversations` 字段：

```jsonl
{
  "conversations": [
    {"role": "user", "content": "请写一篇关于春天的短诗"},
    {"role": "assistant", "content": "春风拂柳绿，燕语呢喃梁。花开庭院处，又是一年芳。"}
  ]
}
```

但这条数据中的 assistant 回答**不会被使用**！RL 训练中模型会自己生成。

### 数据变换的每一步

```
第 1 步：读取 conversations
            │
第 2 步：pre_processing_chat（20% 概率加 system）
            │
第 3 步：随机决定是否开启 thinking 模式 (thinking_ratio=0.5)
   open_thinking=True  → prompt 中包含思考标签
   open_thinking=False → prompt 中不包含思考标签
            │
第 4 步：apply_chat_template(conversations[:-1], add_generation_prompt=True)
   注意: conversations[:-1] 去掉了最后一条（assistant 回答）
   add_generation_prompt=True 在结尾加上 assistant 前缀，等待生成
            │
第 5 步：返回极简结果
```

### Loss Masking 策略

**RLAIF 没有 loss mask。** 这个类根本不做 token 级别的监督学习。它提供的是 prompt，由模型在 RL 训练时自行生成回答并计算 reward。

### 输出格式

```python
return {
    'prompt': prompt_str,   # 格式化后的 prompt 文本（字符串，非 token）
    'answer': ""            # 空字符串 — 训练时由模型自己生成
}
```

这是 5 个 Dataset 中最简单的一个，只返回两个字符串。

### Thinking 模式说明

`thinking_ratio=0.5` 意味着：一半的训练轮次，prompt 中会包含 `` 和 `` 标签，引导模型进入"先思考再回答"模式。另一半则不含这些标签，让模型学会两种模式。

### 核心代码速览

```python
def create_chat_prompt(self, conversations):
    conversations = pre_processing_chat(conversations)
    use_thinking = random.random() < self.thinking_ratio   # 50% 概率
    return self.tokenizer.apply_chat_template(
        conversations[:-1],                                # 去掉回答
        tokenize=False,
        open_thinking=use_thinking,                        # 随机开启思考
        add_generation_prompt=True                         # 加生成前缀
    )

def __getitem__(self, index):
    sample = self.samples[index]
    prompt = self.create_chat_prompt(sample['conversations'])
    return {'prompt': prompt, 'answer': ""}
```

---

## 5. AgentRLDataset — Agent 强化学习数据

### 这个类解决了什么问题？

现代大模型不仅要会聊天，还要会**使用工具**（搜索、计算器、代码执行等）。AgentRL 训练让模型学会根据问题和可用工具，正确地调用工具并给出答案。

### 原始 JSONL 数据长什么样？

除了 `conversations` 外，还包含 `gt`（ground truth / 标准答案）：

```jsonl
{
  "conversations": [
    {"role": "system", "content": "你可以使用工具", "tools": "[{\"name\": \"search\", \"description\": \"搜索信息\"}]"},
    {"role": "user", "content": "今天天气如何？"}
  ],
  "gt": "调用 search 工具查询今日天气，然后回答用户"
}
```

### 数据变换的每一步

```
第 1 步：手动逐行读取 JSONL（不是用 datasets 库）
            │
第 2 步：parse_conversations()
   ├─ 提取 messages 列表（去掉最后一条）
   └─ 从 system 消息中解析 tools 字段（JSON 字符串 → Python dict）
            │
第 3 步：返回
```

### Loss Masking 策略

**AgentRL 没有 loss mask。** 它返回的是原始结构化的数据（消息列表 + 工具定义 + 标准答案），具体怎么做 tokenization 和 loss 计算由 `train_agent.py` 决定。

### 为什么手动读取 JSONL？

其他 4 个类都用 HuggingFace `datasets.load_dataset`，只有 AgentRL 用纯 Python `json.loads`：

```python
with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        self.samples.append(json.loads(line.strip()))
```

原因：Agent RL 数据的 `tools` 字段是嵌套的 JSON 字符串（`"[{...}]"`），`datasets` 库的 Schema 定义处理这种"字符串里套 JSON"的结构不够灵活。直接用 `json.loads` 最可靠。

### 输出格式

```python
return {
    'messages': messages,   # 对话消息列表（list of dict）
    'tools': tools,         # 工具定义（dict 或 None）
    'gt': sample['gt']      # 标准答案（字符串）
}
```

返回原始 Python 对象，不做 tokenization。

### 核心代码速览

```python
def parse_conversations(self, conversations):
    messages = []
    tools = None
    for message in conversations:
        message = dict(message)
        if message.get("role") == "system" and message.get("tools"):
            tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
        messages.append(message)
    return messages[:-1], tools        # 去掉最后一条 + 提取工具

def __getitem__(self, index):
    sample = self.samples[index]
    messages, tools = self.parse_conversations(sample['conversations'])
    return {'messages': messages, 'tools': tools, 'gt': sample['gt']}
```

---

## 可视化对比：5 个 Dataset 的数据流

```
PretrainDataset:
  JSONL {"text": "..."}
    → tokenize → [BOS] + ids + [EOS] → pad → input_ids, labels(-100)

SFTDataset:
  JSONL {"conversations": [...]}
    → pre_processing (加system)
    → apply_chat_template → 文本
    → post_processing (清思考)
    → tokenize → pad → input_ids
    → generate_labels (只标assistant) → input_ids, labels

DPODataset:
  JSONL {"chosen": [...], "rejected": [...]}
    → apply_chat_template (各自) → post_processing → tokenize
    → generate_loss_mask (各自) → x/y/mask 对
    → {x_chosen, y_chosen, mask_chosen, x_rejected, y_rejected, mask_rejected}

RLAIFDataset:
  JSONL {"conversations": [...]}
    → pre_processing
    → apply_chat_template(conversations[:-1], add_generation_prompt=True)
    → {prompt: str, answer: ""}

AgentRLDataset:
  JSONL {"conversations": [...], "gt": "..."}
    → parse_conversations (提取tools) → {messages, tools, gt}
```

---

## 综合对比表

| 特性 | PretrainDataset | SFTDataset | DPODataset | RLAIFDataset | AgentRLDataset |
|------|:-:|:-:|:-:|:-:|:-:|
| **训练阶段** | 预训练 | 监督微调 | 偏好优化 | RL 对齐 | Agent RL |
| **数据源格式** | `.json` / `.jsonl` | `.jsonl` | `.jsonl` | `.jsonl` | `.jsonl` |
| **数据加载方式** | `load_dataset` | `load_dataset + Features` | `load_dataset` | `load_dataset` | 纯 Python `json.loads` |
| **默认 max_length** | 512 | 1024 | 4096 | 1024 | 1024 |
| **使用 chat template** | 否 | 是 | 是 | 是 | 否（手动解析） |
| **pre_processing_chat** | 否 | 是 | 否 | 是（内部调用） | 否 |
| **post_processing_chat** | 否 | 是 | 是 | 否 | 否 |
| **Thinking 模式支持** | 否 | 否 | 否 | 是（50% 概率） | 否 |
| **工具使用支持** | 否 | 是（tool_calls） | 否 | 否 | 是（tools 定义） |
| **输出格式** | `(input_ids, labels)` | `(input_ids, labels)` | `{6 个 key 的 dict}` | `{prompt, answer}` | `{messages, tools, gt}` |
| **Loss Masking** | PAD 位 = -100 | assistant 回复区 = tokens<br>其余 = -100 | assistant 回复区 = 1<br>其余 = 0 | 无（RL 生成式） | 无（结构化返回） |
| **是否做 tokenization** | 是 | 是 | 是 | 否（返回文本） | 否（返回原始对象） |

---

## Loss Masking 策略总结

这是 LLM 训练中最重要也最容易被忽视的概念之一。一句话概括：**-100 就是告诉模型"这个位置别学"。**

| 场景 | 什么被 mask？ | 为什么？ |
|------|-------------|---------|
| 所有数据集的 padding 部分 | PAD token 位置 | 预测 "PAD 的下一个词" 没有意义 |
| SFT / DPO 的 user 消息 | user 发言的所有 token | 模型不需要学"用户会怎么提问" |
| SFT / DPO 的 system 消息 | system 消息的所有 token | system 是给定条件，不需要预测 |
| SFT / DPO 的 assistant 回复 | **不 mask**（正常计算 loss） | 这才是模型该学的内容 |

---

## 关键设计原则

1. **格式统一**: 对话类数据全部通过 `apply_chat_template` 转文本，保证一致性
2. **损失隔离**: SFT/DPO 精确控制只算 assistant 部分的 loss，避免模型学不该学的东西
3. **数据增强**: system prompt 随机注入 + 空思考标签随机保留，提升模型鲁棒性和泛化能力
4. **按需转换**: 预训练/SFT/DPO 做完整 tokenization（监督学习需要）；RL 类只返回 prompt 或原始对象（由训练脚本决定怎么处理）

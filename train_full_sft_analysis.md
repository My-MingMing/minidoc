# train_full_sft.py 初学者完全指南

> 文件路径: `trainer/train_full_sft.py`
> 代码总量: 约 170 行
> 预备知识: 读过 model_minimind_analysis.md 或了解基本的 "下一个词预测" 概念即可

> **阅读提示**: 这份文档按照 "先懂概念 --> 再看代码" 的顺序写。如果你看到某段代码觉得困惑，先停下来看它上面的概念解释，然后再回来看代码。

---

## 一、什么是 SFT？为什么需要它？

### 从一个比喻开始

想象你在教一个小孩：

- **预训练（Pretrain）** 相当于让小孩读完整个图书馆的书。他认识了很多字，知道了各种知识，但你问他 "请帮我写一首关于秋天的诗"，他可能只会接着往下念，而不知道怎么"执行"你的指令。
- **SFT（Supervised Fine-Tuning，监督微调）** 相当于让小孩做 "问答题练习" —— 你给他问题和参考答案，让他学会怎么正确回答。

### 预训练 vs SFT 对比

| | 预训练（Pretrain） | SFT（监督微调） |
|---|---|---|
| **目标** | 学习 "语言规律" | 学习 "听懂指令" |
| **数据** | 纯文本：`"今天天气不错..."` | 对话：`{"role": "user", "content": "写首诗"}, {"role": "assistant", "content": "秋风送爽..."}` |
| **训练方式** | 整段文本一律预测下一个词 | 只让模型学 assistant 的部分，忽略 user 提问 |
| **学习率** | 5e-4（大胆学） | 1e-5（小心改） |
| **序列长度** | 340 | 768（对话通常更长） |
| **batch size** | 32 | 16 |
| **起点** | 随机初始化 | 加载预训练权重 |

### 一个具体例子

**预训练数据**：

```json
{"text": "Python 是一种高级编程语言，由 Guido van Rossum 于 1989 年发明。"}
```

--> 模型学到的是：看到 "Python 是一种高级" 就预测后面是 "编程语言"。

**SFT 数据**：

```json
{
  "conversations": [
    {"role": "user", "content": "Python 是谁发明的？"},
    {"role": "assistant", "content": "Python 由 Guido van Rossum 于 1989 年发明。"}
  ]
}
```

--> 模型需要学会：当用户问 "Python 是谁发明的？" 时，回答 "Python 由 Guido van Rossum..."。模型只对 assistant 部分的回答负责，不学用户提问的部分。

### 训练管线全景

```
随机初始化         预训练权重         SFT 权重         最终模型
    |                 |                |                |
    v                 v                v                v
  [Pretrain] ----> [Full SFT] ----> [LoRA/DPO/PPO] --> 上线
  学语言建模         学对话格式          学对齐/工具       可用
  2小时/3090        2小时/3090        视阶段而定
```

**SFT 是整个管线中承上启下的关键环节。** 没有 SFT，预训练模型只是一个"话痨复读机"，不会按照指令行动。

---

## 二、SFT 和预训练的参数差异速查

两个脚本共享完全相同的 9 步初始化管线，以下是 **所有不同的参数默认值**：

| 参数 | train_pretrain.py | train_full_sft.py | 为什么改？ |
|------|---:|---:|------|
| `--batch_size` | 32 | **16** | 对话序列更长，每样本显存更大 |
| `--learning_rate` | 5e-4 | **1e-5** | 预训练好的权重不能大改，否则遗忘 |
| `--max_seq_len` | 340 | **768** | 对话含多轮交互，需要更长窗口 |
| `--accumulation_steps` | 8 | **1** | 单 GPU 上 16 已经够小，不需要累积 |
| `--data_path` | `pretrain_t2t_mini.jsonl` | `sft_t2t_mini.jsonl` | 数据格式不同 |
| `--from_weight` | `'none'` | **`'pretrain'`** | 必须从预训练权重开始 |
| `--save_weight` | `'pretrain'` | `'full_sft'` | 输出不同的权重命名 |

**有效 batch size 对比**：

- 预训练：32 x 8 累积 = 256（大数据，大 batch）
- SFT：16 x 1 累积 = 16（小数据，小 batch）

**学习率对比**：

- 预训练：5e-4 --> 5e-5（cosine 衰减到 10%）
- SFT：1e-5 --> 1e-6（cosine 衰减到 10%）

学习率降低一个数量级，是为了**防止灾难性遗忘**——模型不应该丢掉预训练时学会的语言知识。

---

## 三、核心难点：SFTDataset 怎么做 label masking

这是 SFT 和预训练在数据层面**最大的不同**。

### 问题

给模型看一段完整的对话（包含 user + assistant），但 loss 只计算 assistant 说的那些 token。怎么做？

### 答案：用 -100 做屏蔽标记

PyTorch 的 `cross_entropy` 函数接受一个 `ignore_index` 参数。当你把某些 label 设为 -100，这些位置的 loss 就不会被计算。

### SFTDataset 的工作流程

**Step 1：整段对话拼成一条长序列**

tokenizer 的 `apply_chat_template` 方法会把多轮对话拼成一个长字符串，例如：

```
<bos> assistant
[用户问题] assistant
[助手回答]
<sos> user
用户：Python 是谁发明的？
助手：Python 由 Guido van Rossum 于 1989 年发明。

变成 token: [BOS, ..., assistant\n, ..., user\n, ..., assistant\n, P, y, t, h, o, n, ..., 。\n, EOS]
```

**Step 2：用 sliding window 找到所有 assistant 的回答区域**

```python
# SFTDataset 记住 assistant 开头和结尾的 token 模式
self.bos_id = tokenizer('<bos>assistant\n', ...).input_ids  # 例如 [1, 500, 13]
self.eos_id = tokenizer('<eos>\n', ...).input_ids            # 例如 [2, 13]

# 在完整的 token 序列中滑动查找这些模式
# <bos>assistant\n   -->   <eos>\n   =  第一个 assistant 回答
# （如果有第二轮）assistant\n   -->   <eos>\n   =  第二个 assistant 回答
```

**Step 3：生成 labels——非 assistant 区域填 -100**

```python
def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)    # 默认全部屏蔽
    i = 0
    while i < len(input_ids):
        # 找到 assistant 开始标记？
        if input_ids[i:i + len(self.bos_id)] == self.bos_id:
            start = i + len(self.bos_id)          # 真正回答的起点
            end = start
            # 找到回答结束标记
            while end < len(input_ids):
                if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                    break
                end += 1
            # 只给 assistant 回答区域填真实 token
            for j in range(start, min(end + len(self.eos_id), self.max_length)):
                labels[j] = input_ids[j]
            i = end + len(self.eos_id)
        else:
            i += 1
    return labels
```

### 直观理解

```
完整 token 序列:    [BOS] [助] [手] [用] [户] [你] [好] [助] [手] [P] [y] [t] [h] [o] [n] [EOS]
                     ^^^^^^^^  user 部分  ^^^^^^^^   assistant 部分（要学的）

labels 应该变成:    [-100][-100][-100][-100][-100][-100][-100][-100][P][y][t][h][o][n][-100]
                    ^^^^^^^^^^^^ 不学（填-100） ^^^^^^^^^^^^^^  要学（填真实token）^^^^^^
```

模型在计算 loss 时，cross_entropy 会自动跳过所有 -100 的位置。

> **为什么不是 "只提取 assistant 部分来训练"？** 因为模型需要看到完整的上下文（包括 user 说了什么），才能学会 "根据 user 输入来生成 assistant 输出"。如果只喂 assistant 的回答，模型就不知道用户问了什么。

---

## 四、SFTDataset 完整生命周期

```
JSONL 文件
  {"conversations": [{"role": "user", ...}, {"role": "assistant", ...}]}
        |
        v
  +-----------------------+
  |  SFTDataset.__init__   |
  |  - 加载 JSONL 数据     |
  |  - 记录 bos_id/eos_id  |
  |    （assistant标记）   |
  +-----------------------+
        |
        v
  +-----------------------+
  |  SFTDataset.__getitem__|    <-- 每次 DataLoader 取一条
  |                        |
  |  1. pre_processing_chat|
  |     概率性插入 system  |     <-- 20%的概率注入 system prompt
  |     prompt              |         让模型适应 system 指令格式
  |                        |
  |  2. create_chat_prompt |
  |     apply_chat_template |     <-- HF tokenizer 把多轮对话拼成字符串
  |                        |
  |  3. post_processing_chat|
  |     清理< think >标签   |     <-- 80%概率删除空思考块
  |                        |
  |  4. 分词 + pad 到固定长度    |
  |                        |
  |  5. generate_labels    |
  |     找assistant区间      |     <-- 核心：非assistant区域填-100
  |     填真实token或-100  |
  |                        |
  +-----------------------+
        |
        v
  return (input_ids, labels)    送到模型进行训练
        input_ids: 完整的对话 token 序列
        labels:    只有 assistant 部分有真实 token，其余为 -100
```

---

## 五、完整训练流程（9 步）

```
┌─────────────────────────────────────────────────────────┐
│  Step 1: 初始化环境和随机种子                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │ init_distributed_mode()  ← 多GPU/DDP初始化        │   │
│  │ setup_seed(42 + rank)    ← 每GPU用不同种子        │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                                │
│  Step 2: 配置、模型参数、检查checkpoint                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ lm_config = MiniMindConfig(hidden=768, layers=8)│   │
│  │ ckp = lm_checkpoint(...)  ← 如果是续训，加载状态  │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                                │
│  Step 3: 设置混合精度                                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │ autocast_ctx = torch.cuda.amp.autocast(BF16)    │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                                │
│  Step 4: 配置 wandb/swanlab 日志                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ wandb.init(project="MiniMind-Full-SFT", ...)     │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                                │
│  Step 5: 定义模型、数据、优化器                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │ model, tokenizer = init_model(config, 'pretrain') │   │  ← 加载预训练权重
│  │ train_ds = SFTDataset(path, tokenizer, max_len=768)│   │  ← JSONL 格式
│  │ optimizer = AdamW(lr=1e-5)                       │   │  ← 小学习率
│  └─────────────────────────────────────────────────┘   │
│                         ↓                                │
│  Step 6: 从 checkpoint 恢复状态                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │ model.load_state_dict(ckp)                      │   │
│  │ optimizer.load_state_dict(ckp)                  │   │
│  │ scaler.load_state_dict(ckp)                     │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                                │
│  Step 7: 编译和分布式包装                                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │ model = torch.compile(model)  （可选）           │   │
│  │ model = DistributedDataParallel(model)  （多GPU） │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                                │
│  Step 8: 开始训练（epoch × step 循环）                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │  for epoch in range(epochs):                     │   │
│  │    │                                           │   │
│  │    for step, (input_ids, labels) in loader:    │   │
│  │      │                                         │   │
│  │      ├─ 计算学习率 (cosine 衰减, 1e-5 → 1e-6)   │   │
│  │      │                                         │   │
│  │      ├─ 前向传播: loss = model(input, labels)   │   │
│  │      │   - 用 cross_entropy 算交叉熵             │   │
│  │      │   - labels 中 -100 的位置自动跳过         │   │
│  │      │   - 如果是Moe模型，加上 aux_loss          │   │
│  │      │                                         │   │
│  │      ├─ loss / accumulation_step               │   │
│  │      │                                         │   │
│  │      ├─ loss.backward()（累积梯度）             │   │
│  │      │                                         │   │
│  │      ├─ 每 accumulation_step 步:                │   │
│  │      │   ├─ 梯度裁剪 (max_norm=1.0)             │   │
│  │      │   ├─ 优化器 step                         │   │
│  │      │   └─ 清零梯度                            │   │
│  │      │                                         │   │
│  │      ├─ 每 100 步: 打印日志（loss、学习率）     │   │
│  │      │                                         │   │
│  │      └─ 每 1000 步: 保存 checkpoint             │   │
│  │                                                 │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                                │
│  Step 9: 清理分布式进程                                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ dist.destroy_process_group()                    │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 六、训练循环（train_epoch）详解

### train_epoch 只做了三件事

```
训练循环（train_epoch）
     │
     ├── 1. 前向传播 + 求 loss
     │   input_ids  -->  model(input_ids, labels) -->  loss
     │
     ├── 2. 反向传播 + 优化器步进
     │   scaler.scale(loss).backward()
     │   if step % accumulation_steps == 0:
     │       clip_grad_norm(1.0)  ← 梯度裁剪
     │       scaler.step(optimizer)
     │       scaler.update()
     │       optimizer.zero_grad()
     │
     └── 3. 记录和保存
         每 100 步   --> 打印/记录日志
         每 1000 步  --> 保存模型权重和 checkpoint
```

### 为什么每一步的 loss 要除以 accumulation_steps？

```python
loss = loss / args.accumulation_steps
```

**直观理解**：

```
普通训练:   loss1 --> backward() --> optimizer.step() --> optimizer.zero_grad()

梯度累积:   loss1/acc --> backward()
            loss2/acc --> backward()
            loss3/acc --> backward()
            ...
            loss_acc --> optimizer.step() --> optimizer.zero_grad()

等价于: (loss1 + loss2 + ... + loss_acc) / acc 的平均 loss
```

除以 `accumulation_steps` 让每个小 step 贡献等量的梯度，这样累积后的总梯度等于这些 step 的平均值。

### 梯度裁剪 —— 防止训练"脱轨"

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 1.0
```

**为什么要做这个？**

想象你在走一条山路。如果坡度太陡（梯度太大），你一步可能跨过头、掉下悬崖。梯度裁剪就是把每一步的最大步长限制在 1.0 以内，防止训练不稳定。

### 学习率是怎么变化的？

```python
lr = get_lr(global_step, total_steps, base_lr=1e-5)
```

余弦曲线：

```
学习率
1e-5 ┤*  ← 开始
     │ *
     │  *
5e-6 ┤   *    ← 中间
     │    * *
1e-6 ┤      **  ← 结束
     └───────────
      步  0    总步数
```

公式：`lr = base_lr × (0.1 + 0.45 × (1 + cos(π × step / total)))`

- 最开始时是基础学习率的 100%（1e-5）
- 训练结束时衰减到 10%（1e-6）

> 为什么不是从 0 开始慢慢升？因为这里 SFT 已经是微调阶段了，模型已经有了较好的初始权重，不需要预热（warmup）。直接按最大的学习率开始，慢慢减小。

---

## 七、SFT 特有的数据处理细节

### pre_processing_chat —— 概率注入 System Prompt

```python
if random.random() < add_system_ratio:  # 20% 的概率
    return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
```

**为什么？** 让模型在 SFT 阶段也学习理解和遵循 system 指令。在数据中随机插入 system prompt，模拟了用户可能给模型设定行为准则的场景。

### post_processing_chat —— 清理空思考标签

```python
if '<think>\n\n</think>\n\n' in prompt_content and random.random() > 0.8:
    prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
```

**为什么？** 有些数据中包含 model 的 "思考" (`<think>` 标签)，但思考内容是空的。80% 的概率删除这些无意义的空思考块，让模型学不到无用的模式。

---

## 八、运行示例

### 基本运行

```bash
# 预训练 --> SFT（最常见的用法）
python -m trainer.train_full_sft \
    --from_weight pretrain \
    --data_path ../dataset/sft_t2t_mini.jsonl \
    --epochs 2 --batch_size 16 --learning_rate 1e-5
```

### 分布式多 GPU

```bash
torchrun --nproc_per_node=4 -m trainer.train_full_sft \
    --from_weight pretrain \
    --epochs 2
```

### 使用 MoE 架构

```bash
python -m trainer.train_full_sft --use_moe 1
```

### 恢复训练

```bash
python -m trainer.train_full_sft --from_resume 1 --from_weight full_sft
```

---

## 九、从 SFT 到下一步

```
预训练 (64M)        SFT (64M)         LoRA (少量参数)       DPO/PPO/GRPO
     │                 │                   │                  │
     │                 │                   │                  │
  学语言建模          学会对话格式          领域适配            人类偏好对齐
  猜下一个词          听懂指令             小改动              优化输出质量
     │                 │                   │                  │
     v                 v                   v                  v
  pretrain.pth --> full_sft.pth --> lora_adapter.pth --> aligned_model.pth
```

**SFT 是核心中间阶段**。没有 SFT，预训练模型不会"听话"；有了好的 SFT，后续的对齐（DPO/PPO）和领域适应（LoRA）才有好的起点。
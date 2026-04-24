# train_pretrain.py 预训练完全指南

> 文件路径: `trainer/train_pretrain.py`（约 170 行）
> 预备知识: 读过 `model_minimind_analysis.md`、了解语言模型在 "猜下一个词" 即可

---

> **阅读提示**: 和模型指南一样，遵循 "先懂概念 → 再看代码" 的顺序。困惑时就回看概念部分。

---

## 先回答一个问题：什么是预训练？

打个比方。想象你要教一个刚出生的婴儿说话。

你不会一开始就教他 "请帮我写一封邮件"（那是后面的 **微调/对话训练** 要做的事）。你做的是：**给他看大量的书、文章、网页，让他一个字一个字地读，然后猜下一个字是什么。**

读得越多，他掌握的语法、常识、知识就越多。这就是**预训练**。

```
婴儿阶段的 MiniMind
====================

刚出生: 随机权重 → "天___" → 可能猜出 "车"、"猫"、"+-!@"
          （谁说的词？完全瞎猜）

读了 100 万字后:
          "天___" → "空" ✓
          "苹果是一种水___" → "果" ✓
          "因为下雨，所以我带了___" → "伞" ✓
```

**预训练 = 自监督学习 = 让模型通过 "猜下一个词" 学会语言规律和世界知识。**

为什么叫 "自监督"？因为我们不需要人手工标注答案。给定一句话 "我喜欢吃苹果"，监督信号自动从文本本身产生：

```
文本:  我 | 喜 | 欢 | 吃 | 苹 | 果 | 。
       ↓   ↓   ↓   ↓   ↓   ↓   ↓
问题:  我__?  喜欢__?  欢__?  ...
答案:   喜    欢     吃     苹    果    。
```

每一个字本身就是上下一个字的 "正确答案"。数据自己标注了自己。

---

## 数据长什么样？

预训练数据是一个 JSONL 文件（每行一条 JSON），每行只有两个字段：

```jsonl
{"text": "天文学是研究宇宙天体的科学"}
{"text": "Python 是一种高级编程语言"}
{"text": "今天的天气真不错"}
```

这就是全部。不需要问题-答案对，不需要标签。纯文本，够了。

---

## 从文本到训练样本：PretrainDataset 做了什么

数据文件 → 模型输入，这中间需要几个步骤。来看 `PretrainDataset` 的核心代码：

```python
def __getitem__(self, index):
    sample = self.samples[index]
    tokens = self.tokenizer(str(sample['text']),
        add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
    tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
    input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[input_ids == self.tokenizer.pad_token_id] = -100
    return input_ids, labels
```

一步一步拆解。以句子 "今天天气真好" 为例。

### Step 1：分词（Tokenizer）

```python
tokens = tokenizer("今天天气真好").input_ids
# 假设得到: [2, 101, 83, 42]（每个数字对应一个字/词）
```

### Step 2：加上开始和结束标记（BOS / EOS）

```
BOS = "开始" 标记（告诉模型：一段新文本开始了）
EOS = "结束" 标记（告诉模型：这段话说完了）

tokens = [BOS] + [2, 101, 83, 42] + [EOS]
# = [1, 2, 101, 83, 42, 2]    （假设 BOS=1, EOS=2）
```

**为什么加 BOS/EOS？** 就像文章段落开头空两格、末尾打句号。这些特殊标记让模型理解文本的边界。

### Step 3：Padding（补齐到统一长度）

深度学习框架要求一个 batch 内所有数据长度一致。如果最大长度 `max_seq_len = 340`：

```
短句子:  [1, 2, 101, 83, 42, 2]            ← 只有 6 个 token
补齐后:  [1, 2, 101, 83, 42, 2, 0, 0, ..., 0]  ← 补齐到 340 个，0 是 PAD 标记
```

### Step 4：-100 掩码

```python
labels = input_ids.clone()
labels[input_ids == tokenizer.pad_token_id] = -100
```

这是**关键的一步**。Padding 位置的 `0` 变成了 `-100`。为什么？

```
训练时，模型在算 loss（交叉熵）：

input_ids:  [1, 2, 101, 83, 42, 2, 0, 0, ...]
labels:     [1, 2, 101, 83, 42, 2, -100, -100, ...]

当 labels = -100 时，cross_entropy 会 直 接 跳 过 这 个 位 置！
```

**直觉**：你不想让模型学 "在 PAD 后面应该输出 PAD"。那些补齐的空位没有实际含义，在计算损失时必须忽略。`-100` 就是 PyTorch 内置的 "忽略此位置" 标记。

最终每个样本的样子：

```
┌─────────────────────────────────────────────────┐
│ input_ids: [BOS, 词1, 词2, ..., EOS, PAD, ...]  │  形状: [340]
│ labels:    [BOS, 词1, 词2, ..., EOS, -100, ...] │  形状: [340]
└─────────────────────────────────────────────────┘
```

---

## 核心问题：训练循环到底在做什么？

抛开所有工程细节，预训练的核心只需三步：

```
1. 给模型一句话 → 让它猜下一个词
2. 看它猜对了没有 → 算一个"错误分数"(loss)
3. 根据错误分数调整参数 → 下次猜得更准
```

重复几百万次。这就是全部。

现在来看具体代码是怎么实现这三步的。

---

## 训练循环详解 train_epoch

### 一次迭代的完整流程

```python
for step, (input_ids, labels) in enumerate(loader):

    # ① 设置当前步的学习率
    lr = get_lr(当前总步数, 总步数, 最大学习率)

    # ② 前向传播：把数据喂给模型
    res = model(input_ids, labels=labels)
    loss = res.loss / 8          # 为什么要除以8？后面讲梯度累积

    # ③ 反向传播：算出每个参数该改多少
    loss.backward()

    # ④ 每8步更新一次参数（梯度累积）
    if step % 8 == 0:
        clip_grad(model, max_norm=1.0)   # 剪掉太大的梯度
        optimizer.step()                  # 更新参数
        optimizer.zero_grad()             # 清空梯度
```

**每一步发生的事情：**

```
数据 ──→ [模型前向计算] ──→ logits ──→ [对比真实标签] ──→ loss
                                                    ↓
                                            ← [反向传播算梯度]
                                                    ↓
                                            ← [优化器更新参数]
```

### 梯度累积：用小显存模拟大 batch

```python
loss = loss / 8          # 第 37 行
# backward() 运行 8 次...
# 第 8 次才 optimizer.step()
```

打个比方。你的车一次只能装 32 箱货（`batch_size=32`，GPU 显存限制），但你想要 256 箱的效果（稳定训练）。怎么办？**分 8 趟运，每趟卸下一部分货物堆在仓库里，等 8 趟都到了再统一发货**。

```
第1步: loss/8 → backward → 梯度存放在参数的 .grad 中
第2步: loss/8 → backward → 梯度累加到 .grad
...
第8步: loss/8 → backward → .step()（全部梯度到位，更新参数）
```

效果等同于 `batch_size = 32 × 8 = 256`，但显存占用和 32 一样。

---

## 超参数直觉指南

这些参数在 `argparse` 中定义。每个都配上 "人话" 解释。

| 参数 | 默认值 | 大白话 | 调大/调小的影响 |
|------|--------|--------|-----------------|
| `batch_size` | 32 | 每次给模型看几句话再考试 | 太大 → 显存不够；太小 → 训练不稳定 |
| `epochs` | 2 | 把整本书看几遍 | 越多 → 越拟合，但可能过拟合 |
| `learning_rate` | 5e-4 | 每次改参数的步幅 | 太大 → 震荡；太小 → 学得太慢 |
| `accumulation_steps` | 8 | 积累几步才更新一次 | 增大 = 模拟更大 batch，训练更稳定但更慢 |
| `grad_clip` | 1.0 | 梯度的天花板 | 防止某一步梯度过大导致训练崩溃 |
| `max_seq_len` | 340 | 每句话最多多少个 token | 长句 → 能学更长上下文，但显存占用 O(n²) |
| `hidden_size` | 768 | 每个 token 的向量维度 | 越大越聪明，但参数呈平方增长 |
| `num_hidden_layers` | 8 | Transformer 堆多少层 | 越深越强，但越慢 |
| `log_interval` | 100 | 每隔几步打印日志 | 太小刷屏，太大看不到中间状态 |
| `save_interval` | 1000 | 每隔几步保存检查点 | 频繁保存更安全，但磁盘 I/O 有开销 |
| `num_workers` | 8 | 数据加载线程数 | 太少 → GPU 等数据；太多 → CPU/内存压力大 |

### 学习率是怎么变化的？

```python
lr = lr_max * (0.1 + 0.45 * (1 + cos(pi * step / total_steps)))
```

这叫做 **带底值的余弦衰减**。画出来是这样的：

```
学习率
5e-4 |●  ← 一开始就用最大值
     |   ●
     |      ●
     |         ●
2.75e-4|           ●  ← 训练到一半
     |              ●
     |                 ●
     |                    ●
0.5e-4|                       ●● ← 最后停在最大值的10%
      +----------------------------→ 训练进度
      开始                          结束
```

为什么不用 0 当底值？因为训练末尾学习率完全归零的话，模型可能错过最后的微调机会。保持一点点学习率让参数还能小幅度调整。

---

## 日志输出：你在终端看到了什么？

每隔 `log_interval` 步（默认100步），你会看到类似这样的输出：

```
Epoch:[1/2](100/1000), loss: 8.2341, logits_loss: 8.1200, aux_loss: 0.1141, lr: 0.00048500, epoch_time: 15.2min
```

每个字段的含义：

| 字段 | 含义 | 正常范围 |
|------|------|----------|
| `loss` | 总损失（下一个词预测有多错） | 预训练初期 8~10，后期逐渐下降 |
| `logits_loss` | 纯交叉熵损失 | 和 loss 接近（不用 MoE 时基本相等） |
| `aux_loss` | MoE 负载均衡损失 | 不用 MoE 时 = 0 |
| `lr` | 当前学习率 | 从 5e-4 慢慢降到 5e-5 |
| `epoch_time` | 预计本轮还剩多少分钟 | 随训练推进越来越准 |

**如何判断训练是否正常？**

- loss 应该在缓慢下降（不会直线下降，会波动）
- 如果 loss 突然飙到 20+ → 学习率可能太大
- 如果 loss 一直不动 → 学习率可能太小或数据有问题

---

## 主函数：训练前的准备

打开 `train_pretrain.py` 的 `__main__` 部分，你会看到一条清晰的初始化链。用颜色标注重要程度，我们一次看完。

### 轻量级启动（不用管的部分）

```python
# Step 1: 初始化分布式环境（单机单卡时会直接跳过）
local_rank = init_distributed_mode()

# Step 2: 创建保存目录
os.makedirs(args.save_dir, exist_ok=True)
```

`init_distributed_mode()` 检查是否有 `RANK` 环境变量。如果你只是用 `python train_pretrain.py` 跑单卡训练，它会立刻返回 0，完全跳过分布式初始化。

### 核心配置（需要理解的部分）

```python
# 创建模型配置
lm_config = MiniMindConfig(
    hidden_size=args.hidden_size,       # 768: 词向量维度
    num_hidden_layers=args.num_hidden_layers,  # 8: 层数
    use_moe=bool(args.use_moe)          # 是否用 MoE 架构
)

# 混合精度设置（默认用 bfloat16）
dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

# 创建模型、tokenizer
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

# 创建数据集
train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

# 创建优化器
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
```

这里的 `AdamW` 和 `bfloat16` 值得多说两句。

**AdamW**：当前 LLM 训练的标准优化器。它会自动调整每个参数的学习率——经常出现的特征学慢点，不常见的特征学快点。

**bfloat16**：比 float32 省一半时间和显存，同时比 float16 稳定得多。3090/A100 等显卡原生支持。

### 训练循环开始

```python
for epoch in range(start_epoch, args.epochs):
    # 每轮换新随机种子 → 数据顺序打乱
    setup_seed(42 + epoch)
    indices = torch.randperm(len(train_ds)).tolist()

    # 创建数据加载器
    batch_sampler = SkipBatchSampler(indices, args.batch_size, skip=0)
    loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                        num_workers=8, pin_memory=True)

    # 进入训练！
    train_epoch(epoch, loader, len(loader), 0)
```

**`SkipBatchSampler` 是什么？** 当你是续训（训练中断过）时，它帮你跳过已经训练过的 batch，从断点继续。正常从头训练时它的 skip=0，等于不用跳。

**`num_workers=8, pin_memory=True`**：8 个线程在前台准备下一批数据，GPU 计算的同时 CPU 不闲着。`pin_memory` 让 CPU→GPU 的数据传输走专用通道，更快。

---

## 续训：中断了怎么办？

如果训练中途停了（断电、报错、想换机器），你可以无缝恢复：

```bash
# 加上这个参数会自动检测 checkpoint 并续训
python -m trainer.train_pretrain --from_resume 1
```

续训恢复的东西比你想的多：

```
┌─────────────────────────────────────────────┐
│ 续训时恢复的完整状态                          │
│                                             │
│  1. 模型权重（参数）                           │
│  2. Adam 优化器状态（一阶/二阶动量）            │
│  3. 学习率调度器当前进度                        │
│  4. 当前 epoch 和 step                        │
│  5. GradScaler 状态（如果用的 float16）        │
│                                             │
│ 全 部 恢 复 = 跟 中 断 那 一 秒 完 全 一 样     │
└─────────────────────────────────────────────┘
```

脚本同时保存两类文件：

| 文件 | 内容 | 用途 |
|------|------|------|
| `pretrain_768.pth` | 仅模型权重（转成 float16） | 推理、后续微调 |
| `pretrain_768_resume.pth` | 模型+优化器+全部训练状态 | 续训 |

---

## 一张图看懂完整训练流程

```
┌──────────────────────────────────────────────────────────┐
│                    预训练完整流程图                          │
│                                                          │
│  JSONL 数据                                               │
│  {"text": "..."}  ← 纯文本，每行一段                       │
│       ↓                                                  │
│  PretrainDataset                                         │
│  │ 1. tokenizer 切词                                      │
│  │ 2. 加 BOS 和 EOS                                       │
│  │ 3. padding 补齐到 max_seq_len                          │
│  │ 4. padding 位置的 labels 设为 -100                      │
│       ↓                                                  │
│  DataLoader (batch_size=32, 8个worker)                    │
│       ↓                                                  │
│  ┌─────────────────────────────────────┐                 │
│  │  每一个 step:                        │                 │
│  │                                    │                 │
│  │  ① 计算当前学习率（余弦衰减）           │                 │
│  │  ② 前向: model(input_ids) → logits   │                 │
│  │  ③ 交叉熵: logits vs labels          │                 │
│  │  ④ 反向: loss.backward()             │                 │
│  │  ⑤ 每8步: clip → step → zero_grad    │                 │
│  │                                    │                 │
│  │  每100步: 打印日志                    │                 │
│  │  每1000步: 保存checkpoint             │                 │
│  └─────────────────────────────────────┘                 │
│       ↓ 循环 2 个 epoch                                  │
│  最终模型: pretrain_768.pth                               │
└──────────────────────────────────────────────────────────┘
```

---

## 常见问题

### Q1: loss 为什么一开始那么高（8~10）？

交叉熵 loss = -log(正确答案的概率)。随机初始化的模型对 6400 个词基本均匀猜，每个词概率约 1/6400。

```
-loss ≈ -log(1/6400) ≈ 8.76
```

所以 8~10 是完全正常的起始值。随着训练推进，loss 会慢慢下降。

### Q2: batch_size=32，但有效 batch 是 256？

对。因为 `accumulation_steps=8`，每 8 步才更新一次参数。等效效果 = 每 "伪步" 看了 32 × 8 = 256 条数据。

### Q3: 为什么最大序列长度是 340 这么奇怪的数字？

中文里 1 个 token 大约对应 1.5~1.7 个字符。340 个 token 大约是 500~600 个汉字，够覆盖大多数短句和段落。数字大了显存压力 O(n²) 增长。

### Q4: 可以只训练 1 个 epoch 吗？

可以。默认 2 个 epoch 是为了让模型多看一遍数据。但数据量很大时（比如 10GB JSONL），1 个 epoch 就够了，2 个过拟合。

### Q5: 预训练完的模型能直接聊天吗？

不能。预训练出来的 MiniMind 会续写、会补全句子，但不会遵守指令。它需要后续的**微调（SFT）**阶段来学习对话格式。流程是：

```
预训练（学语言规律和知识） → SFT微调（学对话格式） → 能用 ✓
```

---

## 附录：进阶内容（初学者可以先跳过）

以下内容涉及工程优化细节。第一次阅读可以跳过，等你理解了上面的概念再回来。

### A. 多 GPU 分布式训练（DDP）

DDP（Distributed Data Parallel）让多张显卡各看一部分数据，各自算梯度后同步。代码用 3 行搞定：

```python
model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
model = DistributedDataParallel(model, device_ids=[local_rank])
```

`freqs_cos` 和 `freqs_sin` 是预计算的位置编码表，不需要 GPU 间同步，排除它们能省通信开销。

启动命令：

```bash
# 4 张 GPU 并行训练
torchrun --nproc_per_node=4 -m trainer.train_pretrain
```

每张卡只看数据集的 1/4，各自独立计算，每步后通过 AllReduce 同步梯度。

### B. 混合精度训练（AMP）

```python
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    res = model(input_ids, labels=labels)

scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
scaler.scale(loss).backward()
```

bfloat16 用 16 位存储（float32 是 32 位），速度和显存直接减半。GradScaler 只在 float16 时启用（bfloat16 动态范围大，不需要缩放保护）。

### C. Checkpoint 原子保存

```python
torch.save(state_dict, ckp_tmp)      # 先写到临时文件
os.replace(ckp_tmp, ckp_path)        # 原子替换
```

如果写到一半被 Ctrl+C 中断，`.tmp` 文件损坏不影响原来的 checkpoint。`os.replace` 是原子操作——要么完全成功、要么不生效。

### D. torch.compile 加速

```python
model = torch.compile(model)   # PyTorch 2.0+ 功能
```

通过 JIT 编译把 Python 计算图转成优化的 Triton 内核，通常加速 20-50%。但首次运行有编译开销。

### E. 模型权重解包

保存时需要拿到原始模型的 state_dict，但 `model` 可能被三层包装：

```python
raw_model = model.module                        # 去 DDP 包装
raw_model = getattr(raw_model, '_orig_mod', raw_model)  # 去 torch.compile 包装
state_dict = raw_model.state_dict()
```

保存时 `v.half().cpu()` 把权重转为 float16 并移到 CPU，省磁盘空间且不占 GPU 显存。

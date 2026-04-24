# eval_llm.py 深度解析——模型推理与评估入口

> 先懂概念，再看代码。本文从整体流程出发，逐步深入每个细节。

---

> 文件路径: `eval_llm.py`
> 代码总量: 约 94 行
> 预备知识: PyTorch 推理基础、Transformers tokenizer、模型生成（generate）
> 阅读建议: 这是使用 MiniMind 模型的"第一站"——训练完之后，这个脚本负责让模型"开口说话"。建议先了解 `model_minimind.py` 的模型结构，再来看这个推理入口。

---

## 目录

1. [这个脚本做什么——全景图](#1-这个脚本做什么全景图)
2. [模型加载：五种权重都能用](#2-模型加载五种权重都能用)
3. [两种使用模式：自动测试 vs 交互对话](#3-两种使用模式自动测试-vs-交互对话)
4. [对话与生成流程](#4-对话与生成流程)
5. [命令行参数一览](#5-命令行参数一览)
6. [实用技巧与注意事项](#6-实用技巧与注意事项)

---

## 1. 这个脚本做什么——全景图

在 MiniMind 的训练流水线中，`eval_llm.py` 处于最下游——它是**训练产出物的验收窗口**：

```
预训练(train_pretrain.py) → SFT(train_full_sft.py) → DPO/PPO/GRPO → eval_llm.py 验收
                                                                         ↑
                                                              训练完了，来这里"面试"模型
```

具体功能：

1. **加载模型权重**——支持原生 PyTorch `.pth`、HuggingFace Transformers、LoRA 叠加、MoE 架构
2. **自动测试**——用 8 个预设中文 prompt 逐个测试，观察模型回答质量
3. **交互对话**——手动输入问题，与模型实时聊天
4. **性能统计**——报告每轮生成的 tokens/s 速度

---

## 2. 模型加载：五种权重都能用

### 2.1 加载路径的分支逻辑

```python
def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        # 路径一：原生 PyTorch 权重
        model = MiniMindForCausalLM(MiniMindConfig(...))
        model.load_state_dict(torch.load(ckp))
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, lora_path)
    else:
        # 路径二：HuggingFace Transformers 格式
        model = AutoModelForCausalLM.from_pretrained(args.load_from)
    return model.half().eval().to(args.device), tokenizer
```

判断规则很简单：`load_from` 参数包含 `'model'` 就走原生路径，否则走 Transformers 路径。

### 2.2 原生路径下的权重拼装

```
   ┌─────────────────────────────────────────────────────────┐
   │                  load_from = 'model'                     │
   │                                                         │
   │  ① MiniMindConfig 根据命令行参数构建配置                   │
   │     hidden_size, num_hidden_layers, use_moe, rope_scaling│
   │                                                         │
   │  ② 权重文件名 = {weight}_{hidden_size}[_moe].pth         │
   │     例: out/full_sft_768.pth                             │
   │                                                         │
   │  ③ 可选：叠加 LoRA 权重                                   │
   │     apply_lora(model) → 注入 A/B 矩阵                    │
   │     load_lora(model, lora_path) → 加载权重                │
   └─────────────────────────────────────────────────────────┘
```

不同训练阶段产出的权重通过 `--weight` 参数区分：

| `--weight` 值 | 对应训练阶段 | 典型文件名 |
|---|---|---|
| `pretrain` | 自监督预训练 | `out/pretrain_768.pth` |
| `full_sft` | 监督微调（默认） | `out/full_sft_768.pth` |
| `rlhf` | 偏好对齐 | `out/rlhf_768.pth` |
| `reason` | 推理能力增强 | `out/reason_768.pth` |
| `ppo_actor` | PPO 训练 | `out/ppo_actor_768.pth` |
| `grpo` | GRPO 训练 | `out/grpo_768.pth` |
| `spo` | SPO 训练 | `out/spo_768.pth` |

### 2.3 Transformers 路径

当 `load_from` 指向一个 Transformers 格式目录（如 `minimind-v3` 或 HuggingFace Hub 路径）时，直接调用 `AutoModelForCausalLM.from_pretrained()`，由 Transformers 库自动处理权重加载。

### 2.4 推理优化

```python
return model.half().eval().to(args.device), tokenizer
```

三步准备：
- `.half()` → 转为 FP16 半精度，显存减半，推理加速
- `.eval()` → 关闭 Dropout 等训练专用层
- `.to(device)` → 送到 GPU（如果有的话）

---

## 3. 两种使用模式：自动测试 vs 交互对话

### 3.1 启动时的选择

```python
input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
```

### 3.2 模式 0：自动测试

自动模式使用 8 个预设 prompt，覆盖多个能力维度：

| # | Prompt | 考察能力 |
|---|---|---|
| 1 | "你有什么特长？" | 自我认知 |
| 2 | "为什么天空是蓝色的" | 科学常识 |
| 3 | "请用Python写一个计算斐波那契数列的函数" | 代码生成 |
| 4 | "解释一下'光合作用'的基本过程" | 科学解释 |
| 5 | "如果明天下雨，我应该如何出门" | 条件推理 |
| 6 | "比较一下猫和狗作为宠物的优缺点" | 对比分析 |
| 7 | "解释什么是机器学习" | 概念解释 |
| 8 | "推荐一些中国的美食" | 推荐能力 |

### 3.3 模式 1：交互对话

手动模式下使用 `iter(lambda: input('💬: '), '')` 构造无限输入迭代器，用户输入空行即退出。

---

## 4. 对话与生成流程

### 4.1 多轮对话管理

```python
conversation = conversation[-args.historys:] if args.historys else []
conversation.append({"role": "user", "content": prompt})
```

- `historys=0`（默认）：每轮对话互相独立，不携带历史
- `historys=N`（N 为偶数）：保留最近 N 条消息（N/2 轮 Q&A）

### 4.2 pretrain 模式 vs chat 模式

```python
if 'pretrain' in args.weight:
    inputs = tokenizer.bos_token + prompt          # 裸文本，没有对话模板
else:
    inputs = tokenizer.apply_chat_template(        # 对话模板
        conversation, tokenize=False,
        add_generation_prompt=True,
        open_thinking=bool(args.open_thinking)
    )
```

- **预训练权重**：模型只学过"续写文本"，所以直接用 BOS + 原始 prompt
- **SFT/对齐权重**：模型学过对话格式，需要用 chat template 包装成标准对话

### 4.3 生成调用

```python
generated_ids = model.generate(
    inputs=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=args.max_new_tokens,
    do_sample=True,
    streamer=streamer,          # 流式输出
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    top_p=args.top_p,
    temperature=args.temperature,
    repetition_penalty=1        # 重复惩罚=1 即不惩罚
)
```

关键参数含义：

| 参数 | 作用 |
|---|---|
| `do_sample=True` | 启用采样（而非贪心解码） |
| `top_p` | nucleus 采样——只从累计概率前 p 的 token 中采样 |
| `temperature` | 温度越高输出越随机，越低越确定 |
| `streamer` | `TextStreamer` 实现边生成边打印，用户体验更好 |
| `repetition_penalty=1` | 不额外惩罚重复 token |

### 4.4 速度统计

```python
gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s')
```

用生成前后的 token 数差值除以耗时，直观反映模型在当前硬件上的吞吐量。

### 4.5 完整生成流程图

```
  用户输入 prompt
       │
       ▼
  构建 conversation 列表（含历史）
       │
       ▼
  ┌─── pretrain 权重? ───┐
  │ 是                    │ 否
  ▼                      ▼
  BOS + 原始文本     apply_chat_template()
  │                      │
  └──────┬───────────────┘
         ▼
    tokenizer 编码为 input_ids
         │
         ▼
    model.generate()  ──→  TextStreamer 实时打印
         │
         ▼
    decode → 追加到 conversation
         │
         ▼
    打印 [Speed]: xx tokens/s
```

---

## 5. 命令行参数一览

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--load_from` | `'model'` | 模型加载路径（`'model'` = 原生 torch，其他路径 = Transformers） |
| `--save_dir` | `'out'` | 权重文件所在目录 |
| `--weight` | `'full_sft'` | 权重名称前缀 |
| `--lora_weight` | `'None'` | LoRA 权重名称（`'None'` = 不使用） |
| `--hidden_size` | `768` | 模型隐藏层维度 |
| `--num_hidden_layers` | `8` | Transformer 层数 |
| `--use_moe` | `0` | 是否启用 MoE 架构 |
| `--inference_rope_scaling` | `False` | 启用 YaRN RoPE 外推（4 倍窗口） |
| `--max_new_tokens` | `8192` | 最大生成 token 数 |
| `--temperature` | `0.85` | 采样温度 |
| `--top_p` | `0.95` | nucleus 采样阈值 |
| `--open_thinking` | `0` | 是否开启自适应思考模式 |
| `--historys` | `0` | 携带历史对话轮数（须为偶数） |
| `--show_speed` | `1` | 是否显示生成速度 |
| `--device` | 自动检测 | `cuda` 或 `cpu` |

---

## 6. 实用技巧与注意事项

### 6.1 如何测试不同训练阶段的效果

```bash
# 测试预训练效果（只会续写文本）
python eval_llm.py --weight pretrain

# 测试 SFT 效果（会对话了）
python eval_llm.py --weight full_sft

# 测试 DPO 对齐效果
python eval_llm.py --weight rlhf

# 测试带 LoRA 的效果
python eval_llm.py --weight full_sft --lora_weight lora_identity

# 测试 MoE 架构
python eval_llm.py --weight full_sft --use_moe 1

# 使用 Transformers 格式
python eval_llm.py --load_from ./minimind-v3
```

### 6.2 thinking 模式

当 `--open_thinking 1` 时，chat template 会指示模型在回答前先进行内部推理（`<think>...</think>`），类似 CoT（Chain of Thought）。这有助于提升复杂推理任务的质量，但会增加生成长度。

### 6.3 随机种子

```python
setup_seed(random.randint(0, 31415926))
```

每轮对话使用随机种子，保证可复现性的同时又有一定多样性。如果想完全复现某次输出，可以固定种子值。

### 6.4 关键依赖关系

```
eval_llm.py
├── model.model_minimind  → MiniMindConfig, MiniMindForCausalLM
├── model.model_lora      → apply_lora, load_lora
├── trainer.trainer_utils  → setup_seed, get_model_params
└── transformers           → AutoTokenizer, AutoModelForCausalLM, TextStreamer
```

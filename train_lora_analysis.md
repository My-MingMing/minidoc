# MiniMind LoRA 微调脚本 (train_lora.py) 深度代码分析

> 文件路径: `trainer/train_lora.py`  
> 配套模块: `model/model_lora.py`  
> 代码总量: 约 183 行

---

## 目录

1. [LoRA 原理简介（低秩适配）](#1-lora-原理简介低秩适配)
2. [本脚本做了什么](#2-本脚本做了什么)
3. [apply_lora 函数的作用分析](#3-apply_lora-函数的作用分析)
4. [参数冻结逻辑](#4-参数冻结逻辑)
5. [与 full_sft 的完整参数对比表](#5-与-full_sft-的完整参数对比表)
6. [train_epoch 中的关键差异](#6-train_epoch-中的关键差异)
7. [LoRA 参数量统计方式](#7-lora-参数量统计方式)
8. [使用场景举例](#8-使用场景举例)
9. [完整训练流程图](#9-完整训练流程图)

---

## 1. LoRA 原理简介（低秩适配）

### 1.1 核心思想

LoRA（Low-Rank Adaptation，低秩适配）是微软于 2021 年提出的一种 Parameter-Efficient Fine-Tuning（PEFT，参数高效微调）技术。它的核心洞察是：

> 预训练大模型在下游任务微调时，参数变化量 Delta_W 的内部秩（intrinsic rank）远低于参数的原始维度。

因此，可以用两个低秩矩阵的乘积来近似表示全量参数变化：

```
Delta_W = B x A
```

其中：
- W_0 是原始预训练权重（冻结不动）
- A 在 R^(d x r)、B 在 R^(r x d) 是可训练的低秩矩阵
- r 称为 LoRA Rank（秩），通常取 4、8、16、64 等远小于 d 的值

前向传播时，实际输出为：

```
h = W_0 * x + Delta_W * x = W_0 * x + B * A * x
```

### 1.2 为什么有效

| 角度 | 说明 |
|------|------|
| 理论直觉 | 大模型本身具有高度冗余，微调只需在一个低维子空间内做适配，不需要修改全部参数 |
| 参数压缩 | 原始全参更新需 d^2 个参数；LoRA 仅需 2*d*r 个。当 d=768, r=16 时，压缩比约 24 倍 |
| 训练效率 | 只需在优化器中追踪 LoRA 参数的梯度，显存占用大幅降低 |
| 零推理开销 | 推理时可将 B*A 加回 W_0（merge_lora），部署模型与原模型结构完全一致 |

### 1.3 初始化策略

LoRA 论文建议的初始化方式在 `model/model_lora.py` 中得到了精确复现：

```python
# 矩阵 A：小的随机高斯噪声，std=0.02
self.A.weight.data.normal_(mean=0.0, std=0.02)

# 矩阵 B：全零初始化
self.B.weight.data.zero_()
```

这样初始化的关键意义在于：**训练开始时，LoRA 分支的输出为零，模型的初始行为与原预训练模型完全一致，不会破坏已有知识。**

### 1.4 LoRA Rank 的选择

在本实现中，apply_lora 默认 rank=16：
- 小 rank（4-8）：参数更少、更快，但表达能力有限
- 中 rank（16）：在效率与效果间取得平衡，本项目使用此值
- 大 rank（64+）：更接近全参数微调，但显存和计算开销增大

---

## 2. 本脚本做了什么

`train_lora.py` 是 MiniMind 项目中用于 LoRA 微调的训练入口脚本。它的主要作用链如下：

```
已训练好的 SFT 模型 (full_sft)
          |
          v
   加载模型权重 + Tokenizer
          |
          v
   对每个方形 Linear 层 挂载 LoRA 适配器
          |
          v
   冻结全部原始参数，仅放通 LoRA 参数
          |
          v
   用领域专用数据（如医疗语料）仅训练 LoRA 参数
          |
          v
   保存 LoRA 权重（不含原模型参数）
```

具体来说：

1. **加载基础模型**（第 127 行）：调用 init_model(lm_config, args.from_weight, device) 从已完成的 SFT 权重（full_sft）加载模型和 Tokenizer
2. **挂载 LoRA 适配器**（第 128 行）：调用 apply_lora(model)，在模型的每一个方形 nn.Linear 层上挂接旁路 LoRA 分支
3. **参数冻结**（第 138-144 行）：遍历所有参数，将名中包含 'lora' 的参数设 requires_grad = True，其余全部 requires_grad = False
4. **仅优化 LoRA 参数**（第 150 行）：optim.AdamW(lora_params, lr=1e-4) -- 优化器只接收 LoRA 参数列表
5. **训练与保存**：在 train_epoch 中，梯度裁剪仅作用于 lora_params，checkpoint 保存调用 save_lora(model, path) 仅导出 LoRA 权重

**整体语义**：在不修改原始 19M+ 参数 SFT 模型的前提下，插入极少量（通常 <1M）的可训练旁路参数，使模型获得新领域的专业知识。

---

## 3. apply_lora 函数的作用分析

### 3.1 完整源码（model/model_lora.py 第 21-32 行）

```python
def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora
```

### 3.2 逐步拆解

| 步骤 | 代码行为 | 作用 |
|------|----------|------|
| 遍历所有子模块 | for name, module in model.named_modules() | 递归访问模型中的每一层（Embedding、Linear、Norm、Block 等） |
| 条件筛选 | isinstance(module, nn.Linear) and weight.shape[0] == weight.shape[1] | 只对**方形矩阵**（输入输出维度相同）的 Linear 层挂 LoRA。过滤掉 lm_head（768x6400，非方形）、MoE gate 等特殊层 |
| 创建 LoRA 旁路 | lora = LoRA(in_dim, out_dim, rank=16) | A: 768x16, B: 16x768 |
| 绑定为子属性 | setattr(module, "lora", lora) | 将 LoRA 实例挂到原模块的 .lora 属性上。这使得 named_parameters() 遍历时会自动包含 xxx.lora.A.weight 和 xxx.lora.B.weight |
| 前向改写 | 将 module.forward 替换为 forward_with_lora | **闭包捕获** original_forward 和 lora 到默认参数 layer1 和 layer2 中。使用默认参数而非外部变量引用，避免了 Python 闭包延迟绑定（late binding）导致的 bug -- 这是此实现的一个亮点 |

### 3.3 前向传播改造前后的对比

```
改造前（原始 Linear 层）:
          x
          |
     [----v----]
     |   W_0   |  <-- 原始权重，768x768
     [----|----]
          |
          v
         h = W_0 . x

改造后（挂载 LoRA）:
          x
          |
     [----v---------]
     |     W_0      |
     [----|----]    |
          |         |
          v         v
     W_0 . x  [B--A]  <-- A: 768x16, B: 16x768
          |    [|-|]
          |     |
          [----v--]
                |
          h = W_0 . x + B.A.x
```

### 3.4 挂载覆盖范围

以 MiniMind 默认配置（hidden_size=768, num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4）为例：

| 模块 | 维度 | 方形? | 是否挂载 LoRA |
|------|------|-------|--------------|
| q_proj | 768 -> 768 | 是 | **挂载** |
| k_proj | 768 -> 256 | 否 | 不挂载 |
| v_proj | 768 -> 256 | 否 | 不挂载 |
| o_proj | 768 -> 768 | 是 | **挂载** |
| gate_proj | 768 -> 2432 | 否 | 不挂载 |
| up_proj | 768 -> 2432 | 否 | 不挂载 |
| down_proj | 2432 -> 768 | 否 | 不挂载 |
| embed_tokens | embedding 层 | 非 Linear | 跳过 |
| lm_head | 768 -> 6400 | 否 | 不挂载 |

因此，**每层实际挂载 2 个 LoRA 分支（q_proj + o_proj），总计 16 个 LoRA 模块**。

> 注：如果 num_key_value_heads 与 num_attention_heads 相等，则 k_proj 也会是方形矩阵。具体挂载数量取决于模型配置。

---

## 4. 参数冻结逻辑

### 4.1 核心代码（第 138-144 行）

```python
lora_params = []
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True       # LoRA 参数：参与梯度计算
        lora_params.append(param)        # 收集到优化器输入列表
    else:
        param.requires_grad = False      # 原始参数：冻结
```

### 4.2 为什么只训练 LoRA 参数

| 原因 | 详细解释 |
|------|----------|
| 显存效率 | 优化器状态（AdamW 中每参数需要 2 个额外状态：一阶矩和二阶矩估计）的显存开销与可训练参数量成正比。只优化 LoRA 参数意味着优化器状态从 ~19M 降到 ~0.5M，节省约 38 倍的优化器显存 |
| 计算效率 | 反向传播中，冻结参数的梯度计算可以被 PyTorch 跳过（autograd 引擎检测到 requires_grad=False 的子图），减少约 95% 以上的梯度计算量 |
| 防灾难性遗忘 | 原始 SFT 模型已经学习到了通用的对话与指令跟随能力，修改其权重会导致已有能力退化。冻结原参数确保"旧知识"不被破坏 |
| 即插即用 | LoRA 权重是独立的旁路模块，不同领域的 LoRA（如医疗、法律）可以在同一基础模型上自由切换，无需训练多个完整副本 |
| 存储效率 | LoRA 权重文件通常只有几百 KB 到几 MB，而完整模型权重是数十 MB。便于分发和部署 |

### 4.3 优化器行为

第 150 行：

```python
optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
```

优化器 AdamW 的 params 参数仅接收 lora_params 列表。这意味着：
- **step() 调用时**，只有 LoRA 参数会被更新
- 即使原模型参数的 grad 字段不为 None（由于闭包在前向中被引用），它们也不会被优化器触及
- 由于 requires_grad=False，这些"冻结参数"在反向传播中根本不会计算梯度，因此 grad 通常为 None

---

## 5. 与 full_sft 的完整参数对比表

以下是 `train_lora.py` 与 `train_full_sft.py` 在关键维度上的逐项对比：

| 对比维度 | train_lora.py | train_full_sft.py | 说明 |
|----------|---------------|-------------------|------|
| 脚本功能 | LoRA 高效微调 | 全参数指令微调 | |
| 模型来源 from_weight | 默认 full_sft | 默认 pretrain | LoRA 在 SFT 之上做领域适配 |
| LoRA 挂载 | 调用 apply_lora(model) | 不挂载 | |
| 参数策略 | 冻结全部原参数，仅训练 LoRA | 全部参数可训练 | |
| train_epoch 签名 | (epoch, loader, iters, **lora_params**, start_step, wandb) | (epoch, loader, iters, start_step, wandb) | 多传一个 lora_params 参数 |
| 梯度裁剪目标 | clip_grad_norm_**(lora_params**, ...) | clip_grad_norm_**(model.parameters()**, ...) | LoRA 仅裁剪 LoRA 自身参数梯度 |
| 模型保存方式 | save_lora(model, path) | torch.save(state_dict, ckp) | LoRA 仅保存 LoRA 旁路权重，文件极小 |
| 恢复方式 strict | strict=False | strict=默认 True | 容忍部分差异 |
| 默认 epochs | **10** | **2** | LoRA 需要更多轮收敛 |
| 默认 batch_size | **32** | **16** | 参数量少、显存充裕，可加大 batch |
| 默认 learning_rate | **1e-4** | **1e-5** | LoRA 学习率通常比全参高 10 倍 |
| 默认 max_seq_len | **340** | **768** | 领域适配通常不需要很长的上下文 |
| 默认 data_path | lora_medical.jsonl | sft_t2t_mini.jsonl | 专用数据 vs 通用指令数据 |
| 默认 log_interval | **10** | **100** | 更频繁地监控训练动态 |
| checkpoint 前缀 | args.lora_name | args.save_weight | LoRA 使用自定义名称前缀 |

---

## 6. train_epoch 中的关键差异

### 6.1 额外参数：lora_params

train_lora.py 的 train_epoch 函数签名相比 full_sft 多了一个 lora_params 参数：

```python
# train_lora.py
def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):

# train_full_sft.py
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
```

### 6.2 差异一：梯度裁剪仅作用于 LoRA 参数

```python
# train_lora.py 第 44 行
torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)

# train_full_sft.py 第 43 行
torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
```

**原理说明**：
- 梯度裁剪用于防止训练过程中梯度爆炸
- 在全参微调中，需要裁剪所有参数的梯度
- 在 LoRA 微调中，由于只有 LoRA 参数会计算 gradient，裁剪 lora_params 是唯一有意义的操作
- 对已冻结的参数调用 clip_grad_norm_ 会无效或报错，因为其梯度为 None

同样的差异也出现在 train_epoch 末尾的"剩余梯度处理块"（第 69-74 行），那里同样使用 lora_params 而非 model.parameters()。

### 6.3 差异二：模型保存仅存 LoRA 权重

```python
# train_lora.py 第 59-63 行
if (step % args.save_interval == 0 or step == iters) and is_main_process():
    model.eval()
    lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}.pth'
    # LoRA只保存LoRA权重
    save_lora(model, lora_save_path)
    lm_checkpoint(...)  # 同时保存完整 checkpoint
    model.train()
```

save_lora 函数（model/model_lora.py 第 45-53 行）的行为分解：

```python
def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model)  # 兼容 torch.compile 包装
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name  # 去 DDP 前缀
            lora_state = {f'{clean_name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)
```

关键细节：
1. **DDP 兼容**：name.startswith("module.") 去掉 DistributedDataParallel 包装的前缀
2. **torch.compile 兼容**：getattr(model, '_orig_mod', model) 获取编译前的原始模型
3. **仅保存 LoRA 子模块**：通过 hasattr(module, 'lora') 过滤
4. **half 精度存储**：v.cpu().half() 以 float16 格式保存到 CPU，节省磁盘空间
5. **文件命名**：格式为 {lora_name}_{hidden_size}.pth，如 lora_medical_768.pth

---

## 7. LoRA 参数量统计方式

### 7.1 统计代码（第 131-135 行）

```python
# 模型总参数（原始参数 + LoRA 参数）
total_params = sum(p.numel() for p in model.parameters())

# 仅 LoRA 参数（按名称过滤）
lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)

Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")
```

### 7.2 LoRA 参数量手动估算

以 MiniMind 默认配置为例：

| 参数 | 数值 |
|------|------|
| hidden_size (d) | 768 |
| LoRA Rank (r) | 16 |
| 每个 LoRA 分支参数量 | d*r + r*d = 768*16 + 16*768 = 24,576 |

挂载 16 个 LoRA 分支（每层 2 个 x 8 层）：

```
LoRA 总参数量 = 16 * 24,576 = 393,216 ≈ 0.393 M
```

对比 MiniMind (hidden_size=768, 8 层) 总参数量约 19M：

```
LoRA 参数占比 ≈ 0.393 / 19 * 100% ≈ 2.07%
```

这意味着用约 2% 的可训练参数量，即可以完成领域适配。

---

## 8. 使用场景举例

### 8.1 医疗领域适配 (lora_medical)

训练前，SFT 模型知道如何对话，但缺乏医学专业知识：

```
用户: 我头痛、发热、喉咙痛，可能是什么病？
SFT模型: 建议你多休息，多喝水。如果症状持续，请咨询医生。
```

挂接 lora_medical 后：

```
用户: 我头痛、发热、喉咙痛，可能是什么病？
SFT + LoRA_Medical: 这些症状可能提示上呼吸道感染，常见原因包括病毒性咽炎、扁桃体炎等。
建议：1) 注意体温变化；2) 多饮水休息；3) 如出现高热、呼吸困难等症状需及时就医。
请注意，我提供的建议不能替代专业医生的诊断。
```

数据格式：dataset/lora_medical.jsonl 中的对话对，包含医患对话、疾病描述、药品说明等。

### 8.2 身份识别 / 角色扮演 (lora_identity)

让模型模拟特定角色（如面试官、客服代表、教学助手）：

```
python train_lora.py --lora_name lora_interviewer --data_path ../dataset/lora_interview.jsonl --epochs 15 --batch_size 32 --learning_rate 1e-4
```

### 8.3 法律领域适配 (lora_law)

让模型具备法律条文检索、法律咨询等能力。

### 8.4 代码生成适配 (lora_code)

基础模型已有代码能力，但可以针对特定语言（如 Rust、SQL）做针对性强化。

### 8.5 多 LoRA 灵活切换

由于 LoRA 权重是独立的，推理时可以自由切换：

```python
load_lora(model, "lora_medical_768.pth")      # 挂接医疗LoRA
# 推理 -> 模型具备医疗知识

load_lora(model, "lora_interviewer_768.pth")  # 替换为面试LoRA
# 推理 -> 模型具备面试风格
```

甚至可以结合 merge_lora 函数将 LoRA 权重合并回原始模型，得到"固化"后的完整模型。

---

## 9. 完整训练流程图

```
+---------------------------------------------------------------+
|            MiniMind LoRA Fine-Tuning 训练流程                   |
+---------------------------------------------------------------+

1. 初始化阶段
   |-- init_distributed_mode()          分布式进程组初始化
   |-- setup_seed(42 + rank)            固定随机种子
   |-- os.makedirs(save_dir)            创建保存目录
   +-- MiniMindConfig 实例化            模型超参数初始化

2. 检查是否有续训 checkpoint
   +-- lm_checkpoint(from_resume)       如有 ckp 则读取
       |-- 返回: ckp_data (含 model / optimizer / scaler / epoch / step)
       +-- 支持中断恢复训练

3. 混合精度设置
   |-- dtype = bfloat16 | float16
   +-- autocast_ctx = nullcontext | amp.autocast

4. Wandb / SwanLab 日志配置（可选）
   +-- 支持断线续记 (resume='must')

================================================================
5. 模型加载 + LoRA 挂载 + 参数冻结  **** 核心步骤 ****

   5a -- model, tokenizer = init_model(config, 'full_sft', device)
           |-- 从 full_sft 权重加载完整的 SFT 模型和 Tokenizer

   5b -- apply_lora(model, rank=16)
           |-- 遍历所有子模块，对每个方形 Linear 层插入 LoRA 旁路
               |-- LoRA(in, out, rank=16)
                   |-- A: Linear(768, 16), 高斯初始化
                   |-- B: Linear(16, 768), 零初始化
                   |-- 前向: h = W0.x + B(A(x))

   5c -- total_params = sum(p.numel() for p in model.parameters())
           |-- ≈ 19M（基础 SFT 全部参数，含新增 LoRA 模块）

   5d -- lora_params_count = sum(... if 'lora' in name)
           |-- ≈ 0.39M（LoRA 可训练参数）

   5e -- 参数冻结循环:
           |-- 'lora' in name --> requires_grad=True  (可训练)
           |-- 否则           --> requires_grad=False (冻结)
================================================================

6. 数据加载器 & 优化器
   |-- SFTDataset(data_path, tokenizer, max_len=340)
   |-- DistributedSampler（分布式训练场景）
   |-- optimizer = AdamW(lora_params, lr=1e-4)  仅优化LoRA参数
   +-- GradScaler = GradScaler(enabled=float16)

7. 状态恢复（如从 checkpoint 续训）
   |-- model.load_state_dict(ckp['model'], strict=False)
   |-- optimizer.load_state_dict(ckp['optimizer'])
   +-- start_epoch, start_step 从 ckp 读取

8. 编译 & 分布式包装
   |-- torch.compile(model)               (可选)
   +-- DistributedDataParallel(model)     (分布式时)

================================================================
9. 训练循环
================================================================

   for epoch in range(start_epoch, args.epochs):
   |
   |-- train_sampler.set_epoch(epoch)  打乱分布式数据
   |-- setup_seed(42 + epoch)
   |-- SkipBatchSampler --> DataLoader
   |
   +-- train_epoch(epoch, loader, iters, lora_params):
       |
       +-- for each batch (input_ids, labels):
           |
           |-- [1] 前向传播
           |   res = model(input_ids, labels=labels)
           |   loss = (res.loss + res.aux_loss) / accumulation_steps
           |
           |-- [2] 反向传播
           |   scaler.scale(loss).backward()
           |
           |-- [3] 梯度裁剪（当达到累积步数时）
           |   if step % accumulation_steps == 0:
           |       unscaler(optim)
           |       clip_grad_norm_**(lora_params**, grad_clip)  ★仅LoRA
           |       step(optim)
           |       update(scaler)
           |       zero_grad(set_to_none=True)
           |
           |-- [4] 日志输出（每 log_interval=10 步）
           |   Logger(loss, logits_loss, aux_loss, lr, time)
           |   wandb.log(...)
           |
           |-- [5] 保存模型（每 save_interval=1000 步）
           |   save_lora(model, path)  ★仅保存LoRA旁路权重
           |   lm_checkpoint(...)      ★保存完整checkpoint
           |
           +-- [6] 处理最后一个batch的剩余梯度
               (当最后一步不整除积累步数时执行剩余梯度更新)

10. 清理分布式进程
    +-- dist.destroy_process_group()

================================================================
输出产物:
   |-- out/lora_medical_768.pth       LoRA 旁路权重 (~数百KB-MB)
   |-- checkpoints/MiniMind-LoRA/     完整 checkpoint (含优化器状态)
   +-- Wandb / SwanLab 训练曲线        可视化监控面板
================================================================
```

---

## 附录：关键命令行参数一览

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| --lora_name | lora_medical | LoRA 权重文件名的前缀标识 |
| --epochs | 10 | 总训练轮次 |
| --batch_size | 32 | 每批次样本数 |
| --learning_rate | 1e-4 | 初始学习率（比全参高 10 倍） |
| --accumulation_steps | 1 | 梯度累积步数，等效增大 batch |
| --grad_clip | 1.0 | 梯度裁剪的 L2 范数阈值 |
| --log_interval | 10 | 每多少步打印日志 |
| --save_interval | 1000 | 每多少步保存模型 |
| --max_seq_len | 340 | 序列最大截断长度 |
| --from_weight | full_sft | 加载的基础模型权重 |
| --data_path | ../dataset/lora_medical.jsonl | 训练数据路径 |
| --hidden_size | 768 | 隐藏层维度 |
| --num_hidden_layers | 8 | Transformer 层数 |
| --use_moe | 0 | 是否启用 MoE 架构 |
| --use_compile | 0 | 是否使用 torch.compile |
| --from_resume | 0 | 是否自动检测并续训 |

---

*文档生成日期: 2026-04-02*

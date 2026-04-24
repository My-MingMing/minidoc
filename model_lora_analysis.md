# model_lora.py 初学者完全指南

> 文件路径: `model/model_lora.py`
> 代码总量: 约 66 行
> 配套模块: `trainer/train_lora.py`, `model/model_minimind.py`

---

## 在开始之前：为什么需要 LoRA？

想象一下这个场景：你训练了一个语言模型 MiniMind，它有约 26M 个参数。现在你想让它学会写诗。

### 方案 A：全量微调（Full Fine-tuning）

在写诗数据上重新训练**所有** 26M 参数。

**问题**：
- 需要 26M 个参数的优化器状态（Adam 优化器需要保存梯度和动量，实际显存占用是参数量的 4-6 倍）
- 如果你还想让它学会写小说、写代码、翻译……每个任务存一份 26M 的模型副本
- 10 个任务 = 260M 参数 × 10 = **260M 参数**（约 500MB~1GB 磁盘）

### 方案 B：LoRA（低秩适配）

冻结模型的 26M 参数，只在某些层旁边加一个小型的"旁路适配器"（LoRA 模块）。训练时只更新适配器。

**好处**：
- 主模型参数不动（`requires_grad = False`），不占优化器显存
- 适配器很小（rank=16 时约占总参数的 3-5%），显存省 60%+
- 不同任务各存各的适配器，切换任务只需 `load_lora()` 换一套小权重
- 10 个任务 = 26M 参数（1份主模型）+ 0.9M × 10（10份适配器）= **约 35M 参数**

**LoRA 的数学直觉：**

> 全量微调：`W_new = W_old + ΔW`，ΔW 是一个完整的矩阵（d×d）
>
> LoRA：`ΔW = B @ A`，其中 B 是 (d×r)、A 是 (r×d)。当 r 远小于 d 时，参数量 `2×d×r` 远小于 `d×d`
>
> 比如 d=768, r=16：全量 ΔW = 768×768 = 589,824 个参数；LoRA 的 A+B = 2×768×16 = 24,576 个参数。**省了 24 倍！**

---

## 代码逐块解析

### 第一部分：LoRA 类

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # 低秩矩阵的秩，控制"适配器"有多小
        self.A = nn.Linear(in_features, rank, bias=False)  # 矩阵 A: (in × rank)
        self.B = nn.Linear(rank, out_features, bias=False) # 矩阵 B: (rank × out)
        # 矩阵 A 用高斯分布随机初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵 B 用全 0 初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))  # 等价于 x @ A^T @ B^T
```

**A 和 B 的初始化为什么不同？**

这是 LoRA 最核心的设计。让我们追踪一下训练的第一步：

```
初始时:  B = 0  →  B @ A = 0  →  LoRA 输出 = 0

原始输出 + LoRA 输出 = 原始输出 + 0 = 原始输出
```

**这意味着：在微调的第一步，模型的行为完全没有改变！**

训练开始后，B 从零开始，A 有微小的随机扰动。梯度会更新 B 和 A，LoRA 的输出从零逐渐增长。这相当于从一个**已知的、好的起点**开始微调，而不是从一个随机的扰动开始。

> **类比**：微调一个模型就像给一辆已经能开的车改装引擎。LoRA 的零初始化确保改装的第一步——车子还是能正常开。然后你慢慢调优，而不是上来就拆掉整个发动机。

如果 A 和 B 都用随机初始化：`B @ A ≠ 0`，第一步就引入了随机扰动。模型行为瞬间改变，可能导致训练不稳定。

---

### 第二部分：apply_lora —— 动态挂载适配器

```python
def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定（Python 闭包陷阱修复）
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora
```

**这 15 行代码在做一件什么事？**

遍历模型的所有层，找到方阵 Linear 层，给每个层动态挂载一个 LoRA 适配器。**不需要改动原模型的代码。**

**为什么要检查 `weight.shape[0] == weight.shape[1]`（方阵）？**

```
模型中的 Linear 层（默认 GQA 配置: num_attention_heads=8, num_key_value_heads=4）:
├── Attention 层: q_proj(768→768)                          方阵 ✓
│                o_proj(768→768)                          方阵 ✓
│                k_proj(768→384)                          非方阵 ✗（GQA 下 KV head 少）
│                v_proj(768→384)                          非方阵 ✗（GQA 下 KV head 少）
├── FFN 层: gate_proj(768→2432)                           非方阵 ✗
│         up_proj(768→2432)                              非方阵 ✗
│         down_proj(2432→768)                            非方阵 ✗
└── LM Head: lm_head(768→6400)                           非方阵 ✗
```

只有方阵才满足 `in_features == out_features`，意味着输入输出维度一致，适合 LoRA 的低秩分解。在 GQA（分组查询注意力）配置下，`k_proj` 和 `v_proj` 的输出维度是 `num_key_value_heads × head_dim = 4 × 96 = 384`，不等于输入维度 768，因此**不是方阵**，不会被 `apply_lora` 选中。实际只有 `q_proj` 和 `o_proj` 会挂载 LoRA。

**为什么 `forward_with_lora` 要用默认参数？**

这是一个经典的 Python 陷阱。如果不这样做：

```python
# ❌ 错误写法
def forward_with_lora(x):
    return original_forward(x) + lora(x)
# 当循环结束以后，original_forward 和 lora 都指向最后一个模块
```

Python 闭包中的变量是**延迟查找**的。在 for 循环里创建闭包时，函数不会"捕获"当前循环的变量值，而是等到**被调用时**再去查找。结果就是——所有层的 `forward_with_lora` 都指向了最后一个 lora 对象。

```python
# ✅ 正确写法：用默认参数在定义时绑定
def forward_with_lora(x, layer1=original_forward, layer2=lora):
    return layer1(x) + layer2(x)
# 默认参数在函数定义时就取值固定下来了，不会再变
```

**挂载后的效果：**

```
原始 Linear 层                 挂载 LoRA 后
x ─→ original_forward ─→ y    x ─→ original_forward ─→ y1 ─┐
                                                       ─→ y1 + y2  (最终输出)
                                        x ─→ lora.A ─→ lora.B ─→ y2  ─┘
```

训练时，`train_lora.py` 会冻结原始模型参数（`requires_grad = False`），只保留 LoRA 参数的梯度。这样反向传播时只更新 A 和 B。

---

### 第三部分：load_lora —— 加载训练好的 LoRA 权重

```python
def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    # 去掉 DDP 的 "module." 前缀
    state_dict = {(k[7:] if k.startswith('module.') else k): v
                  for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 从全局 state_dict 中提取属于此模块 LoRA 的部分
            lora_state = {k.replace(f'{name}.lora.', ''): v
                         for k, v in state_dict.items()
                         if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)
```

**state_dict 中的 key 长什么样？**

```
model.layers.0.self_attn.q_proj.lora.A.weight
model.layers.0.self_attn.q_proj.lora.B.weight
model.layers.0.self_attn.k_proj.lora.A.weight
model.layers.1.mlp.gate_proj.lora.A.weight
...
```

**加载流程：**

```
1. torch.load('lora_output.pth')   ← 加载全部 LoRA 权重
2. 去掉 'module.' 前缀             ← 兼容分布式训练保存的权重
3. 遍历所有有 lora 的层:
   ├── 从大字典里过滤出属于这层 LoRA 的 key
   ├── 把 key 里的 "model.layers.0.self_attn.q_proj.lora." 去掉
   │   变成 "A.weight" 和 "B.weight"（LoRA 类认识的格式）
   └── module.lora.load_state_dict(lora_state)  ← 加载
```

**为什么要过滤 key？**

因为 LoRA 权重保存在**一个全局的 state_dict** 里（所有 LoRA 模块的权重混在一起），每个 lora 模块只能加载属于自己的一部分。

---

### 第四部分：save_lora —— 只保存适配器的权重

```python
def save_lora(model, path):
    # 处理 torch.compile 的 _orig_mod 包装
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half()
                         for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)
```

**三个值得注意的细节：**

1. **`getattr(model, '_orig_mod', model)`**: 如果模型用 `torch.compile(model)` 编译过，实际的模块结构会被包一层 `_orig_mod`。这行代码解包到原始模型。

2. **`.cpu().half()`**: 把 LoRA 权重转到 CPU 上并转换为 FP16（半精度）。LoRA 参数本来就很少（约 0.9M），转 FP16 后文件更小，对精度影响微乎其微。这是节省磁盘空间的好做法。

3. **只保存 LoRA 部分**: 主模型的 26M 参数不保存——那些权重本来就没变过。这就像只保存了"改装配件"，而不是整个"车"。

---

### 第五部分：merge_lora —— 合并回标准模型

```python
def merge_lora(model, lora_path, save_path):
    # Step 1: 先加载 LoRA 权重
    load_lora(model, lora_path)

    raw_model = getattr(model, '_orig_mod', model)
    # Step 2: 复制原始权重（排除 LoRA 参数）
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items()
                  if '.lora.' not in k}

    # Step 3: 把 LoRA 的效果叠加到对应的 Linear 权重上
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                # 🔥 W_merged = W_original + B @ A
                state_dict[f'{name}.weight'] += (
                    (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
                )

    torch.save(state_dict, save_path)
```

**为什么要合并？**

合并后得到一个"标准"模型文件——推理时不需要加载 LoRA 代码了，用任何框架（HuggingFace、vllm、Ollama……）都能加载。

**合并原理：**

```
训练好的 LoRA 推理:
  output = W_original(x) + B(A(x))
         = x @ W^T     + x @ A^T @ B^T
         = x @ (W + B@A)^T

合并后的推理:
  output = x @ W_merged^T
  其中 W_merged = W + B@A

数学上完全等价！
```

**矩阵乘法的形状检查：**

```
B: (out_features, rank) = (768, 16)
A: (rank, in_features)   = (16, 768)
B @ A: (768, 16) @ (16, 768) = (768, 768)  ← 和原始 weight 形状一致
```

这里 `@` 是矩阵乘法。`(768, 16) @ (16, 768)` — 中间的 16 约掉，得到 `(768, 768)`。这正是原始 Linear 层权重的形状。

---

## LoRA 完整使用流程

```
第1步: 加载预训练模型
       model = MiniMindForCastiaLM.from_pretrained("pretrain")
       模型有 26M 参数，都能正常工作
                                                    ↓
第2步: 挂载 LoRA
       apply_lora(model, rank=16)
       → 每个方阵 Linear 多了 A(d×r) + B(r×d) 两个小矩阵
       → 新增 ~0.9M 参数（占约 3.5%）
                                                    ↓
第3步: 冻结主模型，只训练 LoRA
       for p in model.parameters(): p.requires_grad = False
       for n, p in model.named_parameters():
           if 'lora' in n: p.requires_grad = True
       → 只有 A 和 B 的 0.9M 参数参与反向传播
                                                    ↓
第4步: 微调训练 (train_lora.py)
       用写诗数据训练，更新 A 和 B
       → 训练快、显存省、不会"坏忘"主模型
                                                    ↓
第5步: 保存 LoRA 权重
       save_lora(model, 'lora_poetry.pth')
       → 文件很小（约 2MB），只有 A 和 B 的权重
                                                    ↓
第6步（可选）: 合并
       merge_lora(model, 'lora_poetry.pth', 'merged.pth')
       → 得到一个标准的 26M 参数文件
       → 可以用任何框架加载，不需要 LoRA 代码
```

### 多任务切换场景

```
基础模型 (26M): pretrain.pth  ← 只存一份
写诗的 LoRA (2MB):     lora_poetry.pth  ← 按需加载
写代码的 LoRA (2MB):   lora_coding.pth  ← 按需加载
翻译的 LoRA (2MB):     lora_translate.pth  ← 按需加载

想写诗:
  model = load_model("pretrain.pth")
  apply_lora(model, rank=16)
  load_lora(model, "lora_poetry.pth")   # 加载写诗适配器

想写代码:
  load_lora(model, "lora_coding.pth")   # 换一套适配器
  # 不需要重新加载基础模型！
```

---

## 参数量估算：LoRA 到底省了多少？

以 `hidden_size=768`、`rank=16` 的 MiniMind 为例：

**哪些层有 LoRA？**

| 层类型 | 维度 | 是否方阵 | 每层 LoRA 参数量 | 8 层总参数量 |
|--------|------|---------|----------------|------------|
| q_proj | 768→768 | ✓ 方阵 | 2×768×16=24,576 | 196,608 |
| o_proj | 768→768 | ✓ 方阵 | 2×768×16=24,576 | 196,608 |
| k_proj | 768→384 | ✗ GQA 非方阵 | — | — |
| v_proj | 768→384 | ✗ GQA 非方阵 | — | — |
| gate_proj | 768→2432 | ✗ 非方阵 | — | — |
| up_proj | 768→2432 | ✗ 非方阵 | — | — |
| down_proj | 2432→768 | ✗ 非方阵 | — | — |

实际 LoRA 参数约 **2 层 × 8 块 × 24,576 ≈ 0.39M**，占模型总参数 ~26M 的约 **1.5%**。

> 注意：在 GQA 配置下（`num_key_value_heads < num_attention_heads`），`k_proj` 和 `v_proj` 的输出维度为 `num_key_value_heads × head_dim = 384`，不等于输入维度 768，所以不满足 `weight.shape[0] == weight.shape[1]` 的方阵检查。只有 `q_proj` 和 `o_proj`（均为 768→768）会被挂载 LoRA。

---

## 总结

`model_lora.py` 仅 66 行实现了一套完整的 LoRA 系统：

| 函数/类 | 做了什么 | 关键 trick |
|---------|---------|-----------|
| `LoRA` 类 | A+B 低秩矩阵适配器 | B 全零初始化，第一步不扰动模型 |
| `apply_lora` | 动态挂载 LoRA 到原模型 | 默认参数绑定，修复闭包陷阱 |
| `load_lora` | 从文件加载 LoRA 权重 | 过滤 key，兼容 DDP 前缀 |
| `save_lora` | 只保存 LoRA 权重 | 转 FP16 省空间，处理 `torch.compile` |
| `merge_lora` | 合并 LoRA 回标准模型 | `W_merged = W + B@A`，数学等价 |

**LoRA 的核心公式就一行：**

```
output = x @ W^T + x @ A^T @ B^T
         ─────    ─────────────
         原始模型    低秩适配 (rank=d 时退化为全量微调)
```

理解了这个公式，剩下的都是工程细节。

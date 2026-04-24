# model_minimind.py 初学者完全指南

> 文件路径: `model/model_minimind.py`
> 代码总量: 约 280 行
> 预备知识: 了解 Python 基础、知道什么是神经网络即可

---

> **阅读提示**: 这份文档按照 "先懂概念 → 再看代码" 的顺序写。如果你看到某段代码觉得困惑，先停下来看它上面的概念解释，然后再回来看代码。

---

## 你需要先知道的 5 个核心概念

### 1. 语言模型在做什么？

语言模型的工作其实只有一件事 —— **猜下一个词**。

你给它一句 "我喜欢吃苹果"，它预测下一个字大概率是 "的"、"子"、"。" 之类。然后它把预测出的词加回去，再继续猜下一个，不断重复：

```
输入:  我喜欢吃___ → 模型猜出 "苹果"
输入:  我喜欢吃苹果___ → 模型猜出 "，"
输入:  我喜欢吃苹果，___ → 模型猜出 "你呢"
```

这就是**自回归**（Auto-regressive）的含义：每一步的产出变成下一步的输入。MiniMind 做的一切工作，都是为了让这种 "猜词" 更准确。

### 2. 词是怎么变成数字的？—— Embedding

计算机只能处理数字。所以模型看到的第一件事是把 token（字/词）转成数字。但不是简单地编号，而是转成一个**向量**（一组数字）。

比如 "苹果" 可能被编码为 `[0.3, -0.1, 0.7, 0.5, ...]`（768 个数字，默认 `hidden_size=768`）。这不是随便来的，这 768 个数字捕捉了 "苹果" 的语义信息。训练好的模型里，意思相近的词（"苹果" 和 "香蕉"）的向量会很接近。

### 3. 什么是注意力（Attention）？

注意力是 Transformer 的核心。打个比方：

> 当你在阅读一句话 "小明把苹果放在了桌子上，因为他饿了" 时，如果要看 "他" 指谁，你的眼睛会自然地看向 "小明"。这就是**注意力**。

简单说：**模型在生成每个词时，会自动决定 "该看前面哪些词"**。比如生成本句末尾的词时，模型可能 "注意到" 句首的主语，也可能 "注意到" 最近的形容词，这由 attention 机制自己学。

### 4. 为什么需要位置编码（Position Encoding）？

语言是有顺序的。"猫追狗" 和 "狗追猫" 意思完全不同。但最原始的神经网络（全连接层）会把输入当成一袋东西，不管顺序。

**RoPE（旋转位置编码）** 的解决思路很巧妙：给每个位置上的词向量做一次 "旋转"。位置越远的词，旋转角度越大。这样当模型计算两个词的相似度时，自然地包含了 "它们相距多远" 的信息。

```
位置0: [0.3, -0.1, 0.7, ...]  ← 不旋转
位置1: [0.2,  0.3, 0.65,...]  ← 旋转一点
位置2: [-0.1, 0.5, 0.55,...]  ← 旋转更多
```

### 5. GQA 是什么？（Grouped Query Attention）

原始的注意力中，"查询"（Q）、"键"（K）、"值"（V）各自用独立的一组头和权重。GQA 的想法很务实：

- Q（我在找什么）需要很多种（多头数 = 8）
- K/V（我有什么）可以共用较少的头（KV 头数 = 4）

这样 Q 的 8 个头中，每 2 个共享同一对 K/V。**8 头 ÷ 4 对 = 每组复用 2 次**。这大大节省了显存和时间，但效果几乎不降。

---

## 模型架构全景：从一个字到一句输出

```
输入文本: "我喜欢吃"
    ↓
[分词器] → 切分成 tokens: [101, 278, 132, 890]
    ↓
[词嵌入层] → 每个 token 变成 768 维向量: [4, 768]
    ↓
[Transformer Blocks × 8 层] ← 每一层:
    │   1. RMSNorm (归一化)
    │   2. Attention (注意力: "该关注哪些词？")
    │   3. 残差连接 (把输入加回来，防止信息丢失)
    │   4. RMSNorm (再次归一化)
    │   5. SwiGLU FFN (前馈网络: "加工理解后的信息")
    │   6. 残差连接
    ↓
[最终归一化] → 输出 768 维向量
    ↓
[LM Head] → 投影回词表大小: [4, 6400]（每个位置对 6400 个词各打一个分）
    ↓
Softmax → 变成概率分布 → 采样出下一个词
```

---

## 代码逐块解析

### 第一部分：配置类 MiniMindConfig

```python
class MiniMindConfig(PretrainedConfig):
```

**这行代码在做什么？**

定义一个配置类，存储模型的所有超参数（层数、维度、头数……）。继承 `PretrainedConfig` 是 HuggingFace 的标准做法，这样模型就能用 `from_pretrained("权重路径")` 直接加载。

**为什么要继承 PretrainedConfig？**

因为 HuggingFace 生态提供了一整套工具（保存/加载权重、推送到 Hub 等）。有了这个继承，MiniMind 就能无缝使用这些功能。

**关键参数速查表：**

| 参数 | 默认值 | 大白话解释 |
|------|--------|-----------|
| `hidden_size` | 768 | 每个 token 变成多长的向量 |
| `num_hidden_layers` | 8 | Transformer 堆多少层（越多越聪明，但越慢） |
| `vocab_size` | 6400 | 词表大小，模型认识的 "单词量" |
| `num_attention_heads` | 8 | Q 有几个注意力头（并行看不同的关系） |
| `num_key_value_heads` | 4 | KV 用几个头（< Q 头数 = GQA，省显存） |
| `max_position_embeddings` | 32768 | 最多能读多长的上下文 |
| `use_moe` | False | 是否用 Mixture-of-Experts（可选的稀疏架构） |
| `rope_theta` | 1e6 | RoPE 的频率基数，越大位置编码范围越广 |

**一个有趣的细节 —— π 缩放（第 26 行）：**

```python
self.intermediate_size = math.ceil(hidden_size * math.pi / 64) * 64
```

FFN（前馈网络）的中间维度不是整数随便设的，而是 `hidden_size × π` 向上取整到 64 的倍数。这是 Llama 系列的做法，让中间层的维度恰好是 GPU block 大小的整数倍，计算更高效。

当 `hidden_size = 768` 时：`768 × 3.14159 / 64 ≈ 37.7 → ceil → 38 → 38 × 64 = 2432`

---

### 第二部分：RMSNorm（层归一化的轻量版本）

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)
```

**RMSNorm 是什么？**

RMSNorm 做的是 **"把一堆数字缩放到统一的量级"**。

想象一下：某一层输出的数字范围是 `[-100, 200]`，传到下一层后，经过计算又到了 `[-500, 800]`。越往后走数字越大，最后可能溢出（特别是 FP16 半精度下）。RMSNorm 就是来"踩刹车"的。

**公式拆解：**

```python
# 1. x.pow(2).mean(-1)    → 算均方（每个数的平方的平均值）
# 2. + self.eps            → 加一个极小数防止除以 0
# 3. torch.rsqrt()         → 取倒数平方根 = 1 / sqrt(均方)
# 4. x * rsqrt             → 用均方的倒数缩放 x

# 最终: x / sqrt(mean(x²) + ε)
```

> **为什么不减均值？** 传统 LayerNorm 会先减均值再做。RMSNorm 跳过了这一步——在 LLM 中实验发现不减均值效果也差不多，反而计算更少了。

**`forward` 中的两个小心机：**

1. `x.float()`：先转成 float32 做归一化，防止半精度下溢出
2. `.type_as(x)`：算完再转回原来的精度（FP16/BF16）

---

### 第三部分：RoPE 位置编码

这是整份代码中**最难的部分**，我们慢慢来。

#### 为什么需要位置编码？

一句话总结：**让模型知道 "词 A 在词 B 前面 3 个位置"。**

没有位置信息时，模型看到的只是一堆向量，分不清 "猫追狗" 和 "狗追猫"。

#### RoPE 的核心思想

```
传统方法: 位置信息 = 一个偏移向量，加到词向量上
RoPE 方法: 位置信息 = 旋转角度，把词向量在 2D 平面上旋转
```

RoPE 把 768 维向量看成 384 对 2 维点。每个位置做不同的旋转：
- 位置 0 的向量转 0°
- 位置 1 的向量转 θ°
- 位置 2 的向量转 2θ°
- ……

当计算两个向量的点积时，它们的夹角就自然包含了 **相对位置信息**。

#### 代码拆解

```python
def precompute_freqs_cis(dim: int, end: int = 32768, rope_base: float = 1e6, rope_scaling: dict = None):
    # Step 1: 计算基础频率（每个维度的旋转速度）
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
```

这行公式的含义：

```
维度0（低频）: 旋转很慢，跨越很远才有明显变化    ← 捕捉长距离关系
维度1: 稍快一些
维度2: 再快一些
...
维度d/2（高频）: 每次旋转很多，相邻位置就变了  ← 捕捉短距离关系
```

`rope_base = 1e6` 控制了频率范围。base 越大，频率范围越广，位置编码的区分能力越强。

```python
    # Step 2: （可选）YaRN 窗口扩展
    if rope_scaling is not None:
        # 把需要长上下文的维度除以 factor，让旋转变慢
        freqs = freqs * (1 - ramp + ramp / factor)
```

**YaRN 的直觉**：

> 模型训练时只见过 2048 长度的句子。但你想让它读 32000 字？这时候高频维度的旋转太快了（超过了训练时见过的角度）。YaRN 把高频维度的旋转速度除以 16 倍（factor=16），让它适应更长的序列。

```python
    # Step 3: 把频率变成 cos/sin 查表
    t = torch.arange(end)                      # 位置索引 [0, 1, 2, ..., end-1]
    freqs = torch.outer(t, freqs)              # 外积: [end, dim/2] 位置×频率矩阵
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)  # [end, dim]
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)  # [end, dim]
```

**为什么 cos/sin 要 concat 两次？**

因为后面要把 768 维向量切成前后两半 `[x1, x2]`，然后做 `x1 * cos - x2 * sin` 和 `x2 * cos + x1 * sin`。每半都需要完整的 cos/sin。

#### apply_rotary_pos_emb —— 实际把位置信息 "旋转" 到 Q 和 K 上

```python
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed
```

**rotate_half 在做什么？** 把向量后半部分取反，然后前后交换。这实际上是 2D 旋转矩阵在代码中的体现：

```
[cos(θ)  -sin(θ)] [x1]    [x1*cos(θ) - x2*sin(θ)]
[sin(θ)   cos(θ)] [x2]  = [x1*sin(θ) + x2*cos(θ)]

在代码中: result = x * cos + rotate_half(x) * sin
```

这个数学操作就是复数乘法 `e^(iθ) · (x1 + ix2)` 的实数版本。两个位置的向量做了旋转后，它们的点积值自动包含了它们的**相对距离**。

---

### 第四部分：GQA 中的 repeat_kv

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim)
            .reshape(bs, slen, num_key_value_heads * n_rep, head_dim))
```

**为什么需要这个函数？**

Q 有 8 个头，KV 只有 4 个头。要计算注意力得让头数对齐：

```
Q:  [头1, 头2, 头3, 头4, 头5, 头6, 头7, 头8]
K:  [头A,          头B,          头C,          头D]   ← 只有 4 个
```

Q 的头1 应该关注 K 的头A，Q 的头2 也应该关注 K 的头A（因为它们共享同一组 KV）。

`repeat_kv` 把 KV "复制" 到与 Q 对齐：
```
K:  [头a, 头a, 头B, 头B, 头C, 头C, 头D, 头D]   ← 8 个了，和 Q 对齐了
```

**高效实现**：`expand` 是零拷贝操作（不会真的复制内存，只是改变 view），最后 `reshape` 时才变连续。比直接 `repeat` 省内存。

---

### 第五部分：Attention（注意力层）

```python
class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # 4 个全连接层，都无偏置
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        # Q 和 K 各自的逐头 RMSNorm
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn
```

**四个 Linear 的作用：**

| 层 | 输入 | 输出 | 比喻 |
|---|------|------|------|
| `q_proj` | 768 维向量 | 8 × head_dim | 我的"查询卡"——我在找什么信息 |
| `k_proj` | 768 维向量 | 4 × head_dim | 每个词的"标签"——我有什么信息 |
| `v_proj` | 768 维向量 | 4 × head_dim | 每个词的"内容"——我的实际含义 |
| `o_proj` | 8 × head_dim | 768 维向量 | 把注意力结果映射回原来的维度 |

**为什么 Q/K/V 都没有 bias？**

实验表明：加了 attention 后 bias 几乎不提供额外信息，反而多了一些参数。去掉能省计算量。

#### Attention 的 forward —— 从输入到输出

```python
def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
    bsz, seq_len, _ = x.shape

    # Step 1: 投影
    xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

    # Step 2: reshape 成多头形式 (batch, seq_len, num_heads, head_dim)
    xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
    xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
    xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

    # Step 3: Q/K 归一化（防止内积过大，softmax 饱和）
    xq, xk = self.q_norm(xq), self.k_norm(xk)

    # Step 4: 应用 RoPE（加入位置信息）
    cos, sin = position_embeddings
    xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

    # Step 5: 如果有历史 KV cache，拼接到前面
    if past_key_value is not None:
        xk = torch.cat([past_key_value[0], xk], dim=1)
        xv = torch.cat([past_key_value[1], xv], dim=1)
    past_kv = (xk, xv) if use_cache else None

    # Step 6: 把 KV 的 4 头复制到 8 头（GQA 对齐）
    xq, xk, xv = (xq.transpose(1, 2),
                  repeat_kv(xk, self.n_rep).transpose(1, 2),
                  repeat_kv(xv, self.n_rep).transpose(1, 2))
    # 现在 xq/xk/xv 的形状: [batch, num_heads, seq_len, head_dim]

    # Step 7: 计算注意力（两条路径）
    if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
        # 路径 A: PyTorch 内置的融合 attention，更快
        # 注意：当使用 causal 模式且有 KV cache 时禁用，因为 SDPA 的 is_causal 不支持 Q/K 序列长度不等的场景
        output = F.scaled_dot_product_attention(xq, xk, xv,
            dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
    else:
        # 路径 B: 手写注意力公式（推理时 KV cache 激活后走这里）
        scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, T, T]
        if self.is_causal:
            # 上三角填 -inf，使位置 i 只能看到 j <= i
            scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"),
                                                     device=scores.device).triu(1)
        if attention_mask is not None:
            # 无效位置填极小值
            scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
```

**手写的注意力公式是怎么工作的？**

```
scores = q @ k^T / sqrt(d)   ← 每个 token 跟所有 token 算相似度，除以 sqrt(d) 稳定梯度
output = softmax(scores) @ v  ← 把 scores 变成权重，加权求和 v
```

除以 `sqrt(d)` 的原因：假设两个随机向量点积的方差是 `d`，不除的话方差会很大，softmax 的梯度会趋近于 0。

**causal mask 的直觉：**

```
假设序列长度为3，scores 矩阵:
     t0    t1    t2
t0   0.3  -inf  -inf   ← t0 只能看自己
t1   0.5   0.2  -inf   ← t1 能看 t0 和 t1
t2   0.1   0.4   0.6   ← t2 能看 t0, t1, t2  (全部可见)
```

这就是为什么叫 "causal"（因果的）—— 每个位置只能看到它前面（或本身）的信息，不能 "偷看" 后面的词。

---

### 第六部分：前馈神经网络

#### FeedForward（SwiGLU 结构）

```python
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]  # PyTorch 没有 silu，所以借用了 HF 的字典

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

**为什么叫 SwiGLU？** 它由两部分组成：

```
gate_proj(x) → swish(·)     "阀门" — 决定要激活哪些信息
up_proj(x)                    "放大器" — 升维提取信息
两者相乘 → down_proj(·)      "过滤 + 压缩"
```

你可以把它理解成一个**门控放大器**：gate 决定哪些信息重要，up 把这些信息放大到一个更高的空间（768 → 2432），down 再把结果压回原来的维度。

**跟传统 FFN 的对比：**

```
传统 FFN:  x → Linear1 → ReLU → Linear2 → output
SwiGLU:    x → gate_proj → swish ──┐
           x → up_proj ────────────┤──×──→ down_proj → output

SwiGLU 多了一个 gate 分支，计算量大约增加 50%，但表达能力更强
```

---

#### MOEFeedForward（稀疏专家混合）

这是模型可选的高级功能（`use_moe=True` 时替代 FeedForward）。

**直觉：为什么需要 MoE？**

想象一个公司：

- **稠密 FFN**：每个问题都让所有员工来回答，把意见综合起来。慢，浪费。
- **MoE FFN**：有个前台（gate/router）判断你的问题属于哪类，只派给相关专家（expert）。高效，能力强。

MoE 的核心优势：**参数量大，但推理时只用一小部分**。

```python
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # Router / Gate: H 维输入 → num_experts 个分数
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # N 个独立的 FeedForward 专家
        self.experts = nn.ModuleList(
            [FeedForward(config, intermediate_size=config.moe_intermediate_size)
             for _ in range(config.num_experts)]
        )
```

**forward 流程一步一步走：**

```python
def forward(self, x):
    batch_size, seq_len, hidden_dim = x.shape
    x_flat = x.view(-1, hidden_dim)
    # 例: [2, 5, 768] → [10, 768]（2 个句子，每句 5 个 token，共 10 个 token）

    # Gate 打分
    scores = F.softmax(self.gate(x_flat), dim=-1)
    # [10, 4] → 每个 token 对 4 个专家各打一个概率分数

    # Top-k 选专家
    topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
    # 假设 k=1，每个 token 选 1 个专家: topk_idx = [0, 2, 1, 0, 3, ...]

    # 概率归一化（可选）
    if self.config.norm_topk_prob:
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

    # 每个专家分别处理属于它的 token
    y = torch.zeros_like(x_flat)
    for i, expert in enumerate(self.experts):
        mask = (topk_idx == i)                # 属于第 i 个专家的 token
        if mask.any():
            token_idx = mask.any(dim=-1).nonzero().flatten()
            weight = topk_weight[mask].view(-1, 1)
            # 专家处理 + 权重加权
            y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
        elif self.training:
            # 如果某专家完全没分到 token，用 0*params 让它参与计算图
            y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

    # 辅助损失（训练时防止"垄断"）
    if self.training:
        self.aux_loss = (load * scores.mean(0)).sum() * num_experts * router_aux_loss_coef

    return y.view(batch_size, seq_len, hidden_dim)
```

**`index_add_` 是在做什么？**

这是一个按索引把结果"加回去"的操作，比 for 循环快很多：

```
假设有 10 个 token，专家编号分别是: [0, 2, 1, 0, 3, 1, 0, 2, 1, 0]

专家0 处理 token #0,3,6,9  →  index_add_ 把结果放回去原位
专家1 处理 token #2,5,8    →  index_add_ 把结果放回去原位
专家2 处理 token #1,7      →  index_add_ 把结果放回去原位
专家3 处理 token #4        →  index_add_ 把结果放回去原位

最后 y 包含了所有 token 的加权专家输出
```

**空专家兜底（为什么写 `y[0, 0] += 0 * params`？）：**

```python
y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
```

这行代码很"脏"，但很聪明：
- 数学上等于 `+ 0`，不影响输出
- 但让该专家的参数也存在于计算图中
- 这样 PyTorch 仍会给这个专家计算梯度（虽然梯度值为 0），防止它永远被忽略

**辅助损失是什么？**

```python
load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
# [4] → 每个专家分到 token 的比例，4 个专家加起来 = 1
self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
```

如果 Router 总是选专家 0（load = [0.8, 0.05, 0.1, 0.05]），这个辅助损失就会很大。这迫使 Router 学会"公平分配"——每个专家都要有活干。

---

### 第七部分：MiniMindBlock（Transformer 的一层）

```python
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 根据配置选择普通 FFN 或 MoE
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
```

**Pre-Norm 架构（Pre-Normalization）：**

```
输入 x
  ↓
+── ← 残差连接 (residual)
│   ↓
│  RMSNorm (input_layernorm)     ← 注意：在 Attention 之前做归一化
│   ↓
│  Attention                     ← 计算注意力
│   ↓
│  + x                           ← 残差相加
│   ↓
│+── ← 残差连接
│   ↓
│  RMSNorm (post_attention_layernorm) ← 在 FFN 之前做归一化
│   ↓
│  FFN / MoE                     ← 前馈网络
│   ↓
│  + x                           ← 残差相加
│   ↓
输出 (传给下一层)
```

**为什么叫 "Pre-Norm"？** 因为归一化在 **子层之前** 做（Norm → Attention → +x），而不是之后（Attention → Norm → +x）。Pre-Norm 的梯度流更好，训练更稳定。

---

### 第八部分：MiniMindModel（主干模型）

```python
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 预计算 RoPE 的 cos/sin 表（一次计算，训练/推理共用）
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim, end=config.max_position_embeddings,
            rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
```

**`register_buffer` 是什么意思？**

```python
self.register_buffer("freqs_cos", freqs_cos, persistent=False)
```

- `buffer` 是跟随模型在 GPU/CPU 间移动的 Tensor，但**不是**可训练参数
- `persistent=False` 表示不保存到 state_dict（因为重启时会自动重算，没必要存）
- 这样做的好处：初始化时一次性计算，推理时直接切片取用，不用每一步都重算

---

#### MiniMindModel 的 forward

```python
def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
    batch_size, seq_length = input_ids.shape
    # 兼容 HF cache 格式
    if hasattr(past_key_values, 'layers'): past_key_values = None
    past_key_values = past_key_values or [None] * len(self.layers)
    # 从 cache 长度推算起始位置
    start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

    # Step 1: 词嵌入 + dropout
    hidden_states = self.dropout(self.embed_tokens(input_ids))

    # Step 2: 取当前位置区间的 RoPE
    position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length],
                          self.freqs_sin[start_pos:start_pos + seq_length])

    # Step 3: 逐层通过 Transformer block
    presents = []  # KV cache 收集
    for layer, past_key_value in zip(self.layers, past_key_values):
        hidden_states, present = layer(
            hidden_states, position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask
        )
        presents.append(present)

    # Step 4: 最终归一化
    hidden_states = self.norm(hidden_states)

    # Step 5: 如果使用 MoE，收集所有层的辅助损失
    aux_loss = sum(
        [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
        hidden_states.new_zeros(1).squeeze()
    )

    return hidden_states, presents, aux_loss
```

**返回三个值：**

| 返回值 | 形状 | 含义 |
|--------|------|------|
| `hidden_states` | `[batch, seq_len, hidden_size]` | 每个 token 的最终表示 |
| `presents` | `list of tuples` | 每层的 KV cache，用于下一步增量推理 |
| `aux_loss` | scalar | MoE 的辅助损失（稠密模型时 = 0） |

---

### 第九部分：MiniMindForCausalLM（完整的因果语言模型）

```python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 🔥 Weight Tying: 让 embedding 和 lm_head 共用同一组权重
        self.model.embed_tokens.weight = self.lm_head.weight
```

**为什么要共享 embedding 和 lm_head 的权重？**

1. **节省参数**：两个大矩阵变成一个，减少 ~2 × 768 × 6400 = ~9.8M 参数
2. **一致性**：输入和输出使用同一套向量空间，输入中 "猫" 和 "狗" 相似，输出中它们的概率也应该相似
3. **防止过拟合**：少了一组参数需要学习

> **注意**：`self.model.embed_tokens.weight = self.lm_head.weight` 这个赋值让两个 Module 指向同一块内存。修改其中一个，另一个也跟着变。

---

#### forward（训练/推理共用的核心入口）

```python
def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False,
            logits_to_keep=0, labels=None, **kwargs):
    hidden_states, past_key_values, aux_loss = self.model(
        input_ids, attention_mask, past_key_values, use_cache, **kwargs)
    # 根据 logits_to_keep 裁剪位置（推理时只取最后一个 token）
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])
    loss = None
    if labels is not None:
        x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
        loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
    return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits,
                                     past_key_values=past_key_values, hidden_states=hidden_states)
```

**训练时的 Loss 是怎么计算的？**

```
输入:    "我喜欢吃___"
labels:  "喜欢 吃 苹果___"  (向左偏移一位)

logits[t] = 模型在位置 t 对下一个词的预测
loss  = 交叉熵(logits[:-1], labels[1:])
      = sum( -log(模型给正确答案的概率) ) / 总 token 数
```

`ignore_index=-100` 表示 labels 中值为 -100 的位置不参与 loss 计算（通常用于 padding 或 prompt 部分）。

---

### 第十部分：generate（文本生成）

这是整个模型最"好玩"的部分 —— 从输入一句提示词，让模型自己写出后续内容。

```python
@torch.inference_mode()  # 不做梯度计算，省显存、加速
def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192,
             temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2,
             streamer=None, use_cache=True, num_return_sequences=1,
             do_sample=True, repetition_penalty=1.0, **kwargs):
```

#### 参数大白话解释

| 参数 | 含义 | 效果 |
|------|------|------|
| `max_new_tokens` | 最多生成多少个新词 | 设太小 → 答案不完整；设太大 → 模型会啰嗦或跑题 |
| `temperature` | 温度（创造性） | 低（≈0）→ 保守、保守输出最高概率的词；高（≈1.5）→ 更有创造力但可能不合理 |
| `top_p` | Nucleus 采样阈值 | 只保留累积概率 < top_p 的候选。越低 → 保守 |
| `top_k` | 只保留概率前 k 高的词 | k=1 → 贪婪；k=50 → 比较开放 |
| `repetition_penalty` | 重复检测 | > 1.0 时对已出现过的词降权，1.2 通常不错 |
| `do_sample` | 是否采样 | True = 随机（有创造性）；False = 贪心（稳定可复现） |

#### 生成过程一步步走

```python
# Step 1: 准备输入
input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
# 例: [BOS, 101, 278, 132] → 如果 num_return_sequences > 1 就复制多份

finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
# 标记哪些句子已经生成完了
```

```python
# Step 2: 逐 token 生成循环
for _ in range(max_new_tokens):
    # 2a: 只输入新增的 token（复用 KV cache）
    past_len = past_key_values[0][0].shape[1] if past_key_values else 0
    outputs = self.forward(input_ids[:, past_len:], attention_mask,
                          past_key_values, use_cache=use_cache, **kwargs)

    # attention_mask 扩展一位
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1)

    # 2b: 取最后一个位置的 logits，做 temperature 缩放
    logits = outputs.logits[:, -1, :] / temperature

    # 2c: 重复惩罚
    if repetition_penalty != 1.0:
        for i in range(input_ids.shape[0]):
            logits[i, torch.unique(input_ids[i])] /= repetition_penalty
            # 对已出现过的词的 logits 除以 penalty（>1 则等效降权）

    # 2d: top-k 过滤 — 去掉概率排名在 k 以外的词
    if top_k > 0:
        logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')

    # 2e: top-p 过滤 — 去掉累积概率超过 p 的"尾部"词
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
        # 右移一位，保证至少保留第一个（概率最高的）token
        mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
        logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')

    # 2f: 采样或贪心选词
    next_token = (torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                  if do_sample else torch.argmax(logits, dim=-1, keepdim=True))

    # 已完成的句子强制输出 eos
    if eos_token_id is not None:
        next_token = torch.where(finished.unsqueeze(-1),
                                 next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)

    # 2g: 拼接新 token
    input_ids = torch.cat([input_ids, next_token], dim=-1)
    past_key_values = outputs.past_key_values if use_cache else None

    # 2h: 检查 EOS，更新 finished 状态
    if eos_token_id is not None:
        finished |= next_token.squeeze(-1).eq(eos_token_id)
        if finished.all():
            break   # 所有句子都完成了，提前退出
```

```python
# Step 3: 输出
if streamer: streamer.end()
if kwargs.get("return_kv"):
    return {'generated_ids': input_ids, 'past_kv': past_key_values}
return input_ids
```

**KV Cache 为什么能加速？**

```
第1步: 输入 "我喜欢" → 模型过一遍所有层，得到 KV(我喜欢)
第2步: 只输入 "苹" → 模型只算新 token，历史 KV 复用
第3步: 只输入 "果" → 同样复用所有历史
...
```

没有 KV cache 时，每一步都要把整个序列重新过一遍所有层。有了 KV cache，每一步只算新增 token 的 KV。对于 8 层的 MiniMind，这带来约 8 倍的推理加速。

**streamer 是什么？**

```python
if streamer: streamer.put(next_token.cpu())
```

`streamer` 是一个异步输出队列。有了它，生成和打印/显示可以并行执行——模型在算下一个 token 的同时，前一个 token 已经被打印到屏幕或渲染到网页上。这就是为什么你看到 ChatGPT 的回复是"一个字一个字冒出来"的。

---

## 总结：从代码回到全景

```
MiniMindForCausalLM
│
├── model (MiniMindModel)
│   ├── embed_tokens (nn.Embedding) — 词→向量，与 lm_head 共享权重
│   ├── lm_head (nn.Linear) — 向量→词表分数
│   ├── dropout
│   ├── layers (nn.ModuleList[MiniMindBlock × 8])
│   │   └── MiniMindBlock (Pre-Norm 架构)
│   │       ├── RMSNorm → Attention (GQA + RoPE + QK Norm + SDPA/手写)
│   │       └── RMSNorm → FeedForward (SwiGLU) 或 MOEFeedForward
│   └── norm (RMSNorm) — 最后一层归一化
│
└── generate() — 自回归文本生成
    ├── KV Cache 增量推理
    ├── temperature + top-k + top-p 采样
    ├── repetition penalty
    └── streamer 流式输出
```

### 学习建议

1. **先跑起来再看代码**: `python scripts/chat_api.py --from_weight full_sft`，直观感受一下能做什么
2. **从 generate 开始读代码**：它是自包含的，不涉及训练逻辑，最容易理解
3. **再看 Attention**：这是 Transformer 最核心的模块，理解它等于理解了 60%
4. **最后看 MoE 和训练脚本**：这些是进阶内容

MiniMind 的 280 行代码虽然看起来短，但每一行都对应了现代 LLM 的关键设计原则：RoPE 位置编码、GQA 注意力、RMSNorm 归一化、SwiGLU 前馈层。这些跟 GPT-4、Claude、Gemini 等大型模型在**架构理念上是一致的**，只是规模不同。理解 MiniMind 的架构，就相当于拥有了理解所有 Transformer-based LLM 的钥匙。

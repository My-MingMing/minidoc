# convert_model.py 深度解析——模型格式转换工具箱

> 先懂概念，再看代码。本文从"为什么需要格式转换"出发，逐步讲解每个转换函数的用途和实现。

---

> 文件路径: `scripts/convert_model.py`
> 代码总量: 约 145 行
> 预备知识: PyTorch 模型保存/加载机制、HuggingFace Transformers 模型格式、LoRA 原理
> 阅读建议: MiniMind 训练使用原生 PyTorch `.pth` 格式，但发布和部署通常需要 HuggingFace Transformers 格式。这个脚本就是连接两个世界的桥梁。建议先了解 `model_minimind.py` 和 `model_lora.py`，再来看这个转换工具。

---

## 目录

1. [为什么需要格式转换](#1-为什么需要格式转换)
2. [6 个转换函数总览](#2-6-个转换函数总览)
3. [torch → Transformers（MiniMind 格式）](#3-torch--transformersminiMind-格式)
4. [torch → Transformers（Qwen3 兼容格式）](#4-torch--transformersqwen3-兼容格式)
5. [Transformers → torch](#5-transformers--torch)
6. [LoRA 合并](#6-lora-合并)
7. [Chat Template 转换](#7-chat-template-转换)
8. [Transformers v5 兼容处理](#8-transformers-v5-兼容处理)
9. [使用方法](#9-使用方法)

---

## 1. 为什么需要格式转换

MiniMind 的训练流水线全部使用原生 PyTorch 格式（`.pth` 文件，即 `state_dict`）。但当你想要：

| 目标 | 需要的格式 |
|---|---|
| 上传 HuggingFace Hub | Transformers 格式 |
| 用 `AutoModelForCausalLM.from_pretrained()` 加载 | Transformers 格式 |
| 被 vLLM / TGI 等推理框架加载 | Transformers 格式 |
| 兼容 Qwen3 生态（量化、微调工具链） | Qwen3 兼容格式 |
| 将 LoRA 适配器合并进基模 | 合并操作 |

```
训练侧                           部署侧
┌──────────┐    convert_model    ┌──────────────────────┐
│ .pth 文件 │ ──────────────→   │ Transformers 格式     │
│ (PyTorch)│                    │ (config.json +        │
│          │ ←──────────────    │  model.safetensors +  │
└──────────┘    convert_model    │  tokenizer 文件)      │
                                └──────────────────────┘
```

---

## 2. 6 个转换函数总览

| # | 函数名 | 方向 | 用途 |
|---|---|---|---|
| 1 | `convert_torch2transformers_minimind` | torch → HF | 保留 MiniMind 自定义模型类 |
| 2 | `convert_torch2transformers` | torch → HF | 转为 Qwen3 兼容格式（更广泛的生态兼容） |
| 3 | `convert_transformers2torch` | HF → torch | 从 Transformers 格式反向提取 state_dict |
| 4 | `convert_merge_base_lora` | base + LoRA → torch | 将 LoRA 权重合并进基模 |
| 5 | `convert_jinja_to_json` | .jinja → JSON 字符串 | 将 Jinja 模板转为 JSON 中可嵌入的格式 |
| 6 | `convert_json_to_jinja` | JSON → .jinja | 从 tokenizer_config.json 提取 chat template |

---

## 3. torch → Transformers（MiniMind 格式）

```python
def convert_torch2transformers_minimind(torch_path, transformers_path, dtype=torch.float16):
```

### 3.1 核心流程

```
  .pth 文件
     │
     ▼
  ① 注册 MiniMind 自定义类
     MiniMindConfig.register_for_auto_class()
     MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
     │
     ▼
  ② 创建模型实例 + 加载权重
     lm_model = MiniMindForCausalLM(lm_config)
     lm_model.load_state_dict(state_dict)
     │
     ▼
  ③ 转为目标精度（默认 FP16）
     lm_model = lm_model.to(dtype)
     │
     ▼
  ④ 保存模型 + tokenizer
     lm_model.save_pretrained(transformers_path)
     tokenizer.save_pretrained(transformers_path)
     │
     ▼
  ⑤ Transformers v5 兼容补丁（如需要）
```

### 3.2 为什么要注册 Auto 类

```python
MiniMindConfig.register_for_auto_class()
MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
```

注册后，`AutoModelForCausalLM.from_pretrained()` 就能自动识别并加载 MiniMind 模型——即使加载方不知道 MiniMind 的存在，只要指定 `trust_remote_code=True`。

### 3.3 输出目录结构

```
transformers_path/
├── config.json                 ← MiniMindConfig 序列化
├── model.safetensors           ← 模型权重
├── tokenizer_config.json       ← tokenizer 配置
├── tokenizer.json              ← tokenizer 词表
├── special_tokens_map.json     ← 特殊 token 映射
└── *.py                        ← 自定义模型代码（trust_remote_code 需要）
```

---

## 4. torch → Transformers（Qwen3 兼容格式）

```python
def convert_torch2transformers(torch_path, transformers_path, dtype=torch.float16):
```

### 4.1 为什么转为 Qwen3 格式

MiniMind 的模型结构（GQA + RoPE + SwiGLU）与 Qwen3 高度相似。转为 Qwen3 格式后可以直接使用 Qwen3 生态的所有工具，无需 `trust_remote_code=True`。

### 4.2 Dense vs MoE 分支

```python
if not lm_config.use_moe:
    # Dense 模型 → Qwen3ForCausalLM
    qwen_config = Qwen3Config(...)
    qwen_model = Qwen3ForCausalLM(qwen_config)
else:
    # MoE 模型 → Qwen3MoeForCausalLM
    qwen_config = Qwen3MoeConfig(
        ...,
        num_experts=lm_config.num_experts,
        num_experts_per_tok=lm_config.num_experts_per_tok,
        moe_intermediate_size=lm_config.moe_intermediate_size
    )
    qwen_model = Qwen3MoeForCausalLM(qwen_config)
```

### 4.3 MoE 专家权重重组（Transformers v5）

Transformers v5 改变了 MoE 的存储格式——从每个专家独立存储变为堆叠存储：

```python
# v4: experts.0.gate_proj.weight, experts.0.up_proj.weight, ...
# v5: experts.gate_up_proj (shape: [num_experts, 2*intermediate, hidden])
#     experts.down_proj     (shape: [num_experts, hidden, intermediate])

# 转换：将各专家的 gate_proj 和 up_proj 拼接成 gate_up_proj
new_sd[f'{p}.gate_up_proj'] = torch.cat([
    torch.stack([state_dict[f'{p}.{e}.gate_proj.weight'] for e in range(num_experts)]),
    torch.stack([state_dict[f'{p}.{e}.up_proj.weight'] for e in range(num_experts)])
], dim=1)
new_sd[f'{p}.down_proj'] = torch.stack([
    state_dict[f'{p}.{e}.down_proj.weight'] for e in range(num_experts)
])
```

### 4.4 配置映射

| MiniMind 参数 | Qwen3 参数 | 说明 |
|---|---|---|
| `hidden_size` | `hidden_size` | 直接映射 |
| `num_attention_heads` | `num_attention_heads` | 直接映射 |
| `num_key_value_heads` | `num_key_value_heads` | GQA 配置 |
| `intermediate_size` | `intermediate_size` | FFN 中间维度 |
| `rms_norm_eps` | `rms_norm_eps` | 归一化 epsilon |
| `rope_theta` | `rope_theta` | RoPE 基频 |
| - | `tie_word_embeddings=True` | 输入/输出共享权重 |
| - | `use_sliding_window=False` | 不使用滑窗注意力 |

---

## 5. Transformers → torch

```python
def convert_transformers2torch(transformers_path, torch_path):
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save({k: v.cpu().half() for k, v in model.state_dict().items()}, torch_path)
```

最简单的转换：加载 Transformers 模型，提取 `state_dict`，保存为 `.pth`。所有权重统一转为 FP16（`.half()`）并移到 CPU。

---

## 6. LoRA 合并

```python
def convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path):
    lm_model = MiniMindForCausalLM(lm_config).to(device)
    lm_model.load_state_dict(torch.load(base_torch_path))  # 加载基模权重
    apply_lora(lm_model)                                     # 注入 LoRA 结构
    merge_lora(lm_model, lora_path, merged_torch_path)       # 合并并保存
```

### 合并流程

```
  基模权重 (full_sft_768.pth)
     │
     ▼
  apply_lora() → 注入 A/B 矩阵结构
     │
     ▼
  merge_lora() → 加载 LoRA 权重 → W_merged = W_base + A×B
     │
     ▼
  保存合并后的权重 (merge_identity_768.pth)
```

合并后的模型不再需要 LoRA 模块，推理时直接加载即可，避免了额外的 LoRA 计算开销。

---

## 7. Chat Template 转换

### 7.1 Jinja → JSON

```python
def convert_jinja_to_json(jinja_path):
    with open(jinja_path, 'r') as f:
        template = f.read()
    escaped = json.dumps(template)   # 自动转义换行、引号等
    print(f'"chat_template": {escaped}')
```

将可读的 Jinja2 模板文件转为可以直接粘贴到 `tokenizer_config.json` 中的 JSON 字符串。

### 7.2 JSON → Jinja

```python
def convert_json_to_jinja(json_file_path, output_path):
    config = json.load(open(json_file_path))
    template = config['chat_template']          # 从 JSON 中提取
    with open(output_path, 'w') as f:
        f.write(template)                       # 写为独立 .jinja 文件
```

反向操作：从 `tokenizer_config.json` 提取 `chat_template` 字段，保存为独立的 `.jinja` 文件方便编辑。

---

## 8. Transformers v5 兼容处理

两个 `torch2transformers` 函数都包含 v5 兼容补丁：

```python
if int(transformers.__version__.split('.')[0]) >= 5:
    # 1. tokenizer_config.json: 添加 tokenizer_class 和 extra_special_tokens
    json.dump({
        ...config,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "extra_special_tokens": {}
    }, ...)

    # 2. config.json: 修正 rope_scaling 格式
    config['rope_theta'] = lm_config.rope_theta
    config['rope_scaling'] = None
    del config['rope_parameters']   # v5 改名为 rope_parameters，但低版本不兼容
```

这些补丁确保转换出的模型同时兼容 Transformers v4 和 v5。

---

## 9. 使用方法

### 9.1 脚本配置（底部的 `__main__` 块）

```python
if __name__ == '__main__':
    lm_config = MiniMindConfig(
        hidden_size=768,
        num_hidden_layers=8,
        max_seq_len=8192,
        use_moe=False
    )

    # 当前激活的转换
    torch_path = f"../out/full_sft_{lm_config.hidden_size}.pth"
    transformers_path = '../minimind-3'
    convert_torch2transformers(torch_path, transformers_path)
```

### 9.2 常用转换命令

```bash
cd scripts

# 转换为 Qwen3 兼容格式（推荐，生态兼容性最好）
python convert_model.py

# 若要转换为 MiniMind 原生格式，取消注释对应代码块

# 若要合并 LoRA，取消注释 merge lora 代码块
```

### 9.3 注意事项

- `lm_config` 的参数必须与训练时一致（hidden_size, num_hidden_layers, use_moe 等）
- MoE 模型需设置 `use_moe=True`
- 路径是相对于 `scripts/` 目录的，注意 `../` 前缀

### 9.4 关键依赖关系

```
convert_model.py
├── model.model_minimind  → MiniMindConfig, MiniMindForCausalLM
├── model.model_lora      → apply_lora, merge_lora
└── transformers          → Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
```

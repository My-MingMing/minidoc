# rollout_engine.py 详解 — RL 训练中的 "采样引擎"

> 面向初学者的逐层解读 | 约 194 行 | 源文件：`trainer/rollout_engine.py`

---

## 先问一个问题：为什么需要一个 "rollout engine"？

在 PPO / GRPO 这类 **基于强化学习** 的训练方法中，模型需要 "自己写回复"，然后对这个回复打分、更新参数。这和普通的监督学习有本质区别：

```
监督学习（SFT）：
  输入问题 -> 已知标准答案 -> 计算差距 -> 更新参数

强化学习（PPO/GRPO）：
  输入问题 -> 模型自己写回复 -> 给回复打分 -> 更新参数
                ^^^^^^^^^
              这一步就叫 "rollout"（采样/展开）
```

**Rollout** 就是让当前模型用自己的策略生成一批回复，供评分和参数更新使用。

### 为什么需要 per-token log probability？

PPO 需要比较 "新策略" 和 "旧策略" 生成同一个 token 的概率差异：

```
PPO 核心公式（简化版）：
  ratio = P_new(token) / P_old(token)
  用对数概率改写：log_ratio = log_P_new(token) - log_P_old(token)
```

另一个重要用途是 **KL 惩罚**：限制新模型偏离原始模型太远，KL 散度需要 per-token log probability。

---

## 整体架构

```
┌─────────────────────────────────────┐
│         训练脚本 (PPO/GRPO)          │
│   engine.rollout(prompt_ids, ...)    │
│   engine.update_policy(model)        │
└──────────────┬──────────────────────┘
               │  统一接口（抽象基类）
       ┌───────┴───────┐
       ▼               ▼
┌─────────────┐  ┌─────────────┐
│   Torch     │  │   SGLang    │
│ Rollout     │  │  Rollout    │
│  Engine     │  │   Engine    │
└─────────────┘  └─────────────┘
  进程内直接调用    远程 HTTP 服务
  简单、零部署      高效、可热切换权重
```

### 为什么用抽象基类？

```python
# 第 46-55 行
class RolloutEngine(ABC):
    @abstractmethod
    def rollout(self, prompt_ids, attention_mask, num_generations,
                max_new_tokens, temperature=0.8) -> RolloutResult:
        pass

    @abstractmethod
    def update_policy(self, model):
        pass
```

这就像 **USB 接口**：不管插入 U 盘（Torch）还是 SSD（SGLang），宿主（训练脚本）只需知道 "调用 rollout() 就能生成回复"。

**两个抽象方法的含义：**

| 方法 | 含义 | 调用时机 |
|------|------|---------|
| `rollout()` | 让模型根据 prompt 生成回复 | 每个训练 batch 采样阶段 |
| `update_policy()` | 同步最新的模型权重 | 模型参数更新后，告诉引擎"我变了" |

---

## RolloutResult — 引擎的"返回值信封"

```python
# 第 37-42 行
@dataclass
class RolloutResult:
    output_ids: Tensor          # 完整序列：[B, P+R]（prompt + 回复）
    completion_ids: Tensor      # 仅回复部分：[B, R]
    per_token_logps: Tensor     # 每个回复 token 的 log 概率：[B, R]
    completions: List[str]      # 人类可读的文本列表
```

类比：像餐厅端来的托盘，上面放了四样东西：

| `output_ids` | 完整序列，KL 散度需要 |
| `completion_ids` | RL 训练输入——模型自己写的部分 |
| `per_token_logps` | PPO 算 ratio 和 KL 的核心数据 |
| `completions` | 日志打印、reward 打分 |

---

## TorchRolloutEngine — 简单直接的本地引擎

### 它是什么？

直接用训练脚本里的模型做推理，不依赖任何外部服务。

```python
# 第 59-64 行
class TorchRolloutEngine(RolloutEngine):
    def __init__(self, policy_model, tokenizer, device="cuda", autocast_ctx=None):
        self.policy_model = policy_model  # 模型直接传进来
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx  # 混合精度上下文
```

### rollout() 工作流

```
第 1 步：生成回复
  with torch.no_grad():                        ← 不计算梯度
    output_ids = model.generate(               ← 调用模型的生成方法
        do_sample=True,                        ← 随机采样，非贪心
        temperature=0.8,                       ← 控制"创意程度"
        num_return_sequences=num_generations,  ← GRPO 需要多条回复
        ...
    )

第 2 步：拆分出回复部分
  prompt_len = prompt_ids.size(1)
  completion_ids = output_ids[:, prompt_len:]

第 3 步：计算 per-token log probability
  per_token_logps = compute_per_token_logps(...)

第 4 步：解码成可读文本
  completions = tokenizer.batch_decode(completion_ids)

第 5 步：打包返回
  return RolloutResult(output_ids, completion_ids, per_token_logps, completions)
```

### update_policy()

```python
# 第 92-93 行
def update_policy(self, model):
    self.policy_model = model  # 直接替换引用
```

极其简单：直接赋值。因为 TorchRolloutEngine 在同一进程内运行，拿到引用就能用。

### 特点

| 优点 | 缺点 |
|------|------|
| 零配置、开箱即用 | 大批量推理慢 |
| 代码简单、易调试 | 无连续 batching 优化 |
| 与训练同进程 | 共享 GPU 内存 |

---

## SGLangRolloutEngine — 高性能远程引擎

### 它是什么？

SGLang 是一个 **专为大模型推理优化的 HTTP 服务**，独立运行在一个端口上。训练脚本通过 HTTP 调用它。

```
启动命令（源文件第 2 行注释）：
  python -m sglang.launch_server --model-path ./minimind-3 \
      --attention-backend triton --host 0.0.0.0 --port 8998
```

```
训练脚本（进程 A）              SGLang 服务（进程 B，port 8998）
     │                                  │
     │  POST /generate                  │
     │  {input_ids: [...]} ────────>    │
     │                        推理生成   │
     │  <── {output_ids, logprobs}      │
     │                                  │
     │  POST /update_weights_from_disk  │
     │  {model_path: "./sglang_ckpt"}►  │
     │                        加载新权重 │
     │  <── 200 OK                      │
```

### 为什么需要远程引擎？

```
Torch 方式的问题：
  训练和推理共用一个模型对象 → 互相干扰，内存争抢

SGLang 方式的优势：
  训练脚本和推理服务完全独立
  SGLang 可以做连续 batching、张量并行等高级优化
  热切换权重：不需要重启推理服务
```

### SGLang 的热更新机制（重点！）

`update_policy()` 的完整流程（第 168-182 行）：

```
第 1 步：解包模型
  unwrapped = model.module if isinstance(model, DDP) else model
  （分布式训练时需 .module 拿到真实模型）

第 2 步：保存权重到磁盘
  ├─ lm_head.weight 去参数包裹（避免原地修改问题）
  ├─ state_dict 转 half 精度（节省磁盘空间）
  ├─ save_pretrained() 到 self.shared_ckpt_path（默认 "./sglang_ckpt"）
  └─ 同时保存 tokenizer（SGLang 解码需要）

第 3 步：HTTP 通知 SGLang 加载
  POST /update_weights_from_disk
  Body: {"model_path": "/绝对路径/sglang_ckpt"}
  → SGLang 从磁盘读取新权重并替换，无需重启！

第 4 步：检查响应
  200 = 成功，非 200 = 打印警告
```

**类比：热更新就像给飞行中的飞机换引擎**

```
普通方式：停机 -> 卸载旧权重 -> 加载新权重 -> 重启（训练中断）
热更新方式：写磁盘 -> HTTP 通知 -> 后台切换（训练几乎不等待）
```

### 辅助方法

```python
def flush_cache(self):   # POST /flush_cache   — 清空 KV cache
def health(self):        # GET  /health         — 检查服务是否活着
```

### SGLang rollout 的数据流

```
1. 去 padding：只保留 attention mask 为 1 的有效 token
2. 复制多份：每个 prompt 按 num_generations 复制
3. 发请求：POST /generate，关键参数 return_logprob=true
4. 解析响应：提取 output_ids 和 output_token_logprobs
5. 补齐长度：batch 内各回复长度不同，pad 到相同长度再拼成 Tensor
```

---

## compute_per_token_logps — RL 的"概率计算器"

这是文件中最重要的计算函数（第 21-33 行）。

### 它做了什么？

给定完整序列（prompt + 回复），计算每个回复 token 的对数条件概率：

```
log P(token_i | token_1, ..., token_{i-1})
```

### 为什么需要它？

```
模型生成了回复："我喜欢 编程"

RL 算法需要知道：
  "看到 prompt 后，对'我'的打分是 -1.2（对数概率）"
  "看到 '我'之后，对'喜欢'的打分是 -0.8"
  "看到 '我喜欢'之后，对'编程'的打分是 -2.1"

RL 用这些数字计算新旧策略的差距。

交叉熵 loss= 平均值（一个标量），但 RL 需要逐个 token 的向量。
```

### 核心原理

```
语言模型做什么？
  输入：[prompt + 回复序列]
  输出：每个位置上，下一个 token 的概率分布（logits [B, L, V]）

我们要什么？
  从 logits 中取出实际生成 token 对应的概率值
  ┌────────────────────────────────────────────┐
  │ logits[0]  → 预测 "我" 的分布              │
  │ logits[1]  → 预测 "喜欢" 的分布            │
  │ logits[2]  → 预测 "编程" 的分布            │
  │                                            │
  │ 实际 token: "我", "喜欢", "编程", <eos>    │
  │                                            │
  │ gather 操作 = 从每个分布中取出对应 token     │
  │   的值                                      │
  └────────────────────────────────────────────┘
```

### 代码解读

```python
# 步骤 1：处理 DDP 包裹
unwrapped = model.module if isinstance(model, DDP) else model

# 步骤 2：前向传播拿 logits
# logits_to_keep=n_keep+1 → 只保留 reply 部分的 logits（省内存）
logits = unwrapped(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]

# 步骤 3：gather 提取实际 token 的对数概率
for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
    # log_softmax → log(P(token))
    # gather     → 按实际 token ID 取出对应的 log prob
    per_token_logps.append(
        torch.gather(
            logits_row.log_softmax(dim=-1),  # [seq_len, V]
            dim=1,
            index=ids_row.unsqueeze(1)       # [seq_len, 1] 要取的位置
        ).squeeze(1)
    )

# 步骤 4：拼成 [B, R] 返回
return torch.stack(per_token_logps)
```

---

## 工厂函数 — 一行代码创建引擎

```python
# 第 197-212 行
def create_rollout_engine(
    engine_type="torch",       # "torch" | "sglang"
    policy_model=None, tokenizer=None,
    sglang_base_url=None, sglang_model_path=None, sglang_shared_path=None,
) -> RolloutEngine:
    if engine_type == "torch":
        return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)
    elif engine_type == "sglang":
        return SGLangRolloutEngine(sglang_base_url, sglang_model_path, sglang_shared_path)
```

使用示例：
```python
# Torch 引擎（默认）
engine = create_rollout_engine("torch", model, tokenizer)

# SGLang 引擎（服务需已运行）
engine = create_rollout_engine("sglang",
    sglang_base_url="http://localhost:8998",
    sglang_model_path="./minimind-3",
    sglang_shared_path="./sglang_ckpt")
```

---

## 我该用哪个引擎？

```
┌──────────────────────────────────────────────┐
│               你是什么情况？                   │
│                                               │
│  刚学习 / 调试 / 单卡训练                      │
│        ↓                                      │
│  TorchRolloutEngine                           │
│  ✓ 零配置    ✓ 易调试   ✗ 大批量慢            │
│                                               │
│  大规模训练 / 多卡 / 追求速度                  │
│        ↓                                      │
│  SGLangRolloutEngine                          │
│  ✓ 速度快   ✓ 热切换   ✗ 需启服务            │
└──────────────────────────────────────────────┘
```

| 场景 | 推荐 | 原因 |
|------|------|------|
| 第一次跑 PPO/GRPO | Torch | 少一个服务，少一个出错的可能 |
| 笔记本调试 | Torch | 无需额外启动任何东西 |
| 多卡分布式训练 | SGLang | 训练和推理分离，不抢 GPU |
| 大批量推理 | SGLang | 连续 batching 效率高 |
| 频繁更新权重 | SGLang | 热切换，不停服 |

---

## RL 训练中的完整流程

Rollout engine 在整个训练循环中的位置：

```
1. 加载 prompt → 从数据集取一批问题

2. Rollout ⭐ → engine.rollout(prompt_ids, ...)
   模型生成回复，返回 RolloutResult（回复 + log prob）

3. 计算 Reward → 用 Reward Model 或规则给回复打分

4. 计算 Loss → 用 log prob 算 ratio → PPO loss → 梯度下降

5. 同步权重 ⭐ → engine.update_policy(model)
   Torch: 直接替换 | SGLang: 存盘 + HTTP 通知

6. 回到步骤 1
```

步骤 2 和 5 就是 rollout_engine.py 的位置——**训练算法和模型推理之间的桥梁**。

---

## 小结

| 概念 | 一句话解释 |
|------|-----------|
| Rollout | 让模型自己生成回复，用于 RL 训练 |
| RolloutEngine 基类 | 统一接口，像 USB 标准一样解耦 |
| TorchRolloutEngine | 进程内推理，简单零配置 |
| SGLangRolloutEngine | 远程 HTTP 推理，高性能热切换 |
| compute_per_token_logps | 每个 token 对数概率，PPO ratio 的原材料 |
| RolloutResult | 引擎返回值容器 |
| hot-swap | 存盘 + HTTP 通知，不停服更新模型 |

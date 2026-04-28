# train_ppo.py 初学者完全指南

> 文件路径: `trainer/train_ppo.py`
> 代码总量: 约 444 行
> 预备知识: 了解 Python 基础，知道什么是训练和微调即可
> 阅读提示: 按照 "先懂概念 → 再看代码" 的顺序写

---

## 一、什么是 PPO？用一个现实类比来理解

PPO（Proximal Policy Optimization，近端策略优化）是强化学习领域最经典的算法之一。它的核心直觉可以用一个类比来理解。

### 学车类比

想象你正在学开车：

```
第一天:
  教练告诉你: "上次你转弯时打了 30 度的方向盘"（旧策略）
  你自己尝试: "这次我打了 45 度的方向盘"（新策略）
  结果: 转弯更平顺了 → 得分 +3 分！
  → 结论: 45 度比 30 度好，下次往 45 度调
  → ⚠️ 但不能太激进 → 下次试试 48 度就好，别跳到 90 度

第二天:
  教练又告诉你: "上次你用了 45 度"（新的旧策略）
  你自己尝试: "这次我打了 50 度"
  结果: 差点撞上护栏 → 得分 -2 分！
  → 结论: 50 度不好，往回缩一点
```

**这个过程的三个关键要素：**

1. **尝试**：在旧策略基础上做一点改变
2. **反馈**：得到"更好还是更差"的评分
3. **克制**：每次只改一点点，不能一步到位

这就是 PPO 的全部核心思想。

### 翻译到 LLM 对齐

```
学车的你      →  LLM 模型（Actor）
教练评分      →  Reward Model（奖励模型）
方向盘角度    →  生成每个 token 的概率
"只改一点点"  →  PPO 的裁剪机制（Clipping）
```

PPO 要做的事情：让模型在已有的 SFT 能力之上，通过"尝试 → 反馈 → 微调"的循环，逐渐学会生成更高质量、更符合偏好的回答。

---

## 二、PPO 的 4 模型架构

如果说 GRPO 是"精简三人组"，那 PPO 就是"豪华四重奏"。理解这四个模型的角色，是理解 PPO 代码的前提。

### 四个角色的职责

```
训练时，模型们是这样配合的:

┌──────────────────────────────────────────────────────────┐
│                                                         │
│  给模型一个 Prompt: "请解释量子力学"                      │
│                                                         │
│  ┌──────────┐                                           │
│  │  Actor   │  生成回答: "量子力学是研究微观世界..."        │
│  │ (演员)   │  它是唯一的主角，权重每步都在更新              │
│  └────┬─────┘                                           │
│       │ 拿着这个新回答，去找其他三个模型                    │
│       │                                                 │
│       ↓                                                 │
│  ┌──────────────┐     ┌──────────┐     ┌──────────┐      │
│  │   Critic     │     │Reference │     │  Reward  │      │
│  │  (评论家)    │     │  (老师)  │     │  (评委)  │      │
│  │              │     │          │     │          │      │
│  │ 估量当前状态  │     │ 这个新回  │     │ 给最终    │      │
│  │ 的"价值"     │     │ 答和老回  │     │ 回答打    │      │
│  │             │     │ 答有多    │     │ 一个总    │      │
│  │ 回答：7 分   │     │ 接近？    │     │ 分：8 分  │      │
│  │              │     │ KL=0.12  │     │          │      │
│  └──────────────┘     └──────────┘     └──────────┘      │
│                                                         │
└──────────────────────────────────────────────────────────┘
```

### 每个模型详解

| 角色 | 技术名 | 结构 | 训练状态 | 大白话 |
|--|--|--|--|--|
| **Actor** | Policy 网络 | MiniMindForCausalLM | **可训练** | 负责生成回答，不断更新自己 |
| **Critic** | Value 网络 | MiniMind + value_head | **可训练** | 估算当前回答状态有多"好" |
| **Reference** | Ref 网络 | MiniMindForCausalLM | 冻结 | SFT 版本的自己，用来防止走偏 |
| **Reward** | RM 网络 | 外部模型（InternLM2-1.8B-Reward） | 冻结 | 给最终回答打一个总分 |

### 为什么需要 Critic 模型？

**Critic 是 PPO 里的"价值评估员"。** 它的工作不是判断最终答案好不好，而是预测 **"生成到当前这一步，最终大约能拿多少分"**。

这听起来有点反直觉。让我们用数字来说明：

```
假设模型正在生成一个 5 个 token 的回答:

Token 1: "量"     Critic: "嗯...目前走向不错，预测最终约 5 分"
Token 2: "子"     Critic: "继续看好，预测最终约 6 分"
Token 3: "力"     Critic: "有点偏了，预测最终约 3 分"
Token 4: "学"     Critic: "拉回来了，预测最终约 5 分"
Token 5: "。"     Critic: "就这样了，最终 5 分"
```

**为什么要逐 token 估计？** 因为奖励是最后才给的（整个回答生成完才打分）。但训练时我们需要告诉模型"这个 token 生成得好不好"。Critic 就提供了这个逐步骤的价值估计。

**这就是 PPO 比 GRPO 更重但也更精细的原因：** Critic 可以做到逐 token 级别的微调，而 GRPO 只能在回答级别比较。

---

## 三、Reward Model（奖励模型）详解：它是怎么来的？

这是很多初学者最困惑的地方：PPO 里的奖励信号到底从哪来？奖励模型本身又是怎么训练出来的？

### 3.1 MiniMind 使用的 Reward Model：InternLM2-1.8B-Reward

MiniMind 没有自己从头训练一个 Reward Model，而是直接使用了上海人工智能实验室（Shanghai AI Laboratory）开源的 **InternLM2-1.8B-Reward**。这是一个专门为"给回答打分"而训练的模型。

| 属性 | 详情 |
|--|--|
| **模型名** | `internlm/internlm2-1_8b-reward` |
| **参数量** | ~1.8B（约 18 亿参数） |
| **基座模型** | InternLM2-Chat-1.8B-SFT |
| **架构** | LLaMA 风格 Transformer（RMSNorm + SwiGLU + RoPE） |
| **输出** | 单一标量分数（float），表示回答质量 |
| **语言** | 中英双语 |
| **开源地址** | [HuggingFace](https://huggingface.co/internlm/internlm2-1_8b-reward) |
| **论文** | [arXiv:2403.17297](https://arxiv.org/abs/2403.17297) |
| **许可证** | Apache-2.0（代码）；模型权重学术免费，商用需申请 |

### 3.2 Reward Model 是怎么训练出来的？

Reward Model 的训练分两步：先有人类偏好数据，再用这些数据训练一个打分模型。

#### 第一步：收集偏好数据

```
对于同一个问题，给出两个回答（chosen / rejected）:

问题: "请解释什么是黑洞"

回答 A (chosen ✓):
  "黑洞是一种天体，它的引力场极其强大，连光都无法逃脱..."
  → 人类标注员认为：准确、完整、有帮助

回答 B (rejected ✗):
  "黑洞就是一个洞，很黑的那种"
  → 人类标注员认为：敷衍、不准确、没有帮助

这样的 (问题, chosen, rejected) 三元组就是一条偏好数据
```

InternLM2-1.8B-Reward 的训练数据包含 **超过 240 万条偏好对**，来源包括：

| 数据来源 | 说明 |
|--|--|
| **人工标注数据** | 专业标注员对回答进行排序和比较 |
| **AI 合成数据** | 用强模型生成好/坏回答对，再经过筛选 |
| **覆盖领域** | 对话、写作、诗歌、摘要、编程、数学、格式化输出 |
| **语言分布** | 中文 + 英文，双语均衡 |
| **偏好维度** | 有用性（helpfulness）和无害性（harmlessness）均衡覆盖 |

#### 第二步：Reward Model 的训练过程

Reward Model 的核心训练思路是 **排序学习（Learning to Rank）**：

```
训练目标: 让模型学会 "给好回答打高分，给差回答打低分"

Loss 函数: 改进的 Ranking Loss（灵感来自 Focal Loss）

  L = -log(σ(r_chosen - r_rejected)) × 难度衰减系数

  其中:
  - r_chosen: 模型给"好回答"的打分
  - r_rejected: 模型给"差回答"的打分
  - σ: sigmoid 函数，把差值映射到 (0, 1)
  - 难度衰减系数: 对容易区分的样本降低权重，对难区分的样本加大权重
```

**InternLM2 的独特设计——条件式奖励模型（Conditional Reward Model）：**

传统做法（如 LLaMA 2）会训练多个 Reward Model，分别负责不同维度（有用性、安全性等）。InternLM2 的创新是只用**一个模型**，通过**不同的系统提示（System Prompt）** 来切换评分维度：

```
系统提示 1: "请评估这个回答的有用性和准确性"
  → RM 关注: 信息完整度、事实准确性、逻辑清晰度

系统提示 2: "请评估这个回答的安全性和无害性"
  → RM 关注: 是否包含有害内容、偏见、不当建议

同一个模型，不同的"评判角度"
```

这种设计的好处是：一个模型就能覆盖多个偏好维度，节省显存和部署成本。

#### 训练流程总结

```
InternLM2-Chat-1.8B-SFT (基座)
        │
        ▼  加载预训练权重
  ┌──────────────────────┐
  │  添加 Score Head      │
  │  (hidden_size → 1)   │
  └──────┬───────────────┘
         │
         ▼  用 240 万偏好对训练
  ┌──────────────────────────────────────────┐
  │  训练数据: (query, chosen, rejected)      │
  │  Loss: Ranking Loss + Focal 难度衰减      │
  │  多维度: 条件式 System Prompt 切换         │
  │  工具: XTuner 框架                         │
  └──────┬───────────────────────────────────┘
         │
         ▼  训练完毕
  InternLM2-1.8B-Reward (冻结，直接使用)
```

### 3.3 Reward Model 在 MiniMind 中怎么加载？

在 `trainer/trainer_utils.py` 中有一个封装类 `LMForRewardModel`：

```python
class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        # 用 HuggingFace 的 AutoModel 加载，trust_remote_code=True 是必须的
        # 因为 InternLM2 使用了自定义模型代码
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()  # 推理模式，不训练
```

**关键点：**
- `trust_remote_code=True`：InternLM2 在 HuggingFace 上发布了自定义的模型代码（包含 `get_score` 方法），必须信任远程代码才能加载
- `torch.float16`：用半精度加载以节省显存（RM 有 1.8B 参数，float16 约占 3.6GB 显存）
- `.eval()`：冻结为推理模式，不参与梯度计算

### 3.4 Reward Model 是怎么打分的？

`LMForRewardModel.get_score()` 方法的完整流程：

```python
@torch.no_grad()
def get_score(self, messages, response):
    # 1. 拼接历史对话为单一文本
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
    last_query = messages[-1]['content'] if messages else ""
    
    # 2. 构造评分格式：把多轮对话压缩成 "历史 + 当前问题" 的形式
    message_context = (
        f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" 
        if history_text else last_query
    )
    
    # 3. 组装成 InternLM2-Reward 期望的格式
    eval_messages = [
        {"role": "user", "content": message_context},    # 问题
        {"role": "assistant", "content": response}        # 回答
    ]
    
    # 4. 调用 InternLM2 内置的 get_score 方法
    score = self.model.get_score(self.tokenizer, eval_messages)
    
    # 5. 裁剪到 [-3.0, +3.0]，防止极端值
    return max(min(score, 3.0), -3.0)
```

**为什么要把历史对话拼成一条消息？**

InternLM2-Reward 的 `get_score` API 接受的是标准的 `[user, assistant]` 二元消息格式。但 MiniMind 的 prompt 可能包含多轮对话和系统提示。所以封装层把所有历史压缩成一个 user 消息，加上中文分隔符"以上是对话历史。我的新问题是："，确保 RM 能正确理解上下文。

### 3.5 InternLM2-Reward 的 API 全貌

InternLM2-Reward 实际上提供了四个便捷方法：

| 方法 | 说明 | MiniMind 是否使用 |
|--|--|--|
| `model.get_score(tokenizer, chat)` | 给一个对话打分，返回 float | **是** |
| `model.get_scores(tokenizer, [chat1, chat2, ...])` | 批量打分 | 否（逐条调用） |
| `model.compare(tokenizer, chat1, chat2)` | 比较两个回答谁更好 | 否 |
| `model.rank(tokenizer, [chat1, ...])` | 对多个回答排序 | 否 |

MiniMind 只使用了最基础的 `get_score`。如果想优化性能，可以改用 `get_scores` 进行批量打分，减少推理次数。

### 3.6 InternLM2-Reward 的评测性能

在 RewardBench 基准上的表现：

| 模型 | 总分 | Chat | Chat Hard | Safety | Reasoning |
|--|--|--|--|--|--|
| InternLM2-**1.8B**-Reward | **80.6** | 95.0 | 58.1 | 81.8 | 87.4 |
| InternLM2-7B-Reward | 86.6 | 98.6 | 66.7 | 88.3 | 92.8 |
| InternLM2-20B-Reward | 89.5 | 98.6 | 74.1 | 89.4 | 95.7 |

MiniMind 选用 1.8B 版本是在**打分质量**和**显存占用**之间的折中。在 3090（24GB）上同时运行 4 个 MiniMind 模型（~64M × 4）和 1 个 InternLM2-1.8B-Reward（~3.6GB），显存压力可控。

### 3.7 为什么不自己训练 Reward Model？

| 方案 | 优点 | 缺点 |
|--|--|--|
| **自训 RM** | 完全可控，可以针对领域定制 | 需要大量人类偏好标注数据，训练成本高 |
| **外部 RM**（MiniMind 方案） | 开箱即用，240 万偏好对训练，质量有保障 | RM 和 Actor 大小不匹配（1.8B vs 64M），可能存在分布差异 |

MiniMind 选用外部 RM 的原因：
1. **数据成本**：标注偏好数据极其昂贵，不适合教学项目
2. **质量保证**：InternLM2-1.8B-Reward 经过大规模验证
3. **教学聚焦**：项目重点是教会读者理解 PPO 算法，而非 RM 训练
4. **中文支持**：InternLM2 对中文的支持很好，适合中文 LLM 项目

---

## 四、GAE 优势估计：Critic 是"分数预测员"

GAE（Generalized Advantage Estimation，广义优势估计）是 PPO 的核心计算模块。理解它需要先理解一个概念。

### 什么是"优势"？

```
假设 Critic 预测 "当前状态值 5 分"
但最终实际得了 8 分

优势 = 实际得分 - 预测得分 = 8 - 5 = +3

+3 的优势意味着: 这个决策比预期的好了 3 分 → 应该加强
```

**优势（Advantage）** 回答的问题是：**这个行为比平常水平好还是差？**

### GAE 的直觉

GAE 的精妙之处在于：它不是简单地用 `当前奖励 - 预测值`，而是**把未来所有步骤的预期偏差都考虑进来**。

```
一个回答生成的 token 序列:

Token t:   即时奖励 r_t = 0 (中间 token 没有即时分)
           Critic 预测 V(s_t) = 5
           Critic 预测 V(s_{t+1}) = 6

           TD 误差 δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
                       = 0 + 1.0 * 6 - 5 = +1

           GAE A_t = δ_t + (γλ) * δ_{t+1} + (γλ)² * δ_{t+2} + ...

           （把所有未来步骤的偏差加权累加进来）
```

**γ (gamma) 和 λ (lam) 的直觉：**

```
γ = 1.0（不折扣）: 未来的偏差和当前的偏差同等重要
γ = 0.5（重度折扣）: 只关心眼前这一步，长远的不重要

λ = 0.95（高权重）: 更依赖 Critic 的预测（低方差，但可能有偏差）
λ = 1.0（完全蒙特卡洛）: 完全相信最终结果（无偏差，但高方差）

MiniMind 的默认选择:
  γ = 1.0  → 文本生成没有"时间偏好"，最后一个 token 的奖励
             和第一个 token 的贡献同等重要
  λ = 0.95 → 稍稍依赖 Critic 预测，避免完全蒙特卡洛的高方差
```

### 为什么 gamma=1.0（无折扣）？

在游戏等 RL 场景中，gamma 通常设为 0.99。意思是"未来的奖励不如现在重要"。但文本生成不一样：

```
游戏里: 现在吃金币 → 立刻爽，10步后吃金币 → 可能游戏已经结束了
文本生成: 最后一个 token 的奖励和第一个 token 同样重要
          "量子力学" 这句话，少了任何一个字都不完整
```

所以文本生成中 `gamma=1.0`，所有 token 的贡献一视同仁。

### 奖励是怎么分配到 token 序列上的？

这是一个关键细节。外部奖励（RM 打分 + 规则奖励）是**整个回答级别**的，但 PPO 需要**逐 token** 的信号。MiniMind 的做法是：

```python
token_rewards = torch.zeros_like(old_resp_logp)   # [B, R]，全零
last_idx = resp_lengths - 1                        # 每个回答最后一个 token 的位置
token_rewards[torch.arange(B), last_idx] += rewards  # 只在最后一个 token 放奖励
```

```
假设回答有 5 个 token，总奖励 = 2.5:

token_rewards = [0.0, 0.0, 0.0, 0.0, 2.5]
                                       ↑ 只有这里有值

然后 GAE 把这个奖励"反向传播"给前面的 token:
  A_4 = δ_4 = 2.5 + γ·0 - V_4        (最后一步，nv=0)
  A_3 = δ_3 + γλ·A_4                   (从后往前累积)
  A_2 = δ_2 + γλ·A_3
  ...
```

这就是"GAE 把末尾奖励往前传播"的含义。Critic 预测的 V(s_t) 在这个过程中起到了"基线"的作用——它帮助区分"这个 token 确实好"和"只是运气好（最终奖励高但跟这个 token 关系不大）"。

### 代码实现解读

```python
# GAE 计算（从后往前遍历序列）
gen_len = old_resp_values.size(1)
lastgaelam = torch.zeros(B, device=args.device)
advs_rev = []
for t in reversed(range(gen_len)):
    nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0  # 下一时刻价值
    delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
    lastgaelam = delta + args.gamma * args.lam * lastgaelam
    advs_rev.append(lastgaelam)
advantages = torch.stack(advs_rev[::-1], dim=1)  # 反转回来
```

**关键细节：**

1. **从后往前遍历**：因为 `A_t = δ_t + γλ * A_{t+1}`，后面的优势要先算好，才能用来算前面的
2. **最后一个 token 的特殊值**：`nv = 0.0`，因为回答结束后没有更多价值了
3. **reward 只在最后一个 token 上有值**：中间 token 的 `token_rewards` 为 0，只有 EOS 位置被加上了最终的外部奖励分
4. **returns 的计算**：`returns = advantages + old_resp_values`，这是 Critic 的训练目标（实际回报）

### 优势归一化

```python
adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
adv_var = ((advantages - adv_mean) ** 2 * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask
```

算完 GAE 之后，还把优势做了**均值 0、方差 1 的归一化**。这和 GRPO 做 z-score 的原理是一样的——保证优势分布稳定，防止某些 batch 的优势值特别大导致训练不稳定。

**注意 mask 的使用：** 归一化只在 `resp_policy_mask`（有效回答 token）范围内进行，padding 位置不参与统计。

---

## 五、PPO 裁剪目标函数（Clipped Objective）详解

这是 PPO 训练的核心公式，也是 `PPO` 这个名字中 `P`（Proximal / 近端）的出处。

### 公式全貌

```python
ratio = torch.exp(log_ratio)  # ratio = π_new(a|s) / π_old(a|s)
policy_loss = torch.max(
    -advantages * ratio,
    -advantages * torch.clamp(ratio, 1 - ε, 1 + ε)
)
```

> 注意这里用了 `max` 而不是 GRPO 的 `min`，因为 GRPO 的损失带负号后取了 `min`，这里 PPO 直接用了 `max(-x, -y) = -min(x, y)` 的等价形式。

### 用具体数字走一遍

假设：
- 优势 A = +2.0（这个回答比预期好）
- 裁剪参数 ε = 0.2
- `ratio = 3.0`（新模型对这个 token 的概率估计是旧模型的 3 倍）

```
未裁剪项:  -A * ratio  = -2.0 * 3.0 = -6.0
裁剪项:    ratio 被 clamp 到 [0.8, 1.2] → clamped = 1.2
           -A * clamped = -2.0 * 1.2 = -2.4

取 max:    max(-6.0, -2.4) = -2.4
```

**含义解读：**

```
未裁剪的值 -6.0 表示: "因为 ratio=3 很高，大力惩罚"（-(-6.0) = 大力鼓励）
但裁剪把它限制到 -2.4: "别太激动，适度就行"

为什么要限制？
  如果 ratio=3 完全是"运气好"（偶然碰到一个高分回答），
  那 -6.0 的梯度会让模型疯狂提高这个 token 的概率，
  可能下批就因为过拟合而搞砸了。

  裁剪到 -2.4: "你确实做得不错，但我们谨慎一点，只更新这么多。"
```

再看一个反方向的例子。假设优势 A = -2.0（这个回答比预期差），ratio = 0.3：

```
未裁剪项:  -A * ratio  = -(-2.0) * 0.3 = 0.6
裁剪项:    ratio 被 clamp 到 [0.8, 1.2] → clamped = 0.8 (下界)
           -A * clamped = -(-2.0) * 0.8 = 1.6

取 max:    max(0.6, 1.6) = 1.6
```

**含义解读：**

```
新模型降低了这个差回答的概率（ratio=0.3），这是好事。
未裁剪项 0.6: "你降低了差评的概率 → 小幅奖励"
裁剪项 1.6:    "降得太低了，应该更狠一点 → 更大奖励"
最终取 1.6:    鼓励模型更坚决地避免这个差的回答
```

**裁剪的双面作用：**

```
优势为正（A > 0）时:
  ratio > 1+ε → 用裁剪值 → 防止对"好回答"学太猛
  1-ε < ratio <= 1+ε → 用原值 → 正常学习
  
优势为负（A < 0）时:
  ratio < 1-ε → 用裁剪值 → 鼓励模型更狠地避开"差回答"
  ratio >= 1-ε → 用原值 → 正常学习
```

### 为什么叫 "Proximal"（近端）？

因为裁剪保证了新旧策略之间的比率不会超过 `[1-ε, 1+ε]`。也就是说，**新旧策略必须足够"近"**——每次更新只能走一小步。这就是 "Proximal Policy Optimization" 中 "Proximal" 的含义。

### clipfrac 监控指标

代码中还计算了 `clipfrac`：

```python
clipfrac = ((((ratio - 1.0).abs() > args.clip_epsilon).float() * resp_policy_mask[inds]).sum()
            / resp_policy_mask[inds].sum().clamp(min=1))
```

这是"被裁剪的 token 比例"。如果 clipfrac 很高（比如 > 0.3），说明新旧策略已经差距很大，更新步幅可能太激进了。如果 clipfrac 接近 0，说明策略几乎没变，学习可能太保守。**理想范围大约在 0.05~0.20**。

---

## 六、KL 散度的双重机制

PPO 里有**两种不同的 KL 相关机制**，作用对象和目的完全不同。很多初学者容易混淆，这里详细区分。

### 机制一：KL 早停（Early Stopping）

```python
log_ratio = mb_resp_logp - old_resp_logp
approx_kl = (0.5 * (log_ratio ** 2) * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)

# 同步各卡 approx_kl（防止 DDP 死锁）
approx_kl_val = approx_kl.detach().clone()
if dist.is_initialized():
    dist.all_reduce(approx_kl_val, op=dist.ReduceOp.AVG)

if approx_kl_val > args.early_stop_kl:  # 默认阈值 0.25
    stop_ppo = True
```

| 属性 | 说明 |
|--|--|
| **参考对象** | 旧策略（本轮 rollout 开始时的 Actor 权重） |
| **计算公式** | `0.5 * (log(π_new/π_old))²`（二阶近似） |
| **作用** | 如果**同一批数据上** PPO 更新了太多轮，策略偏离太大，**立刻停止** |
| **参数** | `early_stop_kl = 0.25` |
| **何时触发** | 在 mini-batch PPO 更新循环的每个 step 都检查 |

**直觉**：PPO 会在同一批 rollout 数据上多次更新（`ppo_update_iters=2`）。如果更新了一轮后策略已经变化很大，第二轮就不更新了。

### 机制二：KL 惩罚（KL Penalty against Reference）

```python
kl_ref_penalty = ((torch.exp(ref_resp_logp - mb_resp_logp) 
                   - (ref_resp_logp - mb_resp_logp) - 1.0)
                  * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
policy_loss = (... policy_loss部分 ... + args.kl_coef * kl_ref_penalty)
```

| 属性 | 说明 |
|--|--|
| **参考对象** | Reference 模型（SFT 版本，整个训练过程冻结不变） |
| **计算公式** | `E[exp(log_ref - log_π) - (log_ref - log_π) - 1]`（Schulman KL 估计器） |
| **作用** | 持续惩罚当前策略相对于初始 SFT 状态的偏离 |
| **参数** | `kl_coef = 0.02` |
| **何时生效** | 始终参与 loss 计算，是 policy_loss 的一部分 |

**直觉**：无论 PPO 训练了多少步，都不能让模型忘记 SFT 阶段学会的东西。KL 惩罚就像一根"弹性绳"，把 Actor 拉向 Reference。

### 两种 KL 的对比

| | KL 早停 | KL 惩罚 |
|--|--|--|
| 参考对象 | 旧策略（本次 rollout 前的 policy） | Reference 模型（SFT 初始版） |
| 计算方式 | `0.5 * (log_ratio)²`（二阶近似） | `e^x - x - 1`（Schulman 估计器） |
| 作用范围 | 同一批数据内的多轮更新 | 整个训练过程的全局约束 |
| 作用方式 | 超阈则 **停止更新** | 作为 **loss 项** 持续生效 |
| 参数 | `early_stop_kl = 0.25` | `kl_coef = 0.02` |
| 防止什么 | 防止"在一批数据上过拟合" | 防止"整体训练偏离 SFT 太远" |

### 为什么早停时必须保持 forward-backward 循环？

```python
if stop_ppo:
    loss = (...) * 0.0   # 归零 loss
else:
    loss = (...) / args.accumulation_steps
loss.backward()  # 仍然执行 backward
```

这是 DDP（分布式训练）的硬性要求。在多 GPU 训练中，每张卡必须同步参与 forward 和 backward。如果某张卡因为 `approx_kl > threshold` 而提前 break，其他卡还在等待梯度同步，就会**死锁**。

所以代码做了两件事：
1. 用 `dist.all_reduce(approx_kl_val, op=dist.ReduceOp.AVG)` 确保所有卡**同时**判断是否早停
2. 即使早停，也走完 `loss * 0.0` → `loss.backward()` 的全流程，只是梯度为零，不影响参数

---

## 七、复合奖励机制

MiniMind 的奖励不是单纯依赖 Reward Model，而是一个**规则奖励 + RM 打分**的组合。

### 奖励计算的完整流程

```python
def calculate_rewards(prompts, responses, reward_model):
    rewards = torch.zeros(len(responses), device=args.device)
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        # ① 长度奖励: 回答太短或太长都扣分
        rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5

        if '</think>' in response:
            # ② 思考长度奖励: 思考内容要适中
            thinking_content, answer_content = response.split('</think>', 1)
            rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
            # ③ 思考格式奖励: 恰好一个 </think> 标签
            rewards[i] += 0.25 if response.count('</think>') == 1 else -0.25
            answer = answer_content.strip()
        else:
            answer = response

        # ④ 重复惩罚: trigram 重复度
        rewards[i] -= rep_penalty(answer)  # 0 ~ -0.5

        # ⑤ RM 打分: 外部奖励模型的评价
        score = reward_model.get_score(messages, answer)  # -3.0 ~ +3.0
        reward_model_scores.append(score)
    
    rewards += reward_model_scores  # 加上 RM 分
    return rewards
```

### 各奖励项详解

| # | 奖励项 | 范围 | 条件 | 设计意图 |
|--|--|--|--|--|
| ① | 长度奖励 | +0.5 / -0.5 | 20~800 字符 | 避免模型"偷懒"（极短回答）或"灌水"（极长回答） |
| ② | 思考长度 | +1.0 / -0.5 | 20~300 字符的 thinking | 鼓励适度推理，不要空想也不要冗长 |
| ③ | 思考格式 | +0.25 / -0.25 | 恰好 1 个 `</think>` | 防止模型输出多个或零个思考标签 |
| ④ | 重复惩罚 | 0 ~ -0.5 | trigram 重复比例 | 惩罚重复说同样的话 |
| ⑤ | RM 打分 | -3.0 ~ +3.0 | 外部 RM 评价 | 整体回答质量的主要信号 |

**总奖励范围：** 约 `[-4.75, +4.75]`

### 重复惩罚函数详解

```python
def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())  # 分词：字母数字词 + 标点
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]  # 所有 trigram
    # 重复 trigram 数 / 总 trigram 数 × cap × 2，但不超过 cap
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0
```

**示例：**
```
text = "我喜欢苹果我喜欢苹果我喜欢苹果"
trigrams = [("我","喜欢","苹果"), ("喜欢","苹果","我"), ("苹果","我","喜欢"), ...]
重复 trigram 很多 → 返回接近 0.5 的惩罚
```

### 为什么需要规则奖励？

纯粹依赖 RM 打分有几个问题：

1. **RM 可能被"骗"**：模型可能学会输出一些 RM 给高分但实际质量不高的"讨好"文本
2. **格式要求难从 RM 学到**：thinking 标签的格式、长度约束这些结构性要求，用规则直接编码更可靠
3. **重复问题**：RM 有时对重复不敏感，但用户体验极差

规则奖励 + RM 打分的组合，是 RLHF 工程实践中非常常见的做法。

---

## 八、Critic 模型详解

Critic 是 PPO 独有的组件，也是 PPO 和 GRPO 的核心差异。

### 模型结构

```python
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        # 替换 lm_head (vocab_size维度输出) 为 value_head (单一标量输出)
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids, attention_mask, **kwargs):
        # 复用 Transformer body 提取特征
        outputs = self.model(input_ids, attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])  # 应用 RMSNorm
        # value_head: [B, T, hidden_size] → [B, T, 1] → [B, T]
        return self.value_head(hidden_states).squeeze(-1)
```

**结构对比：**

```
Actor (MiniMindForCausalLM):
  embed_tokens → [N层 Transformer] → RMSNorm → lm_head(hidden → vocab_size)
                                                 ↑ 输出每个 token 的概率

Critic (CriticModel):
  embed_tokens → [N层 Transformer] → RMSNorm → value_head(hidden → 1)
                                                 ↑ 输出每个位置的价值估计
```

### Critic 的初始化

```python
# 从 SFT 权重加载（与 Actor 使用相同的初始权重）
ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
state_dict = torch.load(ckp, map_location=args.device)
critic_model = CriticModel(lm_config)
critic_model.load_state_dict(state_dict, strict=False)  # strict=False!
```

**关键细节：**

- `strict=False`：因为权重文件里有 `lm_head.weight`，但 CriticModel 用的是 `value_head.weight`。`strict=False` 让 PyTorch 忽略这个不匹配
- Transformer body 的权重被正确加载（共享了 SFT 的表征能力）
- `value_head`（`nn.Linear(768, 1)`）是**随机初始化**的，需要在训练中从零学习

### Critic 的学习率为什么更高？

```python
actor_optimizer = optim.AdamW(actor_model.parameters(), lr=3e-7)   # Actor: 3e-7
critic_optimizer = optim.AdamW(critic_model.parameters(), lr=5e-7)  # Critic: 5e-7
```

Critic 的学习率是 Actor 的约 1.67 倍。原因：
1. `value_head` 从零初始化，需要更快地收敛到合理的值
2. Critic 的任务（预测回报值）相对简单，不容易过拟合
3. Actor 已经有了 SFT 基础，更新需要更谨慎

### Value Loss 的裁剪

```python
value_loss = 0.5 * (torch.max(
    (mb_resp_values - returns) ** 2,                                    # 未裁剪
    (torch.clamp(mb_resp_values, 
                 old_resp_values - args.cliprange_value,
                 old_resp_values + args.cliprange_value) - returns) ** 2  # 裁剪
) * resp_value_mask).sum() / resp_value_mask.sum().clamp(min=1)
```

和 Policy Loss 类似，Value Loss 也做了裁剪。防止 Critic 的预测值在一次更新中变化过大，保持训练稳定。`cliprange_value=0.2` 意味着新的 Critic 预测不能比旧预测偏离超过 ±0.2。

---

## 九、训练数据：RLAIF Dataset

### 数据格式

PPO 训练使用 `RLAIFDataset`，数据存储在 `dataset/rlaif.jsonl` 中：

```json
{"conversations": [
    {"role": "user", "content": "什么是量子纠缠？"},
    {"role": "assistant", "content": "量子纠缠是...（这部分会被忽略）"}
]}
```

**重要：只有 prompt 被使用，assistant 的回答被完全丢弃。** 因为 PPO 的核心是让模型自己生成回答（rollout），然后通过奖励信号来学习。

### 数据处理流程

```python
class RLAIFDataset(Dataset):
    def create_chat_prompt(self, conversations):
        # 1. 预处理：20% 概率随机添加系统提示（中英各5条）
        conversations = pre_processing_chat(conversations)
        
        # 2. 按概率决定是否开启思考模式（默认 90%）
        use_thinking = random.random() < self.thinking_ratio
        
        # 3. 用 HF tokenizer 的 chat template 生成 prompt
        return self.tokenizer.apply_chat_template(
            conversations[:-1],          # 丢弃最后一条（assistant 的回答）
            tokenize=False,
            open_thinking=use_thinking,   # 是否在 prompt 末尾加 <think> 标签
            add_generation_prompt=True    # 添加生成触发标记
        )
    
    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])
        return {'prompt': prompt, 'answer': ""}  # answer 始终为空
```

**最终 prompt 格式示例（90% 概率的 thinking 模式）：**

```
<|im_start|>user
什么是量子纠缠？<|im_end|>
<|im_start|>assistant
<think>
```

**10% 概率的直接回答模式：**

```
<|im_start|>user
什么是量子纠缠？<|im_end|>
<|im_start|>assistant
```

### thinking_ratio 参数的意义

`thinking_ratio=0.9` 意味着 90% 的 prompt 会带上 `<think>` 标签。这是一种**混合训练策略**：
- 大部分时间模型学习"先思考再回答"的模式
- 少部分时间模型学习直接回答
- 训练后模型能同时支持两种模式

---

## 十、完整训练流程

### 初始化阶段

```
1. init_distributed_mode()        → DDP 初始化（多GPU时）
2. MiniMindConfig                 → 模型配置（768维、8层、GQA等）
3. lm_checkpoint (检查模式)       → 检查是否有续训 checkpoint
4. 混合精度设置                    → bfloat16 或 float16
5. wandb/swanlab 初始化           → 训练日志（可选）
6. init_model() × 2              → Actor + Reference（都从 full_sft 加载）
7. Reference 冻结                 → .eval().requires_grad_(False)
8. CriticModel                    → 从 full_sft 加载 body，value_head 随机初始化
9. LMForRewardModel               → InternLM2-1.8B-Reward（外部奖励模型）
10. create_rollout_engine()       → torch 原生 或 SGLang HTTP 服务
11. RLAIFDataset                  → 训练数据加载
12. AdamW × 2                     → Actor / Critic 分别的优化器
13. CosineAnnealingLR × 2        → 学习率调度器
14. 续训状态恢复                   → 加载所有 checkpoint 数据
15. torch.compile（可选）          → 编译加速
16. DDP 包装                      → DistributedDataParallel
17. rollout_engine.update_policy() → 用初始 Actor 权重初始化引擎
```

### 每轮训练循环

```
┌──────────────────────────────────────────────────┐
│  1. Rollout: Actor 对每个 prompt 生成 1 个回答    │
│     (temperature=0.8, 保证生成多样性)              │
│                                                  │
│  2. Reward: 计算复合奖励                          │
│     规则奖励 (长度/思考/格式/重复)                  │
│     + InternLM2-Reward 打分                       │
│                                                  │
│  3. Mask 构建:                                    │
│     区分 prompt / response / padding / EOS        │
│     resp_policy_mask, resp_value_mask             │
│                                                  │
│  4. Rollout 推理 (no_grad):                       │
│     Critic → old_values [B, R]                    │
│     Actor  → old_logp [B, R]                      │
│     Ref    → ref_logp [B, R]                      │
│     token_rewards[最后一个token] += 外部奖励        │
│                                                  │
│  5. GAE 优势估计:                                 │
│     倒序遍历 → TD 误差 → 累积优势                  │
│     returns = advantages + old_values             │
│     归一化 advantages                              │
│                                                  │
│  6. Mini-batch PPO 更新 (ppo_update_iters=2):     │
│     for epoch in range(2):                        │
│       随机打乱 batch 索引                          │
│       for mini_batch:                             │
│         a. 重跑 Actor → 新 logp                   │
│         b. 重跑 Critic → 新 values                │
│         c. 计算 approx_kl → 检查早停              │
│         d. 计算 clipped policy_loss               │
│         e. 计算 KL penalty (vs Reference)         │
│         f. 计算 clipped value_loss                │
│         g. loss = policy + vf_coef*value + kl     │
│         h. loss.backward()                        │
│         i. grad clip → optimizer.step()           │
│                                                  │
│  7. 日志 & 保存:                                  │
│     Log: reward, kl_ref, approx_kl, clipfrac...  │
│     Save: actor 权重 + 完整续训 checkpoint         │
│     更新 rollout engine 策略                       │
└──────────────────────────────────────────────────┘
```

### 数据流图

```
                  prompt (str)
                      │
                      ▼
          ┌──── Rollout Engine ────┐
          │   Actor.generate()     │
          │   temp=0.8, top-p      │
          └────────┬───────────────┘
                   │ gen_out [B, P+R]
                   │ responses_text [B]
                   ▼
          ┌──── calculate_rewards ────┐
          │  规则: 长度/思考/重复      │
          │  RM: InternLM2 打分       │
          └────────┬──────────────────┘
                   │ rewards [B]
                   ▼
    ┌─── no_grad rollout inference ───┐
    │  Critic(gen_out) → values       │
    │  Actor(gen_out)  → old_logp     │
    │  Ref(gen_out)    → ref_logp     │
    └──────────┬──────────────────────┘
               │
               ▼
    ┌───── GAE computation ──────┐
    │  token_rewards[last] = R   │
    │  reverse scan → δ → A → N │
    │  returns = A + V           │
    └──────────┬─────────────────┘
               │ advantages [B, R]
               │ returns [B, R]
               ▼
    ┌── mini-batch PPO update ──┐
    │  for epoch in 2:          │
    │    shuffle → split → step │
    │    policy_loss + value_loss│
    │    + KL penalty           │
    │    backward → clip → step │
    └───────────────────────────┘
```

---

## 十一、Loss 计算完整公式

### 总 Loss 公式

```
Total Loss = Policy Loss + vf_coef × Value Loss + aux_loss
             ────────────────────────────────────────────
                         accumulation_steps
```

其中 `aux_loss` 是 MoE 的负载均衡损失（不使用 MoE 时为 0）。

### Policy Loss 展开

```
Policy Loss = Clipped Surrogate + kl_coef × KL Penalty

Clipped Surrogate:
  ratio = exp(log_π_new - log_π_old)
  surr1 = -advantages × ratio
  surr2 = -advantages × clamp(ratio, 1-ε, 1+ε)
  = mean(max(surr1, surr2))     // 对有效 token 求均值

KL Penalty (Schulman 估计器):
  kl_ref = mean(exp(log_ref - log_new) - (log_ref - log_new) - 1)
  
Total Policy Loss = Clipped Surrogate + 0.02 × kl_ref
```

### Value Loss 展开

```
v_unclipped = (V_new - returns)²
v_clipped = (clamp(V_new, V_old - 0.2, V_old + 0.2) - returns)²
Value Loss = 0.5 × mean(max(v_unclipped, v_clipped))
```

### 默认系数

```
vf_coef = 0.5    →  Value Loss 的权重
kl_coef = 0.02   →  KL 惩罚的权重
ε = 0.2          →  Policy 裁剪范围
cliprange_value = 0.2  →  Value 裁剪范围
```

---

## 十二、Rollout Engine（推理引擎）

PPO 训练的一个独特需求是：每一步都要让 Actor 生成回答（rollout）。这个过程可以很慢，所以 MiniMind 提供了两种引擎。

### TorchRolloutEngine（进程内推理）

```python
class TorchRolloutEngine:
    def rollout(self, prompt_ids, attention_mask, num_generations, max_new_tokens, temperature):
        # 直接调用 model.generate()
        gen_ids = self.model.generate(
            prompt_ids, attention_mask, 
            max_new_tokens=max_new_tokens,
            temperature=temperature, do_sample=True
        )
        # 计算 per-token log-probs
        logps = compute_per_token_logps(self.model, gen_ids, n_keep, attention_mask)
        return RolloutResult(output_ids, completion_ids, per_token_logps, completions)
```

**优点**：简单，无需额外服务。**缺点**：慢（因为生成是自回归的，每次只能生成一个 token）。

### SGLangRolloutEngine（远程推理服务）

```python
class SGLangRolloutEngine:
    def rollout(self, prompt_ids, ...):
        # 发送 HTTP 请求到 SGLang 服务器
        response = requests.post(f"{self.base_url}/generate", json={
            "input_ids": ids_list, "sampling_params": {...}, "return_logprob": True
        })
        return RolloutResult(...)
    
    def update_policy(self, model):
        # 把最新的 Actor 权重保存到磁盘
        model.save_pretrained(shared_path)
        # 通知 SGLang 热加载新权重
        requests.post(f"{self.base_url}/update_weights_from_disk", json={"model_path": shared_path})
```

**优点**：SGLang 支持 continuous batching 和 PagedAttention，推理速度快很多。
**缺点**：需要额外启动 SGLang 服务，配置更复杂。

### 选择哪个？

```bash
# 简单开发/调试：用 torch 引擎
python -m trainer.train_ppo --rollout_engine torch

# 正式训练/追求速度：用 SGLang 引擎
# 先启动 SGLang 服务（另一个终端）
python -m sglang.launch_server --model-path ./model --port 8997
# 再启动训练
python -m trainer.train_ppo --rollout_engine sglang
```

---

## 十三、分布式训练（DDP）细节

### 基本设置

```python
local_rank = init_distributed_mode()  # NCCL 后端
setup_seed(42 + rank)                 # 每张卡不同的种子

# DDP 包装
actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}  # RoPE 不需要同步
actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
```

### 关键的 DDP 安全机制

1. **KL 早停同步**：`dist.all_reduce(approx_kl_val, op=dist.ReduceOp.AVG)` 确保所有卡对"是否早停"达成一致
2. **早停后仍执行 backward**：`loss * 0.0` 保证 DDP 通信不死锁
3. **RoPE 排除**：`freqs_cos`, `freqs_sin` 是预计算的常量，不需要跨卡同步

### 运行命令

```bash
# 单 GPU
python -m trainer.train_ppo --from_weight full_sft

# 多 GPU（4 卡）
torchrun --nproc_per_node=4 -m trainer.train_ppo --from_weight full_sft
```

---

## 十四、Checkpoint 与续训

### 保存内容

每 `save_interval` 步保存两个文件：

| 文件 | 内容 | 用途 |
|--|--|--|
| `out/ppo_actor_{dim}.pth` | 仅 Actor 权重（half精度） | 推理/部署 |
| `checkpoints/ppo_actor_{dim}_resume.pth` | 完整训练状态 | 续训 |

完整状态包含：
- Actor 模型权重 + 优化器状态 + 学习率调度器
- Critic 模型权重 + 优化器状态 + 学习率调度器
- epoch、step、world_size、wandb_id

### 续训机制

```bash
# 启用自动续训
python -m trainer.train_ppo --from_resume 1
```

续训时会：
1. 加载所有模型/优化器/调度器状态
2. 自动跳过已训练的 batch（`SkipBatchSampler`）
3. 如果 GPU 数量变化，自动调整 step 计数
4. wandb 日志自动接续

### 保存格式

```python
# Actor 权重保存（原子写入，防止中断导致损坏）
torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)

# 完整状态通过 lm_checkpoint 保存
lm_checkpoint(lm_config, weight='ppo_actor', model=actor_model, optimizer=actor_optimizer,
              epoch=epoch, step=step, wandb=wandb,
              scheduler=actor_scheduler, critic_model=critic_model,
              critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
```

---

## 十五、梯度累积与学习率调度

### 梯度累积

```python
loss = (...) / args.accumulation_steps   # loss 除以累积步数
loss.backward()                           # 梯度累积

grad_accum_step += 1
if grad_accum_step % args.accumulation_steps == 0:
    clip_grad_norm_(actor_model.parameters(), args.grad_clip)   # 梯度裁剪
    clip_grad_norm_(critic_model.parameters(), args.grad_clip)
    actor_optimizer.step()                  # 更新参数
    critic_optimizer.step()
    actor_scheduler.step()                  # 更新学习率
    critic_scheduler.step()
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
```

**注意：** 循环结束后还有一个"尾部处理"，确保最后一批未满的梯度也被应用：

```python
if grad_accum_step % args.accumulation_steps != 0:
    # 强制执行一次 step，不丢弃尾部梯度
    clip_grad_norm_(...); optimizer.step(); ...
```

### 学习率调度

```python
# 总优化器步数的精确计算
mb_factor = max(1, ceil(batch_size / mini_batch_size))  # mini-batch 拆分倍数
total_optimizer_steps = ceil(iters × epochs × ppo_update_iters × mb_factor / accumulation_steps)

# 余弦退火调度
actor_scheduler = CosineAnnealingLR(
    actor_optimizer, T_max=total_optimizer_steps, eta_min=lr / 10
)
```

学习率从 `lr` 按余弦曲线衰减到 `lr / 10`：
- Actor: 3e-7 → 3e-8
- Critic: 5e-7 → 5e-8

---

## 十六、命令行参数速查

| 参数 | 默认值 | 含义 |
|--|--|--|
| `--batch_size` | 2 | 同时处理的 prompt 数量 |
| `--learning_rate` | 3e-7 | Actor 学习率（RL 阶段学习率很低） |
| `--critic_learning_rate` | 5e-7 | Critic 学习率 |
| `--clip_epsilon` | 0.2 | PPO 裁剪参数（策略比率限制在 [0.8, 1.2]） |
| `--vf_coef` | 0.5 | Value loss 在总 loss 中的权重 |
| `--kl_coef` | 0.02 | KL 惩罚系数（对 Reference 的偏离惩罚） |
| `--gamma` | 1.0 | GAE 折扣因子（文本生成不折扣） |
| `--lam` | 0.95 | GAE 平滑参数 |
| `--cliprange_value` | 0.2 | Value 函数裁剪范围 |
| `--ppo_update_iters` | 2 | 同一批 rollout 数据重复更新的次数 |
| `--early_stop_kl` | 0.25 | KL 早停阈值（超过停止更新） |
| `--mini_batch_size` | 2 | 每个 mini-batch 的大小 |
| `--max_seq_len` | 768 | Prompt 最长 |
| `--max_gen_len` | 1024 | 生成最长 |
| `--rollout_engine` | `"sglang"` | 生成引擎（"torch" 或 "sglang"） |
| `--reward_model_path` | `internlm2-1_8b-reward` | 外部奖励模型路径 |
| `--thinking_ratio` | 0.9 | 开启 thinking 模式的概率 |
| `--accumulation_steps` | 1 | 梯度累积步数 |
| `--grad_clip` | 1.0 | 梯度裁剪阈值 |
| `--save_interval` | 10 | 保存间隔（步数） |
| `--from_weight` | `full_sft` | 基于哪个权重训练 |
| `--from_resume` | 0 | 是否续训（0=否，1=是） |
| `--use_wandb` | False | 启用 wandb/swanlab 日志 |
| `--use_compile` | 0 | 启用 torch.compile 加速 |
| `--debug_mode` | False | 打印训练调试采样 |
| `--debug_interval` | 20 | 调试打印间隔 |
| `--dtype` | `bfloat16` | 混合精度类型 |

### 运行示例

```bash
# 基础运行
python -m trainer.train_ppo --from_weight full_sft

# 使用 torch 引擎（不需要 SGLang 服务）
python -m trainer.train_ppo --from_weight full_sft --rollout_engine torch

# 更保守的训练（更小的 KL 容忍度）
python -m trainer.train_ppo --early_stop_kl 0.1 --kl_coef 0.05

# 调试模式（打印每 20 步的 prompt/response/reward）
python -m trainer.train_ppo --debug_mode --debug_interval 20

# 多 GPU 训练
torchrun --nproc_per_node=4 -m trainer.train_ppo --from_weight full_sft

# 续训（自动检测 checkpoint）
python -m trainer.train_ppo --from_resume 1
```

---

## 十七、代码逐块阅读

### Critic 模型定义（第 36-48 行）

```python
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids, attention_mask, **kwargs):
        hidden_states = self.model.norm(self.model(input_ids, attention_mask, **kwargs)[0])
        return self.value_head(hidden_states).squeeze(-1)  # [B, T]
```

复用 Transformer body，只替换 lm_head 为 value_head。

### 初始化（第 375-384 行）

```python
actor_model, _ = init_model(lm_config, base_weight, device=args.device)
ref_model, _ = init_model(lm_config, base_weight, device=args.device)
ref_model = ref_model.eval().requires_grad_(False)
state_dict = torch.load(ckp, map_location=args.device)
critic_model = CriticModel(lm_config)
critic_model.load_state_dict(state_dict, strict=False)
```

4 个模型的初始化，和架构描述完全一致。

### GAE 计算（第 146-157 行）

```python
for t in reversed(range(gen_len)):
    nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
    delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
    lastgaelam = delta + args.gamma * args.lam * lastgaelam
    advs_rev.append(lastgaelam)
advantages = torch.stack(advs_rev[::-1], dim=1)
```

倒序遍历 → TD 误差 → GAE 累积 → 归一化（均值0/方差1）。

### Policy + Value Loss（第 198-210 行）

```python
policy_loss = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1-ε, 1+ε))
value_loss = 0.5 * torch.max((V - R)², (clamp(V) - R)²)
loss = policy_loss + vf_coef * value_loss + kl_coef * kl_penalty
loss.backward()
```

PPO clipped objective + Critic value loss + KL 惩罚。

---

## 十八、PPO 和 GRPO 的核心差异

| 维度 | PPO | GRPO |
|--|--|--|
| **模型数量** | 4 个（Actor + Critic + Ref + RM） | 3 个（Policy + Ref + RM） |
| **优势计算** | GAE（逐 token，需要 Critic） | 组内 z-score（回答级） |
| **优势精度** | 高（逐 token 反馈） | 中（回答整体反馈） |
| **生成数量** | 每个 prompt 生成 **1** 个回答 | 每个 prompt 生成 **多个** 回答 |
| **显存需求** | 高（4 个模型，其中 RM 有 1.8B 参数） | 低（3 个模型） |
| **代码行数** | ~444 行 | ~332 行 |
| **安全机制** | 裁剪 + KL 早停 + KL 惩罚 | 裁剪 + KL 惩罚 |
| **适用场景** | 精细控制、资源充足 | 资源受限、快速上手 |

**核心差异总结：** GRPO 通过"同一 prompt 的多个回答相互比较"来估计优势，不需要 Critic。PPO 通过 Critic 预测每步价值，用 GAE 做逐 token 级别的优势估计，更精细但更重。

---

## 十九、显存占用估算

在 3090（24GB）上的典型显存分布：

| 模型 | 参数量 | 精度 | 显存（近似） |
|--|--|--|--|
| Actor | ~64M | bfloat16 | ~128MB（+ 优化器 ~384MB） |
| Critic | ~64M | float32 | ~256MB（+ 优化器 ~384MB） |
| Reference | ~64M | bfloat16 | ~128MB |
| Reward (InternLM2-1.8B) | ~1.8B | float16 | ~3.6GB |
| 激活值 / 中间变量 | - | - | ~2-6GB |
| **合计** | | | **~7-11GB** |

这意味着在 3090 上跑 PPO 是可行的，但显存余量不算太大。如果 `batch_size` 或 `max_gen_len` 设得很大，可能会 OOM。

---

## 二十、学习建议

1. **先跑通 GRPO**（代码更简单，逻辑更直观）
2. **再学 PPO**，重点关注 GRPO **没有的东西**：Critic、GAE、Value Loss
3. **对比优势计算**：一个"和小组比"（GRPO），一个"Critic 预测偏差"（PPO）
4. **理解 Reward Model**：搜索 InternLM2-1.8B-Reward 的 HuggingFace 页面，试试它的 `get_score` API
5. **调参实验**：
   - 尝试 `gamma=0.99` 看折扣在文本生成中是否有害
   - 调整 `kl_coef` 看模型的"自由度"和"保守度"的权衡
   - 修改规则奖励的阈值，观察对生成行为的影响
6. **观察训练指标**：
   - `reward` 上升 → 模型在学习生成更好的回答
   - `kl_ref` 持续增大 → 模型在远离 SFT 初始状态，考虑增大 `kl_coef`
   - `clipfrac` 很高 → 更新步幅太大，考虑减小学习率
   - `approx_kl` 经常触发早停 → `ppo_update_iters` 可能设太大
7. **Debug 模式**：用 `--debug_mode` 实际查看模型的 prompt、response 和 reward，建立直观感受

PPO 大约 444 行代码，是四种 RL 方法中最复杂的一个。但它的核心思想——**在旧策略附近做小步更新，确保每一步都不会走太远**——是稳定训练 LLM 的通用原则。理解 PPO 后，再看 GRPO 会有一种"原来少了一个 Critic 也能做"的豁然开朗感。

---

## 附录A：Reward Model 训练数据来源参考

如果你想自己训练 Reward Model，以下是常见的偏好数据集：

| 数据集 | 规模 | 语言 | 说明 |
|--|--|--|--|
| [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) | ~170K | 英文 | Anthropic 的人类偏好对话数据 |
| [OpenAssistant oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | ~160K | 多语言 | 社区标注的多轮对话排名数据 |
| [Stanford SHP](https://huggingface.co/datasets/stanfordnlp/SHP) | ~385K | 英文 | Reddit 真实投票数据 |
| [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) | ~64K prompts | 英文 | GPT-4 评价的多模型回答 |
| InternLM2 私有数据 | ~2.4M | 中英 | 人工标注 + AI 合成，未公开 |

MiniMind 使用现成的 InternLM2-1.8B-Reward 而非自训，是教学项目中非常合理的工程决策。

## 附录B：PPO 相关论文

| 论文 | 说明 |
|--|--|
| [Schulman et al., 2017](https://arxiv.org/abs/1707.06347) | PPO 原始论文 |
| [Schulman et al., 2015](https://arxiv.org/abs/1506.02438) | GAE（广义优势估计）论文 |
| [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155) | InstructGPT，将 PPO 用于 LLM 对齐的里程碑 |
| [Cai et al., 2024](https://arxiv.org/abs/2403.17297) | InternLM2 技术报告（含 Reward Model 训练细节） |

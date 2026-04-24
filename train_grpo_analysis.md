# train_grpo.py 初学者完全指南

> 文件路径: `trainer/train_grpo.py`
> 代码总量: 约 332 行
> 预备知识: 了解 Python 基础，知道什么是训练和微调即可
> 阅读提示: 按照 "先懂概念 → 再看代码" 的顺序写

---

## 一、什么是对齐（Alignment）？SFT 和 RL 有什么区别？

训练一个大语言模型就像培养一个学生，需要经过三个阶段：

```
阶段1: 预训练 (Pretrain)
  → 读万卷书：在海量文本上学习 "世界的规律"
  → 学会语法、常识、推理模式
  → 但还不会对话，不知道怎么 "当助手"

阶段2: 监督微调 (SFT)
  → 名师示范：给它看 "好回答" 的样例
  → 学会 "用户问了什么 → 我应该怎么回答"
  → 但只会模仿，不会 "自己想清楚再回答"

阶段3: 强化学习 (RL)
  → 考试反馈：给它出题，打完分后告诉它 "这个比那个好"
  → 学会 "什么样的回答得分更高"
  → 学会推理、偏好对齐、自我纠错
```

**SFT 和 RL 的核心区别：**

| | SFT（监督微调） | RL（强化学习） |
|--|--|--|
| 训练目标 | 让模型"模仿好回答" | 让模型"追求高分数" |
| 数据格式 | 一问一答 | 一问多答 + 评分 |
| 学习方式 | 照着做 | 自己试，然后对比 |
| 学会的能力 | 遵循指令的格式 | 推理深度、偏好对齐 |
| 类比 | 抄作业 | 做题后看标准答案 |

SFT 让模型学会了"听话"——知道用户提问后该怎么回答。但它缺乏一个关键能力：**不知道自己的回答到底好不好**。RL 要补的就是这个短板：让模型学会自我评估，追求更好的回答质量。

---

## 二、GRPO 是什么？用最简单的方式理解

GRPO 全称 **Group Relative Policy Optimization**（组相对策略优化），是 DeepSeek 在 DeepSeek-Math 和 DeepSeek-R1 中使用的核心算法。

它的核心直觉非常好懂：**同一个问题，让模型写好几个答案，然后比较哪个更好。**

### GRPO 的工作流程（直觉版）

```
问题: "1+1=?"

模型生成4个答案:
  答案A: "2"           → 奖励: 3.0
  答案B: "1+1=3"      → 奖励: -1.0
  答案C: "等于二"      → 奖励: 2.0
  答案D: "42"         → 奖励: -2.0

组内平均分 = (3.0 - 1.0 + 2.0 - 2.0) / 4 = 0.5
组内标准差 = ... （计算波动大小）

然后算 z-score（标准化分数）:
  答案A 的优势: (3.0 - 0.5) / 标准差  >> 正数 → 鼓励！
  答案B 的优势: (-1.0 - 0.5) / 标准差  << 负数 → 抑制！
  答案C 的优势: (2.0 - 0.5) / 标准差   正数 → 鼓励！
  答案D 的优势: (-2.0 - 0.5) / 标准差  更负 → 强烈抑制！
```

**这就是 GRPO 的全部核心思想。** 不需要额外的价值网络，不需要 Critic 模型，只需要一组回答之间的相对排名就够了。

---

## 三、为什么 GRPO 不需要 Critic 模型？

这是 GRPO 最大的卖点。让我们对比一下 PPO 和 GRPO 的架构差异。

### PPO 需要 4 个模型

```
PPO 训练时的模型阵容:
┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────┐
│  Actor   │  │   Critic     │  │Reference │  │  Reward  │
│ (主角)   │  │ (价值评估员)  │  │(老版本)  │  │  (评委)  │
│ 可训练   │  │  可训练       │  │ 冻结      │  │ 冻结     │
└──────────┘  └──────────────┘  └──────────┘  └──────────┘
```

Actor 生成回答，Critic 判断"当前状态好不好"，Reference 确保不偏离原始能力，Reward 给回答打分。

**问题：** Critic 是一个和 Actor 一样大的 Transformer 模型。训练 PPO 意味着显存里要同时塞两套完整的 Transformer。

### GRPO 只需要 3 个模型（少了 Critic）

```
GRPO 训练时的模型阵容:
┌──────────┐               ┌──────────┐  ┌──────────┐
│  Policy  │               │Reference │  │  Reward  │
│ (主角)   │               │(老版本)  │  │  (评委)  │
│ 可训练   │               │ 冻结      │  │ 冻结     │
└──────────┘               └──────────┘  └──────────┘
```

**GRPO 不需要 Critic 的原因：**

1. **组内比较代替了价值预测**。PPO 需要一个 Critic 来预测"当前这个回答大约值多少分"，GRPO 直接拿了几个回答来做比较——"A 比 B 好 5 分"这个信息足够指导优化，不需要单独的价值预测模型。

2. **显存省了很多**。少一整个 Transformer（含 Critic head），大约节省了 40%~50% 的显存。这对于在单卡上跑 RL 训练是决定性的。

3. **训练流程简化了**。PPO 要协调 Actor 和 Critic 的学习率、同步更新节奏、调两个优化器。GRPO 只有一个模型要优化，代码量少了 1/3。

---

## 四、组内归一化：z-score 是怎么算出来的

这是 GRPO 最关键的计算步骤，也是代码中最核心的逻辑。

```python
# rewards: [B * num_generations]，B 是 prompt 数，num_generations 是每个 prompt 生成的答案数
grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
std_r  = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)   # [B*num_gen]
advantages = (rewards - mean_r) / (std_r + 1e-4)  # [B*num_generations]
```

**分步拆解：**

**Step 1 — 分组：** 假设有 2 个 prompt，每个生成 4 个回答，共 8 个奖励。

```
平展的 rewards:        [3.0, -1.0, 2.0, -2.0,  1.0, 0.5, -0.5, 3.0]
reshape 为 [B, G]:    [[3.0, -1.0, 2.0, -2.0],  ← 第1个prompt的4个回答
                       [1.0, 0.5, -0.5, 3.0]]   ← 第2个prompt的4个回答
```

**Step 2 — 算均值：** 每个 prompt 各自的平均分（不是全局平均）。

```
组1 均值 = (3.0 + (-1.0) + 2.0 + (-2.0)) / 4 = 0.5
组2 均值 = (1.0 + 0.5 + (-0.5) + 3.0) / 4 = 1.0
```

**Step 3 — 算标准差：** 每组分数的波动幅度。

```
组1 标准差 ≈ 2.08
组2 标准差 ≈ 1.41
```

**Step 4 — z-score 归一化：**

```
答案A: (3.0 - 0.5) / 2.08 ≈ +1.20       ← 比这组平均水平好很多
答案B: (-1.0 - 0.5) / 2.08 ≈ -0.72    ← 比这组平均水平差
答案C: (2.0 - 0.5) / 2.08 ≈ +0.72     ← 中上
答案D: (-2.0 - 0.5) / 2.08 ≈ -1.20    ← 这组最差
```

**为什么要减去组内均值？** 想象一道题特别难，所有回答都很差（奖励全是负数）。如果直接拿负数当优势信号，模型会觉得"全都不行"。但减去均值后，最差的那个变成负优势（抑制），最好的那个变成正优势（鼓励）——**它不关心绝对分数，只关心相对排名。**

**为什么要除以标准差？** 这保证了优势的尺度稳定。如果某组回答差异很大（标准差大），z-score 会缩小优势值；如果回答差异很小（标准差小），z-score 会放大优势值。这相当于一个"自动调档"。

---

## 五、损失函数详解：GRPO 是怎么"学"的

GRPO loss 的完整公式看起来吓人，但拆开看其实很简单。

```
L_GRPO = -E [ min(r * A, clip(r, 1-ε, 1+ε) * A) - β * KL(π || π_ref) ]
```

这其实由**三件独立的事情**组成：

### 第一件：策略比率（Policy Ratio）

```python
ratio = torch.exp(per_token_logps - old_per_token_logps)
# ratio = π_new(y|x) / π_old(y|x)
```

这个比值回答了一个问题：**新模型对这个回答的概率估计，比旧模型高还是低？**

- ratio > 1：新模型觉得这个答案更好了
- ratio = 1：没变
- ratio < 1：新模型觉得这个答案更差了

### 第二件：裁剪（Clipping）—— 别让模型学太狠

```python
clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
per_token_loss = torch.min(
    ratio * advantages,                    # 不裁剪的好处
    clipped_ratio * advantages             # 裁剪后的保守估计
)
# 取两者中较小的（更保守的）
```

这是一个**安全机制**。假设某个回答的优势 A = +2（非常好），新模型对它的 ratio = 3（大幅提高了概率）。

```
不裁剪:  ratio * A = 3 * 2 = 6         ← 大幅鼓励
裁剪后:  clipped_ratio * A = 1.2 * 2 = 2.4  ← 只适度鼓励（ε=0.2）
取最小值: min(6, 2.4) = 2.4             ← 选保守的
```

**为什么要裁剪？** 如果不裁剪，模型可能对一些偶然的高分回答"用力过猛"，把所有概率都压上去。裁剪限制了每步更新的幅度，保证训练稳定。

```
当 A > 0 时: min(ratio, clipped_ratio)  → 防止过度鼓励
当 A < 0 时: max(ratio, clipped_ratio)  → 防止过度惩罚
（因为后面取了负号，所以 min 的效果反过来）
```

### 第三件：KL 惩罚 —— 别忘记自己原本是谁

```python
kl_div = ref_per_token_logps - per_token_logps       # ref_logp - new_logp
per_token_kl = torch.exp(kl_div) - kl_div - 1        # 具体 KL 计算公式
per_token_loss = -(... - args.beta * per_token_kl)   # 减去 KL 项
```

**KL 惩罚解决了一个关键问题：** 模型在 RL 训练中有可能"走偏"。比如为了追求高分，开始说一堆正确但没用的废话。KL 惩罚就是拉住它，不让它偏离 SFT 后学到的"基本人设"太远。

- `ref_per_token_logps`：老模型（SFT 版本）对每个 token 的概率
- `per_token_logps`：新模型对每个 token 的概率
- 差异越大，KL 惩罚越大 → 损失增加 → 梯度会把模型拉回来

`beta` 参数控制 KL 惩罚的强度。beta=0 表示完全不管，beta 越大模型越保守。MiniMind 默认 `beta=0.1`。

---

## 六、CISPO：另一种训练策略（可选 Loss 变体）

GRPO 脚本除了标准 loss，还支持一个叫 **CISPO**（Clipped Importance Sampled Policy Optimization）的变体。

```python
# 标准 GRPO Loss
clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
per_token_loss = -(torch.min(ratio * A, clipped_ratio * A) - beta * KL)

# CISPO Loss（替代方案）
clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
per_token_loss = -(clamped_ratio * advantages * per_token_logps - beta * KL)
```

**CISPO 和标准 GRPO 的三个区别：**

| | 标准 GRPO | CISPO |
|--|--|--|
| 裁剪方式 | 双侧裁剪 `[1-ε, 1+ε]`，对称 | 单侧裁剪 `[0, ε_high]`，只限制上界 |
| detach | 无 | 裁剪后的 ratio 被 detach，不传梯度 |
| 目标函数 | `min(ratio*A, clip*A)` | `clamped_ratio * A * log_p` |

**CISPO 的直觉：**

```
标准 GRPO: "更新幅度不能超过 ±20%"
CISPO:    "更新幅度不能超过上限500%"（epsilon_high=5.0）
```

CISPO 更激进地限制了过大的比率更新（通过 detach），同时在单侧裁剪上更宽松。这意味着：
- 对于优势为负的回答（该抑制的），CISPO 会更大胆地打压
- 对于优势为正的回答（该鼓励的），CISPO 会更大胆地提升
- 但不会让单个回答主导训练（detached 的裁剪起到了保护）

**什么时候用 CISPO？** 默认配置已经是 `"cispo"`。如果你发现训练不稳定（loss 剧烈抖动），可以尝试切换回 `"grpo"`。

---

## 七、奖励模型（Reward Model）是怎么工作的？

GRPO 训练不需要人类标注的对错标签。它用一个预训练好的 Reward Model 给回答自动打分。

### 奖励由 4 部分组成

```python
def calculate_rewards(prompts, responses, reward_model):
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        # 1. 长度奖励: 回答不长不短（20~800 字符）
        rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5

        # 2. 思考过程奖励: 有</think>标签且思考内容适中（20~300 字符）
        if '</think>' in response:
            thinking_content, answer_content = response.split('</think>', 1)
            rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
            rewards[i] += 0.25 if response.count('</think>') == 1 else -0.25
            answer = answer_content.strip()

        # 3. 重复度惩罚: 防止车轱辘话来回说
        rewards[i] -= rep_penalty(answer)

        # 4. 外部奖励模型打分: RM 对回答质量打分
        score = reward_model.get_score(messages, answer)
        rewards[i] += score

    return rewards
```

**各部分分数范围：**

| 组成部分 | 分数范围 | 含义 | 大白话 |
|--|--|--|--|
| 长度奖励 | +0.5 / -0.5 | 回答长度合适 | "说够了但别废话" |
| 思考奖励 | +1.0 / -0.5 | 有思考过程 | "先想好再回答" |
| 格式奖励 | +0.25 / -0.25 | 标签格式正确 | "格式别搞错" |
| 重复惩罚 | 0 ~ -0.5 | 重复内容程度 | "别车轱辘话" |
| RM 打分 | -3 ~ +3 | 回答整体质量 | 评委评分 |

**为什么不需要人类标注？**

这就是 RLAIF（Reinforcement Learning from AI Feedback）的核心理念：与其花大量人力去标注"哪个回答更好"，不如训练一个 Reward Model 来代替人类评委。这个 Reward Model 本身是在人类标注的偏好数据上训练出来的，所以它"继承了"人类的判断标准，但可以 7x24 小时自动打分。

### 重复惩罚 `rep_penalty` 是怎么工作的

```python
def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0
```

这函数检测的是 "连续 3 个词（trigram）有多少是重复出现过的"。比如：

```
回答: "我觉得这个方法很好，这个方法非常有效，这个方法..."

trigrams:
  ("我觉得","这个","方法")    # 第1次出现
  ("这个","方法","很好")      # 第1次出现
  ("这个方法","很好","这个")  # 第1次出现
  ...
  ("这个","方法","很好")      ← 重复！(第2次)
  ...

重复 trigrams 越多，惩罚越大（上限 0.5 分）
```

---

## 八、参考模型（Reference Model）的作用

GRPO 训练除了可训练的 Policy 模型，还加载了一个权重相同但**冻结**的 Reference 模型：

```python
model, tokenizer = init_model(lm_config, base_weight, device=args.device)
ref_model, _ = init_model(lm_config, base_weight, device=args.device)
ref_model = ref_model.eval().requires_grad_(False)
```

| | Policy（策略模型） | Reference（参考模型） |
|--|--|--|
| 初始权重 | 从 SFT 权重加载 | 从 SFT 权重加载 |
| 训练状态 | 可训练 | 冻结 |
| 用途 | 生成回答 | 计算 KL 惩罚 |

为什么需要 Reference？想象你在学书法。刚开始字和老师的差不多，但后来你可能会"自由发挥"——某个评分标准给了高分，但你已经完全不像会写汉字的人了。Reference 通过 KL 惩罚拉住模型："可以变得更好，但不能完全不像原来的你。"

---

## 九、完整训练流程

```
初始化:
┌────────────────────────────────────────┐
│  Policy  (从 SFT 加载，可训练)           │
│  Reference (从 SFT 加载，冻结)           │
│  Reward Model (外部评委，冻结)           │
└────────────────────────────────────────┘

每轮训练循环:
┌────────────────────────────────────────┐
│  1. Rollout: 每个 prompt 生成 G 个回答  │
│     同时记录 per_token_logp             │
│  2. Reward: 规则函数 + RM 打分          │
│  3. Advantage: 组内 z-score 归一化      │
│  4. Loss: ratio → 裁剪 → KL 惩罚        │
│  5. Update: 反向传播 + 梯度裁剪 + 优化   │
│  6. Save: 定期保存策略权重               │
└────────────────────────────────────────┘
```

---

## 十、GRPO 和 PPO 的完整对比

| 维度 | PPO | GRPO |
|--|--|--|
| **Critic 模型** | 需要 | **不需要** |
| **优势计算** | GAE（逐 token） | 组内 z-score（回答级） |
| **每个 prompt 生成数** | 1 个 | `num_generations` 个 |
| **优化器数量** | 2 个（Actor + Critic） | **1 个**（仅 Policy） |
| **代码量** | ~443 行 | ~332 行 |
| **显存** | 高（两套 Transformer） | **低**（一套） |
| **著名应用** | ChatGPT (RLHF) | DeepSeek-R1 |

---

## 十一、命令行参数速查

| 参数 | 默认值 | 含义 |
|--|--|--|
| `--num_generations` | 6 | 每个 prompt 生成的回答数 |
| `--batch_size` | 2 | 同时处理的 prompt 数 |
| `--learning_rate` | 3e-7 | RL 阶段学习率很低 |
| `--beta` | 0.1 | KL 惩罚系数 |
| `--epsilon` | 0.2 | 裁剪参数（下界 = 0.8） |
| `--epsilon_high` | 5.0 | CISPO 专用比率上限 |
| `--loss_type` | `"cispo"` | "grpo" 或 "cispo" |
| `--max_seq_len` | 768 | Prompt 最长 |
| `--max_gen_len` | 1024 | 生成最长 |
| `--rollout_engine` | `"sglang"` | "torch" 或 "sglang" |

```bash
# 基础运行
python -m trainer.train_grpo --from_weight full_sft

# 调试模式（打印每个样本的细节）
python -m trainer.train_grpo --debug_mode
```

---

## 十二、代码逐块阅读

### 初始化模型（第 271-278 行）

```python
model, tokenizer = init_model(lm_config, base_weight, device=args.device)
ref_model, _ = init_model(lm_config, base_weight, device=args.device)
ref_model = ref_model.eval().requires_grad_(False)
reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
```

三个模型的初始化与前面描述一致。Reward Model 用的是外部模型（默认 `internlm2-1_8b-reward`），不是 MiniMind 自带的。

### 优势计算（第 121-124 行）

```python
grouped_rewards = rewards.view(-1, args.num_generations)
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
std_r  = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)
advantages = (rewards - mean_r) / (std_r + 1e-4)
```

z-score 计算，完整代码只有 4 行。

### Loss 计算（第 131-143 行）

```python
kl_div = ref_per_token_logps - per_token_logps
per_token_kl = torch.exp(kl_div) - kl_div - 1
ratio = torch.exp(per_token_logps - old_per_token_logps)
if args.loss_type == "cispo":
    clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
    per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
else:
    clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
    per_token_loss = -(torch.min(ratio * advantages.unsqueeze(1), clipped_ratio * advantages.unsqueeze(1)) - args.beta * per_token_kl)
policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
loss = (policy_loss + aux_loss) / args.accumulation_steps
loss.backward()
```

注意这里只用了**一个优化器**（没有 Critic），backward 只传回 Policy 模型。

### 优化器更新（第 146-151 行）

```python
if step % args.accumulation_steps == 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

---

## 十三、总结

GRPO 的核心优势可以总结为一句话：**用组内比较代替价值网络，用更少的资源做 RL 对齐训练。**

对于初学者来说，建议的入门路径是：

1. **先跑通一次训练**：`python -m trainer.train_grpo --debug_mode` 观察具体的奖励和优势计算
2. **看奖励分布**：理解 `rewards → advantages → loss` 的数据流
3. **调整 num_generations**：观察不同组大小对训练稳定性和速度的影响
4. **尝试两种 loss**：对比 `"grpo"` 和 `"cispo"` 在你的数据上的表现
5. **调整 KL 系数 beta**：太小模型会"放飞自我"，太大学不到新东西

GRPO 大约 332 行代码，但其承载的思想影响深远。DeepSeek-R1（一个拥有 671B 参数的巨型模型）在推理能力上的突破，核心算法就是这套"让模型自己生成多个答案，然后和自己比"的朴素思路。这说明：**好算法不一定要复杂，找到正确的视角比堆砌模块更重要。**

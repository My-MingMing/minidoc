# train_ppo.py 初学者完全指南

> 文件路径: `trainer/train_ppo.py`
> 代码总量: 约 443 行
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
| **Reward** | RM 网络 | 外部模型（如 internlm2-reward） | 冻结 | 给最终回答打一个总分 |

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

## 三、GAE 优势估计：Critic 是"分数预测员"

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

### 优势归一化

```python
adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
adv_var = ((advantages - adv_mean) ** 2 * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask
```

算完 GAE 之后，还把优势做了**均值 0、方差 1 的归一化**。这和 GRPO 做 z-score 的原理是一样的——保证优势分布稳定，防止某些 batch 的优势值特别大导致训练不稳定。

---

## 四、PPO 裁剪目标函数（Clipped Objective）详解

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

---

## 五、KL 早停机制（Early Stopping）

除了裁剪，PPO 还有另一层安全网：KL 早停。

```python
log_ratio = mb_resp_logp - old_resp_logp
approx_kl = (0.5 * (log_ratio ** 2) * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)

# 同步各卡 approx_kl（防止 DDP 死锁）
approx_kl_val = approx_kl.detach().clone()
if dist.is_initialized():
    dist.all_reduce(approx_kl_val, op=dist.ReduceOp.AVG)

if approx_kl_val > args.early_stop_kl:  # 默认阈值 0.25
    stop_ppo = True
    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) * 0.0  # 归零但不中断通信
```

### KL 早停 vs KL 惩罚

PPO 里其实有两种 KL 相关的机制，不要混淆：

| | KL 早停 | KL 惩罚 |
|--|--|--|
| 参考对象 | 旧策略（本次更新前的 policy） | Reference 模型（SFT 版） |
| 计算方式 | `0.5 * (log_ratio)²` | 完整 KL 公式 |
| 作用 | 如果**同一批数据上** PPO 更新了太多轮，策略偏离太大，**立刻停止** | 持续惩罚相对于 SFT 初始状态的偏离 |
| 参数 | `early_stop_kl = 0.25` | `kl_coef = 0.02` |

**KL 惩罚的代码：**

```python
kl_ref_penalty = ((torch.exp(ref_resp_logp - mb_resp_logp) 
                   - (ref_resp_logp - mb_resp_logp) - 1.0)
                  * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
policy_loss = (... policy_loss部分 ... + args.kl_coef * kl_ref_penalty)
```

这个公式用的是 `e^x - x - 1`，这是 KL 散度的一种近似形式，其中 `x = ref_logp - new_logp = log(ref/new)`。当新策略和 Reference 完全一致时（x=0），惩罚为 0。偏差越大，惩罚越大。

### 为什么早停时必须保持 forward-backward 循环？

```python
if stop_ppo:
    loss = (...) * 0.0   # 归零 loss
else:
    loss = (...) / args.accumulation_steps
loss.backward()  # 仍然执行 backward
```

这是 DDP（分布式训练）的硬性要求。如果某张卡提前退出，其他卡还在等待梯度同步，就会死锁。所以即使 loss 归零了，也要走完 forward 和 backward 的全流程，保证所有卡的通信步调一致。

---

## 六、复合奖励机制

```python
def calculate_rewards(prompts, responses, reward_model):
    # 1. 长度奖励: +0.5 / -0.5（20~800 字符）
    rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
    # 2. 思考奖励: +1.0 / -0.5（20~300 字符内容 + 恰好一个</think> 标签）
    # 3. 重复惩罚: 0 ~ -0.5（trigram 重复度）
    # 4. RM 打分: -3.0 ~ +3.0
```

**Reward 是怎么放到 token 序列上的？**

```python
token_rewards = torch.zeros_like(old_resp_logp)
last_idx = resp_lengths - 1
token_rewards[torch.arange(B), last_idx] += rewards
```

中间 token 奖励全为 0，只有最后一个 token（EOS 位置）被加上外部奖励。GAE 负责把这个"最终分"往前传播到每一步。

---

## 七、Critic 模型详解

Critic 是 PPO 独有的组件：

```python
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        self.value_head = nn.Linear(params.hidden_size, 1)  # 单标量输出

    def forward(self, input_ids, attention_mask, **kwargs):
        hidden_states = self.model.norm(self.model(input_ids, attention_mask, **kwargs)[0])
        return self.value_head(hidden_states).squeeze(-1)  # [B, T]
```

Critic 从 SFT 权重加载（复用 Transformer body），`lm_head` 被替换为 `value_head`。它有独立的学习率（5e-7，高于 Actor 的 3e-7），因为 value_head 从零初始化需要更快收敛。Value 损失也做了裁剪（`cliprange_value=0.2`），防止 Critic 预测过度偏离旧值。

---

## 八、完整训练流程

```
初始化: Policy/Actor (SFT, 可训练) | Critic (SFT, 可训练) | Reference (SFT, 冻结) | Reward (冻结)

每轮训练循环:
┌──────────────────────────────────────────────────┐
│  1. Rollout: Actor 对每个 prompt 生成 1 个回答    │
│  2. Reward: Reward Model + 规则函数打分          │
│  3. Mask: 区分 prompt / response / EOS 位置       │
│  4. 收集数据 (no_grad):                           │
│     Critic → old_values, Actor → old_logp        │
│     Reference → ref_logp, token_rewards[EOS]     │
│  5. GAE 优势估计: 倒序遍历 → TD 误差 → 归一化     │
│  6. Mini-batch PPO 更新:                          │
│     a. 重跑 Actor → 新 logp, Critic → 新 values   │
│     b. 检查 KL 早停 → 超阈则 break                │
│     c. policy_loss (clipped) + value_loss + KL    │
│     d. backward → grad clip → 优化器更新           │
│  7. Save: 定期保存权重                            │
└──────────────────────────────────────────────────┘
```

---

## 九、命令行参数速查

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

### 运行示例

```bash
# 基础运行
python -m trainer.train_ppo --from_weight full_sft

# 更保守的训练（更小的 KL 容忍度）
python -m trainer.train_ppo --early_stop_kl 0.1 --kl_coef 0.05

# 调试模式
python -m trainer.train_ppo --debug_mode
```

---

## 十、代码逐块阅读

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

## 十一、PPO 和 GRPO 的核心差异

| 维度 | PPO | GRPO |
|--|--|--|
| **模型数量** | 4 个（Actor + Critic + Ref + RM） | 3 个（Policy + Ref + RM） |
| **优势计算** | GAE（逐 token，需要 Critic） | 组内 z-score（回答级） |
| **优势精度** | 高（逐 token 反馈） | 中（回答整体反馈） |
| **显存需求** | 高（两套 Transformer） | 低（一套 Transformer） |
| **代码行数** | ~443 行 | ~332 行 |
| **安全机制** | 裁剪 + KL 早停 + KL 惩罚 | 裁剪 + KL 惩罚 |
| **适用场景** | 精细控制、资源充足 | 资源受限、快速上手 |

---

## 十二、学习建议

1. **先跑通 GRPO**（代码更简单，逻辑更直观）
2. **再学 PPO**，重点关注 GRPO **没有的东西**：Critic、GAE、Value Loss
3. **对比优势计算**：一个"和小组比"（GRPO），一个"Critic 预测偏差"（PPO）
4. **调参实验**：尝试 `gamma=0.99` 看折扣在文本生成中是否有害
5. **观察 KL 曲线**：不则 `early_stop_kl` 太高，每轮都停则 `ppo_update_iters` 太多

PPO 大约 443 行代码，是四种 RL 方法中最复杂的一个。但它的核心思想——**在旧策略附近做小步更新，确保每一步都不会走太远**——是稳定训练 LLM 的通用原则。理解 PPO 后，再看 GRPO 会有一种"原来少了一个 Critic 也能做"的豁然开朗感。

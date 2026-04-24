# train_distillation.py 逐行导读

> 知识蒸馏训练脚本 | `trainer/train_distillation.py` | 约 245 行
> **先懂概念，再看代码** --- 本文为中文 LLM 初学者设计

---

## 一、什么是知识蒸馏？为什么要把大模型的知识 "蒸馏" 到小模型？

想象你是一位经验丰富的**特级教师**（大模型），教出了一群成绩优秀的学生。现在学校要新建一所分校，预算有限，只能雇一位**年轻老师**（小模型）。怎么办？

最笨的办法是让年轻老师重新摸索教学经验 --- 这需要大量时间和数据。聪明的办法是：**让年轻老师坐在特级教师的课堂旁边，不仅学"正确答案"，还学"特级教师思考问题的方式"**。

这就是**知识蒸馏（Knowledge Distillation）** 的核心思想：

```
┌─────────────────────┐          ┌─────────────────────┐
│   Teacher 大模型      │          │   Student 小模型      │
│   (特级教师)          │          │   (年轻老师)          │
│   参数多、能力强       │──知识──▶ │   参数少、速度快       │
│   推理慢、成本高       │   蒸馏   │   推理快、部署便宜      │
└─────────────────────┘          └─────────────────────┘
```

**蒸馏的本质**：小模型不仅要学习训练数据中的"标准答案"（hard labels），还要模仿大模型输出的"概率分布"（soft labels）。大模型输出中暗含了丰富的**类间关系** --- 哪些答案是"虽然不对但有点沾边"的，哪些是"完全不着边际"的 --- 这些信息对小模型的泛化能力至关重要。

### 一个直观例子

假设输入是 "法国的首都是___"。Teacher 模型的输出分布可能如下：

```
T=1.0 时：  {"巴黎": 0.85, "里昂": 0.08, "伦敦": 0.04, "香蕉": 0.03}
```

Student 如果只学"标准答案是巴黎"（one-hot），它只知道"巴黎是对的"。但如果它模仿 Teacher 的整个分布，它还能学到：
- "里昂"是第二可能的答案（也是法国城市，只是不是首都），说明答案应该跟法国有关
- "伦敦"也有一点点概率（也是欧洲首都，但属于英国）
- "香蕉"概率极低（完全不相关）

**这些隐含知识让小模型在遇到没见过的题目时，也能做出更合理的猜测**。

---

## 二、整体架构：学生跟着老师学

```
  训练数据 (JSONL 对话)
        │
        ▼
  ┌─────────────┐    ┌─────────────┐
  │ Student 模型 │    │ Teacher 模型 │
  │ (可训练)     │    │ (冻结/eval)  │
  └──────┬──────┘    └──────┬──────┘
         │                  │
         ▼                  ▼
  student_logits      teacher_logits
         │                  │
         │         (no_grad, detach)
         │                  │
         ▼                  ▼
  ┌──────────────────────────────────┐
  │         混合损失计算              │
  │                                  │
  │  loss = α × CE_loss             │
  │       + (1 - α) × KL_divergence │
  │                                  │
  │  CE_loss  : 只更新 Student       │
  │  KL_loss  : Student 模仿 Teacher │
  └────────────────┬─────────────────┘
                   ▼
            反向传播 → 只更新 Student 参数
```

**几个关键点**：
- Teacher 模型全程处于 `eval()` 模式，**不参与梯度计算**，所以它不会产生额外的显存开销
- Student 模型从头到尾都是训练模式，只有它的参数会被更新
- 两个模型可以有不同的隐藏维度和层数 --- Small Student 向 Big Teacher 学习是完全合理的

---

## 三、核心概念：温度参数（Temperature）与 KL 散度

这是蒸馏最难理解的部分，我们用类比来拆解。

### 3.1 什么是 Temperature（温度）？

Temperature（温度 T）控制着 softmax 输出的"陡峭"程度：

```
原始 logits:    [5.0, 2.0, 1.0, 0.5]

T = 1.0 (标准):  softmax → [0.83, 0.10, 0.04, 0.03]  ← 差距很大
T = 2.0 (升温):  softmax → [0.53, 0.24, 0.15, 0.09]  ← 差距缩小
T = 5.0 (更热):  softmax → [0.36, 0.24, 0.20, 0.19]  ← 接近平坦
```

**类比**：想象一座山峰。T=1 时，最高的山峰非常尖锐（正确答案概率极高）；T 越大，山峰越平缓，所有选项的差距变小。

**为什么要升温？** 当分布太平坦时（T 很大），那些原本概率只有 0.01 的"错误答案"也能变得可观。这些小概率中藏着一个词和其他词之间的**语义关系** --- 这正是 Teacher 要传授给 Student 的"暗知识（Dark Knowledge）"。

Hinton（蒸馏的提出者）有一个精妙的观察：

> 如果 Student 正确回答 1000 道题，但每道题的概率分布都非常极端（99.9% vs 0.001%），这看起来是"满分"，但实际上它并没有学到类间关系。升温后的 soft labels 迫使 Student 去学习那些微妙的相关性。

### 3.2 为什么要乘以 T 的平方（T^2）？

这个问题问得很好。先看代码：

```python
def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    kl = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    return (temperature ** 2) * kl
```

**为什么需要 T^2？用直觉理解**：

假设我们把 T 从 1 增大到 3。经过 T 除法的 softmax 变得非常平滑，导致每个概率值之间的差距缩小到原来的 1/3。而 KL 散度的计算涉及到这些差距的对数，因此梯度也会缩小约 1/3。

同时，我们输入除以了 T，这又导致梯度再缩小 1/T。所以总梯度缩小了 **1/T x 1/T = 1/T^2**。

**乘以 T^2 就是对梯度缩放做补偿**，确保蒸馏损失的梯度和交叉熵损失的梯度在同一个量级上。如果不乘 T^2，增大温度后蒸馏损失的贡献会变得非常小，起不到蒸馏的效果。

### 3.3 KL 散度是在做什么？

KL 散度（Kullback-Leibler Divergence）衡量的是**两个概率分布之间的"距离"**。

```
KL(Student || Teacher) = Σ Teacher(y) × log(Teacher(y) / Student(y))
```

**通俗理解**：
- 如果 Teacher 说"巴黎"的概率是 0.5，但 Student 只给了 0.1 --- 差距很大，KL 散度高，Student 需要大幅调整
- 如果 Teacher 说"巴黎"的概率是 0.5，Student 给了 0.45 --- 差距小，KL 散度低
- KL 散度 = 0 时，Student 和 Teacher 的分布完全一致

---

## 四、为什么 MoE Teacher 蒸馏到 Dense Student 是一个实用模式？

MoE（Mixture of Experts，混合专家）模型通过在推理时激活部分专家来实现"参数大但计算量小"的设计。但它仍有缺点：

```
MoE Teacher 模型                    Dense Student 模型
┌──────────────────────┐           ┌──────────────────┐
│   ├─专家1 (激活)      │           │   一个统一的      │
│   ├─专家2 (激活)      │   蒸馏    │   密集前馈层       │
│   ├─专家3 (未激活)    │  ──────▶  │   (参数少，       │
│   ├─专家4 (未激活)    │           │    推理统一，      │
│   └─ 路由网络         │           │    部署友好)       │
└──────────────────────┘           └──────────────────┘
```

**这个模式的价值在于**：

1. **压缩推理**：MoE 模型运行时可能需要加载多个专家权重到显存中，而 Dense 模型只需加载一份固定权重
2. **统一性**：MoE 的路由决策引入了额外的复杂性和不确定性，Dense 模型的推理过程更加确定和稳定
3. **性能接近**：通过蒸馏，Dense Student 可以在很多任务上达到 MoE Teacher 的 90% 以上精度，但推理速度快数倍
4. **部署成本低**：Dense 模型更容易在边缘设备、小显存 GPU 或 CPU 上部署

代码中默认配置体现了这一模式：

```python
student_use_moe = 0   # 学生用密集模型（dense）
teacher_use_moe = 1   # 老师用混合专家模型（MoE）
```

---

## 五、Alpha 超参数：平衡真相与老师的话

**Alpha（α）是你最需要调好的参数**。它决定了学习过程中"相信真实标签"和"模仿老师"各占多大比重：

```
总损失 = α × CE_loss + (1 - α) × KL_distill
```

```
α = 1.0 ━━━━━━━━━━━━━━━━━━━━━━━●━━━━━━━━━━━━━━━
         纯 SFT 训练           0% 蒸馏
         （不看 Teacher，只看训练数据）

α = 0.5 ━━━━━━━━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━━
         一半 CE    一半蒸馏
         （默认值，兼顾两者）

α = 0.3 ━━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          30% CE    70% 蒸馏
          （让 Student 更多模仿 Teacher）

α = 0.0 ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         纯蒸馏
         （完全模仿 Teacher，不看标准答案）
```

**如何选择合适的 α？**

| 场景 | 推荐 α | 原因 |
|------|--------|------|
| Teacher 远强于 Student | 0.2 ~ 0.3 | Teacher 的知识质量更高，让 Student 多学 |
| 两者能力接近 | 0.5 | 均衡策略，既看数据也看老师 |
| Student 已经学得不错，只需微调 | 0.7 ~ 0.8 | 以真实标签为主，蒸馏做辅助 |
| 没有 Teacher（只想跑 SFT） | 1.0 | 纯交叉熵训练 |

在代码中，alpha 作为命令行参数传入：

```python
parser.add_argument('--alpha', default=0.5, type=float,
    help="CE损失权重，总损失=alpha*CE+(1-alpha)*KL")
```

---

## 六、训练循环逐段解读

现在有了概念基础，我们走进代码。训练的主循环在 `train_epoch` 函数中（第 38-143 行）。

### 第一阶段：前向传播 -- 两个模型各跑一遍

```python
# Student 前向（可训练，计算梯度）
res = model(input_ids)
student_logits = res.logits[..., :-1, :].contiguous()

# Teacher 前向（冻结，不计算梯度）
with torch.no_grad():
    teacher_logits = teacher_model(input_ids).logits[..., :-1, :].contiguous()
    vocab_size_student = student_logits.size(-1)
    teacher_logits = teacher_logits[..., :vocab_size_student]  # 对齐词表
```

**注意** `[..., :-1, :]` 的含义 --- 去掉最后一个 token 的 logits。这是因为在自回归语言模型中，第 i 个位置的 logit 用来预测第 i+1 个 token，所以我们要对除了最后一个位置之外的所有位置计算损失。

**词表对齐**：如果 Teacher 的词表比 Student 大（常见于 MoE 模型使用了更大的 embedding），我们需要截断 Teacher 的 logits，只保留前 `vocab_size_student` 个维度。

### 第二阶段：计算两个损失

```python
# 1) Ground-Truth 交叉熵损失 --- 让 Student 说"正确答案"
shift_labels = labels[..., 1:].contiguous()
ce_loss = F.cross_entropy(
    student_logits.view(-1, student_logits.size(-1)),
    shift_labels.view(-1),
    ignore_index=-100,
    reduction='none'
)

# 2) 蒸馏 KL 损失 --- 让 Student 模仿 Teacher
distill_loss = distillation_loss(
    student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
    teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
    temperature=temperature
)
```

**损失掩码（loss_mask）**：`-100` 是 `cross_entropy` 的忽略索引，对应 padding 位置或不需要学习的位置。蒸馏时也只用 `loss_mask_flat == 1` 的有效位置计算 KL 散度。

### 第三阶段：混合并反向传播

```python
# 总损失 = α × CE + (1 - α) × 蒸馏
loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps

scaler.scale(loss).backward()
```

除以 `accumulation_steps` 是梯度累积 --- 累积 N 步后做一次参数更新，等效于把 batch size 扩大 N 倍。

### 第四阶段：梯度裁剪与优化

```python
if step % args.accumulation_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

**梯度裁剪**：防止梯度爆炸。在蒸馏训练中尤其重要，因为 KL 散度的梯度可能很大。默认 `grad_clip = 1.0`。

### 训练循环全景图

```
Epoch 开始
   │
   ▼
┌─────────────────────────────────┐
│  for step, (input_ids, labels)  │
│     in enumerate(loader):       │
│                                 │
│  Student 前向 ──▶ student_logits│──┐
│  Teacher 前向 ──▶ teacher_logits│  │  在同一 batch 上
│                                 │  │
│  CE 损失 (labels 监督) ─────────┘  │
│  KL 蒸馏损失 (Teacher 模仿) ──────┘
│                                 │
│  总损失 = α * CE + (1-α) * KL   │
│  loss.backward()                │
│                                 │
│  每 accumulation_steps 步：      │
│    梯度裁剪 → optimizer.step()  │
│                                 │
│  每 log_interval 步：           │
│    打印日志 (loss, ce, distill) │
│                                 │
│  每 save_interval 步：          │
│    保存 checkpoint              │
└─────────────────────────────────┘
   │
   ▼
Epoch 结束
保存 Student 模型权重
```

---

## 七、命令行参数一览

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--student_hidden_size` | 768 | Student 隐藏层维度 |
| `--student_num_layers` | 8 | Student 层数 |
| `--teacher_hidden_size` | 768 | Teacher 隐藏层维度 |
| `--teacher_num_layers` | 8 | Teacher 层数 |
| `--student_use_moe` | 0 | Student 是否使用 MoE |
| `--teacher_use_moe` | **1** | Teacher 默认使用 MoE |
| `--from_student_weight` | `full_sft` | Student 初始化权重 |
| `--from_teacher_weight` | `full_sft` | Teacher 权重 |
| `--alpha` | **0.5** | CE 权重（总损失 = α * CE + (1-α) * KL） |
| `--temperature` | **1.5** | 蒸馏温度（推荐 1.0~2.0） |
| `--epochs` | 6 | 蒸馏轮数 |
| `--batch_size` | 32 | 每批次样本数 |
| `--accumulation_steps` | 1 | 梯度累积步数 |
| `--grad_clip` | 1.0 | 梯度裁剪阈值 |
| `--learning_rate` | 5e-6 | 学习率（远低于预训练） |

**注意学习率**：蒸馏的学习率（5e-6）比预训练（5e-4）低两个数量级。这是因为 Student 已经有预训练权重基础，蒸馏只是"微调"，不是"从头学"。

---

## 八、常见训练命令

```bash
# 基础蒸馏：MoE Teacher → Dense Student
python -m trainer.train_distillation \
    --epochs 6 --batch_size 32 \
    --alpha 0.5 --temperature 1.5

# 更强蒸馏：让 Student 更多模仿 Teacher
python -m trainer.train_distillation \
    --epochs 6 --alpha 0.3 --temperature 2.0

# 不同架构蒸馏：大 Teacher → 小 Student
python -m trainer.train_distillation \
    --teacher_hidden_size 1024 --teacher_num_layers 12 \
    --student_hidden_size 512  --student_num_layers 6 \
    --alpha 0.3

# 分布式训练
torchrun --nproc_per_node=4 -m trainer.train_distillation ...
```

---

## 九、数学公式总结

```
蒸馏损失函数（完整形式）：

Teacher 软化概率：    p_T(y) = softmax(z_T / T)_y
Student 软化对数概率： q_S(y) = log_softmax(z_S / T)_y

KL 蒸馏损失：   L_distill = T^2 × Σ p_T(y) × log(p_T(y) / q_S(y))

交叉熵损失：    L_CE = -Σ y_true × log(softmax(z_S))

总损失：        L_total = α × L_CE + (1 - α) × L_distill
```

其中：
- `z_T` 和 `z_S` 分别是 Teacher 和 Student 的原始 logits
- `T` 是温度参数
- `α` 是 CE 权重系数
- `y` 遍历词表中的所有 token

# trainer_utils.py 深度代码分析

> 文件路径: `trainer/trainer_utils.py`
> 代码总量: 约 177 行
> 性质: 训练工具函数集合，被所有 `train_*.py` 脚本共享使用

---

## 目录

1. [模块总览](#1-模块总览)
2. [函数逐项解析](#2-函数逐项解析)
   - [2.1 get_model_params](#21-get_model_params)
   - [2.2 is_main_process / Logger](#22-is_main_process--logger)
   - [2.3 get_lr](#23-get_lr)
   - [2.4 init_distributed_mode](#24-init_distributed_mode)
   - [2.5 setup_seed](#25-setup_seed)
   - [2.6 lm_checkpoint](#26-lm_checkpoint)
   - [2.7 init_model](#27-init_model)
3. [类逐项解析](#3-类逐项解析)
   - [3.1 SkipBatchSampler](#31-skipbatchsampler)
   - [3.2 LMForRewardModel](#32-lmforrewardmodel)
4. [跨脚本调用关系图](#4-跨脚本调用关系图)

---

## 1. 模块总览

`trainer_utils.py` 是 MiniMind 训练框架的**基础设施层**，它不涉及具体的训练逻辑，而是提供所有训练脚本（pretrain / full_sft / lora / dpo / ppo / grpo / distillation / agent）所共用的通用工具函数。

### 导入依赖

| 模块 | 用途 |
|------|------|
| `os`, `sys` | 路径操作，`sys.path` 动态追加 |
| `random`, `math`, `numpy` | 随机种子 / 数学函数 |
| `torch`, `torch.distributed` | 分布式训练支持 |
| `DistributedDataParallel` | PyTorch DDP 包装器 |
| `AutoTokenizer`, `AutoModel`, `AutoModelForSequenceClassification` | HuggingFace 自动加载 |
| `MiniMindForCausalLM` | 项目核心模型类 |

### 路径设置

```python
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

声明当前文件属于 `trainer` 包，并将项目根目录加入 `sys.path`，允许 `trainer/` 目录下的脚本正确导入 `model/` 等模块。

---

## 2. 函数逐项解析

### 2.1 get_model_params

```python
def get_model_params(model, config):
```

**目的**: 打印模型的参数量统计，对 MoE（Mixture of Experts）模型做特殊处理，分别显示总参数和激活参数。

**逻辑拆解**:

1. 计算模型的总参数量（以百万计）: `total = sum(p.numel()) / 1e6`
2. 从 `config` 中提取 MoE 相关配置（兼容不同命名风格）:
   - `n_routed`: 专家总数（尝试 `n_routed_experts` 和 `num_experts`）
   - `n_active`: 每个 token 激活的专家数
   - `n_shared`: 共享专家数
3. 分别统计单个专家的参数（通过遍历参数名匹配 `mlp.experts.0.`）和共享专家的参数
4. 计算基座参数量: `base = total - expert_params * n_routed - shared_expert_params * n_shared`
5. 计算激活参数量: `active = base + expert_params * n_active + shared_expert_params * n_shared`
6. 如果存在 MoE（激活参数 < 总参数），则输出 `Model Params: XX.XXM-A YY.YYM`，否则只输出总参数量

**输出示例**:
- 非 MoE 模型: `Model Params: 26.28M`
- MoE 模型: `Model Params: 120.50M-A 45.20M`

> **关键设计**: MoE 模型虽然总参数大，但每次前向只激活部分专家，所以实际计算开销接近活跃参数量。

---

### 2.2 is_main_process / Logger

```python
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def Logger(content):
    if is_main_process():
        print(content)
```

**目的**: 分布式训练时的统一日志输出控制。

- `is_main_process()`: 判断当前进程是否为主进程。在未启用 DDP 时（单卡训练）始终返回 `True`
- `Logger()`: 只有主进程才打印输出，避免多卡环境中每个 GPU 都打印相同的日志导致刷屏

---

### 2.3 get_lr

```python
def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))
```

**目的**: 余弦退火学习率调度。

**公式**:
```
lr_current = lr_base * [0.1 + 0.45 * (1 + cos(π * step / total))]
```

**行为**:
- 起点（step=0）: `lr_current = lr_base * [0.1 + 0.45 * (1+1)] = lr_base * 1.0` → **100% 起始**
- 中点: `lr_current = lr_base * [0.1 + 0.45 * (1+0)] = lr_base * 0.55`
- 终点: `lr_current = lr_base * [0.1 + 0.45 * (1-1)] = lr_base * 0.1` → **10% 最低**

**特点**: 这是一个简单的单调余弦衰减曲线，从 100% 线性地衰减到 10%，没有 warmup 阶段（warmup 在训练脚本的 `train_epoch` 中单独实现）。

---

### 2.4 init_distributed_mode

```python
def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank
```

**目的**: 初始化 DDP 分布式训练环境。

**逻辑**:
1. 检查环境变量 `RANK`: 如果未设置（值为 -1），则返回 `0` 表示单卡模式
2. 如果 `RANK` 存在，说明是通过 `torchrun` 或 `torch.distributed.launch` 启动的分布式任务
3. 使用 `nccl` 后端初始化进程组（NVIDIA GPU 通信）
4. 从 `LOCAL_RANK` 环境变量获取当前进程对应的 GPU 编号
5. 将该 GPU 设为默认设备

**返回值**: `local_rank`，主进程为 0，后续脚本通过此值判断是否执行数据保存/日志打印等主进程操作。

---

### 2.5 setup_seed

```python
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**目的**: 完全固定随机种子，确保训练可复现。

**覆盖范围**:

| 随机源 | 设置方式 |
|--------|----------|
| Python 内置 `random` | `random.seed()` |
| NumPy | `np.random.seed()` |
| PyTorch CPU | `torch.manual_seed()` |
| PyTorch GPU (当前) | `torch.cuda.manual_seed()` |
| PyTorch GPU (所有) | `torch.cuda.manual_seed_all()` |
| cuDNN 确定性 | `deterministic = True` |
| cuDNN 自动调优 | `benchmark = False` |

> **注意**: `benchmark = False` 会牺牲一点性能（不再寻找最优卷积算法），但换来完全一致的运行结果。

---

### 2.6 lm_checkpoint

```python
def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None,
                   epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
```

**目的**: 统一的检查点保存/加载函数，同时支持模型权重和训练断点恢复。

#### 保存模式（`model` 不为 None）

保存两种文件：

| 文件 | 格式 | 用途 |
|------|------|------|
| `{weight}_{hidden_size}{_moe}.pth` | 纯模型权重 | 推理时使用 |
| `{weight}_{hidden_size}{_moe}_resume.pth` | 完整状态字典 | 断点恢复训练 |

**保存细节**:
1. 自动解包 `DistributedDataParallel` 和 `torch.compile` 包装（`_orig_mod`）
2. 权重转半精度（`half()`）以节省磁盘空间
3. 原子写入（先写 `.tmp` 再 `os.replace`），防止写入中断导致文件损坏
4. 自动提取 WandB run ID

**resume 状态字典包含**:
```python
{
    'model': state_dict,        # 模型权重
    'optimizer': optimizer.state_dict(),  # 优化器状态
    'epoch': epoch,             # 当前轮次
    'step': step,               # 已训练步数
    'world_size': N,            # GPU 数量
    'wandb_id': '...',          # WandB 实验 ID
    **kwargs                    # 用户自定义额外状态
}
```

**GPU 数量自适应** (第 110-114 行):
如果恢复时 GPU 数量与保存时不同，自动调整 step:
```
step_new = step_old * world_size_old / world_size_new
```
这确保了总的训练量（步数 × GPU 数）保持一致。

#### 加载模式（`model` 为 None）

1. 优先查找 `_resume.pth` 文件
2. 如果不存在则返回 `None`
3. 返回的字典包含所有用于恢复训练的完整状态

---

### 2.7 init_model

```python
def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model',
               save_dir='../out', device='cuda'):
```

**目的**: 模型和 tokenizer 的一站式初始化工具。

**流程**:
```
AutoTokenizer.from_pretrained(tokenizer_path)
        ↓
MiniMindForCausalLM(lm_config)     ← 创建模型
        ↓
如果 from_weight != 'none':        ← 从权重文件加载
    torch.load(weight_path)
    model.load_state_dict(..., strict=False)
        ↓
get_model_params()                  ← 打印参数统计
Logger(Trainable Params)            ← 打印可训练参数
        ↓
返回 (model.to(device), tokenizer)
```

**权重路径推导**:
```
{save_dir}/{from_weight}_{hidden_size}{_moe}.pth
```
例如: `../out/pretrain_512.pth` 或 `../out/full_sft_768_moe.pth`

**关键点**: `strict=False` 允许部分加载（当 checkpoint 中存在某些当前模型没有的参数时跳过，不发报错）。

---

## 3. 类逐项解析

### 3.1 SkipBatchSampler

```python
class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
```

**目的**: 在断点恢复训练时，跳过已经训练过的数据批次。

**场景**: 训练中断后恢复，step 已知但不能从 epoch 开头重新开始，否则会重复训练已完成的批次导致 loss 异常。

**工作原理**:

```
原始 sampler:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...]
batch_size = 2, skip_batches = 2

完整批次:  [0,1], [2,3], [4,5], [6,7], [8,9], ...
跳过前2批:            ↑跳过          ↓
输出:                             [6,7], [8,9], ...
```

**关键实现**:
- `__iter__`: 逐批次遍历原始 sampler，计数跳过指定数量后 yield 后续批次
- `__len__`: 返回 `(总样本数 + batch_size - 1) // batch_size - skip_batches`

**调用方**: 仅在 `train_grpo.py` 和 `train_ppo.py` 中用于断点恢复时 `SkipBatchSampler(sampler, batch_size, skip_batches=start_step)`

---

### 3.2 LMForRewardModel

```python
class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
```

**目的**: 加载一个独立的奖励/评分模型，用于 RLHF/RLAIF 训练中评估生成回复的质量。

**内部组件**:
| 组件 | 类型 | 说明 |
|------|------|------|
| `tokenizer` | AutoTokenizer | 分词器，支持远程代码 |
| `model` | AutoModel | 预训练评分模型，FP16 精度 |
| `device` | str | 推理设备 |

#### get_score 方法

```python
def get_score(self, messages, response):
```

**输入**:
- `messages`: 多轮对话历史 `[{role, content}, ...]`
- `response`: 模型生成的回复

**处理流程**:
1. 将对话历史格式化为文本: `"{history}\n以上是对话历史。我的新问题是：\n{last_query}"`
2. 构造评分输入: `[{"role": "user", "content": message_context}, {"role": "assistant", "content": response}]`
3. 调用底层评分模型的 `get_score()` 方法
4. 将分数 clamp 到 `[-3.0, 3.0]` 范围

**输出**: 归一化后的标量分数，用于 PPO/GRPO 等强化学习训练中的 reward 信号。

**调用方**: `train_ppo.py`, `train_grpo.py`, `train_dpo.py` 等 RL 相关训练脚本。

---

## 4. 跨脚本调用关系图

```
trainer_utils.py
  │
  ├── get_model_params     ──► init_model, 所有 train_*.py
  ├── is_main_process      ──► Logger, 所有 save/checkpoint 逻辑
  ├── Logger               ──► 所有 train_*.py 的日志输出
  ├── get_lr               ──► train_pretrain, train_full_sft, train_lora
  ├── init_distributed_mode ──► 所有 train_*.py 的 DDP 初始化
  ├── setup_seed           ──► 所有 train_*.py 的随机种子设置
  ├── lm_checkpoint        ──► 所有 train_*.py 的检查点保存/加载
  ├── init_model           ──► train_lora, train_distillation, train_agent
  ├── SkipBatchSampler     ──► train_grpo, train_ppo
  └── LMForRewardModel     ──► train_ppo, train_grpo, train_dpo
```

---

## 总结

`trainer_utils.py` 是整个训练框架的**共享基础设施**，共提供了 10 个核心功能（7 个函数 + 2 个类 + 1 个辅助函数）。它的设计原则是：

1. **通用性**: 所有函数不依赖具体训练目标，可被任意训练脚本调用
2. **安全性**: 检查点保存使用原子写入（`.tmp` → `replace`），防止文件损坏
3. **兼容性**: DDP 单卡/多卡自动检测、`torch.compile` 和 DDP 包装自动解包
4. **可复现**: `setup_seed` 全随机源覆盖

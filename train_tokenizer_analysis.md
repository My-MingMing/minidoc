# train_tokenizer.py 初学者完全指南

> 文件路径: `trainer/train_tokenizer.py`
> 代码总量: 约 170 行
> 预备知识: 了解 Python 基础即可

> **重要提示**: MiniMind 已经自带分词器，不建议重复训练。此脚本仅供学习参考。不同的分词器会导致同一模型的输出完全不同，降低社区复用性。

---

## 一、什么是分词器？为什么大模型需要自己的分词器？

### 1.1 模型只看数字，不吃文字

深度神经网络（包括所有大语言模型）是纯数学运算的——它们输入的是数字张量，输出的也是数字张量。**文字必须被翻译成数字，模型才能处理。**

分词器就是这个"翻译官"：

```
"你好世界"
    ↓ 分词器编码
[1024, 2048, 332]          ← 每段文字变一个整数
    ↓ 再转成嵌入向量
[[0.1, -0.3, ...], ...]    ← 模型真正吃的东西
```

### 1.2 为什么不一个字一个编号？

听起来很简单：给每个字一个编号不就行了？问题在于：

```
切分粒度     例子                问题
─────────────────────────────────────────
按字切      我/喜/欢/吃/苹/果     中文可以，但英文 "transformers" 变成 12 个 token，太碎
按词切      我/喜欢/吃/苹果       英文词数上百万，词表太大放不进模型
按句切      "整个句子"            句子组合无限，词表爆炸
```

现代方案是**子词切分（Subword Tokenization）**——介于字和词之间的"字串片段"：

```
中文： "我喜欢吃苹果" → ["我喜欢", "吃", "苹果"]   3 个 token
英文： "unhappiness"   → ["un", "happiness"]       2 个 token
罕见： "supercalifrag..." → ["su", "per", "ca", ...] 拆开也能表示
```

**核心思想**：常见模式压缩成一个 token（高效），罕见模式拆成小片段（全覆盖）。

### 1.3 类比：乐高积木

想象你在教孩子用乐高建城市：

- **按字切** = 每片积木只占 1 个点（太小，拼一座楼需要成百上千片）
- **按词切** = 每片积木是一整栋预制房子（太大了，想拼个车库都不行）
- **子词切** = 有 1x1 小块，也有预制的门窗、车轮（常见部件成品，特殊部件现拼）

BPE 就是**自动学习"哪些零件最常一起出现，应该做成预制件"的算法。**

---

## 二、BPE（Byte-Pair Encoding）逐步图解

### 2.1 BPE 的核心逻辑

BPE 只做一件事：**反复找"最常相邻出现的两个片段"，把它们合并成一个。**

### 2.2 动手演练

假设训练集只有下面 4 句话：

```
"a b b c"
"a b a b"
"b c a b"
"a a b b"
```

**第 0 轮：按空格拆到最小**

```
a | b | b | c
a | b | a | b
b | c | a | b
a | a | b | b
```

当前词表：`{a, b, c}`，共 3 个。

**第 1 轮：统计所有相邻对**

```
相邻对        出现次数
──────────────────────
("a", "b")      5 次  ← 最高！
("b", "b")      2 次
("b", "c")      2 次
("c", "a")      1 次
("a", "a")      1 次
```

`("a", "b")` 出现最多，合并成 `ab`：

```
ab | b | c
ab | ab
b | c | ab
a | a | bb
```

当前词表：`{a, b, c, ab}`，共 4 个。

**第 2 轮：重新统计**

```
相邻对        出现次数
──────────────────────
("ab", "b")     1 次
("ab", "ab")    1 次
("b", "c")      1 次
...
```

假设出现次数相同，按规则选先扫描到的——合并 `("ab", "b")` 成 `abb`：

```
abb | c
ab  | ab
b   | c | ab
a   | a | bb
```

当前词表：`{a, b, c, ab, abb}`。

**反复执行几百轮之后：**

```
高频组合都变成了预制件（单个 token）
低频组合拆成小片段也能表示
```

### 2.3 MiniMind 用的是 Byte-Level BPE

上面的例子用"空格分隔的字符"做演示，但 MiniMind 的最小单位不是字符，而是**字节（0-255）**。

为什么？因为：

```
中文 "你" → UTF-8 编码为 [0xE4, 0xBD, 0xA0]  三个字节
emoji ""  → UTF-8 编码为 [0xF0, 0x9F, 0xA4, 0x96]  四个字节
```

字节级 BPE 保证了**世界上任何文字、任何符号都能被表示**，不存在"不认识的字"。这也是 GPT-2/3/4、Llama 等主流模型采用的方案。

### 2.4 直观对比：不同分词方式的 Token 数量

同一段话 "我喜欢 Transformers"：

```
按字：我/喜/欢/T/r/a/n/s/f/o/r/m/e/r/s     → 15 个 token
按词：我/喜欢/Transformers                  → 3 个 token（但词表要够大）
BPE  ：我喜欢/Trans/form/ers               → 4 个 token（平衡）
```

---

## 三、代码逐块解析

### 3.1 配置区（第 7-9 行）

```python
DATA_PATH = '../dataset/sft_t2t_mini.jsonl'   # 训练数据
TOKENIZER_DIR = '../model_learn_tokenizer/'   # 输出目录
VOCAB_SIZE = 6400                              # 最终词表大小
SPECIAL_TOKENS_NUM = 36                        # 特殊 token 数量
```

| 配置项 | 值 | 含义 |
|--------|-----|------|
| `DATA_PATH` | JSONL 文件 | 从对话数据中提取纯文本 |
| `VOCAB_SIZE` | 6400 | 词表总共 6400 个条目 |
| `SPECIAL_TOKENS_NUM` | 36 | 36 个特殊 token 占位 |

**为什么词表这么小？** GPT-4 的词表约 10 万，MiniMind 只要 6400。原因有二：

1. 模型只有 64M 参数，大词表会让 embedding 层占用太多比例，留给"真正学习"的容量不够。
2. MiniMind 面向中文场景，6400 已经能较好地覆盖常用中文子词。

### 3.2 数据读取：get_texts 函数（第 12-22 行）

```python
def get_texts(data_path):
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i >= 10000: break  # 只取 10000 行
            try:
                data = json.loads(line)
                contents = [item.get('content') for item in data.get('conversations', [])
                            if item.get('content')]
                if contents:
                    yield "\n".join(contents)
            except json.JSONDecodeError:
                continue
```

这段代码做的事情：**从 JSONL 对话数据中，把所有内容抽出来变成纯文本流。**

```
一行 JSONL 输入:
{"conversations": [
  {"role": "user",      "content": "你好"},
  {"role": "assistant", "content": "你好！我是 MiniMind"}
]}

提取后的 text:
"你好\n你好！我是 MiniMind"
```

**为什么用 `yield`？** 这是 Python 生成器语法——一行一行地读、读完就扔，不需要把所有数据加载到内存。如果训练集有 10GB，用 `return` 会撑爆内存，用 `yield` 只用几 KB。

**为什么限制 10000 行？** 训练分词器不需要海量数据，取 1 万行做抽样已经足够学习常见模式。

### 3.3 分词器的"骨架"（第 25-26 行）

```python
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
```

这行创建了一个空的 BPE 模型，并设置预处理器为 **Byte-Level**。预处理器在 BPE 合并之前先把文字按字节切分：

```
原始文字: "Hello 你好"
字节预处理: H e l l o [空格] <0xE4> <0xBD> <0xA0> <0xE5> <0xA5> <0xBD>
```

所有后续的合并都在这个字节级别上进行。

### 3.4 Trainer 配置（第 43-48 行）

```python
trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=all_special_tokens
)
```

| 参数 | 作用 |
|------|------|
| `vocab_size` | 词表总大小，合并到 6400 个条目就停止 |
| `show_progress` | 显示训练进度条 |
| `initial_alphabet` | 初始字母表，使用字节级的 256 个完整字节值 |
| `special_tokens` | 特殊 token 列表（下面详细讲） |

### 3.5 训练与解码器（第 49-52 行）

```python
texts = get_texts(data_path)              # 获取文本流
tokenizer.train_from_iterator(texts, trainer=trainer)  # 开始训练
tokenizer.decoder = decoders.ByteLevel()  # 设置反向解码器
tokenizer.add_special_tokens(special_tokens_list)      # 注册特殊 token
```

`train_from_iterator` 会遍历所有文本，执行 BPE 合并，直到词表达到 `vocab_size`。

**Decoder 的作用**：把 token ID 序列还原回文字。Byte-Level 解码器能正确处理多字节 UTF-8 字的边界。

### 3.6 保存模型（第 54-56 行）

```python
os.makedirs(tokenizer_dir, exist_ok=True)
tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
tokenizer.model.save(tokenizer_dir)
```

输出两个文件：

- `tokenizer.json` — 完整的分词器（词表 + 合并规则 + 特殊 token）
- `tokenizer.model` / `vocab.json` — BPE 的合并规则和词表（SentencePiece 格式）

### 3.7 配置修正（第 57-76 行）

这段代码做了两件事：

1. **修正 `added_tokens` 标记**：把非核心特殊 token 的 `special` 字段改为 `False`，避免 HF tokenizer 把它们当真正的特殊标记处理。

2. **构建 `added_tokens_decoder`**：为每个特殊 token 创建解码映射。`special=True` 的 token 在解码时会跳过 `lstrip`、空格拼接等处理，确保它们原样出现。

```python
added_tokens_decoder[str(idx)] = {
    "content": token,
    "lstrip": False,      # 不在左侧去掉空格
    "normalized": False,  # 不做文本规范化
    "rstrip": False,      # 不在右侧去掉空格
    "single_word": False, # 不强制当作独立词
    "special": True/False
}
```

### 3.8 写出 tokenizer_config.json（第 78-105 行）

这是 HuggingFace `transformers` 库要求读取的配置文件：

```python
config = {
    "bos_token": "",          # 序列开始标记
    "eos_token": "",          # 序列结束标记
    "pad_token": "",          # 填充标记（batch 对齐用）
    "unk_token": "",          # 未登录词标记
    "model_max_length": 131072,  # 最大支持 128K 上下文
    "chat_template": "...",   # 对话模板（Jinja2 语法）
    ...
}
```

---

## 四、特殊 Token 全解

特殊 Token 不是从训练数据中学来的，而是**人为设计的"控制标记"**，告诉模型"这里发生了什么"。MiniMind 共定义了 36 个特殊 token。

### 4.1 核心对话控制（6 个）

```
Token ID  Token               用途
─────────────────────────────────────────────────────────
0                           序列开始（Begin Of Sequence）
1                           序列结束（End Of Sequence）
2                           未知/未登录词（Unknown）
3                           填充（Padding）
4         <think>            用户消息开始
5                           助手回复开始
```

### 4.2 工具调用标记（6 个）

```
Token ID  Token               用途
─────────────────────────────────────────────────────────
6                            工具调用开始
7         
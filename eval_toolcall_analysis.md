# eval_toolcall.py 深度解析——工具调用评测与演示

> 先懂概念，再看代码。本文从"什么是工具调用"出发，逐步深入评测脚本的每个环节。

---

> 文件路径: `scripts/eval_toolcall.py`
> 代码总量: 约 241 行
> 预备知识: LLM 工具调用（Function Calling）概念、JSON 解析、OpenAI API 格式
> 阅读建议: 工具调用是 LLM 从"只会说话"进化到"能做事"的关键能力。本脚本完整演示了 MiniMind 的 tool calling 全流程。建议先阅读 `model_minimind_analysis.md` 了解模型生成，再来看这个评测脚本。

---

## 目录

1. [什么是工具调用](#1-什么是工具调用)
2. [整体架构](#2-整体架构)
3. [8 个 Mock 工具](#3-8-个-mock-工具)
4. [双后端设计：local vs api](#4-双后端设计local-vs-api)
5. [工具调用解析](#5-工具调用解析)
6. [多轮工具调用循环 run_case](#6-多轮工具调用循环-run_case)
7. [测试用例设计](#7-测试用例设计)
8. [命令行参数一览](#8-命令行参数一览)
9. [使用示例](#9-使用示例)

---

## 1. 什么是工具调用

普通 LLM 只能"说"，不能"做"。工具调用（Tool Calling / Function Calling）让模型能够：

```
用户: "现在几点了？"

普通 LLM:  "抱歉我无法获取实时时间"       ← 只能说话
Tool Call:  <tool_call>{"name":"get_current_time"}</tool_call>
            → 执行工具 → 返回 "2024-01-15 14:30:00"
            → 模型继续: "现在是2024年1月15日14:30"    ← 能做事了
```

完整流程：

```
  用户问题
     │
     ▼
  模型判断: 需要工具吗?
     │
     ├─ 不需要 → 直接回答
     │
     └─ 需要 → 输出 <tool_call>{...}</tool_call>
                    │
                    ▼
               系统解析 + 执行工具
                    │
                    ▼
               将结果注入对话（role: tool）
                    │
                    ▼
               模型基于工具结果生成最终回答
```

---

## 2. 整体架构

```
eval_toolcall.py
│
├── 工具定义层
│   ├── TOOLS[]           ← 8 个工具的 JSON Schema 定义
│   ├── MOCK_RESULTS{}    ← 每个工具的 mock 执行函数
│   └── TEST_CASES[]      ← 8 个预设测试用例
│
├── 推理后端层
│   ├── init_model()      ← local 模式: 加载本地模型
│   ├── generate()        ← local 模式: 本地生成
│   └── chat_api()        ← api 模式: 调用远程 API
│
├── 工具调用解析层
│   ├── parse_tool_calls()         ← 从文本中提取 <tool_call> 标签
│   ├── parse_tool_call_from_text()← 解析为 OpenAI 格式的 tool_calls
│   └── execute_tool()             ← 执行 mock 工具
│
└── 评测循环
    └── run_case()        ← 多轮工具调用循环
```

---

## 3. 8 个 Mock 工具

### 3.1 工具定义（JSON Schema 格式）

每个工具遵循 OpenAI 的 function calling 规范：

```python
{
    "type": "function",
    "function": {
        "name": "calculate_math",
        "description": "计算数学表达式的结果",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式"}
            },
            "required": ["expression"]
        }
    }
}
```

### 3.2 工具一览

| 工具名 | 功能 | 必需参数 | Mock 实现 |
|---|---|---|---|
| `calculate_math` | 计算数学表达式 | expression | `eval()` 计算 |
| `get_current_time` | 获取当前时间 | 无 | `datetime.now()` |
| `random_number` | 生成随机数 | 无（可选 min/max） | `random.randint()` |
| `text_length` | 统计文本长度 | text | `len()` + `split()` |
| `unit_converter` | 单位转换 | value, from_unit, to_unit | 固定系数换算 |
| `get_current_weather` | 获取天气 | location | 返回固定值 |
| `get_exchange_rate` | 查询汇率 | from_currency, to_currency | 返回固定汇率 |
| `translate_text` | 翻译文本 | text, target_language | 返回固定结果 |

### 3.3 Mock vs 真实工具

这些工具都是 mock 实现——`calculate_math` 和 `get_current_time` 会返回真实结果，但 `get_current_weather` 永远返回"晴，22°C"。重点在于**测试模型是否能正确决定调用哪个工具并传递正确的参数**，而非工具本身的实现。

---

## 4. 双后端设计：local vs api

### 4.1 两种推理模式

```
┌───────────────────────────────────────────────┐
│                  eval_toolcall.py              │
│                                               │
│    --backend local          --backend api     │
│         │                        │            │
│         ▼                        ▼            │
│  ┌─────────────┐        ┌──────────────┐      │
│  │ init_model() │        │ OpenAI SDK   │      │
│  │ 加载本地模型  │        │ 连接远程 API  │      │
│  │             │        │              │      │
│  │ generate()  │        │ chat_api()   │      │
│  │ 本地推理    │        │ HTTP 请求     │      │
│  └─────────────┘        └──────────────┘      │
└───────────────────────────────────────────────┘
```

### 4.2 local 模式

```python
def generate(model, tokenizer, messages, tools, args):
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True,
        tools=tools, open_thinking=False
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(args.device)
    generated_ids = model.generate(inputs["input_ids"], ...)
    response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):])
    return response
```

直接在进程内加载模型，使用 `apply_chat_template` 将工具描述注入 prompt。

### 4.3 api 模式

```python
def chat_api(client, messages, tools, args, stream=True):
    response = client.chat.completions.create(
        model=args.api_model,
        messages=messages,
        tools=tools,
        stream=stream, ...
    )
    # 处理流式/非流式响应
    # 如果 API 返回的 tool_calls 为空，回退到文本解析
    if not tool_calls:
        tool_calls = parse_tool_call_from_text(content)
    return content, tool_calls
```

通过 OpenAI SDK 调用远程 API。支持流式和非流式两种模式。

**回退解析策略**：如果 API 没有通过标准 `tool_calls` 字段返回工具调用（例如 MiniMind 的 API 可能将工具调用嵌在文本中），则回退到 `parse_tool_call_from_text()` 从文本中提取 `<tool_call>` 标签。

---

## 5. 工具调用解析

### 5.1 从文本中提取工具调用

```python
def parse_tool_calls(text):
    """用于 local 模式：从模型输出文本中提取 <tool_call> 标签"""
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    calls = []
    for m in matches:
        calls.append(json.loads(m.strip()))  # {"name": "...", "arguments": {...}}
    return calls
```

### 5.2 转换为 OpenAI 格式

```python
def parse_tool_call_from_text(content):
    """用于 api 模式回退：将文本中的 tool_call 转为 OpenAI 标准格式"""
    # 输入: <tool_call>{"name":"get_time","arguments":{}}</tool_call>
    # 输出: [{"id": "call_0", "function": {"name": "get_time", "arguments": "{}"}}]
```

两个解析函数的区别：

| 函数 | 用途 | 输出格式 |
|---|---|---|
| `parse_tool_calls` | local 模式 | `[{"name": "...", "arguments": {...}}]` |
| `parse_tool_call_from_text` | api 模式回退 | `[{"id": "...", "function": {"name": "...", "arguments": "..."}}]` |

### 5.3 工具执行

```python
def execute_tool(call, arguments=None):
    name = call.get("name", "") if isinstance(call, dict) else call
    args = ...  # 解析 arguments（字符串或字典）
    fn = MOCK_RESULTS.get(name)
    return fn(args)
```

根据工具名查找 `MOCK_RESULTS` 字典中的对应 lambda 函数并执行。

---

## 6. 多轮工具调用循环 run_case

这是评测的核心函数，实现了完整的 tool calling 多轮交互：

```python
def run_case(prompt, tools, args, model=None, tokenizer=None, client=None):
    messages = [{"role": "user", "content": prompt}]
    while True:
        # 1. 生成回复
        if args.backend == 'local':
            content = generate(model, tokenizer, messages, tools, args)
            tool_calls = parse_tool_calls(content)
        else:
            content, tool_calls = chat_api(client, messages, tools, args)

        # 2. 没有工具调用 → 结束
        if not tool_calls:
            break

        # 3. 将助手回复加入对话
        messages.append({"role": "assistant", "content": content, ...})

        # 4. 执行每个工具 → 将结果加入对话
        for tc in tool_calls:
            result = execute_tool(tc)
            messages.append({"role": "tool", "content": json.dumps(result)})

        # 5. 回到步骤 1，模型基于工具结果继续生成
```

### 多轮工具调用示例

```
💬: 帮我生成一个1到1000的随机数，然后计算它的平方

🧠: <tool_call>{"name":"random_number","arguments":{"min":1,"max":1000}}</tool_call>
📞 [Tool Calling]: random_number | args={"min":1,"max":1000}
✅ [Tool Called]: {"result": 347}

🧠: <tool_call>{"name":"calculate_math","arguments":{"expression":"347**2"}}</tool_call>
📞 [Tool Calling]: calculate_math | args={"expression":"347**2"}
✅ [Tool Called]: {"result": "120409"}

🧠: 随机数是 347，它的平方是 120409。
```

这个例子展示了**链式工具调用**：模型先调用 `random_number` 获取随机数，再用结果调用 `calculate_math` 计算平方。

---

## 7. 测试用例设计

```python
TEST_CASES = [
    {"prompt": "帮我算一下 256 乘以 37 等于多少",
     "tools": ["calculate_math", "get_current_time"]},
    {"prompt": "现在几点了？",
     "tools": ["get_current_time", "random_number"]},
    {"prompt": "帮我把100公里换算成英里",
     "tools": ["unit_converter", "calculate_math"]},
    # ...
]
```

每个测试用例设计包含：
- **prompt**：用户问题
- **tools**：提供给模型的工具子集（不是全部 8 个）

**为什么不给全部工具？** 考察模型在有限工具集中的选择能力——能否在干扰项存在时选对工具。

| # | 考察能力 | 关键工具 | 干扰工具 |
|---|---|---|---|
| 1 | 数学计算 | calculate_math | get_current_time |
| 2 | 时间查询 | get_current_time | random_number |
| 3 | 单位转换 | unit_converter | calculate_math |
| 4 | 链式调用 | random_number + calculate_math | text_length |
| 5 | 天气查询 | get_current_weather | get_current_time |
| 6 | 汇率查询 | get_exchange_rate | get_current_time |
| 7 | 文本翻译 | translate_text | text_length |
| 8 | 英文+多工具 | get_current_weather + unit_converter | get_current_time |

---

## 8. 命令行参数一览

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--backend` | `'local'` | 推理后端（`local` = 本地模型，`api` = 远程 API） |
| `--load_from` | `'../model'` | 模型加载路径（local 模式） |
| `--save_dir` | `'../out'` | 权重文件目录 |
| `--weight` | `'full_sft'` | 权重名称前缀 |
| `--hidden_size` | `768` | 隐藏层维度 |
| `--num_hidden_layers` | `8` | Transformer 层数 |
| `--use_moe` | `0` | 是否使用 MoE |
| `--max_new_tokens` | `512` | 最大生成长度 |
| `--temperature` | `0.9` | 采样温度 |
| `--top_p` | `0.9` | nucleus 采样 |
| `--show_speed` | `0` | 是否显示生成速度 |
| `--device` | 自动检测 | 运行设备 |
| `--api_base_url` | `localhost:11434/v1` | API 地址（api 模式） |
| `--api_key` | `'sk-123'` | API key（api 模式） |
| `--api_model` | `'jingyaogong/minimind-3:latest'` | API 模型名（api 模式） |
| `--stream` | `1` | API 模式是否流式输出 |

---

## 9. 使用示例

### 9.1 本地模式自动测试

```bash
cd scripts
python eval_toolcall.py --backend local --weight full_sft
# 选择 [0] 自动测试
```

### 9.2 API 模式手动测试

```bash
# 先启动服务器
python serve_openai_api.py --weight full_sft

# 再运行评测
python eval_toolcall.py --backend api --api_base_url http://localhost:8998/v1
# 选择 [1] 手动输入
```

### 9.3 关键依赖关系

```
eval_toolcall.py
├── model.model_minimind    → 本地模型加载（local 模式）
├── trainer.trainer_utils    → setup_seed, get_model_params
├── openai (SDK)            → API 调用（api 模式）
└── transformers            → tokenizer + TextStreamer
```

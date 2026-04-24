# chat_api.py 深度解析——最小化 CLI 聊天客户端

> 先懂概念，再看代码。本文解析 MiniMind 最简洁的客户端脚本。

---

> 文件路径: `scripts/chat_api.py`
> 代码总量: 约 40 行
> 预备知识: OpenAI Python SDK 基础、HTTP API 调用
> 阅读建议: 这是整个 MiniMind 项目中最短的脚本，但它清晰展示了"如何通过 OpenAI 兼容接口与 MiniMind 对话"。建议先阅读 `serve_openai_api_analysis.md` 了解服务端实现，再来看这个客户端。

---

## 目录

1. [定位：客户端与服务端的关系](#1-定位客户端与服务端的关系)
2. [代码全解析](#2-代码全解析)
3. [thinking 模式的流式展示](#3-thinking-模式的流式展示)
4. [多轮对话管理](#4-多轮对话管理)
5. [使用方法](#5-使用方法)

---

## 1. 定位：客户端与服务端的关系

```
┌──────────────────┐              ┌──────────────────────────┐
│  chat_api.py     │   HTTP POST  │  serve_openai_api.py     │
│  (CLI 客户端)    │ ──────────→  │  (API 服务器)             │
│                  │              │                          │
│  使用 OpenAI SDK │   SSE stream │  端口: 11434 或 8998     │
│  发送消息并接收  │ ←──────────  │  承载 MiniMind 模型      │
└──────────────────┘              └──────────────────────────┘
```

`chat_api.py` 是一个**纯客户端脚本**——它本身不加载模型，只通过 HTTP 调用远程（或本地）的 OpenAI 兼容 API。它可以连接：
- 本地的 `serve_openai_api.py`
- Ollama 服务（默认端口 11434）
- 任何 OpenAI 兼容的 API 服务

---

## 2. 代码全解析

整个脚本只有 40 行，我们逐段看：

### 2.1 客户端初始化

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-123",                        # 占位 key（本地服务不校验）
    base_url="http://localhost:11434/v1"      # 默认连接 Ollama 端口
)
stream = True                                 # 启用流式输出
```

### 2.2 对话历史管理

```python
conversation_history_origin = []
conversation_history = conversation_history_origin.copy()
history_messages_num = 0  # 必须设置为偶数（Q+A），为0则不携带历史对话
```

- `history_messages_num = 0`：每轮对话独立（默认）
- `history_messages_num = 4`：携带最近 2 轮对话（2 个 Q + 2 个 A）

### 2.3 对话循环

```python
while True:
    query = input('[Q]: ')
    conversation_history.append({"role": "user", "content": query})
    response = client.chat.completions.create(
        model="minimind-local:latest",
        messages=conversation_history[-(history_messages_num or 1):],
        stream=stream,
        temperature=0.8,
        max_tokens=2048,
        top_p=0.8,
        extra_body={
            "chat_template_kwargs": {"open_thinking": True},
            "reasoning_effort": "medium"
        }
    )
```

关键细节：
- `messages` 截取：`-(history_messages_num or 1)` 确保至少发送当前消息
- `extra_body`：OpenAI SDK 的扩展机制，传递非标准参数给服务端
  - `open_thinking: True`：开启思考模式
  - `reasoning_effort: "medium"`：控制思考深度

### 2.4 流式/非流式响应处理

```python
if not stream:
    assistant_res = response.choices[0].message.content
    print('[A]: ', assistant_res)
else:
    print('[A]: ', end='', flush=True)
    assistant_res = ''
    for chunk in response:
        delta = chunk.choices[0].delta
        r = getattr(delta, 'reasoning_content', None) or ""
        c = delta.content or ""
        if r:
            print(f'\033[90m{r}\033[0m', end="", flush=True)  # 灰色显示推理
        if c:
            print(c, end="", flush=True)                       # 正常显示回答
        assistant_res += c
```

---

## 3. thinking 模式的流式展示

当服务端返回 `reasoning_content` 时，客户端用 ANSI 转义码将推理过程显示为灰色：

```
[Q]: 1+1等于多少？
[A]: 让我想想... 1加1等于2，这是最基础的加法运算...    ← 灰色（推理过程）
答案是2。                                              ← 正常颜色（最终回答）
```

| ANSI 码 | 含义 |
|---|---|
| `\033[90m` | 亮黑色（灰色） |
| `\033[0m` | 重置颜色 |

注意：只有 `content` 部分被追加到 `assistant_res`，`reasoning_content` 只在终端展示但不存入对话历史。

---

## 4. 多轮对话管理

```python
conversation_history.append({"role": "assistant", "content": assistant_res})
```

每轮对话后，助手回复被追加到历史列表中。下一轮发送时通过切片 `[-(history_messages_num or 1):]` 控制携带多少历史。

```
history_messages_num = 0  →  每轮只发当前消息
history_messages_num = 2  →  发送 [上一轮Q, 上一轮A, 当前Q]
history_messages_num = 4  →  发送 [Q1, A1, Q2, A2, 当前Q]
```

**注意**：必须设置为偶数，因为每轮对话包含一个 user 消息和一个 assistant 消息。

---

## 5. 使用方法

### 5.1 前置条件

确保 API 服务器已启动：

```bash
# 方式一：使用 MiniMind 自带服务器
cd scripts && python serve_openai_api.py

# 方式二：使用 Ollama
ollama run minimind-local
```

### 5.2 启动客户端

```bash
python scripts/chat_api.py
```

### 5.3 自定义连接

如需连接不同的服务端，修改脚本中的 `base_url`：

```python
client = OpenAI(
    api_key="sk-123",
    base_url="http://localhost:8998/v1"   # 改为 serve_openai_api.py 的端口
)
```

### 5.4 关键依赖

```
chat_api.py
└── openai (Python SDK)  →  唯一依赖，不直接加载模型
```

# serve_openai_api.py 深度解析——OpenAI 兼容 API 服务器

> 先懂概念，再看代码。本文从"为什么需要 API 服务器"出发，逐步深入每个组件的实现。

---

> 文件路径: `scripts/serve_openai_api.py`
> 代码总量: 约 246 行
> 预备知识: HTTP API 基础、FastAPI 框架、SSE（Server-Sent Events）流式协议、OpenAI Chat Completions API 格式
> 阅读建议: 这个脚本把 MiniMind 模型包装成一个标准的 OpenAI 兼容接口，让任何支持 OpenAI SDK 的客户端都能调用。建议先了解 `model_minimind.py` 的生成逻辑，再来看这个服务封装。

---

## 目录

1. [为什么需要 API 服务器](#1-为什么需要-api-服务器)
2. [整体架构](#2-整体架构)
3. [模型加载 init_model](#3-模型加载-init_model)
4. [请求与响应格式 ChatRequest](#4-请求与响应格式-chatrequest)
5. [流式输出核心 CustomStreamer](#5-流式输出核心-customstreamer)
6. [响应解析 parse_response](#6-响应解析-parse_response)
7. [流式生成 generate_stream_response](#7-流式生成-generate_stream_response)
8. [API 端点 chat_completions](#8-api-端点-chat_completions)
9. [命令行参数一览](#9-命令行参数一览)
10. [使用示例](#10-使用示例)

---

## 1. 为什么需要 API 服务器

直接用 `eval_llm.py` 做推理只能在本地终端使用。如果你想：

- 让 **Web UI**（如 `web_demo.py`）通过网络调用模型
- 让 **第三方工具**（如 ChatBox、Open WebUI、Cursor）接入 MiniMind
- 让 **多个客户端** 同时访问同一个模型
- 测试 MiniMind 的 **工具调用**（tool calling）能力

你需要一个 HTTP 服务器。`serve_openai_api.py` 就是这个角色——它把 MiniMind 包装成一个 **OpenAI 兼容的 API**，任何支持 OpenAI SDK 的客户端都能无缝接入。

```
┌──────────────┐        HTTP POST         ┌────────────────────────┐
│  chat_api.py │  ──────────────────────→  │  serve_openai_api.py   │
│  (CLI 客户端) │                           │                        │
├──────────────┤  /v1/chat/completions     │  FastAPI + MiniMind    │
│  web_demo.py │  ──────────────────────→  │                        │
│  (Web UI)    │                           │  ┌──────────────────┐  │
├──────────────┤        SSE stream         │  │ MiniMind Model   │  │
│  ChatBox     │  ←─────────────────────   │  │ (GPU)            │  │
│  Open WebUI  │                           │  └──────────────────┘  │
│  Cursor 等   │                           │  端口: 8998            │
└──────────────┘                           └────────────────────────┘
```

---

## 2. 整体架构

整个脚本可以分为 **5 个组件**：

```
serve_openai_api.py
│
├── init_model()              ← 模型加载（同 eval_llm.py 逻辑）
├── ChatRequest               ← 请求数据结构（Pydantic 模型）
├── CustomStreamer             ← 把 generate() 的输出接入 Queue
├── parse_response()          ← 解析 <think> 和 <tool_call> 标签
├── generate_stream_response()← 流式生成器，产出 SSE 事件
└── chat_completions()        ← FastAPI 路由，处理 POST 请求
```

---

## 3. 模型加载 init_model

加载逻辑与 `eval_llm.py` 完全一致，支持原生 PyTorch 和 Transformers 两种路径，支持 LoRA 叠加和 MoE 架构：

```python
def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        # 原生 PyTorch 路径
        model = MiniMindForCausalLM(MiniMindConfig(...))
        model.load_state_dict(torch.load(ckp))
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, lora_path)
    else:
        # Transformers 路径
        model = AutoModelForCausalLM.from_pretrained(args.load_from)
    return model.half().eval().to(device), tokenizer
```

注意：服务器脚本的权重路径前缀是 `../`（因为运行目录是 `scripts/`）。

---

## 4. 请求与响应格式 ChatRequest

### 4.1 请求结构

```python
class ChatRequest(BaseModel):
    model: str                         # 模型名称（兼容用，不影响实际推理）
    messages: list                     # 对话消息列表
    temperature: float = 0.7           # 采样温度
    top_p: float = 0.92                # nucleus 采样
    max_tokens: int = 8192             # 最大生成 token 数
    stream: bool = True                # 是否流式输出
    tools: list = []                   # 工具定义列表
    open_thinking: bool = False        # 是否开启思考模式
    chat_template_kwargs: dict = None  # 透传给 chat template 的参数
```

### 4.2 thinking 模式的兼容设计

```python
def get_open_thinking(self) -> bool:
    if self.open_thinking:
        return True
    if self.chat_template_kwargs:
        return self.chat_template_kwargs.get('open_thinking', False) or \
               self.chat_template_kwargs.get('enable_thinking', False)
    return False
```

支持三种方式开启 thinking：
1. 直接设置 `open_thinking: true`
2. 通过 `chat_template_kwargs.open_thinking`
3. 通过 `chat_template_kwargs.enable_thinking`（兼容不同客户端的叫法）

---

## 5. 流式输出核心 CustomStreamer

### 5.1 为什么需要自定义 Streamer

`model.generate()` 是阻塞调用——它跑完所有 token 才返回。但 HTTP 流式响应需要**边生成边发送**。解决方案是用线程 + 队列：

```
┌──────────────┐     Queue      ┌──────────────────┐
│  生成线程     │  ──put()──→   │  HTTP 响应生成器   │
│  model.       │               │  generate_stream_  │
│  generate()  │  ←─get()───   │  response()       │
│              │               │                    │
│  CustomStreamer               │  yield SSE chunks  │
│  .on_finalized_text()        │                    │
└──────────────┘               └──────────────────┘
```

### 5.2 实现

```python
class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)          # 每生成一小段文本就放入队列
        if stream_end:
            self.queue.put(None)      # None 作为结束信号
```

继承 `TextStreamer` 并重写 `on_finalized_text`：
- 每产出一段 finalized text，放入 `Queue`
- 生成结束时放入 `None` 作为终止信号
- `skip_prompt=True`：不输出 prompt 部分
- `skip_special_tokens=True`：不输出特殊 token（如 EOS）

---

## 6. 响应解析 parse_response

模型的原始输出可能包含特殊标签，需要提取和清理：

```python
def parse_response(text):
    # 1. 提取 <think>...</think> 推理内容
    reasoning_content = None
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    elif '</think>' in text:
        # 处理没有 <think> 开标签的情况（模型直接输出推理内容后跟 </think>）
        parts = text.split('</think>', 1)
        reasoning_content = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else ''

    # 2. 提取 <tool_call>...</tool_call> 工具调用
    tool_calls = []
    for i, m in enumerate(re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)):
        try:
            call = json.loads(m.strip())
            tool_calls.append({
                "id": f"call_{int(time.time())}_{i}",
                "type": "function",
                "function": {
                    "name": call.get("name", ""),
                    "arguments": json.dumps(call.get("arguments", {}))
                }
            })
        except Exception:
            pass

    return text.strip(), reasoning_content, tool_calls or None
```

处理三种输出：

| 场景 | 模型原始输出示例 | 解析结果 |
|---|---|---|
| 普通回答 | `你好，我是MiniMind` | content = "你好，我是MiniMind" |
| 带思考 | `<think>先分析...</think>答案是42` | reasoning = "先分析...", content = "答案是42" |
| 工具调用 | `<tool_call>{"name":"get_time"}</tool_call>` | tool_calls = [{name: "get_time"}] |

---

## 7. 流式生成 generate_stream_response

这是整个服务器最复杂的函数，负责**实时**解析生成内容并以 SSE 格式发送。

### 7.1 核心流程

```
apply_chat_template → tokenize → 启动生成线程
                                       │
                                  Queue.get() 循环
                                       │
                          ┌────────────┼────────────┐
                          │                         │
                    thinking 阶段             正常内容阶段
                          │                         │
              yield reasoning_content     yield content
                          │                         │
                    遇到 </think> 切换 ──→          │
                                                    │
                                              生成结束
                                                    │
                                          解析 tool_calls
                                                    │
                                    yield finish_reason (stop/tool_calls)
```

### 7.2 thinking 阶段状态机

```python
thinking_ended = not bool(open_thinking)  # 未开启 thinking 则直接跳过

while True:
    text = queue.get()
    if text is None: break
    full_text += text

    if not thinking_ended:
        pos = full_text.find('</think>')
        if pos >= 0:
            thinking_ended = True
            # 输出剩余 reasoning_content，切换到 content
        else:
            # 持续输出 reasoning_content
    else:
        # 正常输出 content
```

当开启 thinking 时，生成过程分两阶段：
1. **推理阶段**：`</think>` 之前的内容作为 `reasoning_content` 发送
2. **回答阶段**：`</think>` 之后的内容作为 `content` 发送

### 7.3 SSE 输出格式

每个 chunk 都是 JSON，格式兼容 OpenAI：

```json
{"choices": [{"delta": {"reasoning_content": "让我想想..."}}]}
{"choices": [{"delta": {"content": "答案是42"}}]}
{"choices": [{"delta": {"tool_calls": [...]}}]}
{"choices": [{"delta": {}, "finish_reason": "stop"}]}
```

---

## 8. API 端点 chat_completions

### 8.1 路由定义

```python
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
```

完全兼容 OpenAI 的 `/v1/chat/completions` 接口路径。

### 8.2 流式 vs 非流式

```
request.stream == True?
    │
    ├── 是 → StreamingResponse (SSE)
    │        data: {"choices": [{"delta": {...}}]}
    │
    └── 否 → 一次性返回完整 JSON
             {"id": "chatcmpl-xxx", "choices": [{"message": {...}}]}
```

**流式模式**：返回 `StreamingResponse`，媒体类型为 `text/event-stream`，每个 chunk 以 `data: {json}\n\n` 格式发送。

**非流式模式**：等待生成完成后返回完整响应，格式包含：
- `id`：`chatcmpl-{timestamp}`
- `model`：`"minimind"`
- `choices[0].message`：包含 content、reasoning_content（可选）、tool_calls（可选）
- `choices[0].finish_reason`：`"stop"` 或 `"tool_calls"`

---

## 9. 命令行参数一览

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--load_from` | `'../model'` | 模型加载路径 |
| `--save_dir` | `'out'` | 权重文件目录 |
| `--weight` | `'full_sft'` | 权重名称前缀 |
| `--lora_weight` | `'None'` | LoRA 权重名称 |
| `--hidden_size` | `768` | 隐藏层维度 |
| `--num_hidden_layers` | `8` | Transformer 层数 |
| `--max_seq_len` | `8192` | 最大序列长度 |
| `--use_moe` | `0` | 是否使用 MoE |
| `--inference_rope_scaling` | `False` | 启用 RoPE 外推 |
| `--device` | 自动检测 | 运行设备 |

服务器固定监听 `0.0.0.0:8998`。

---

## 10. 使用示例

### 10.1 启动服务器

```bash
cd scripts
python serve_openai_api.py --weight full_sft --hidden_size 768
```

### 10.2 用 curl 测试

```bash
curl http://localhost:8998/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"minimind", "messages":[{"role":"user","content":"你好"}], "stream":false}'
```

### 10.3 用 Python OpenAI SDK 调用

```python
from openai import OpenAI
client = OpenAI(api_key="sk-123", base_url="http://localhost:8998/v1")
response = client.chat.completions.create(
    model="minimind",
    messages=[{"role": "user", "content": "你好"}],
    stream=True
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

### 10.4 带工具调用

```python
response = client.chat.completions.create(
    model="minimind",
    messages=[{"role": "user", "content": "现在几点了？"}],
    tools=[{"type": "function", "function": {"name": "get_time", "parameters": {}}}],
    stream=True
)
```

### 10.5 关键依赖关系

```
serve_openai_api.py
├── fastapi + uvicorn       → HTTP 服务器
├── model.model_minimind    → 模型定义
├── model.model_lora        → LoRA 支持
└── transformers            → tokenizer + TextStreamer 基类
```

# web_demo.py 深度解析——Streamlit Web 交互界面

> 先懂概念，再看代码。本文从用户交互出发，逐步深入每个 UI 组件和生成逻辑的实现。

---

> 文件路径: `scripts/web_demo.py`
> 代码总量: 约 421 行
> 预备知识: Streamlit 框架基础、Transformers 生成流程、HTML/CSS 基础
> 阅读建议: 这是 MiniMind 的"门面担当"——一个完整的 Web 聊天界面。建议先了解 `model_minimind.py` 的生成逻辑和 `serve_openai_api.py` 的工具调用解析，再来看这个一体化的前端+后端实现。

---

## 目录

1. [整体定位](#1-整体定位)
2. [架构概览](#2-架构概览)
3. [UI 布局与配置](#3-ui-布局与配置)
4. [模型加载与发现](#4-模型加载与发现)
5. [对话消息管理](#5-对话消息管理)
6. [流式生成引擎](#6-流式生成引擎)
7. [thinking 模式的折叠展示](#7-thinking-模式的折叠展示)
8. [工具调用循环](#8-工具调用循环)
9. [完整对话流程图](#9-完整对话流程图)

---

## 1. 整体定位

与 `chat_api.py`（CLI 客户端 → 远程 API）不同，`web_demo.py` 是一个**一体化方案**：模型直接在进程内加载，不需要外部 API 服务器。

```
chat_api.py:    客户端 ──HTTP──→ 服务端 ──→ 模型
web_demo.py:    浏览器 ──────→ Streamlit 进程（内含模型）
```

适用场景：
- 本地演示和体验
- 快速测试不同模型权重
- 向非技术人员展示模型能力

---

## 2. 架构概览

```
web_demo.py
│
├── 全局初始化
│   ├── CSS 样式注入
│   ├── LANG_TEXTS 多语言文本
│   ├── TOOLS 工具定义（8 个 mock 工具）
│   └── MODEL_PATHS 自动扫描模型目录
│
├── 侧边栏配置
│   ├── 模型选择（动态扫描）
│   ├── 语言切换（中文/English）
│   ├── 参数调节（历史轮次、生成长度、温度）
│   ├── thinking 开关
│   └── 工具选择（最多 4 个）
│
├── 核心函数
│   ├── load_model_tokenizer()  ← @st.cache_resource 缓存
│   ├── process_assistant_content() ← <think>/<tool_call> HTML 渲染
│   ├── execute_tool()          ← 8 个 mock 工具的执行
│   └── setup_seed()            ← 随机种子
│
└── main() 主循环
    ├── 渲染历史消息
    ├── 接收用户输入
    ├── 流式生成（TextIteratorStreamer + Thread）
    └── 工具调用循环（最多 16 轮）
```

---

## 3. UI 布局与配置

### 3.1 多语言支持

```python
LANG_TEXTS = {
    'zh': {
        'settings': '模型设定调整',
        'history_rounds': '历史对话轮次',
        'max_length': '最大生成长度',
        'temperature': '温度',
        'thinking': '思考',
        ...
    },
    'en': {
        'settings': 'Model Settings',
        ...
    }
}
```

所有界面文本通过 `get_text(key)` 函数动态获取，切换语言时整个界面即时刷新。

### 3.2 侧边栏参数

| 控件 | 参数 | 范围 | 默认值 |
|---|---|---|---|
| Slider | 历史对话轮次 | 0-8（步长2） | 0 |
| Slider | 最大生成长度 | 256-8192 | 8192 |
| Slider | 温度 | 0.6-1.2 | 0.90 |
| Checkbox | thinking 模式 | on/off | off |
| Expander | 工具选择 | 最多4个 | 无 |

### 3.3 工具选择限制

```python
selected_count = sum(1 for tool in TOOLS
                     if st.session_state.get(f"tool_{tool['function']['name']}", False))
for tool in TOOLS:
    checked = st.checkbox(
        short_name,
        disabled=(selected_count >= 4 and not st.session_state.get(f"tool_{name}", False))
    )
```

当已选 4 个工具时，未选中的 checkbox 自动 disabled，防止超选。

---

## 4. 模型加载与发现

### 4.1 自动扫描模型目录

```python
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {}
for d in sorted(os.listdir(script_dir), reverse=True):
    full_path = os.path.join(script_dir, d)
    if os.path.isdir(full_path) and not d.startswith('.') and not d.startswith('_'):
        if any(f.endswith(('.bin', '.safetensors', '.pt')) for f in os.listdir(full_path)):
            MODEL_PATHS[d] = [d, d]
```

扫描 `scripts/` 目录下的所有子目录，如果包含模型权重文件（`.bin`、`.safetensors`、`.pt`），就自动列入可选模型列表。这意味着你只需要把 Transformers 格式的模型目录放在 `scripts/` 下，Web UI 就能自动发现它。

### 4.2 缓存加载

```python
@st.cache_resource
def load_model_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.half().eval().to(device)
    return model, tokenizer
```

`@st.cache_resource` 确保模型只加载一次——切换页面或重新输入不会重复加载，大幅提升体验。

---

## 5. 对话消息管理

### 5.1 双消息列表设计

```python
st.session_state.messages      # 用于 UI 展示（含 HTML 格式化后的内容）
st.session_state.chat_messages # 用于模型推理（原始对话格式）
```

为什么需要两个列表？因为 UI 展示需要 HTML（折叠的 `<details>` 标签、工具调用卡片），但模型推理需要纯文本格式。

### 5.2 历史对话截取

```python
st.session_state.chat_messages = sys_prompt + st.session_state.chat_messages[-(history_chat_num + 1):]
```

- `history_chat_num=0`：只发送系统 prompt + 当前消息
- `history_chat_num=4`：发送系统 prompt + 最近 4 条消息 + 当前消息

### 5.3 系统 prompt 策略

```python
tools = [t for t in TOOLS if t['function']['name'] in selected_tools] or None
sys_prompt = [] if tools else [{"role": "system", "content": "你是MiniMind..."}]
```

关键设计：**有工具时不发送 system prompt**。这是因为工具描述会通过 chat template 注入系统消息位置，如果再额外加 system prompt 会冲突。

---

## 6. 流式生成引擎

### 6.1 生成架构

```
                     Thread
┌──────────────┐   (后台线程)    ┌──────────────────┐
│ Streamlit    │                │ model.generate()  │
│ 主线程       │                │                    │
│              │  TextIterator- │ 逐 token 产出      │
│ for new_text │ ←─ Streamer ──│                    │
│   in streamer│                │                    │
│              │                │                    │
│ placeholder  │                │                    │
│ .markdown()  │                │                    │
└──────────────┘                └──────────────────┘
```

### 6.2 为什么用线程

`model.generate()` 是阻塞调用，如果在 Streamlit 主线程运行会冻结 UI。解决方案：

```python
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
generation_kwargs = {
    "input_ids": inputs.input_ids,
    "max_length": inputs.input_ids.shape[1] + st.session_state.max_new_tokens,
    "streamer": streamer,
    ...
}
Thread(target=model.generate, kwargs=generation_kwargs).start()

answer = ""
for new_text in streamer:      # 迭代器，主线程边收边渲染
    answer += new_text
    placeholder.markdown(process_assistant_content(answer, is_streaming=True),
                         unsafe_allow_html=True)
```

`TextIteratorStreamer` 与 `CustomStreamer`（在 `serve_openai_api.py` 中）原理相同，但它实现了 `__iter__` 接口，可以直接用 `for` 循环迭代。

---

## 7. thinking 模式的折叠展示

### 7.1 四种展示场景

`process_assistant_content()` 函数处理四种 `<think>` 标签的组合：

| 场景 | 条件 | 展示方式 |
|---|---|---|
| 完整思考 | `<think>内容</think>` | 折叠的 `<details>` + "已思考" |
| 思考中（流式） | `<think>内容...`（无结束标签） | 展开的 `<details>` + "思考中..." |
| 无开始标签 | `内容</think>` | 折叠的 `<details>` + "已思考" |
| 隐式思考 | 开启 thinking 但无标签 | 根据内容启发式判断 |

### 7.2 HTML 渲染示例

```html
<details open style="border-left: 2px solid #666; padding-left: 12px;">
  <summary style="cursor: pointer; color: #888;">已思考</summary>
  <div style="color: #aaa; font-size: 0.95em; max-height: 100px; overflow-y: auto;">
    让我分析一下这个问题...
  </div>
</details>
```

使用 `<details>` HTML 标签实现可折叠/展开的思考内容，限制最大高度 100px 并支持滚动。

---

## 8. 工具调用循环

### 8.1 循环逻辑

```python
for _ in range(16):                                     # 最多 16 轮
    tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', answer, re.DOTALL)
    if not tool_calls:
        break                                            # 无工具调用则退出

    # 1. 将助手回复加入对话
    chat_messages.append({"role": "assistant", "content": answer})

    # 2. 执行每个工具调用
    for tc_str in tool_calls:
        tc = json.loads(tc_str.strip())
        result = execute_tool(tc['name'], tc['arguments'])
        chat_messages.append({"role": "tool", "content": json.dumps(result)})

    # 3. 重新生成（模型看到工具结果后继续回答）
    new_prompt = tokenizer.apply_chat_template(chat_messages, **template_kwargs)
    # ... 再次启动 Thread + TextIteratorStreamer ...
```

### 8.2 可视化流程

```
用户提问: "北京天气怎么样？"
    │
    ▼
模型生成: <tool_call>{"name":"get_current_weather","arguments":{"city":"北京"}}</tool_call>
    │
    ▼ 解析 + 执行 mock 工具
工具结果: {"result": "北京: 晴, 7~10°C"}
    │
    ▼ 注入 tool message，重新生成
模型生成: "北京今天天气晴朗，温度7~10°C，适合外出。"
    │
    ▼ 无更多 tool_call，循环结束
```

### 8.3 8 个 Mock 工具

| 工具名 | 短名 | 功能 |
|---|---|---|
| `calculate_math` | 数学 | 计算数学表达式 |
| `get_current_time` | 时间 | 获取当前时间 |
| `random_number` | 随机 | 生成随机数 |
| `text_length` | 字数 | 计算文本长度 |
| `unit_converter` | 单位 | 单位转换 |
| `get_current_weather` | 天气 | 获取天气（mock） |
| `get_exchange_rate` | 汇率 | 获取汇率（mock） |
| `translate_text` | 翻译 | 翻译文本（mock） |

所有工具都是 mock 实现，返回预设或简单计算的结果，仅用于演示工具调用能力。

---

## 9. 完整对话流程图

```
  浏览器访问 Streamlit
       │
       ▼
  ┌─ 侧边栏配置 ─────────────────────────────┐
  │ 选模型 → 选语言 → 调参数 → 开/关 thinking │
  │ → 选工具（最多4个）                        │
  └─────────────────────────────────────────────┘
       │
       ▼
  load_model_tokenizer() ← @st.cache_resource
       │
       ▼
  用户输入 prompt ← st.chat_input()
       │
       ▼
  构建 chat_messages（system + 历史 + 当前）
       │
       ▼
  apply_chat_template(tools=..., open_thinking=...)
       │
       ▼
  tokenize → Thread(model.generate) + TextIteratorStreamer
       │
       ▼
  ┌─ 流式渲染循环 ─┐
  │ for text in     │
  │   streamer:     │──→ process_assistant_content() → placeholder.markdown()
  │   answer += text│
  └────────┬────────┘
           │
           ▼
  ┌─ 工具调用循环（最多16轮）─┐
  │ 有 <tool_call>?           │
  │   ├─ 否 → 结束            │
  │   └─ 是 → execute_tool()  │
  │          → 注入 tool msg   │
  │          → 重新生成        │
  └───────────────────────────┘
           │
           ▼
  保存到 messages + chat_messages
```

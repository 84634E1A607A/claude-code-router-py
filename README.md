# ccr-py

轻量代理服务器，接收 [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) 请求，转发到任意 OpenAI 兼容后端，双向转换格式。客户端看到的是标准 Anthropic API，后端看到的是标准 OpenAI API。

```
Claude Code / 任意 Anthropic 客户端
        |  Anthropic 格式
        v
    ccr-py (本服务)
        |  OpenAI 格式
        v
  OpenAI 兼容后端
```

---

## server.py 功能概览

server.py 是 ccr-py 的主体。它做的事情：

1. 接收 Anthropic 格式的请求，转成 OpenAI 格式发给后端，再把响应转回来。流式和非流式都支持。
2. 流式请求不只是转发字节流，内部有个状态机管理 content block 的开关顺序（详见下文"流式处理内部逻辑"）。
3. 除了主接口 `/v1/messages`，还代理了 models、batch、token 计数、旧版 complete 等端点。
4. 通过 config.json 或环境变量配置 provider 和路由，支持参数覆盖（temperature / max_tokens / thinking 等）。

### 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/v1/messages` | Messages API 主接口（流式 + 非流式） |
| `POST` | `/v1/messages/count_tokens` | token 计数（需配置 `tokenizer_path`） |
| `POST` | `/v1/complete` | 旧版文本补全（转换 `\n\nHuman:`/`\n\nAssistant:` 格式） |
| `GET` | `/v1/models` | 列出模型（代理到 provider） |
| `GET` | `/v1/models/{id}` | 获取单个模型 |
| `POST` | `/v1/messages/batches` | 创建 batch |
| `GET` | `/v1/messages/batches` | 列出 batch |
| `GET` | `/v1/messages/batches/{id}` | 查询 batch 状态 |
| `POST` | `/v1/messages/batches/{id}/cancel` | 取消 batch |
| `DELETE` | `/v1/messages/batches/{id}` | 删除 batch |
| `GET` | `/v1/messages/batches/{id}/results` | 流式返回 batch 结果（JSONL） |
| `POST` | `/tokens/clear` | 代理到上游 adapter 的 `/tokens/clear` |
| `GET` | `/health` | 健康检查 |

### 请求处理流程

非流式：

```
收到 Anthropic 请求
  → 路由：从 config 确定 provider + model
  → 覆盖请求中的 model
  → anthropic_to_openai() 转格式
  → apply_provider_params() 应用 provider 级参数默认值/上限
  → post_json() 发给后端（带重试）
  → openai_to_anthropic() 转回 Anthropic 格式
  → 返回给客户端
```

流式：

```
收到 Anthropic 请求（stream: true）
  → 同上转换步骤
  → open_provider_stream() 先连上游 SSE（这一步就能发现后端报错）
  → 返回 HTTP 200 给客户端
  → stream_openai_to_anthropic() 状态机逐 chunk 转换 SSE 事件
  → 结束时发 message_delta（usage）+ message_stop
```

---

## 快速开始

Local development in this repo uses the `ccr-py` conda environment. If `conda` is not on your `PATH`, use `~/miniconda3/bin/conda`.

```bash
~/miniconda3/bin/conda run -n ccr-py pip install -r requirements.txt
```

把 Anthropic 客户端指向代理：

```bash
export ANTHROPIC_BASE_URL=http://localhost:3456
export ANTHROPIC_API_KEY=any-value   # 由后端验证，ccr-py 不校验
```

## 启动服务

### 开发模式（单进程）

```bash
~/miniconda3/bin/conda run -n ccr-py python main.py --config config.json
```

### 生产部署

```bash
gunicorn server:app \
  -w 32 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:3456 \
  --timeout 850 \
  --graceful-timeout 10 \
  --preload \
  -e CCR_CONFIG=config.json
```

参数说明：

| Flag | 值 | 说明 |
|---|---|---|
| `-w 32` | worker 数 | IO 密集型，一般设 ~2× CPU 核数，按内存和并发调整 |
| `-k uvicorn.workers.UvicornWorker` | worker 类型 | 每个 worker 一个 asyncio 事件循环，适合大量并发流式响应 |
| `--timeout 850` | 秒 | 需大于 `API_TIMEOUT_MS / 1000`（默认 120 s）加上流式响应的最长时间 |
| `--graceful-timeout 10` | 秒 | reload/shutdown 时等待在途请求完成的时间 |
| `--preload` | — | master 进程 fork 前先导入 app，省内存（copy-on-write），也能提前发现 import 错误 |
| `-e CCR_CONFIG=config.json` | 环境变量 | 每个 worker fork 后独立读取配置文件 |

也可以用 `main.py`：

```bash
~/miniconda3/bin/conda run -n ccr-py python main.py --config config.json --workers 32
```

You can point the server at any config path with:

```bash
~/miniconda3/bin/conda run -n ccr-py python main.py --config /path/to/config.json
```

### Worker 数量建议

每个 worker 是独立进程，有自己的 asyncio 事件循环。单个 worker 就能处理大量并发流式连接。经验值：

- IO 密集型从 `2 × nproc` 开始
- 内存紧张就减少（每个 worker ~50 MB）
- 低流量 / 开发环境用 `1` 个就够

### CLI

```
python main.py [--config PATH] [--host HOST] [--port PORT] [--log-level LEVEL]
```

CLI 参数覆盖 `config.json` 中的对应值。

---

## 配置

`config.json` 示例：

```json
{
  "PORT": 3456,
  "API_TIMEOUT_MS": 120000,
  "Providers": [
    {
      "name": "my-provider",
      "api_base_url": "http://your-openai-endpoint/v1/chat/completions",
      "api_key": "$MY_API_KEY",
      "tokenizer_path": "/models/your-tokenizer",
      "max_retries": 3,
      "dp_routing": {
        "enabled": true,
        "server_info_ttl_sec": 30,
        "sticky_mode": "session_system"
      },
      "params": {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 16384,
        "reasoning": {"budget_tokens": 8000}
      }
    }
  ],
  "Router": {
    "default": "my-provider,model-name"
  }
}
```

### 顶层字段

### 示例：SGLang 后端启用 `dp_routing`

当上游是 SGLang，且服务端启用了 data parallel（例如 `dp_size=64`）时，可以让 router 自动探测 `/get_server_info` 返回的 `dp_size`，并按会话注入 `routed_dp_rank`。

```json
{
  "PORT": 3456,
  "API_TIMEOUT_MS": 850000,
  "Providers": [
    {
      "name": "sglang",
      "api_base_url": "http://sglang-host:30000/v1/chat/completions",
      "api_key": "$SGLANG_API_KEY",
      "max_retries": 3,
      "dp_routing": {
        "enabled": true,
        "server_info_ttl_sec": 30,
        "sticky_mode": "session_system"
      },
      "params": {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 65536,
        "reasoning": {"budget_tokens": 10240}
      }
    }
  ],
  "Router": {
    "default": "sglang,/model"
  }
}
```

说明：
- `dp_routing.enabled: true` 会为 `/v1/messages` 打开 `routed_dp_rank` 注入逻辑。
- `sticky_mode: "session"` 使用 `X-Claude-Code-Session-Id` 作为粘性路由 key。
- `sticky_mode: "session_system"` 优先使用 `X-Claude-Code-Session-Id + short hash(messages[0].content[1].text)`，拿不到时退回到标准化 system prompt 的短哈希。
- 可通过 `X-Routed-DP-Rank` 显式指定 rank，便于测试。
- 如果后端报告的 `dp_size <= 1`，router 不会注入 `routed_dp_rank`。

| 字段 | 默认值 | 说明 |
|---|---|---|
| `PORT` | `3456` | 监听端口 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `API_TIMEOUT_MS` | `600000` | 请求超时（毫秒） |
| `LOG_LEVEL` | `info` | 日志级别：`debug` / `info` / `warning` / `error` |

### Provider 字段

| 字段 | 必填 | 说明 |
|---|---|---|
| `name` | 是 | `Router` 中引用的标识符 |
| `api_base_url` | 是 | OpenAI 兼容端点的完整 URL（如 `.../v1/chat/completions`） |
| `api_key` | 是 | 以 `Authorization: Bearer <key>` 发送 |
| `max_retries` | 否 | 遇到 429/5xx 时重试次数（默认 `3`） |
| `dp_routing` | 否 | SGLang 的 DP worker 粘性路由配置（见下文） |
| `params` | 否 | 参数默认值 / 覆盖（见下文） |
| `tokenizer_path` | 否 | HuggingFace tokenizer 路径，用于 `/v1/messages/count_tokens`；也可来自 `CCR_TOKENIZER_PATH` 或 `TOKENIZER_PATH` |

### `dp_routing` — SGLang DP rank 固定路由

这是 provider 级的可选配置，用来把同一个 Claude Code 会话稳定路由到同一个 SGLang DP worker。

| 字段 | 说明 |
|---|---|
| `enabled` | 开启 SGLang `dp_size` 探测和 `/v1/messages` 的 `routed_dp_rank` 注入 |
| `server_info_ttl_sec` | `/get_server_info` 查询结果缓存秒数，默认 `30` |
| `sticky_mode` | `session`（默认）或 `session_system`；后者会在有 `messages[0].content[1].text` 时用其短哈希区分子代理，否则退回到标准化 system prompt 哈希 |

启用后，router 会：
- 从后端 `/get_server_info` 获取 `dp_size`
- 只在 `dp_size > 1` 时注入 `routed_dp_rank`
- 默认使用 `X-Claude-Code-Session-Id` 作为粘性 key
- 在 `session_system` 模式下，优先使用 `X-Claude-Code-Session-Id + short hash(messages[0].content[1].text)`，否则使用 `X-Claude-Code-Session-Id + short hash(normalized system prompt)`
- 支持 `X-Routed-DP-Rank` 手工覆盖，便于调试

`session_system` 仍然只是启发式方案，不是真正的 subagent ID；当首条用户消息里有稳定的子代理标识块时效果最好。

CLI / 环境变量等价项：
- `python main.py ... --dp-routing --dp-server-info-ttl-sec 30 --dp-sticky-mode session_system`
- `CCR_DP_ROUTING_ENABLED=1`
- `CCR_DP_SERVER_INFO_TTL_SEC=30`
- `CCR_DP_STICKY_MODE=session_system`

相关请求头：
- `X-Claude-Code-Session-Id`：用于选择粘性 DP rank 的会话 key
- `X-Routed-DP-Rank`：显式指定 rank，用于测试

启用固定路由时的响应头：
- `X-Router-DP-Rank`
- `X-Router-Sticky-Key`（仅会话粘性路由时返回）

### `params` — provider 级参数覆盖

所有字段可选。对发往该 provider 的每个请求生效。

| 字段 | 说明 |
|---|---|
| `temperature` | 请求未指定时的默认值 |
| `top_p` | 请求未指定时的默认值 |
| `max_tokens` | 默认值；同时也是上限，请求值超过此值会被截断 |
| `reasoning` | `{"budget_tokens": N}` — 请求没有 `thinking` 配置且没有 tools 时自动注入 extended thinking |

### Router

```json
"Router": {
  "default": "provider-name,model-name"
}
```

model 名会注入到每个发出的请求中，也会在 Anthropic 响应中回显。目前只用 `default`。

### 环境变量替换

`config.json` 中的任意字符串值支持 `$VAR` 或 `${VAR}` 替换：

```json
{ "api_key": "$MY_API_KEY" }
```

### 环境变量配置（不用 config.json）

不想写配置文件的话，可以用 `CCR_*` 环境变量：

| 环境变量 | 说明 |
|---|---|
| `CCR_API_BASE_URL` | provider 端点 URL（设了这个就不需要 config.json） |
| `CCR_API_KEY` | API key（也可以用 `API_KEY`） |
| `CCR_MODEL` | 模型名（默认 `/model`） |
| `CCR_MAX_RETRIES` | 重试次数（默认 `3`） |
| `CCR_API_TIMEOUT_MS` | 超时毫秒（默认 `850000`） |
| `CCR_TEMPERATURE` | temperature |
| `CCR_TOP_P` | top_p |
| `CCR_MAX_TOKENS` | max_tokens |
| `CCR_BUDGET_TOKENS` | thinking budget_tokens |
| `CCR_TOKENIZER_PATH` | HuggingFace tokenizer 路径（也可以用 `TOKENIZER_PATH`），用于 count_tokens |
| `CCR_CONFIG_JSON` | 整个 config 的 JSON 字符串（multi-worker 场景下 main.py 自动设置） |
| `CCR_CONFIG` | config 文件路径 |

优先级：`CCR_CONFIG_JSON` > `CCR_*` 环境变量 > `CCR_CONFIG` 文件路径。

---

## 请求转换（Anthropic → OpenAI）

### 采样参数

| Anthropic 字段 | OpenAI 字段 | 备注 |
|---|---|---|
| `model` | `model` | 被 router 配置的 model 覆盖 |
| `max_tokens` | `max_tokens` | 受 provider `params.max_tokens` 上限约束 |
| `temperature` | `temperature` | 未指定时用 provider 默认值 |
| `top_p` | `top_p` | 未指定时用 provider 默认值 |
| `top_k` | `top_k` | 直接透传，不支持的 provider 会忽略 |
| `stop_sequences` | `stop` | 重命名 |
| `stream` | `stream` | 为 true 时额外加 `stream_options: {include_usage: true}` |

### Thinking / 扩展推理

| Anthropic `thinking.type` | 行为 |
|---|---|
| `enabled` | 带 `budget_tokens` 原样转发，provider 需支持 |
| `adaptive` | 原样转发，不带 `budget_tokens`，provider 自行决定 |
| `disabled` | 不发送 `thinking` 字段 |

provider 级默认值（`params.reasoning`）：
- 请求没有 `thinking` 且没有 tools 且 `params.reasoning` 已设置时，注入 `{"type": "enabled", "budget_tokens": N}`

### Tool choice

| Anthropic `tool_choice.type` | OpenAI `tool_choice` |
|---|---|
| `auto` | `"auto"` |
| `any` | `"required"` |
| `tool` | `{"type": "function", "function": {"name": "..."}}` |
| `none` | `"none"` |

任意变体上 `disable_parallel_tool_use: true` → `parallel_tool_calls: false`

### Tools

| Anthropic 字段 | OpenAI 字段 |
|---|---|
| `tool.name` | `function.name` |
| `tool.description` | `function.description` |
| `tool.input_schema` | `function.parameters` |
| `tool.strict` | `function.strict` |

不在已知字段集合（`name`, `description`, `input_schema`, `strict`, `type`, `cache_control`）内的字段会直接透传到 `function` 对象上。

### `output_config`

| Anthropic 字段 | OpenAI 字段 | 备注 |
|---|---|---|
| `output_config.effort` | `reasoning_effort` | `low/medium/high` 直接透传；`max` → `xhigh` |
| `output_config.format` (json_schema) | `response_format` | `{type: "json_schema", json_schema: <schema>}` |

### `metadata`

| Anthropic 字段 | OpenAI 字段 |
|---|---|
| `metadata.user_id` | `user` |

### 消息和内容块

| Anthropic 内容块 | 转换结果 |
|---|---|
| `text`（user） | `{type: "text", text: "..."}` |
| `image`（base64） | `{type: "image_url", image_url: {url: "data:<mt>;base64,..."}}` |
| `image`（URL） | `{type: "image_url", image_url: {url: "..."}}` |
| `tool_result` | OpenAI `role: "tool"` 消息，带 `tool_call_id` |
| `text`（assistant） | `content` 字符串 |
| `tool_use`（assistant） | `tool_calls` 数组 |
| `thinking`（assistant 历史） | 跳过，不发给 provider |

user 消息中如果只有 text 块，会合并成纯字符串（不用数组）。

### 不转换的字段（Anthropic 专有）

| 字段 | 原因 |
|---|---|
| `cache_control` | Anthropic prompt caching |
| `container` | Anthropic 代码执行容器 |
| `inference_geo` | Anthropic 数据驻留路由 |
| `service_tier` | Anthropic 计费层级 |
| `tool.cache_control` | tool 上的 prompt caching |
| `system` 内容块 `cache_control` | 同上 |

---

## 响应转换（OpenAI → Anthropic）

### 非流式

| OpenAI 字段 | Anthropic 字段 | 备注 |
|---|---|---|
| `id` | `id` | |
| `choices[0].message.content` | `content[].text` 块 | |
| `choices[0].message.tool_calls` | `content[].tool_use` 块 | `arguments` 字符串解析为 JSON；支持嵌套（`{function: {name, arguments}}`）和扁平（`{name, arguments}`）两种格式 |
| `choices[0].message.thinking.content` | `content[].thinking` 块 | OpenAI 原生 thinking 格式 |
| `choices[0].message.reasoning_content` | `content[].thinking` 块 | 第三方 provider 常用格式 |
| `choices[0].finish_reason` | `stop_reason` | 见下表 |
| `usage.prompt_tokens` | `usage.input_tokens` | 减去 cache tokens |
| `usage.completion_tokens` | `usage.output_tokens` | |
| `usage.prompt_tokens_details.cached_tokens` | `usage.cache_read_input_tokens` | |
| `usage.prompt_tokens_details.cache_creation_tokens` | `usage.cache_creation_input_tokens` | |

### `finish_reason` → `stop_reason` 映射

| OpenAI `finish_reason` | Anthropic `stop_reason` |
|---|---|
| `stop` | `end_turn` |
| `length` | `max_tokens` |
| `tool_calls` | `tool_use` |
| `content_filter` | `stop_sequence` |
| 其他 | `end_turn` |

### 流式 SSE 事件序列

```
message_start
ping
  [每个 content block:]
  content_block_start
    content_block_delta  (重复)
  content_block_stop
message_delta          ← stop_reason + 最终 usage
message_stop
```

支持的 delta 类型：

| 来源 | Anthropic delta 类型 |
|---|---|
| `delta.content`（文本） | `text_delta` |
| `delta.thinking.content` | `thinking_delta` |
| `delta.thinking.signature` | `signature_delta` |
| `delta.reasoning_content` | `thinking_delta` |
| `delta.tool_calls[].function.arguments` | `input_json_delta` |

流式的 tool_calls 同样兼容嵌套和扁平格式。

---

## 流式处理内部逻辑

`stream: true` 时 ccr-py 不是简单转发字节流。

ccr-py 在返回 HTTP 200 给客户端_之前_先连上游 SSE。如果 provider 在连接阶段就报错，客户端拿到的是正常的 HTTP 错误，而不是一个断掉的流。

转换器（`stream_openai_to_anthropic`）内部跑一个状态机，跟踪当前哪些 content block 是打开的（thinking / text / tool_use）以及它们的 index。block 在收到第一个 delta 时惰性创建，在下一个 block 类型开始或流结束时关闭。Anthropic SSE 的事件顺序（`content_block_start` / `content_block_delta` / `content_block_stop`）由这个状态机保证，不依赖上游 provider。

thinking delta（`delta.thinking` 或 `delta.reasoning_content`）打开一个 thinking block。收到 signature delta 时立即关闭。如果 text delta 到达时 thinking 还开着（没收到 signature），thinking 会被强制关闭。新的 tool call index 出现时，先关闭已有的 text block，再关闭前一个 tool block（如果有），然后才打开新的。客户端不会看到重叠的 block。

`CCR_DEBUG=1` 时，流式路径还会把文本攒到 buffer 里，流结束后检查是否有敏感 token（见下文 Debug 模式）。debug 关闭时没有额外开销。

如果流中途抛异常（HTTP 200 已经发了），ccr-py 会发一个 `event: error` SSE 事件，而不是静默截断。

---

## chat_to_generate_adapter

`chat_to_generate_adapter.py` 是一个独立的 FastAPI 应用，放在原始推理引擎（sglang 等）前面，给它套一个 OpenAI 兼容的 `v1/chat/completions` 接口。ccr-py 可以把这个 adapter 当作 provider。

```
Claude Code
    |  Anthropic 格式
    v
  ccr-py  (server.py, port 3456)
    |  OpenAI 格式
    v
  chat_to_generate_adapter  (port 8890)
    |  tokenized prompt / generate API
    v
  sglang / 推理引擎
```

### 路由策略

三种方式跟后端通信，通过环境变量控制：

| 策略 | 环境变量 | 做什么 |
|---|---|---|
| generate | `USE_GENERATE_API=true` | 用 tokenizer chat template 生成原始 prompt，调 sglang `/generate`。从响应的 XML（`<tool_call>...</tool_call>`）中解析 tool call。 |
| completions | `USE_COMPLETIONS_FOR_CHAT=true` | 同样的 tokenizer prompt 构建，但发到 `/v1/completions`。某些后端上更稳定。 |
| 透传 | 两个都 `false` | 直接把 chat 请求转发到后端的 `/v1/chat/completions`。不需要 tokenizer。 |

generate 和 completions 都需要设 `TOKENIZER_PATH`，因为要调 `tokenizer.apply_chat_template()`。

### Tool 处理

有 tools 时，adapter 会往 system prompt 里注入所有 tool 的 XML 描述。模型的原始输出用正则解析 `<tool_call>` 块及其 `<arg_key>`/`<arg_value>` 对，转回 OpenAI 格式的 `tool_calls`。

流式 + tool 的场景下，整个响应文本会被缓冲，tool call 在最后一次性解析，作为一个 chunk 在 `[DONE]` 前发出。tool call 参数必须完整接收后才能从 XML 中可靠提取，所以没法逐 token 流式发送。

### Thinking / 推理

原始 completion 中的 `<think>...</think>` 块被提取出来，放到 OpenAI 响应的 `reasoning_content` 字段。ccr-py 再把它转成 Anthropic 的 `thinking` 块。

### 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat completions（经 generate / completions / 透传路由） |
| `POST` | `/v1/completions` | 原始文本补全 |
| `POST` | `/tokens/clear` | 代理到 sglang 的 `/tokens/clear` |
| `GET` | `/health` | 健康检查 |

### 启动 adapter

```bash
TOKENIZER_PATH=/path/to/tokenizer \
ROUTER_URL=http://sglang-host:8080 \
MODEL=GLM-5 \
USE_GENERATE_API=true \
gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:8890 \
  chat_to_generate_adapter:app
```

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `ROUTER_URL` | `http://127.0.0.1:8000` | sglang / 推理后端的地址 |
| `MODEL` | `GLM-5` | 响应中回显的模型名 |
| `TOKENIZER_PATH` | （generate/completions 模式必填） | HuggingFace tokenizer 路径 |
| `API_KEY` | `""` | 发给后端的 Bearer token |
| `USE_GENERATE_API` | `true` | 用 `/generate` 端点 |
| `USE_COMPLETIONS_FOR_CHAT` | `true` | 用 `/v1/completions`（`USE_GENERATE_API` 也为 true 时被覆盖） |
| `PORT` | `8890` | 通过 `__main__` 运行时的监听端口 |

---

## main_key 绑定

sglang 的 PD 分离（Prefill-Decode disaggregation）或 radix attention 场景下，同一会话的请求共享 `main_key` 可以跨 turn 复用 KV cache。

adapter 按以下优先级解析 `main_key`：

1. 请求体中的 `main_key` 字段，直接使用。
2. `metadata.user_id.session_id`（`metadata.user_id` 是 dict 时）。
3. `user.session_id`（`user` 是 dict 时）。
4. `user` 是 JSON 字符串时，解析后提取 `session_id`。

都不匹配就不带 `main_key`。

ccr-py 会把 `metadata.user_id` 透传为 OpenAI 的 `user` 字段，所以完整链路是这样的：

```
Anthropic 客户端设置 metadata.user_id = {"session_id": "abc123"}
  → ccr-py 转成 OpenAI: user = {"session_id": "abc123"}
    → adapter 提取 main_key = "abc123"
      → sglang 复用该 session 的 KV cache 前缀
```

---

## Debug 模式

设 `CCR_DEBUG=1` 开启：

- 每个转换后的 OpenAI 请求会打印到 stdout。
- 流式和非流式响应都会扫描敏感 token（`<tool_call>`、`<|user|>`、`<|observation|>` 等）。
- 发现敏感 token 时，请求 + 响应对会以 JSON 文件保存到 `CCR_DEBUG_DIR`（默认 `./debug_dumps/`）。

用来排查模型把内部控制 token 泄漏到输出中的情况。

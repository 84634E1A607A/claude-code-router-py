# ccr-py

`ccr-py` 是一个 FastAPI 代理服务。它接收 Anthropic Messages API 请求，路由到 OpenAI 兼容后端，并在 Anthropic 与 OpenAI 两种格式之间做双向转换。

```text
Anthropic client
    -> Anthropic Messages API
ccr-py
    -> OpenAI-compatible Chat Completions API
provider
```

## What It Serves

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/messages` | Anthropic Messages API，支持流式和非流式 |
| `POST` | `/v1/messages/count_tokens` | 基于 tokenizer 的 token 计数 |
| `POST` | `/v1/complete` | Anthropic text completion 兼容接口 |
| `GET` | `/v1/models` | 透传 provider 模型列表 |
| `GET` | `/v1/models/{id}` | 透传单个模型信息 |
| `POST` | `/v1/messages/batches` | 创建 batch |
| `GET` | `/v1/messages/batches` | 列出 batch |
| `GET` | `/v1/messages/batches/{id}` | 查询 batch |
| `POST` | `/v1/messages/batches/{id}/cancel` | 取消 batch |
| `DELETE` | `/v1/messages/batches/{id}` | 删除 batch |
| `GET` | `/v1/messages/batches/{id}/results` | 返回 Anthropic JSONL batch 结果 |
| `POST` | `/tokens/clear` | 转发到 provider 的 `/tokens/clear` |
| `GET` | `/health` | 健康检查 |

## Request Flow

非流式请求：

```text
Anthropic request
-> route model from config
-> convert to OpenAI payload
-> apply provider params
-> send to provider with retries
-> convert response to Anthropic format
-> return JSON response
```

流式请求：

```text
Anthropic request with stream=true
-> route model from config
-> convert to OpenAI payload
-> open upstream SSE stream
-> convert chunks to Anthropic SSE events
-> emit message_delta and message_stop
```

## Quick Start

Local development in this repo uses the `ccr-py` conda environment.

```bash
~/miniconda3/bin/conda run -n ccr-py pip install -r requirements.txt
```

设置 Anthropic 客户端：

```bash
export ANTHROPIC_BASE_URL=http://localhost:3456
export ANTHROPIC_API_KEY=any-value
```

启动服务：

```bash
~/miniconda3/bin/conda run -n ccr-py python main.py --config config.json
```

也可以通过环境变量提供配置文件路径：

```bash
CCR_CONFIG=/path/to/config.json ~/miniconda3/bin/conda run -n ccr-py python main.py
```

多进程部署：

```bash
gunicorn server:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:3456 \
  --timeout 850 \
  -e CCR_CONFIG=config.json
```

## CLI

```bash
python main.py [--config PATH]
```

## Configuration

配置在加载时经过显式 schema 校验。顶层、provider、`params`、`dp_routing` 都使用固定字段集合。

示例：

```json
{
  "PORT": 3456,
  "API_TIMEOUT_MS": 120000,
  "tokenizer_path": "/models/default-tokenizer",
  "Providers": [
    {
      "name": "primary",
      "model": "/model",
      "api_base_url": "http://provider/v1/chat/completions",
      "api_key": "$PROVIDER_API_KEY",
      "max_retries": 3,
      "tokenizer_path": "/models/provider-tokenizer",
      "dp_routing": {
        "enabled": true,
        "server_info_ttl_sec": 30,
        "session_ttl_sec": 10800.0
      },
      "params": {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 16384,
        "reasoning": {
          "budget_tokens": 8000
        }
      }
    }
  ],
  "Router": {
    "default": "/model"
  }
}
```

### Top-Level Fields

| Field | Type | Default | Purpose |
|---|---|---|---|
| `PORT` | `int` | `3456` | `main.py` 监听端口 |
| `API_TIMEOUT_MS` | `int` | `600000` | 上游请求超时 |
| `tokenizer_path` | `string` | `null` | 默认 tokenizer 路径 |
| `Providers` | `list` | required | provider 列表 |
| `Router` | `object` | required | 场景到模型名的映射 |

### Provider Fields

| Field | Type | Default | Purpose |
|---|---|---|---|
| `name` | `string` | required | provider 名称 |
| `model` | `string` | required | provider 对应的模型名 |
| `api_base_url` | `string` | required | OpenAI 兼容 chat completions URL |
| `api_key` | `string` | `""` | Bearer token |
| `max_retries` | `int` | `3` | 429/5xx 重试次数 |
| `tokenizer_path` | `string` | `null` | provider 级 tokenizer 路径 |
| `params` | `object` | `null` | provider 默认参数 |
| `dp_routing` | `object` | `null` | SGLang DP 粘性路由配置 |

### `params`

| Field | Type | Purpose |
|---|---|---|
| `temperature` | `float` | 请求默认值 |
| `top_p` | `float` | 请求默认值 |
| `max_tokens` | `int` | 请求默认值与上限 |
| `reasoning.budget_tokens` | `int` | thinking 默认预算 |

### `dp_routing`

| Field | Type | Default | Purpose |
|---|---|---|---|
| `enabled` | `bool` | required | 启用 DP 路由 |
| `server_info_ttl_sec` | `int` | `30` | `/get_server_info` 缓存时间 |
| `session_ttl_sec` | `float` | `10800.0` | 会话保留时间 |

### Router

`Router` 是一个字符串映射，值直接写模型名：

```json
{
  "Router": {
    "default": "/model",
    "think": "/model-thinking"
  }
}
```

每个路由值都需要能匹配到 `Providers[*].model`。

### Environment Interpolation

`config.json` 中的字符串支持 `$VAR` 和 `${VAR}`：

```json
{
  "api_key": "$PROVIDER_API_KEY"
}
```

### Config Sources

`main.py`:

1. `--config PATH`
2. `CCR_CONFIG`
3. `config.json`

`server.py` / `gunicorn server:app`:

1. `CCR_CONFIG_JSON`
2. `CCR_CONFIG`
3. `config.json`

当配置来自文件路径时，服务会自动轮询文件变更并热重载。默认轮询间隔为 `1` 秒，可通过 `CCR_CONFIG_RELOAD_INTERVAL_SEC` 调整。`CCR_CONFIG_JSON` 不支持热重载，且监听端口这类进程启动参数不会因配置变更而重新绑定。

| Source | Usage |
|---|---|
| `--config PATH` | `python main.py --config /path/to/config.json` |
| `CCR_CONFIG` | 配置文件路径 |
| `CCR_CONFIG_JSON` | 完整 config JSON 字符串 |

## Anthropic to OpenAI Mapping

### Request Fields

| Anthropic | OpenAI | Notes |
|---|---|---|
| `model` | `model` | 运行时由路由结果写入 |
| `max_tokens` | `max_tokens` | 受 provider `params.max_tokens` 控制 |
| `temperature` | `temperature` | 支持 provider 默认值 |
| `top_p` | `top_p` | 支持 provider 默认值 |
| `top_k` | `top_k` | 透传 |
| `stop_sequences` | `stop` | 字段改名 |
| `stream` | `stream` | 流式时附加 `stream_options.include_usage` |
| `metadata.user_id` | `user` | 用户标识 |

### Thinking

| Anthropic `thinking.type` | OpenAI payload |
|---|---|
| `enabled` | `thinking` with `budget_tokens` |
| `adaptive` | `thinking` |
| `disabled` | omitted |

当请求里没有 `thinking` 且没有 `tools` 时，`params.reasoning.budget_tokens` 会生成默认 `thinking` 配置。

### Tools

| Anthropic | OpenAI |
|---|---|
| `tools[].name` | `tools[].function.name` |
| `tools[].description` | `tools[].function.description` |
| `tools[].input_schema` | `tools[].function.parameters` |
| `tools[].strict` | `tools[].function.strict` |
| `tool_choice.auto` | `"auto"` |
| `tool_choice.any` | `"required"` |
| `tool_choice.none` | `"none"` |
| `tool_choice.tool` | named function choice |

### Content Blocks

| Anthropic block | OpenAI representation |
|---|---|
| `text` | text content |
| `image` base64 | `image_url` with data URL |
| `image` URL | `image_url` |
| `tool_result` | `role: "tool"` |
| assistant `tool_use` | `tool_calls` |
| assistant `thinking` history | thinking blocks are skipped before provider call |

## OpenAI to Anthropic Mapping

### Non-Streaming Response

| OpenAI | Anthropic |
|---|---|
| `choices[0].message.content` | `content[].text` |
| `choices[0].message.tool_calls` | `content[].tool_use` |
| `choices[0].message.thinking.content` | `content[].thinking` |
| `choices[0].message.reasoning_content` | `content[].thinking` |
| `usage.prompt_tokens` | `usage.input_tokens` |
| `usage.completion_tokens` | `usage.output_tokens` |

### Finish Reasons

| OpenAI `finish_reason` | Anthropic `stop_reason` |
|---|---|
| `stop` | `end_turn` |
| `length` | `max_tokens` |
| `tool_calls` | `tool_use` |
| `content_filter` | `stop_sequence` |

### Streaming Events

Anthropic SSE 事件序列：

```text
message_start
ping
content_block_start
content_block_delta
content_block_stop
message_delta
message_stop
```

流式转换会按内容块维护顺序和边界，覆盖 text、thinking、signature、tool input JSON 这些 delta 类型。

## Testing

运行全部测试：

```bash
~/miniconda3/bin/conda run -n ccr-py python -m unittest discover -s tests
```

运行单元测试：

```bash
~/miniconda3/bin/conda run -n ccr-py python tests/test_router.py unit
```

运行集成测试：

```bash
PROVIDER_URL=http://provider/v1/chat/completions \
PROVIDER_KEY=token \
PROVIDER_MODEL=/model \
~/miniconda3/bin/conda run -n ccr-py python tests/test_router.py integration
```

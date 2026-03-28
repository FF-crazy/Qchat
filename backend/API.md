# Backend API

Base URL: `http://127.0.0.1:{port}`

---

## Chat

### GET /models

从缓存读取当前 provider 的模型列表。缓存为空时返回空列表，需先调用 `/models/refresh`。

**Response** `200`

```json
{
  "model_list": [
    {
      "model_name": "gpt-4o",
      "support_context": ["text"]
    }
  ]
}
```

---

### POST /models/refresh

请求上游 provider API 获取最新模型列表，写入缓存并返回。

**Response** `200` — 同 `GET /models`

**Error** `400` — 未选择 provider

---

### POST /messages

发送对话消息。根据 `stream` 字段决定返回方式。

**Request Body**

```json
{
  "model": "gpt-4o",
  "messages": { "context": [] },
  "stream": false,
  "max_tokens": 65536,
  "temperature": 1.0
}
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model | string | (必填) | 模型名称 |
| messages | ContextManager | (必填) | 对话上下文 |
| stream | bool | false | 是否流式返回 |
| max_tokens | int | 65536 | 最大输出 token |
| temperature | float | 1.0 | 采样温度 |

**Response (非流式)** `200`

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1719000000,
  "model": "gpt-4o",
  "choices": [],
  "usage": {}
}
```

**Response (流式)** `200` `text/event-stream`

```
data: {"delta": "Hello"}

data: {"delta": " world"}

data: [DONE]
```

**Error** `400` — 未选择 provider

---

## Local Config

### GET /local/config/current

获取当前运行时配置。

**Response** `200`

```json
{
  "provider": "my_openai",
  "provider_type": "openai",
  "agent": null,
  "stream": false,
  "max_tokens": 65536,
  "temperature": 1.0
}
```

---

### POST /local/config/update

更新运行时配置，返回更新后的完整配置。

**Request Body** — `CurrentLocalConfig` 的任意字段组合

```json
{
  "provider": "my_anthropic",
  "stream": true
}
```

**Response** `200` — 同 `GET /local/config/current`

---

### GET /local/providers

获取所有已加载的 provider 名称列表。

**Response** `200`

```json
["my_openai", "my_gemini", "my_anthropic"]
```

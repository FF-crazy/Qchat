# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Qchat is a FastAPI-based chat aggregator providing a unified API to multiple LLM providers (OpenAI, Anthropic, Gemini). Python 3.13, managed with `uv`.

## Commands

```bash
# Install dependencies
uv sync

# Run server (auto-finds available port 10000-65535)
python -m backend.main

# Run all tests
pytest

# Run a single test file
pytest backend/tests/test_chat.py

# Run a single test
pytest backend/tests/test_chat.py::TestProviders::test_openai_post_returns_message
```

## Architecture

```
Client → FastAPI (api/chat.py) → MessageConverter → MessagePoster → Provider APIs
                                                                      ↕
                                  ContextManager ← decode/encode → Provider Models
```

**Key abstractions:**

- **ContextManager** (`service/context.py`): Central message abstraction. Encodes canonical messages to provider-specific formats (`encode_openai()`, `encode_anthropic()`, `encode_gemini()`) and decodes responses back. Handles both streaming and non-streaming.

- **MessageConverter** (`service/model_converter.py`): Routes `QchatRequest` → provider-specific request format and provider response → `QchatResponse` based on `provider_type`.

- **MessagePoster** (`service/post.py`): HTTP client (httpx async) with provider-specific request building, posting, and streaming. Inner `RequestBuilder` constructs headers/payloads per provider.

- **ConfigManager / FileProcessor / ProviderProcessor** (`service/local.py`): Loads providers from `~/.config/Qchat/provider.toml` and prompts from `~/.config/Qchat/prompts/*.toml`. Manages runtime config (current provider, agent, stream settings).

- **Canonical models** (`models/local.py`): `QchatRequest`, `QchatResponse`, `CanonicalMessage`, `ContentPart`, `Provider`, `Agent`. Provider-specific Pydantic models live in `models/openai.py`, `models/anthropic.py`, `models/gemini.py`.

**Adding a new provider requires:**
1. Provider-specific Pydantic models in `models/<provider>.py`
2. Encode/decode methods in `ContextManager`
3. HTTP methods in `MessagePoster`
4. Routing cases in `MessageConverter`

## Provider Features

All three providers support streaming (SSE), extended thinking, and tool/function calling. Anthropic has the most complete tool call streaming implementation (`decode_anthropic_stream_tool_call`).

## Configuration

TOML-based config in `~/.config/Qchat/`. See `backend/init/provider.example.toml` and `backend/init/prompt.example.toml` for formats. The `FileProcessor` creates this directory structure on first startup.

## Testing

Tests are integration-style and hit live provider APIs (not mocked). They require valid API credentials. pytest config is in `pytest.ini` with `pythonpath = .` and `testpaths = backend/tests`.

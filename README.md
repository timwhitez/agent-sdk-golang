# agent-sdk-golang

[![Go Report Card](https://goreportcard.com/badge/github.com/timwhitez/agent-sdk-golang)](https://goreportcard.com/report/github.com/timwhitez/agent-sdk-golang)
[![GoDoc](https://godoc.org/github.com/timwhitez/agent-sdk-golang?status.svg)](https://godoc.org/github.com/timwhitez/agent-sdk-golang)

> **A minimal, control-first Agent SDK for Go.**  
> Built for developers who want less magic, more control, and a focus on tool execution.

## 📖 Overview

`agent-sdk-golang` is a minimal Agent SDK in Go. At its core, an agent is just a **for-loop around tool calling**: the model proposes tool calls, the runtime executes them, feeds results back, and repeats.

We prioritize explicit control flow over hidden prompts or complex abstractions.

### Why use this?
- **Relationship to `browser-use/agent-sdk`**: This project is **inspired by** [browser-use/agent-sdk](https://github.com/browser-use/agent-sdk). We learned from its “less abstraction, more control, tool-calling-first” philosophy and reimplemented similar ideas in the Go ecosystem.
- **Independent Implementation**: This is **not** an official port. It's an independent implementation tailored for Go's idioms and performance.

## ✨ Key Features

- 🎛 **Control First**: No hidden magic. You control the loop, the prompts, and the tools.
- 🔄 **Streaming Support**: all three built-in protocol clients stream over HTTP SSE; `QueryStream` delivers real-time deltas and Agent events (see [Streaming](#-streaming)).
- 🛠 **Robust Tooling**:
  - Automatic JSON schema generation (with `additionalProperties=false` support).
  - Dependency injection for tools.
  - Ephemeral output cleanup to save context.
  - "Done tool" pattern enforcement.
- 🔌 **Multiple Providers**:
  - **Anthropic Messages** — `anthropic.Client`
  - **OpenAI Chat Completions** — `openai.ChatClient`
  - **OpenAI Responses** — `openai.ResponsesClient`

  Each client supports buffered `Invoke` and HTTP SSE `InvokeStream` (`stream: true`). Protocol details: [agent_docs/providers.md](agent_docs/providers.md).
- 📉 **Context Compaction**: Smart auto-summarization of conversation history when token limits are reached.
- 🎮 **Real-time Steering**: Inject user feedback mid-flight during agent execution (boundary-aware).
- 💾 **Session Management**: Restore and resume conversation history with ease.
- 🛡 **Sandboxed Security**: Built-in safe tools for file reading, writing, editing, and command execution (requires explicit confirmation by default).

## 📦 Installation

```bash
go get github.com/timwhitez/agent-sdk-golang
```

## 🚀 Usage

Here is a simple example of how to initialize an agent and run a query:

```go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/openai"
)

func main() {
	// 1. Initialize LLM Provider
	llm := &openai.ChatClient{
		BaseURL:   "https://api.openai.com/v1",
		APIKey:    os.Getenv("OPENAI_API_KEY"),
		ModelName: "gpt-4o",
	}

	// 2. Initialize Agent with Configuration
	a, err := agent.New(agent.Config{
		LLM:          llm,
		SystemPrompt: "You are a helpful assistant.",
	})
	if err != nil {
		panic(err)
	}

	// 3. Run Query
	answer, err := a.Query(context.Background(), "Hello, who are you?")
	if err != nil {
		panic(err)
	}
	fmt.Println(answer)
}
```

`Query` returns only the aggregated final answer. It still uses SSE underneath when the model implements `llm.StreamingChatModel`, but it does not show deltas as they arrive; use `QueryStream` for that.

## 🔄 Streaming

Three levels, from lowest to highest:

| API | Returns | Use it for |
|---|---|---|
| `ChatModel.Invoke` | one `*llm.Completion` | a single buffered model call |
| `StreamingChatModel.InvokeStream` | `<-chan llm.StreamEvent` for one model call | raw deltas of one call; you run any tool loop yourself |
| `Agent.QueryStream` (and `QueryStreamEnveloped`, `…WithSteering`) | `<-chan agent.Event` for the whole Agent run | real-time text plus the managed tool loop |

`anthropic.Client`, `openai.ChatClient` and `openai.ResponsesClient` all implement `llm.StreamingChatModel`. A wrapper that implements only `ChatModel` hides that capability from the Agent, which then falls back to buffered calls; keep `InvokeStream` on custom wrappers. The SDK never infers streaming support from a type name or provider string.

Reading the outcome correctly:

- **InvokeStream**: append each `StreamTextDeltaEvent.Delta` once. Return the error from `InvokeStream` itself and from any `StreamErrorEvent.AsError()`. `StreamDoneEvent` is the provider's normal terminal; its `StopReason` is the provider's own value and can still report a length limit. A channel that closes without `StreamDoneEvent` is incomplete, not a success. Tool-call argument deltas are partial JSON: never execute them before the stream ends (the Agent does this for you).
- **QueryStream**: print `TextDeltaEvent` deltas; `FinalResponseEvent.Content` repeats that text, so print it only if no delta arrived. `ErrorEvent` ends the run with a failure. `FinalResponseEvent.Status == "partial"` is a bounded fallback answer, not a normally completed task, and `DroppedEvents`/`DroppedCriticalEvents` say the delivered stream is incomplete or inconsistent with history.
- Cancel the `context` to stop a request; an early return should always cancel so the HTTP body is closed.

[`examples/streaming`](examples/streaming) is a runnable example for all three protocols in both modes (configuration via `STREAM_MODEL`, `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`, optional `STREAM_BASE_URL`; missing settings fail before any request). Its tests use local HTTP fixtures only.

## 📂 Layout

- `sdk/`: Core SDK implementation (agent, llm, tools, tokens).
- `examples/streaming`: streaming usage for the three built-in protocols.


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

# 中文说明

> **Go 语言实现的极简 Agent SDK。**  
> 专为想要掌控一切、拒绝黑盒魔法的开发者设计。

## 📖 概述

`agent-sdk-golang` 是一个用 Go 实现的极简 Agent SDK。它的核心本质非常简单：**一个围绕工具调用的 for 循环**。模型提出工具调用请求，运行时执行这些工具，将结果反馈给模型，如此循环往复。

我们推崇显式的控制流，拒绝隐藏的提示词（Prompts）和过度的抽象。

### 项目背景
本项目的设计与实现**受到** [browser-use/agent-sdk](https://github.com/browser-use/agent-sdk) 的启发。我们参考了它“少抽象、可控、以工具调用为中心”的设计哲学，并在 Go 生态中进行了重新实现。
> **注意**：这不是官方的 Go 版本移植，也不存在从属关系。接口与行为细节可能有所不同。

## ✨ 核心能力

- 🎛 **掌控一切**：没有隐藏的魔法。你完全控制循环、提示词和工具行为。
- 🔄 **流式支持**：三个内置协议客户端均通过 HTTP SSE 流式输出；`QueryStream` 实时提供增量与 Agent 事件（见 [流式调用](#-流式调用)）。
- 🛠 **强大的工具系统**：
  - 自动生成 JSON Schema（支持 `additionalProperties=false`）。
  - 工具依赖注入（DI）。
  - Ephemeral（临时）输出清理，节省上下文。
  - 强制 "Done tool" 模式。
- 🔌 **多模型支持**：
  - **Anthropic Messages** — `anthropic.Client`
  - **OpenAI Chat Completions** — `openai.ChatClient`
  - **OpenAI Responses** — `openai.ResponsesClient`

  每个客户端都支持缓冲的 `Invoke` 与 HTTP SSE 的 `InvokeStream`（`stream: true`）。协议细节见 [agent_docs/providers.md](agent_docs/providers.md)。
- 📉 **上下文压缩**：当达到 Token 限制时，自动对历史记录进行摘要压缩。
- 🎮 **实时干预 (Real-time Steering)**：在 Agent 执行过程中（工具调用边界）实时注入用户反馈，纠正行为。
- 💾 **会话管理**：支持通过 `InitialMessages` 轻松恢复和继续历史会话。
- 🛡 **安全沙盒**：内置安全的文件读写、编辑、搜索和命令执行工具（默认需要确认，CLI 可用 `-y` 开启全自动模式）。

## 📦 安装

```bash
go get github.com/timwhitez/agent-sdk-golang
```

## 🚀 使用示例

```go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/openai"
)

func main() {
	// 1. 初始化 LLM
	llm := &openai.ChatClient{
		BaseURL:   "https://api.openai.com/v1",
		APIKey:    os.Getenv("OPENAI_API_KEY"),
		ModelName: "gpt-4o",
	}

	// 2. 初始化 Agent
	a, err := agent.New(agent.Config{
		LLM:          llm,
		SystemPrompt: "You are a helpful assistant.",
	})
	if err != nil {
		panic(err)
	}

	// 3. 执行查询
	answer, err := a.Query(context.Background(), "Hello, who are you?")
	if err != nil {
		panic(err)
	}
	fmt.Println(answer)
}
```

`Query` 只返回聚合后的最终答案。模型实现 `llm.StreamingChatModel` 时它在底层仍使用 SSE，但不会实时展示增量；需要实时输出请使用 `QueryStream`。

## 🔄 流式调用

三个层次，由低到高：

| API | 返回 | 适用场景 |
|---|---|---|
| `ChatModel.Invoke` | 一个 `*llm.Completion` | 单次缓冲模型调用 |
| `StreamingChatModel.InvokeStream` | 单次模型调用的 `<-chan llm.StreamEvent` | 单次调用的原始增量；工具循环需自行实现 |
| `Agent.QueryStream`（及 `QueryStreamEnveloped`、`…WithSteering`） | 整个 Agent 执行过程的 `<-chan agent.Event` | 实时文本 + 托管的工具循环 |

`anthropic.Client`、`openai.ChatClient`、`openai.ResponsesClient` 都实现了 `llm.StreamingChatModel`。只实现 `ChatModel` 的 wrapper 会对 Agent 隐藏该能力，Agent 随之退回缓冲调用；自定义 wrapper 请保留 `InvokeStream`。SDK 不会根据类型名或 Provider 字符串推断是否支持流式。

正确判断结果：

- **InvokeStream**：每个 `StreamTextDeltaEvent.Delta` 只追加一次。同时处理 `InvokeStream` 本身返回的 error 与 `StreamErrorEvent.AsError()`。`StreamDoneEvent` 是 Provider 的正常终态，其 `StopReason` 为 Provider 原值，仍可能表示输出被长度截断。未收到 `StreamDoneEvent` 就关闭的通道表示响应不完整，不是成功。工具参数增量是不完整的 JSON，流结束前绝不能执行（Agent 会替你处理）。
- **QueryStream**：打印 `TextDeltaEvent` 增量；`FinalResponseEvent.Content` 重复了这些文本，只有未收到任何增量时才打印它。`ErrorEvent` 表示失败结束。`FinalResponseEvent.Status == "partial"` 是有界的兜底答案，不是正常完成的任务；`DroppedEvents`/`DroppedCriticalEvents` 表示交付的事件流不完整或与历史不一致。
- 通过取消 `context` 终止请求；提前返回时务必 cancel，确保 HTTP body 被关闭。

[`examples/streaming`](examples/streaming) 是覆盖三种协议、两种模式的可运行示例（通过 `STREAM_MODEL`、`OPENAI_API_KEY`/`ANTHROPIC_API_KEY` 及可选的 `STREAM_BASE_URL` 配置；缺少配置会在发起请求前报错）。其测试只使用本地 HTTP fixture。

## 📂 目录结构

- `sdk/`：SDK 核心实现（agent/llm/tools/tokens）。
- `examples/streaming`：三种内置协议的流式用法示例。

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

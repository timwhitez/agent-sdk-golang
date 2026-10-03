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
- 🛡 **Sandboxed Security**: Built-in file and command tools. File operations validate sandbox paths; risky operations require a host-supplied confirmation policy (see [Sandbox confirmation](#sandbox-confirmation)).

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

### Sandbox confirmation

This is a library: your host supplies a `sandbox.Confirmer` with `Confirm(ctx context.Context, action, detail string) (bool, error)`, backed by its user interaction or permission policy. The SDK provides no CLI `-y` confirmation bypass; `examples/streaming` is a separate usage example without sandbox tools. The text-only quickstart above does not need sandbox dependencies.

When adding `sandbox.Tools()`, register both the sandbox and your confirmer, then pass the container as `agent.Config.Deps`. This helper accepts your configured model, sandbox root and host policy:

```go
package example

import (
    "context"

    "github.com/timwhitez/agent-sdk-golang/sdk/agent"
    "github.com/timwhitez/agent-sdk-golang/sdk/llm"
    "github.com/timwhitez/agent-sdk-golang/sdk/tools"
    "github.com/timwhitez/agent-sdk-golang/sdk/tools/sandbox"
)

func NewSandboxAgent(model llm.ChatModel, root string, policy sandbox.Confirmer) (*agent.Agent, error) {
    box, err := sandbox.New(root)
    if err != nil {
        return nil, err
    }
    deps := tools.NewContainer()
    tools.Provide(deps, sandbox.Key, func(context.Context) (*sandbox.Sandbox, error) {
        return box, nil
    })
    tools.Provide(deps, sandbox.ConfirmKey, func(context.Context) (sandbox.Confirmer, error) {
        return policy, nil
    })
    return agent.New(agent.Config{LLM: model, Tools: sandbox.Tools(), Deps: deps})
}
```

For confirmation-gated operations such as `bash`, `write` and `webfetch`, `(true, nil)` approves the action, `(false, nil)` denies it with `sandbox.ErrToolDenied`, and an error prevents execution. A missing confirmer, a confirmer dependency-provider error, or a provider returning a nil `Confirmer` interface fails closed with `sandbox.ErrMissingConfirmer`. Read/list/search do not require confirmation. The Agent sends tool failures back to the model; a final answer or `done` message containing that error does not mean the command ran. The sandbox validates file paths and sets the shell working directory; it does not provide OS-level process isolation.

[`examples/confirmation/example_test.go`](examples/confirmation/example_test.go) is a runnable library example with a local fake model and a host-owned mock confirmer. It only approves one fixed harmless command in its own temporary directory. Its tests check approval, denial, missing dependency and confirmation error through the real Agent → sandbox → `done` path, including command effects and error results. No provider or credentials are used:

```sh
go test ./examples/confirmation -v -count=1
```

## 🔄 Streaming

Three levels, from lowest to highest:

| API | Returns | Use it for |
|---|---|---|
| `ChatModel.Invoke` | one `*llm.Completion` | a single buffered model call |
| `StreamingChatModel.InvokeStream` | `<-chan llm.StreamEvent` for one model call | raw deltas of one call; you run any tool loop yourself |
| `Agent.QueryStream` (and `QueryStreamWithSteering`) | `<-chan agent.Event` for the whole Agent run | real-time text plus the managed tool loop |
| `Agent.QueryStreamEnveloped` (and `…WithSteering`) | `<-chan agent.EventEnvelope` (the same events with query/Frame correlation) | the same, when you need correlation metadata |
| `Agent.QueryStreamEnvelopedWithReceipt` (takes an optional steering channel like `…WithSteering`) | the same channel plus a `*agent.QueryStreamReceipt` | when you must prove you received the whole stream: after the channel closes, `Summary()` gives the last allocated `Sequence` and the envelopes dropped (and critical ones) over the whole stream, including any after the terminal event (`ok` may turn true just before the close; the values are final once you have observed it) |

`anthropic.Client`, `openai.ChatClient` and `openai.ResponsesClient` all implement `llm.StreamingChatModel`. A wrapper that implements only `ChatModel` hides that capability from the Agent, which then falls back to buffered calls; keep `InvokeStream` on custom wrappers. The SDK never infers streaming support from a type name or provider string.

Reading the outcome correctly:

- **InvokeStream**: append each `StreamTextDeltaEvent.Delta` once. Return the error from `InvokeStream` itself and from any `StreamErrorEvent.AsError()`. `StreamDoneEvent` is the provider's normal terminal; its `StopReason` is the provider's own value and can still report a length limit. A channel that closes without `StreamDoneEvent` is incomplete, not a success. Tool-call argument deltas are partial JSON: never execute them before the stream ends (the Agent does this for you).
- **QueryStream**: print `TextDeltaEvent` deltas as progress. `FinalResponseEvent.Content` is the authoritative answer and is **not** guaranteed to equal the deltas: text streamed before a tool call is progress, a later turn or the `done` tool can deliver a different answer, a turn may stream nothing, and a dropped delta leaves the shown text incomplete. Compare the final answer with what the **last model turn** actually showed (the example resets at each tool call/result) and print it when they differ; never skip it just because some delta arrived earlier in the query. `ErrorEvent` ends the run with a failure. `FinalResponseEvent.Status == "partial"` is a bounded fallback answer, not a normally completed task, and `DroppedEvents`/`DroppedCriticalEvents` say the delivered stream is incomplete or inconsistent with history.
- Cancel the `context` to stop a request; an early return should always cancel so the HTTP body is closed.

[`examples/streaming`](examples/streaming) is a runnable example for all three protocols in both modes (configuration via `STREAM_MODEL`, `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`, optional `STREAM_BASE_URL`; missing settings fail before any request). Its tests use local HTTP fixtures only.

## 📂 Layout

- `sdk/`: Core SDK implementation (agent, llm, tools, tokens).
- `examples/streaming`: streaming usage for the three built-in protocols.
- `examples/confirmation`: host-supplied sandbox confirmation with local fixtures.


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
- 🛡 **安全沙盒**：内置文件与命令工具。文件操作校验沙盒路径；危险操作需要宿主提供确认策略（见 [沙盒确认](#沙盒确认)）。

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

### 沙盒确认

这是一个库：宿主需要实现 `sandbox.Confirmer` 的 `Confirm(ctx context.Context, action, detail string) (bool, error)`，接入自己的用户交互或权限策略。SDK 没有 CLI `-y` 确认跳过开关；`examples/streaming` 是一个独立的用法示例，没有安装沙盒工具。上面的纯文本快速开始不需要沙盒依赖。

添加 `sandbox.Tools()` 时，需注册沙盒和确认器，并将容器传给 `agent.Config.Deps`。下面的函数接收已配置的模型、沙盒根目录和宿主确认策略：

```go
package example

import (
    "context"

    "github.com/timwhitez/agent-sdk-golang/sdk/agent"
    "github.com/timwhitez/agent-sdk-golang/sdk/llm"
    "github.com/timwhitez/agent-sdk-golang/sdk/tools"
    "github.com/timwhitez/agent-sdk-golang/sdk/tools/sandbox"
)

func NewSandboxAgent(model llm.ChatModel, root string, policy sandbox.Confirmer) (*agent.Agent, error) {
    box, err := sandbox.New(root)
    if err != nil {
        return nil, err
    }
    deps := tools.NewContainer()
    tools.Provide(deps, sandbox.Key, func(context.Context) (*sandbox.Sandbox, error) {
        return box, nil
    })
    tools.Provide(deps, sandbox.ConfirmKey, func(context.Context) (sandbox.Confirmer, error) {
        return policy, nil
    })
    return agent.New(agent.Config{LLM: model, Tools: sandbox.Tools(), Deps: deps})
}
```

对于 `bash`、`write`、`webfetch` 等需要确认的操作，`(true, nil)` 表示批准，`(false, nil)` 会返回 `sandbox.ErrToolDenied`，返回 error 则阻止执行。缺少确认器、确认器依赖 provider 返回 error，或返回 nil `Confirmer` 接口时，会以 `sandbox.ErrMissingConfirmer` 拒绝执行。读文件、列目录、搜索不需要确认。Agent 会把工具失败结果交给模型；最终回答或 `done` 消息提到错误，不代表命令已经执行。沙盒校验文件路径并设置 shell 工作目录，不提供操作系统级的进程隔离。

[`examples/confirmation/example_test.go`](examples/confirmation/example_test.go) 是可运行的库示例，使用本地 fake model 和宿主自有 mock 确认器，只批准自有临时目录中的一个固定无害命令。测试走真实 Agent → sandbox → `done` 路径，覆盖批准、拒绝、缺依赖和确认器错误，检查命令效果与错误结果，不使用 Provider 或凭据：

```sh
go test ./examples/confirmation -v -count=1
```

## 🔄 流式调用

三个层次，由低到高：

| API | 返回 | 适用场景 |
|---|---|---|
| `ChatModel.Invoke` | 一个 `*llm.Completion` | 单次缓冲模型调用 |
| `StreamingChatModel.InvokeStream` | 单次模型调用的 `<-chan llm.StreamEvent` | 单次调用的原始增量；工具循环需自行实现 |
| `Agent.QueryStream`（及 `QueryStreamWithSteering`） | 整个 Agent 执行过程的 `<-chan agent.Event` | 实时文本 + 托管的工具循环 |
| `Agent.QueryStreamEnveloped`（及 `…WithSteering`） | `<-chan agent.EventEnvelope`（同样的事件，附带 query/Frame 关联） | 同上，需要关联元数据时使用 |
| `Agent.QueryStreamEnvelopedWithReceipt`（与 `…WithSteering` 一样接受可选 steering channel） | 同一 channel 加 `*agent.QueryStreamReceipt` | 需要证明收到了完整事件流时：channel 关闭后 `Summary()` 给出最后分配的 `Sequence` 及整个流（含终态之后）丢弃的信封数和其中关键事件数 |

`anthropic.Client`、`openai.ChatClient`、`openai.ResponsesClient` 都实现了 `llm.StreamingChatModel`。只实现 `ChatModel` 的 wrapper 会对 Agent 隐藏该能力，Agent 随之退回缓冲调用；自定义 wrapper 请保留 `InvokeStream`。SDK 不会根据类型名或 Provider 字符串推断是否支持流式。

正确判断结果：

- **InvokeStream**：每个 `StreamTextDeltaEvent.Delta` 只追加一次。同时处理 `InvokeStream` 本身返回的 error 与 `StreamErrorEvent.AsError()`。`StreamDoneEvent` 是 Provider 的正常终态，其 `StopReason` 为 Provider 原值，仍可能表示输出被长度截断。未收到 `StreamDoneEvent` 就关闭的通道表示响应不完整，不是成功。工具参数增量是不完整的 JSON，流结束前绝不能执行（Agent 会替你处理）。
- **QueryStream**：`TextDeltaEvent` 增量作为进度打印。`FinalResponseEvent.Content` 是权威最终答案，**不保证**等于此前的增量：工具调用前流出的文本只是进度，后续轮次或 `done` 工具可能给出不同的答案，某一轮可能没有任何增量，丢失的增量也会使已显示文本不完整。应将最终答案与**最后一个模型轮次**实际显示的内容比较（示例在每次工具调用/结果处重置），不同则打印；不能仅因本次 Query 曾收到过增量就跳过最终答案。`ErrorEvent` 表示失败结束。`FinalResponseEvent.Status == "partial"` 是有界的兜底答案，不是正常完成的任务；`DroppedEvents`/`DroppedCriticalEvents` 表示交付的事件流不完整或与历史不一致。
- 通过取消 `context` 终止请求；提前返回时务必 cancel，确保 HTTP body 被关闭。

[`examples/streaming`](examples/streaming) 是覆盖三种协议、两种模式的可运行示例（通过 `STREAM_MODEL`、`OPENAI_API_KEY`/`ANTHROPIC_API_KEY` 及可选的 `STREAM_BASE_URL` 配置；缺少配置会在发起请求前报错）。其测试只使用本地 HTTP fixture。

## 📂 目录结构

- `sdk/`：SDK 核心实现（agent/llm/tools/tokens）。
- `examples/streaming`：三种内置协议的流式用法示例。
- `examples/confirmation`：宿主提供沙盒确认策略的本地 fixture 示例。

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

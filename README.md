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
- 🔄 **Streaming Support**: Built-in `QueryStream` for real-time token and event streaming.
- 🛠 **Robust Tooling**:
  - Automatic JSON schema generation (with `additionalProperties=false` support).
  - Dependency injection for tools.
  - Ephemeral output cleanup to save context.
  - "Done tool" pattern enforcement.
- 🔌 **Multiple Providers**:
  - **Anthropic**
  - **OpenAI Chat Completions**
  - **OpenAI Responses** (Best-effort, non-streaming)
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

## 📂 Layout

- `sdk/`: Core SDK implementation (agent, llm, tools, tokens).


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
- 🔄 **流式支持**：内置 `QueryStream`，支持实时的 Token 和事件流输出。
- 🛠 **强大的工具系统**：
  - 自动生成 JSON Schema（支持 `additionalProperties=false`）。
  - 工具依赖注入（DI）。
  - Ephemeral（临时）输出清理，节省上下文。
  - 强制 "Done tool" 模式。
- 🔌 **多模型支持**：
  - **Anthropic**
  - **OpenAI Chat Completions**
  - **OpenAI Responses**（Best-effort，非流式）
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

## 📂 目录结构

- `sdk/`：SDK 核心实现（agent/llm/tools/tokens）。

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

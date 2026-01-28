# agent-sdk-golang

这是一个用 Go 实现的极简 Agent SDK：本质上就是一个 for-loop，围绕“工具调用（tool calling）→ 执行 → 回填结果 → 继续”的闭环组织起来。

## 与 `browser-use/agent-sdk` 的关系

本项目的设计与实现**受到** `https://github.com/browser-use/agent-sdk` 的启发：我们参考了它“少抽象、可控、以工具调用为中心”的思路，并在 Go 生态下做了重新实现。

需要强调：

- 本项目并非 `browser-use/agent-sdk` 的官方 Go 版本，也不与其团队存在从属或合作关系。
- 这里的代码是独立实现，可能与原项目在 API/行为细节上存在差异。
- 当前仍处于早期阶段，接口与行为可能会调整；如你在生产环境使用，请先自行评估与补充测试。

## 现有能力（概览）

- Agent loop：支持 `Query` / `QueryStream`（事件流）
- 工具系统：schema 生成（`additionalProperties=false`）、DI overrides、ephemeral 输出清理、done tool 模式
- Provider：Anthropic、OpenAI Chat Completions、OpenAI Responses（best-effort，非流式）
- 安全沙盒工具：读/写/编辑/搜索/执行命令（默认需要确认，CLI 可用 `-y` 全放权）

## 目录

- `sdk/`：SDK 主体（agent/llm/tools/tokens）


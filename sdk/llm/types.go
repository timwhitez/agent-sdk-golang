package llm

import (
	"encoding/json"
	"strings"
)

type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ToolChoice controls how the model should choose tools.
// Supported values are provider-dependent, but common values are:
// - "auto": model may call tools
// - "required": model must call at least one tool
// - "none": model must not call tools
// - "<tool_name>": force a specific tool
type ToolChoice string

type ToolDefinition struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]any         `json:"parameters"`
	Strict      bool                   `json:"strict"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type ToolCall struct {
	ID             string       `json:"id"`
	Type           string       `json:"type"` // currently "function"
	Function       FunctionCall `json:"function"`
	ThoughtSig     []byte       `json:"thought_signature,omitempty"` // for Gemini-style providers
}

// ContentBlock is a provider-agnostic content part representation.
// Providers may serialize this differently; the SDK keeps it for multimodal
// messages and for persisting conversation state.
type ContentBlock struct {
	Type      string    `json:"type"`                // "text", "image_url", "document", "thinking", "redacted_thinking"
	Text      string    `json:"text,omitempty"`
	ImageURL  *ImageURL `json:"image_url,omitempty"`
	Source    *DocSrc   `json:"source,omitempty"`
	Thinking  string    `json:"thinking,omitempty"`
	Signature string    `json:"signature,omitempty"`
	Data      string    `json:"data,omitempty"`
}

type ImageURL struct {
	URL       string `json:"url"`
	Detail    string `json:"detail,omitempty"`     // "auto"|"low"|"high"
	MediaType string `json:"media_type,omitempty"` // e.g. "image/png" (Anthropic)
}

type DocSrc struct {
	Data      string `json:"data"`       // base64
	MediaType string `json:"media_type"` // e.g. "application/pdf"
}

// Content represents message content.
// Either Text is non-empty (simple case) or Blocks is non-empty (multimodal).
type Content struct {
	Text   string         `json:"text,omitempty"`
	Blocks []ContentBlock `json:"blocks,omitempty"`
}

func TextContent(s string) Content { return Content{Text: s} }

func (c Content) IsEmpty() bool {
	return strings.TrimSpace(c.Text) == "" && len(c.Blocks) == 0
}

func (c Content) PlainText() string {
	if strings.TrimSpace(c.Text) != "" {
		return c.Text
	}
	if len(c.Blocks) == 0 {
		return ""
	}
	var b strings.Builder
	for _, blk := range c.Blocks {
		if blk.Type == "text" && strings.TrimSpace(blk.Text) != "" {
			if b.Len() > 0 {
				b.WriteByte('\n')
			}
			b.WriteString(blk.Text)
		}
	}
	return b.String()
}

type Message struct {
	Role Role `json:"role"`

	Content Content `json:"content"`
	Name    string  `json:"name,omitempty"`

	// Anthropic prompt caching: when true, provider may cache this message.
	Cache bool `json:"cache,omitempty"`

	// Assistant-only: tool calls produced by the model.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`

	// Tool-only: tool result linkage.
	ToolCallID string `json:"tool_call_id,omitempty"`
	ToolName   string `json:"tool_name,omitempty"`
	IsError    bool   `json:"is_error,omitempty"`

	// Ephemeral tool result handling.
	Ephemeral bool `json:"ephemeral,omitempty"`
	Destroyed bool `json:"destroyed,omitempty"`
}

func (m Message) PlainText() string { return m.Content.PlainText() }

type Usage struct {
	PromptTokens              int  `json:"prompt_tokens"`
	CompletionTokens          int  `json:"completion_tokens"`
	TotalTokens               int  `json:"total_tokens"`
	PromptCachedTokens        *int `json:"prompt_cached_tokens,omitempty"`
	PromptCacheCreationTokens *int `json:"prompt_cache_creation_tokens,omitempty"`
	PromptImageTokens         *int `json:"prompt_image_tokens,omitempty"`
}

type Completion struct {
	Content    Content         `json:"content"`
	Thinking   string          `json:"thinking,omitempty"`
	ToolCalls  []ToolCall      `json:"tool_calls,omitempty"`
	Usage      *Usage          `json:"usage,omitempty"`
	StopReason string          `json:"stop_reason,omitempty"`
	Raw        json.RawMessage `json:"-"`
}

func (c Completion) HasToolCalls() bool { return len(c.ToolCalls) > 0 }

func (c Completion) PlainText() string { return c.Content.PlainText() }

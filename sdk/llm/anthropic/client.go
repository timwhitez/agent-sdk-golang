package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

const defaultBaseURL = "https://api.anthropic.com"

type Client struct {
	HTTPClient *http.Client
	BaseURL    string

	APIKey    string
	AuthToken string

	ModelName string
	MaxTokens int
	Temperature *float64
	TopP        *float64
	Seed        *int

	// Thinking mode (Anthropic extended thinking). If nil or <=0, disabled.
	ThinkingBudgetTokens *int

	// Retry policy.
	MaxRetries            int
	RetryBaseDelay        time.Duration
	RetryMaxDelay         time.Duration
	RetryableStatusCodes  map[int]struct{}

	// Optional Anthropic beta header values, e.g. "prompt-caching-2024-07-31".
	Beta []string

	// Only the last N tool definitions get cache_control (Anthropic cache block limits).
	MaxCachedToolDefinitions int
}

func (c *Client) Provider() string { return "anthropic" }

func (c *Client) Model() string { return c.ModelName }

func (c *Client) Invoke(ctx context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	client := c.httpClient()
	baseURL := strings.TrimRight(c.baseURL(), "/")

	payload, err := c.buildRequest(req)
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	endpoint := baseURL + "/v1/messages"
	lastErr := error(nil)

	maxRetries := c.MaxRetries
	if maxRetries <= 0 {
		maxRetries = 5
	}
	baseDelay := c.RetryBaseDelay
	if baseDelay <= 0 {
		baseDelay = 1 * time.Second
	}
	maxDelay := c.RetryMaxDelay
	if maxDelay <= 0 {
		maxDelay = 60 * time.Second
	}

	for attempt := 0; attempt < maxRetries; attempt++ {
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("anthropic-version", "2023-06-01")
		if len(c.Beta) > 0 {
			httpReq.Header.Set("anthropic-beta", strings.Join(c.Beta, ", "))
		}
		if c.APIKey != "" {
			httpReq.Header.Set("x-api-key", c.APIKey)
		}
		if c.AuthToken != "" {
			httpReq.Header.Set("Authorization", "Bearer "+c.AuthToken)
		}

		resp, err := client.Do(httpReq)
		if err == nil {
			data, readErr := io.ReadAll(resp.Body)
			_ = resp.Body.Close()
			if readErr != nil {
				return nil, readErr
			}

			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				return parseResponse(data)
			}

			msg := strings.TrimSpace(string(data))
			if resp.StatusCode == 429 {
				lastErr = &llm.RateLimitError{Provider: "anthropic", Message: msg}
			} else {
				lastErr = &llm.ProviderError{Provider: "anthropic", StatusCode: resp.StatusCode, Message: msg}
			}
			if c.isRetryableStatus(resp.StatusCode) && attempt < maxRetries-1 {
				c.sleepBackoff(ctx, attempt, baseDelay, maxDelay)
				continue
			}
			return nil, lastErr
		}

		// Network / timeout errors.
		lastErr = err
		if attempt < maxRetries-1 && isRetryableNetErr(err) {
			c.sleepBackoff(ctx, attempt, baseDelay, maxDelay)
			continue
		}
		return nil, err
	}

	if lastErr != nil {
		return nil, lastErr
	}
	return nil, errors.New("anthropic: retry loop ended without result")
}

func (c *Client) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return &http.Client{Timeout: 60 * time.Second}
}

func (c *Client) baseURL() string {
	if c.BaseURL != "" {
		return c.BaseURL
	}
	return defaultBaseURL
}

func (c *Client) isRetryableStatus(code int) bool {
	if c.RetryableStatusCodes == nil {
		return code == 429 || code == 500 || code == 502 || code == 503 || code == 504
	}
	_, ok := c.RetryableStatusCodes[code]
	return ok
}

func (c *Client) sleepBackoff(ctx context.Context, attempt int, baseDelay, maxDelay time.Duration) {
	d := time.Duration(1<<attempt) * baseDelay
	if d > maxDelay {
		d = maxDelay
	}
	// 10% jitter
	jitter := time.Duration(rand.Float64() * float64(d) * 0.1)
	d += jitter
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return
	case <-t.C:
		return
	}
}

func isRetryableNetErr(err error) bool {
	if err == nil {
		return false
	}
	var netErr net.Error
	if errors.As(err, &netErr) {
		return netErr.Timeout() || netErr.Temporary()
	}
	// best-effort string matching
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "timeout") || strings.Contains(msg, "connection") || strings.Contains(msg, "tls")
}

// ---- request/response mapping ----

type cacheControl struct {
	Type string `json:"type"`
}

type toolParam struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`
	CacheCtrl   *cacheControl  `json:"cache_control,omitempty"`
}

type toolChoiceParam struct {
	Type string `json:"type"` // "auto"|"any"|"none"|"tool"
	Name string `json:"name,omitempty"`
}

type contentBlockParam struct {
	Type string `json:"type"`

	Text string `json:"text,omitempty"`

	// tool_use
	ID    string `json:"id,omitempty"`
	Name  string `json:"name,omitempty"`
	Input any    `json:"input,omitempty"`

	// tool_result
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   any    `json:"content,omitempty"`
	IsError   bool   `json:"is_error,omitempty"`

	// thinking
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
	Data      string `json:"data,omitempty"`

	CacheCtrl *cacheControl `json:"cache_control,omitempty"`
}

type messageParam struct {
	Role    string             `json:"role"`
	Content []contentBlockParam `json:"content"`
}

type thinkingParam struct {
	Type        string `json:"type"` // "enabled"
	BudgetTokens int   `json:"budget_tokens"`
}

type requestPayload struct {
	Model     string `json:"model"`
	MaxTokens int    `json:"max_tokens"`

	System any           `json:"system,omitempty"` // string or []contentBlockParam
	Messages []messageParam `json:"messages"`

	Tools      []toolParam      `json:"tools,omitempty"`
	ToolChoice *toolChoiceParam `json:"tool_choice,omitempty"`

	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	Seed        *int     `json:"seed,omitempty"`

	Thinking *thinkingParam `json:"thinking,omitempty"`
}

func (c *Client) buildRequest(req llm.InvokeRequest) (*requestPayload, error) {
	if c.ModelName == "" {
		return nil, fmt.Errorf("anthropic: model is required")
	}
	maxTokens := c.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 8192
	}

	sys, msgs, err := serializeMessages(req.Messages)
	if err != nil {
		return nil, err
	}

	tools := []toolParam(nil)
	if len(req.Tools) > 0 {
		tools = serializeTools(req.Tools, c.MaxCachedToolDefinitions)
	}

	var toolChoice *toolChoiceParam
	if len(tools) > 0 {
		tc := req.ToolChoice
		if tc == "" {
			tc = "auto"
		}
		toolChoice = mapToolChoice(tc)
	}

	// allow per-call temperature override
	temp := c.Temperature
	if req.Temperature != nil {
		temp = req.Temperature
	}

	var thinking *thinkingParam
	if c.ThinkingBudgetTokens != nil && *c.ThinkingBudgetTokens > 0 {
		thinking = &thinkingParam{Type: "enabled", BudgetTokens: *c.ThinkingBudgetTokens}
	}

	return &requestPayload{
		Model:       c.ModelName,
		MaxTokens:   maxTokens,
		System:      sys,
		Messages:    msgs,
		Tools:       tools,
		ToolChoice:  toolChoice,
		Temperature: temp,
		TopP:        c.TopP,
		Seed:        c.Seed,
		Thinking:    thinking,
	}, nil
}

func mapToolChoice(choice llm.ToolChoice) *toolChoiceParam {
	s := string(choice)
	switch s {
	case "auto":
		return &toolChoiceParam{Type: "auto"}
	case "required":
		return &toolChoiceParam{Type: "any"}
	case "none":
		return &toolChoiceParam{Type: "none"}
	default:
		if s == "" {
			return &toolChoiceParam{Type: "auto"}
		}
		return &toolChoiceParam{Type: "tool", Name: s}
	}
}

func serializeTools(tools []llm.ToolDefinition, maxCached int) []toolParam {
	res := make([]toolParam, 0, len(tools))
	cacheCount := maxCached
	if cacheCount <= 0 {
		cacheCount = 0
	}
	cacheStart := len(tools) - cacheCount
	if cacheStart < 0 {
		cacheStart = 0
	}
	for i, t := range tools {
		schema := map[string]any{}
		for k, v := range t.Parameters {
			schema[k] = v
		}
		delete(schema, "title")
		p := toolParam{Name: t.Name, Description: t.Description, InputSchema: schema}
		if i >= cacheStart {
			p.CacheCtrl = &cacheControl{Type: "ephemeral"}
		}
		res = append(res, p)
	}
	return res
}

func serializeMessages(in []llm.Message) (system any, out []messageParam, err error) {
	var sysBlocks []contentBlockParam
	var sysTextParts []string

	for _, m := range in {
		switch m.Role {
		case llm.RoleSystem:
			if strings.TrimSpace(m.Content.Text) != "" {
				blk := contentBlockParam{Type: "text", Text: m.Content.Text}
				if m.Cache {
					blk.CacheCtrl = &cacheControl{Type: "ephemeral"}
					sysBlocks = append(sysBlocks, blk)
				} else {
					sysTextParts = append(sysTextParts, m.Content.Text)
				}
			}
			for _, b := range m.Content.Blocks {
				sysBlocks = append(sysBlocks, toAnthropicBlock(b, m.Cache))
			}
		default:
			mp, e := toAnthropicMessage(m)
			if e != nil {
				return nil, nil, e
			}
			if mp != nil {
				out = append(out, *mp)
			}
		}
	}

	if len(sysBlocks) > 0 {
		// Use structured system blocks when we need cache_control or non-text blocks.
		system = sysBlocks
	} else if len(sysTextParts) > 0 {
		system = strings.Join(sysTextParts, "\n\n")
	}
	return system, out, nil
}

func toAnthropicMessage(m llm.Message) (*messageParam, error) {
	if m.Role == llm.RoleTool {
		// Anthropic expects tool results as role=user with tool_result blocks.
		content := contentBlockParam{
			Type:      "tool_result",
			ToolUseID: m.ToolCallID,
			Content:   m.Content.PlainText(),
			IsError:   m.IsError,
		}
		return &messageParam{Role: "user", Content: []contentBlockParam{content}}, nil
	}

	role := string(m.Role)
	if role != "user" && role != "assistant" {
		return nil, nil
	}

	blocks := []contentBlockParam{}
	if strings.TrimSpace(m.Content.Text) != "" {
		blk := contentBlockParam{Type: "text", Text: m.Content.Text}
		if m.Cache {
			blk.CacheCtrl = &cacheControl{Type: "ephemeral"}
		}
		blocks = append(blocks, blk)
	}
	for _, b := range m.Content.Blocks {
		blocks = append(blocks, toAnthropicBlock(b, m.Cache))
	}

	if m.Role == llm.RoleAssistant && len(m.ToolCalls) > 0 {
		for _, tc := range m.ToolCalls {
			input := any(map[string]any{})
			if strings.TrimSpace(tc.Function.Arguments) != "" {
				var v any
				if json.Unmarshal([]byte(tc.Function.Arguments), &v) == nil {
					input = v
				} else {
					input = map[string]any{"_raw": tc.Function.Arguments}
				}
			}
			blocks = append(blocks, contentBlockParam{
				Type:  "tool_use",
				ID:    tc.ID,
				Name:  tc.Function.Name,
				Input: input,
			})
		}
	}

	if len(blocks) == 0 {
		blocks = append(blocks, contentBlockParam{Type: "text", Text: ""})
	}
	return &messageParam{Role: role, Content: blocks}, nil
}

func toAnthropicBlock(b llm.ContentBlock, inheritCache bool) contentBlockParam {
	blk := contentBlockParam{Type: b.Type}
	if inheritCache {
		blk.CacheCtrl = &cacheControl{Type: "ephemeral"}
	}
	switch b.Type {
	case "text":
		blk.Text = b.Text
	case "thinking":
		blk.Thinking = b.Thinking
		blk.Signature = b.Signature
	case "redacted_thinking":
		blk.Data = b.Data
	default:
		// unsupported blocks are ignored on wire; keep placeholder text.
		blk.Type = "text"
		blk.Text = ""
	}
	return blk
}

type responsePayload struct {
	Content []struct {
		Type      string           `json:"type"`
		Text      string           `json:"text,omitempty"`
		ID        string           `json:"id,omitempty"`
		Name      string           `json:"name,omitempty"`
		Input     json.RawMessage  `json:"input,omitempty"`
		Thinking  string           `json:"thinking,omitempty"`
		Signature string           `json:"signature,omitempty"`
		Data      string           `json:"data,omitempty"`
	} `json:"content"`
	Usage struct {
		InputTokens              int  `json:"input_tokens"`
		OutputTokens             int  `json:"output_tokens"`
		CacheReadInputTokens     *int `json:"cache_read_input_tokens,omitempty"`
		CacheCreationInputTokens *int `json:"cache_creation_input_tokens,omitempty"`
	} `json:"usage"`
}

func parseResponse(data []byte) (*llm.Completion, error) {
	var rp responsePayload
	if err := json.Unmarshal(data, &rp); err != nil {
		return nil, err
	}

	blocks := make([]llm.ContentBlock, 0, len(rp.Content))
	toolCalls := []llm.ToolCall{}
	thinkingParts := []string{}

	for _, blk := range rp.Content {
		switch blk.Type {
		case "text":
			blocks = append(blocks, llm.ContentBlock{Type: "text", Text: blk.Text})
		case "tool_use":
			args := "{}"
			if len(blk.Input) > 0 {
				args = string(blk.Input)
			}
			toolCalls = append(toolCalls, llm.ToolCall{
				ID:   blk.ID,
				Type: "function",
				Function: llm.FunctionCall{
					Name:      blk.Name,
					Arguments: args,
				},
			})
		case "thinking":
			thinkingParts = append(thinkingParts, blk.Thinking)
			blocks = append(blocks, llm.ContentBlock{Type: "thinking", Thinking: blk.Thinking, Signature: blk.Signature})
		case "redacted_thinking":
			blocks = append(blocks, llm.ContentBlock{Type: "redacted_thinking", Data: blk.Data})
		default:
			// ignore unknown
		}
	}

	pt := rp.Usage.InputTokens
	if rp.Usage.CacheReadInputTokens != nil {
		pt += *rp.Usage.CacheReadInputTokens
	}
	usage := &llm.Usage{
		PromptTokens:              pt,
		CompletionTokens:          rp.Usage.OutputTokens,
		TotalTokens:               rp.Usage.InputTokens + rp.Usage.OutputTokens,
		PromptCachedTokens:        rp.Usage.CacheReadInputTokens,
		PromptCacheCreationTokens: rp.Usage.CacheCreationInputTokens,
	}

	return &llm.Completion{
		Content:   llm.Content{Blocks: blocks},
		Thinking:  strings.Join(thinkingParts, "\n"),
		ToolCalls: toolCalls,
		Usage:     usage,
		Raw:       append([]byte(nil), data...),
	}, nil
}

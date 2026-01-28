package openai

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

const defaultBaseURL = "https://api.openai.com"

type ChatClient struct {
	HTTPClient *http.Client
	BaseURL    string
	APIKey     string

	ModelName string

	Temperature       *float64
	TopP              *float64
	Seed              *int
	MaxCompletionTokens *int

	// Reasoning effort for reasoning-capable models (best-effort).
	ReasoningEffort string

	MaxRetries           int
	RetryBaseDelay       time.Duration
	RetryMaxDelay        time.Duration
	RetryableStatusCodes map[int]struct{}

	// If true, include "parallel_tool_calls" when tools are provided.
	ParallelToolCalls bool
}

func (c *ChatClient) Provider() string { return "openai" }

func (c *ChatClient) Model() string { return c.ModelName }

func (c *ChatClient) Invoke(ctx context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
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

	endpoint := baseURL + "/v1/chat/completions"
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
		if c.APIKey != "" {
			httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
		}

		resp, err := client.Do(httpReq)
		if err == nil {
			defer resp.Body.Close()
			data, readErr := io.ReadAll(resp.Body)
			if readErr != nil {
				return nil, readErr
			}

			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				return parseChatCompletion(data)
			}

			msg := strings.TrimSpace(string(data))
			if resp.StatusCode == 429 {
				lastErr = &llm.RateLimitError{Provider: "openai", Message: msg}
			} else {
				lastErr = &llm.ProviderError{Provider: "openai", StatusCode: resp.StatusCode, Message: msg}
			}
			if c.isRetryableStatus(resp.StatusCode) && attempt < maxRetries-1 {
				c.sleepBackoff(ctx, attempt, baseDelay, maxDelay)
				continue
			}
			return nil, lastErr
		}

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
	return nil, errors.New("openai: retry loop ended without result")
}

func (c *ChatClient) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return &http.Client{Timeout: 60 * time.Second}
}

func (c *ChatClient) baseURL() string {
	if c.BaseURL != "" {
		return c.BaseURL
	}
	return defaultBaseURL
}

func (c *ChatClient) isRetryableStatus(code int) bool {
	if c.RetryableStatusCodes == nil {
		return code == 429 || code == 500 || code == 502 || code == 503 || code == 504
	}
	_, ok := c.RetryableStatusCodes[code]
	return ok
}

func (c *ChatClient) sleepBackoff(ctx context.Context, attempt int, baseDelay, maxDelay time.Duration) {
	d := time.Duration(1<<attempt) * baseDelay
	if d > maxDelay {
		d = maxDelay
	}
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
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "timeout") || strings.Contains(msg, "connection") || strings.Contains(msg, "tls")
}

// ---- request mapping ----

type toolFnDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
	Strict      bool           `json:"strict,omitempty"`
}

type toolParam struct {
	Type     string    `json:"type"` // "function"
	Function toolFnDef `json:"function"`
}

type toolChoiceFunction struct {
	Name string `json:"name"`
}

type toolChoiceParam struct {
	Type     string             `json:"type"` // "function"
	Function toolChoiceFunction `json:"function"`
}

type messageParam struct {
	Role       string        `json:"role"`
	Content    any           `json:"content,omitempty"`
	ToolCalls  []llm.ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
}

type chatRequest struct {
	Model    string         `json:"model"`
	Messages []messageParam `json:"messages"`

	Tools      []toolParam `json:"tools,omitempty"`
	ToolChoice any        `json:"tool_choice,omitempty"` // string or toolChoiceParam

	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	Seed        *int     `json:"seed,omitempty"`

	MaxCompletionTokens *int `json:"max_completion_tokens,omitempty"`

	ReasoningEffort string `json:"reasoning_effort,omitempty"`

	ParallelToolCalls *bool `json:"parallel_tool_calls,omitempty"`
}

func (c *ChatClient) buildRequest(req llm.InvokeRequest) (*chatRequest, error) {
	if c.ModelName == "" {
		return nil, fmt.Errorf("openai: model is required")
	}

	msgs := make([]messageParam, 0, len(req.Messages))
	for _, m := range req.Messages {
		mp, err := toChatMessage(m)
		if err != nil {
			return nil, err
		}
		if mp != nil {
			msgs = append(msgs, *mp)
		}
	}

	tools := []toolParam(nil)
	if len(req.Tools) > 0 {
		tools = make([]toolParam, 0, len(req.Tools))
		for _, t := range req.Tools {
			params := cloneMap(t.Parameters)
			if t.Strict {
				params = makeStrictSchema(params)
			}
			tools = append(tools, toolParam{
				Type: "function",
				Function: toolFnDef{
					Name:        t.Name,
					Description: t.Description,
					Parameters:  params,
					Strict:      t.Strict,
				},
			})
		}
	}

	var toolChoice any
	if len(tools) > 0 {
		tc := string(req.ToolChoice)
		if tc == "" {
			tc = "auto"
		}
		switch tc {
		case "auto", "none", "required":
			toolChoice = tc
		default:
			toolChoice = toolChoiceParam{Type: "function", Function: toolChoiceFunction{Name: tc}}
		}
	}

	temp := c.Temperature
	if req.Temperature != nil {
		temp = req.Temperature
	}

	var ptc *bool
	if len(tools) > 0 {
		v := c.ParallelToolCalls
		ptc = &v
	}

	return &chatRequest{
		Model:              c.ModelName,
		Messages:           msgs,
		Tools:              tools,
		ToolChoice:         toolChoice,
		Temperature:        temp,
		TopP:               c.TopP,
		Seed:               c.Seed,
		MaxCompletionTokens: c.MaxCompletionTokens,
		ReasoningEffort:    c.ReasoningEffort,
		ParallelToolCalls:  ptc,
	}, nil
}

func toChatMessage(m llm.Message) (*messageParam, error) {
	role := string(m.Role)
	if role == "system" || role == "user" || role == "assistant" {
		mp := &messageParam{Role: role}
		if m.Role == llm.RoleAssistant {
			if len(m.ToolCalls) > 0 {
				mp.ToolCalls = append([]llm.ToolCall(nil), m.ToolCalls...)
			}
			// content may be empty when tool_calls exist
			if strings.TrimSpace(m.Content.Text) != "" || len(m.Content.Blocks) > 0 {
				mp.Content = contentToOpenAI(m.Content)
			}
			return mp, nil
		}
		mp.Content = contentToOpenAI(m.Content)
		return mp, nil
	}
	if role == "tool" {
		return &messageParam{Role: "tool", Content: m.Content.PlainText(), ToolCallID: m.ToolCallID}, nil
	}
	return nil, nil
}

func contentToOpenAI(c llm.Content) any {
	if len(c.Blocks) == 0 {
		return c.Text
	}
	parts := make([]map[string]any, 0, len(c.Blocks)+1)
	if strings.TrimSpace(c.Text) != "" {
		parts = append(parts, map[string]any{"type": "text", "text": c.Text})
	}
	for _, b := range c.Blocks {
		if b.Type == "text" {
			parts = append(parts, map[string]any{"type": "text", "text": b.Text})
		}
		// images/documents are out of scope for now
	}
	return parts
}

func cloneMap(in map[string]any) map[string]any {
	if in == nil {
		return nil
	}
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

// makeStrictSchema transforms a schema for OpenAI strict mode:
// - all properties become required
// - previously-optional properties become nullable
// - additionalProperties=false
func makeStrictSchema(schema map[string]any) map[string]any {
	s := cloneMap(schema)
	props, _ := s["properties"].(map[string]any)
	if props == nil {
		return s
	}
	requiredSet := map[string]struct{}{}
	if req, ok := s["required"].([]any); ok {
		for _, x := range req {
			if name, ok := x.(string); ok {
				requiredSet[name] = struct{}{}
			}
		}
	} else if req, ok := s["required"].([]string); ok {
		for _, name := range req {
			requiredSet[name] = struct{}{}
		}
	}

	newProps := map[string]any{}
	all := make([]string, 0, len(props))
	for name, propAny := range props {
		all = append(all, name)
		prop, _ := propAny.(map[string]any)
		if prop == nil {
			newProps[name] = propAny
			continue
		}
		_, wasRequired := requiredSet[name]
		newProps[name] = makeStrictProperty(prop, wasRequired)
	}
	s["properties"] = newProps
	// all required
	reqList := make([]any, 0, len(all))
	for _, name := range all {
		reqList = append(reqList, name)
	}
	s["required"] = reqList
	s["additionalProperties"] = false
	return s
}

func makeStrictProperty(prop map[string]any, wasRequired bool) map[string]any {
	p := cloneMap(prop)
	// recurse nested objects
	if t, _ := p["type"].(string); t == "object" {
		p = makeStrictSchema(p)
	}
	if !wasRequired {
		// allow null
		if t, ok := p["type"].(string); ok {
			p["type"] = []any{t, "null"}
			return p
		}
		if arr, ok := p["type"].([]any); ok {
			for _, v := range arr {
				if s, ok := v.(string); ok && s == "null" {
					return p
				}
			}
			p["type"] = append(arr, "null")
			return p
		}
		// fallback
		p["nullable"] = true
	}
	return p
}

// ---- response parsing ----

type chatCompletionResponse struct {
	Choices []struct {
		Message struct {
			Role      string        `json:"role"`
			Content   string        `json:"content"`
			ToolCalls []llm.ToolCall `json:"tool_calls"`
		} `json:"message"`
	} `json:"choices"`
	Usage map[string]any `json:"usage"`
}

func parseChatCompletion(data []byte) (*llm.Completion, error) {
	var r chatCompletionResponse
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, err
	}
	if len(r.Choices) == 0 {
		return nil, fmt.Errorf("openai: empty choices")
	}
	msg := r.Choices[0].Message

	usage := parseUsage(r.Usage)

	return &llm.Completion{
		Content:   llm.TextContent(msg.Content),
		ToolCalls: msg.ToolCalls,
		Usage:     usage,
		Raw:       append([]byte(nil), data...),
	}, nil
}

func parseUsage(u map[string]any) *llm.Usage {
	if u == nil {
		return nil
	}
	pt := intFromAny(u["prompt_tokens"])
	ct := intFromAny(u["completion_tokens"])
	// completion_tokens_details.reasoning_tokens
	if det, ok := u["completion_tokens_details"].(map[string]any); ok {
		ct += intFromAny(det["reasoning_tokens"])
	}
	tt := intFromAny(u["total_tokens"])
	var cached *int
	if det, ok := u["prompt_tokens_details"].(map[string]any); ok {
		v := intFromAny(det["cached_tokens"])
		if v > 0 {
			cached = &v
		}
	}
	return &llm.Usage{PromptTokens: pt, CompletionTokens: ct, TotalTokens: tt, PromptCachedTokens: cached}
}

func intFromAny(v any) int {
	switch x := v.(type) {
	case float64:
		return int(x)
	case int:
		return x
	case int64:
		return int(x)
	case json.Number:
		i, _ := x.Int64()
		return int(i)
	default:
		return 0
	}
}

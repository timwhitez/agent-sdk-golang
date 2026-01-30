package openai

import (
	"bufio"
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

	Temperature         *float64
	TopP                *float64
	Seed                *int
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
	endpoint := openAIEndpoint(baseURL, "chat/completions")
	lastErr := error(nil)

	maxRetries := c.MaxRetries
	if maxRetries <= 0 {
		maxRetries = 10
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
		payload, err := c.buildRequest(req)
		if err != nil {
			return nil, err
		}
		body, err := json.Marshal(payload)
		if err != nil {
			return nil, err
		}

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
			data, readErr := io.ReadAll(resp.Body)
			_ = resp.Body.Close()
			if readErr != nil {
				return nil, readErr
			}

			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				return parseChatCompletion(data)
			}

			retryAfter := parseRetryAfter(resp.Header.Get("Retry-After"))
			msg := strings.TrimSpace(string(data))
			if msg == "" {
				msg = resp.Status
			}
			// Include endpoint to make debugging (base-url/path) easier.
			msg = fmt.Sprintf("%s (POST %s)", msg, endpoint)

			// Automatic downgrade: some OpenAI-compatible providers/models reject reasoning_effort.
			if (resp.StatusCode == 400 || resp.StatusCode == 422) && strings.TrimSpace(c.ReasoningEffort) != "" && looksLikeReasoningUnsupported(msg) {
				// Disable for subsequent calls too.
				c.ReasoningEffort = ""
				// Retry immediately without surfacing the first error.
				if attempt < maxRetries-1 {
					continue
				}
			}
			if resp.StatusCode == 429 {
				lastErr = &llm.RateLimitError{Provider: "openai", Message: msg}
			} else {
				lastErr = &llm.ProviderError{Provider: "openai", StatusCode: resp.StatusCode, Message: msg}
			}
			if c.isRetryableStatus(resp.StatusCode) && attempt < maxRetries-1 {
				c.sleepBackoff(ctx, attempt, baseDelay, maxDelay, retryAfter)
				continue
			}
			return nil, lastErr
		}

		lastErr = err
		if attempt < maxRetries-1 && isRetryableNetErr(err) {
			c.sleepBackoff(ctx, attempt, baseDelay, maxDelay, 0)
			continue
		}
		return nil, err
	}

	if lastErr != nil {
		return nil, lastErr
	}
	return nil, errors.New("openai: retry loop ended without result")
}

// InvokeStream implements true SSE streaming for OpenAI chat/completions.
// It emits text deltas and tool_call deltas.
func (c *ChatClient) InvokeStream(ctx context.Context, req llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	out := make(chan llm.StreamEvent, 128)
	go func() {
		defer close(out)

		client := streamHTTPClient(c.httpClient())
		baseURL := strings.TrimRight(c.baseURL(), "/")
		endpoint := openAIEndpoint(baseURL, "chat/completions")

		maxRetries := c.MaxRetries
		if maxRetries <= 0 {
			maxRetries = 10
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
			payload, err := c.buildRequest(req)
			if err != nil {
				out <- llm.StreamErrorEvent{Err: err}
				return
			}
			payload.Stream = true
			payload.StreamOptions = map[string]any{"include_usage": true}
			body, err := json.Marshal(payload)
			if err != nil {
				out <- llm.StreamErrorEvent{Err: err}
				return
			}

			httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
			if err != nil {
				out <- llm.StreamErrorEvent{Err: err}
				return
			}
			httpReq.Header.Set("Content-Type", "application/json")
			httpReq.Header.Set("Accept", "text/event-stream")
			if c.APIKey != "" {
				httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
			}

			resp, err := client.Do(httpReq)
			if err != nil {
				if attempt < maxRetries-1 && isRetryableNetErr(err) {
					c.sleepBackoff(ctx, attempt, baseDelay, maxDelay, 0)
					continue
				}
				out <- llm.StreamErrorEvent{Err: err}
				return
			}

			if resp.StatusCode < 200 || resp.StatusCode >= 300 {
				data, readErr := io.ReadAll(resp.Body)
				_ = resp.Body.Close()
				if readErr != nil {
					out <- llm.StreamErrorEvent{Err: readErr}
					return
				}
				retryAfter := parseRetryAfter(resp.Header.Get("Retry-After"))
				msg := strings.TrimSpace(string(data))
				if msg == "" {
					msg = resp.Status
				}
				msg = fmt.Sprintf("%s (POST %s)", msg, endpoint)

				// Automatic downgrade: disable reasoning_effort when unsupported.
				if (resp.StatusCode == 400 || resp.StatusCode == 422) && strings.TrimSpace(c.ReasoningEffort) != "" && looksLikeReasoningUnsupported(msg) {
					c.ReasoningEffort = ""
					if attempt < maxRetries-1 {
						continue
					}
				}

				var lastErr error
				if resp.StatusCode == 429 {
					lastErr = &llm.RateLimitError{Provider: "openai", Message: msg}
				} else {
					lastErr = &llm.ProviderError{Provider: "openai", StatusCode: resp.StatusCode, Message: msg}
				}
				if c.isRetryableStatus(resp.StatusCode) && attempt < maxRetries-1 {
					c.sleepBackoff(ctx, attempt, baseDelay, maxDelay, retryAfter)
					continue
				}
				out <- llm.StreamErrorEvent{Err: lastErr}
				return
			}

			stopReason := ""
			err = consumeSSE(resp.Body, func(data string) error {
				data = strings.TrimSpace(data)
				if data == "" {
					return nil
				}
				if data == "[DONE]" {
					return errSSEDone
				}
				var r chatCompletionStreamResponse
				if json.Unmarshal([]byte(data), &r) != nil {
					return nil
				}
				if r.Error != nil && strings.TrimSpace(r.Error.Message) != "" {
					return fmt.Errorf("openai stream error: %s", r.Error.Message)
				}
				if u := parseUsage(r.Usage); u != nil {
					out <- llm.StreamUsageEvent{Usage: *u}
				}
				for _, ch := range r.Choices {
					if ch.FinishReason != "" {
						sr := ch.FinishReason
						if sr == "length" {
							sr = "max_tokens"
						}
						stopReason = sr
					}
					// Preserve whitespace deltas to keep streaming output faithful.
					if ch.Delta.Content != "" {
						out <- llm.StreamTextDeltaEvent{Delta: ch.Delta.Content}
					}
					for _, tc := range ch.Delta.ToolCalls {
						name := strings.TrimSpace(tc.Function.Name)
						args := tc.Function.Arguments
						if name != "" || strings.TrimSpace(args) != "" || strings.TrimSpace(tc.ID) != "" {
							out <- llm.StreamToolCallDeltaEvent{Index: tc.Index, ID: tc.ID, NameDelta: name, ArgumentsDelta: args}
						}
					}
				}
				return nil
			})
			_ = resp.Body.Close()
			if errors.Is(err, errSSEDone) {
				out <- llm.StreamDoneEvent{StopReason: stopReason}
				return
			}
			if err != nil {
				out <- llm.StreamErrorEvent{Err: err}
				return
			}
			out <- llm.StreamDoneEvent{StopReason: stopReason}
			return
		}
		out <- llm.StreamErrorEvent{Err: errors.New("openai stream: retry loop ended without result")}
	}()
	return out, nil
}

type chatCompletionStreamResponse struct {
	Choices []struct {
		Delta struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage map[string]any `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

var errSSEDone = errors.New("_sse_done")

func consumeSSE(r io.Reader, onData func(data string) error) error {
	sc := bufio.NewScanner(r)
	// Large chunks can appear in tool-call argument streaming.
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	dataLines := []string{}
	flush := func() error {
		if len(dataLines) == 0 {
			return nil
		}
		data := strings.Join(dataLines, "\n")
		dataLines = nil
		return onData(data)
	}
	for sc.Scan() {
		line := sc.Text()
		if line == "" {
			if err := flush(); err != nil {
				return err
			}
			continue
		}
		if strings.HasPrefix(line, "data:") {
			dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
	if err := sc.Err(); err != nil {
		return err
	}
	return flush()
}

func streamHTTPClient(base *http.Client) *http.Client {
	if base == nil {
		return &http.Client{Timeout: 0}
	}
	if base.Timeout == 0 {
		return base
	}
	cpy := *base
	cpy.Timeout = 0
	return &cpy
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

// openAIEndpoint builds an endpoint based on the provided baseURL.
//
// It supports common OpenAI-compatible gateway styles:
// - baseURL like "https://api.openai.com" => "/v1/..."
// - baseURL like "https://proxy.example.com/v1" => "/..." (avoid double v1)
// - baseURL like "https://host/api/v3" => "/..." (enterprise version path already encoded)
func openAIEndpoint(baseURL, suffix string) string {
	baseURL = strings.TrimRight(strings.TrimSpace(baseURL), "/")
	suffix = strings.TrimLeft(strings.TrimSpace(suffix), "/")
	if baseURL == "" {
		baseURL = strings.TrimRight(defaultBaseURL, "/")
	}
	if suffix == "" {
		return baseURL
	}

	// If base already ends with /v1, do not add another /v1.
	if strings.HasSuffix(baseURL, "/v1") {
		return baseURL + "/" + suffix
	}
	// If base contains explicit enterprise versioning like /api/v3, assume version is included.
	if strings.Contains(baseURL, "/api/v") {
		return baseURL + "/" + suffix
	}
	return baseURL + "/v1/" + suffix
}

func (c *ChatClient) isRetryableStatus(code int) bool {
	if c.RetryableStatusCodes == nil {
		return code == 429 || code == 500 || code == 502 || code == 503 || code == 504
	}
	_, ok := c.RetryableStatusCodes[code]
	return ok
}

func (c *ChatClient) sleepBackoff(ctx context.Context, attempt int, baseDelay, maxDelay time.Duration, retryAfter time.Duration) {
	d := time.Duration(1<<attempt) * baseDelay
	if d > maxDelay {
		d = maxDelay
	}
	if retryAfter > d {
		d = retryAfter
		if d > maxDelay {
			d = maxDelay
		}
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

func looksLikeReasoningUnsupported(msg string) bool {
	s := strings.ToLower(msg)
	// Common patterns from OpenAI-compatible gateways.
	if strings.Contains(s, "reasoning_effort") {
		return true
	}
	if strings.Contains(s, "unknown") && strings.Contains(s, "reasoning") {
		return true
	}
	if strings.Contains(s, "unsupported") && strings.Contains(s, "reasoning") {
		return true
	}
	if strings.Contains(s, "unrecognized") && strings.Contains(s, "reasoning") {
		return true
	}
	return false
}

func parseRetryAfter(v string) time.Duration {
	v = strings.TrimSpace(v)
	if v == "" {
		return 0
	}
	// Retry-After can be seconds or an HTTP date.
	if secs, err := time.ParseDuration(v + "s"); err == nil {
		if secs > 0 {
			return secs
		}
	}
	if t, err := http.ParseTime(v); err == nil {
		d := time.Until(t)
		if d > 0 {
			return d
		}
	}
	return 0
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
	Role       string         `json:"role"`
	Content    any            `json:"content,omitempty"`
	ToolCalls  []llm.ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
}

type chatRequest struct {
	Model    string         `json:"model"`
	Messages []messageParam `json:"messages"`

	Tools      []toolParam `json:"tools,omitempty"`
	ToolChoice any         `json:"tool_choice,omitempty"` // string or toolChoiceParam

	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	Seed        *int     `json:"seed,omitempty"`

	MaxCompletionTokens *int `json:"max_completion_tokens,omitempty"`

	ReasoningEffort string `json:"reasoning_effort,omitempty"`

	ParallelToolCalls *bool `json:"parallel_tool_calls,omitempty"`

	Stream        bool           `json:"stream,omitempty"`
	StreamOptions map[string]any `json:"stream_options,omitempty"`
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
		Model:               c.ModelName,
		Messages:            msgs,
		Tools:               tools,
		ToolChoice:          toolChoice,
		Temperature:         temp,
		TopP:                c.TopP,
		Seed:                c.Seed,
		MaxCompletionTokens: c.MaxCompletionTokens,
		ReasoningEffort:     c.ReasoningEffort,
		ParallelToolCalls:   ptc,
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
			Role      string         `json:"role"`
			Content   string         `json:"content"`
			ToolCalls []llm.ToolCall `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
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

	// Map OpenAI finish_reason to normalized stop reason
	stopReason := r.Choices[0].FinishReason
	if stopReason == "length" {
		stopReason = "max_tokens"
	}

	return &llm.Completion{
		Content:    llm.TextContent(msg.Content),
		ToolCalls:  msg.ToolCalls,
		Usage:      usage,
		StopReason: stopReason,
		Raw:        append([]byte(nil), data...),
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

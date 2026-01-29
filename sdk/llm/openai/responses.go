package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"strings"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// ResponsesClient implements the OpenAI Responses API (/v1/responses).
// This is best-effort and focuses on tool calling + text output.
type ResponsesClient struct {
	HTTPClient *http.Client
	BaseURL    string
	APIKey     string

	ModelName string

	Temperature     *float64
	TopP            *float64
	Seed            *int
	MaxOutputTokens *int

	ReasoningEffort string

	MaxRetries           int
	RetryBaseDelay       time.Duration
	RetryMaxDelay        time.Duration
	RetryableStatusCodes map[int]struct{}

	// ForceStringInput enables a compatibility mode where `input[].content` is sent as a plain string
	// instead of the official array-of-content-parts form.
	// Some OpenAI-compatible gateways (e.g. certain enterprise proxies) require this.
	ForceStringInput bool
}

func (c *ResponsesClient) Provider() string { return "openai" }

func (c *ResponsesClient) Model() string { return c.ModelName }

func (c *ResponsesClient) Invoke(ctx context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	client := c.httpClient()
	baseURL := strings.TrimRight(c.baseURL(), "/")
	endpoint := openAIEndpoint(baseURL, "responses")
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
				return parseResponses(data)
			}

			retryAfter := parseRetryAfter(resp.Header.Get("Retry-After"))
			msg := strings.TrimSpace(string(data))
			if msg == "" {
				msg = resp.Status
			}
			// Include endpoint to make debugging (base-url/path) easier.
			msg = fmt.Sprintf("%s (POST %s)", msg, endpoint)

			// Automatic downgrade: some gateways reject reasoning settings.
			if (resp.StatusCode == 400 || resp.StatusCode == 422) && strings.TrimSpace(c.ReasoningEffort) != "" && looksLikeReasoningUnsupported(msg) {
				c.ReasoningEffort = ""
				if attempt < maxRetries-1 {
					continue
				}
			}
			// Automatic compat: some gateways require input.content to be a string.
			if (resp.StatusCode == 400 || resp.StatusCode == 422) && strings.Contains(msg, "MissingParameter") && strings.Contains(msg, "input.content") {
				c.ForceStringInput = true
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
	return nil, errors.New("openai responses: retry loop ended without result")
}

func (c *ResponsesClient) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return &http.Client{Timeout: 60 * time.Second}
}

func (c *ResponsesClient) baseURL() string {
	if c.BaseURL != "" {
		return c.BaseURL
	}
	return defaultBaseURL
}

func (c *ResponsesClient) isRetryableStatus(code int) bool {
	if c.RetryableStatusCodes == nil {
		return code == 429 || code == 500 || code == 502 || code == 503 || code == 504
	}
	_, ok := c.RetryableStatusCodes[code]
	return ok
}

func (c *ResponsesClient) sleepBackoff(ctx context.Context, attempt int, baseDelay, maxDelay time.Duration, retryAfter time.Duration) {
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

// ---- request mapping (best-effort) ----

type responsesMessage struct {
	Role       string `json:"role"`
	Content    any    `json:"content"`
	ToolCallID string `json:"tool_call_id,omitempty"`
}

type responsesTool struct {
	Type        string         `json:"type"` // "function"
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
}

type responsesRequest struct {
	Model string             `json:"model"`
	Input []responsesMessage `json:"input"`

	Tools []responsesTool `json:"tools,omitempty"`
	// tool_choice can be a string ("none"|"required") or an object; omit for default "auto".
	ToolChoice any `json:"tool_choice,omitempty"`

	Temperature     *float64 `json:"temperature,omitempty"`
	TopP            *float64 `json:"top_p,omitempty"`
	Seed            *int     `json:"seed,omitempty"`
	MaxOutputTokens *int     `json:"max_output_tokens,omitempty"`

	Reasoning map[string]any `json:"reasoning,omitempty"`

	Stream bool `json:"stream,omitempty"`
}

func (c *ResponsesClient) forceStringInput() bool {
	if c == nil {
		return false
	}
	if c.ForceStringInput {
		return true
	}
	// Auto-detect: enterprise versioned endpoints tend to be OpenAI-compatible but not fully spec-complete.
	base := strings.TrimRight(strings.TrimSpace(c.baseURL()), "/")
	return strings.Contains(base, "/api/v")
}

// InvokeStream implements true SSE streaming for OpenAI responses.
// It emits text deltas and basic tool-call deltas (best-effort).
func (c *ResponsesClient) InvokeStream(ctx context.Context, req llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	out := make(chan llm.StreamEvent, 128)
	go func() {
		defer close(out)

		client := streamHTTPClient(c.httpClient())
		baseURL := strings.TrimRight(c.baseURL(), "/")
		endpoint := openAIEndpoint(baseURL, "responses")

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

				// Automatic downgrade: disable reasoning when unsupported.
				if (resp.StatusCode == 400 || resp.StatusCode == 422) && strings.TrimSpace(c.ReasoningEffort) != "" && looksLikeReasoningUnsupported(msg) {
					c.ReasoningEffort = ""
					if attempt < maxRetries-1 {
						continue
					}
				}
				// Automatic compat: some gateways require input.content to be a string.
				if (resp.StatusCode == 400 || resp.StatusCode == 422) && strings.Contains(msg, "MissingParameter") && strings.Contains(msg, "input.content") {
					c.ForceStringInput = true
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

			idToIndex := map[string]int{}
			nextIndex := 0
			getIndex := func(id string) int {
				id = strings.TrimSpace(id)
				if id == "" {
					idx := nextIndex
					nextIndex++
					return idx
				}
				if v, ok := idToIndex[id]; ok {
					return v
				}
				idx := nextIndex
				nextIndex++
				idToIndex[id] = idx
				return idx
			}

			err = consumeSSE(resp.Body, func(data string) error {
				data = strings.TrimSpace(data)
				if data == "" {
					return nil
				}
				if data == "[DONE]" {
					return errSSEDone
				}
				var root map[string]any
				if json.Unmarshal([]byte(data), &root) != nil {
					return nil
				}
				typ, _ := root["type"].(string)
				switch typ {
				case "response.output_text.delta":
					// Preserve whitespace deltas to keep streaming output faithful.
					if d, ok := root["delta"].(string); ok && d != "" {
						out <- llm.StreamTextDeltaEvent{Delta: d}
					}
				case "response.output_item.added":
					item, _ := root["item"].(map[string]any)
					itType, _ := item["type"].(string)
					if itType == "function_call" || itType == "tool_call" {
						id, _ := item["id"].(string)
						if id == "" {
							id, _ = item["call_id"].(string)
						}
						name, _ := item["name"].(string)
						idx := getIndex(id)
						if strings.TrimSpace(name) != "" || strings.TrimSpace(id) != "" {
							out <- llm.StreamToolCallDeltaEvent{Index: idx, ID: id, NameDelta: name}
						}
					}
				case "response.function_call_arguments.delta":
					itemID, _ := root["item_id"].(string)
					if itemID == "" {
						itemID, _ = root["id"].(string)
					}
					if d, ok := root["delta"].(string); ok && strings.TrimSpace(d) != "" {
						idx := getIndex(itemID)
						out <- llm.StreamToolCallDeltaEvent{Index: idx, ID: itemID, ArgumentsDelta: d}
					}
				case "response.completed":
					respObj, _ := root["response"].(map[string]any)
					if u := usageFromResponses(respObj); u != nil {
						out <- llm.StreamUsageEvent{Usage: *u}
					}
				case "response.error", "error":
					if e, ok := root["error"].(map[string]any); ok {
						if m, ok := e["message"].(string); ok && strings.TrimSpace(m) != "" {
							return fmt.Errorf("openai responses stream error: %s", m)
						}
					}
				}
				return nil
			})
			_ = resp.Body.Close()
			if errors.Is(err, errSSEDone) {
				out <- llm.StreamDoneEvent{}
				return
			}
			if err != nil {
				out <- llm.StreamErrorEvent{Err: err}
				return
			}
			out <- llm.StreamDoneEvent{}
			return
		}
		out <- llm.StreamErrorEvent{Err: errors.New("openai responses stream: retry loop ended without result")}
	}()
	return out, nil
}

func usageFromResponses(resp map[string]any) *llm.Usage {
	if resp == nil {
		return nil
	}
	u, _ := resp["usage"].(map[string]any)
	if u == nil {
		return nil
	}
	pt := intFromAny(u["input_tokens"])
	ct := intFromAny(u["output_tokens"])
	tt := intFromAny(u["total_tokens"])
	if tt == 0 {
		tt = pt + ct
	}
	return &llm.Usage{PromptTokens: pt, CompletionTokens: ct, TotalTokens: tt}
}

func (c *ResponsesClient) buildRequest(req llm.InvokeRequest) (*responsesRequest, error) {
	if c.ModelName == "" {
		return nil, fmt.Errorf("openai responses: model is required")
	}

	stringContent := c.forceStringInput()
	input := make([]responsesMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		role := string(m.Role)
		if role != "system" && role != "user" && role != "assistant" && role != "tool" {
			continue
		}
		// Ark (and some OpenAI-compatible gateways) do not accept role "tool" in responses input.
		// Best-effort: convert tool outputs into a user message so the model can continue.
		if role == "tool" {
			role = "user"
			prefix := "[tool_result]"
			if strings.TrimSpace(m.ToolName) != "" {
				prefix += " name=" + m.ToolName
			}
			if strings.TrimSpace(m.ToolCallID) != "" {
				prefix += " id=" + m.ToolCallID
			}
			if m.IsError {
				prefix += " error=true"
			}
			// prepend to content
			m = llm.Message{Role: llm.RoleUser, Content: llm.TextContent(prefix + "\n" + m.Content.PlainText())}
		}
		txt := strings.TrimSpace(m.Content.PlainText())
		if txt == "" {
			// Some providers reject empty content; skip.
			continue
		}
		var content any
		if stringContent {
			content = txt
		} else {
			content = []map[string]any{{"type": "input_text", "text": txt}}
		}
		msg := responsesMessage{Role: role, Content: content}
		// tool_call_id is only valid for role "tool"; skip for compatibility.
		input = append(input, msg)
	}
	if len(input) == 0 {
		// Always send at least one user message to satisfy strict gateways.
		var content any = "(empty)"
		if !stringContent {
			content = []map[string]any{{"type": "input_text", "text": "(empty)"}}
		}
		input = append(input, responsesMessage{Role: "user", Content: content})
	}

	toolsList := []responsesTool(nil)
	if len(req.Tools) > 0 {
		toolsList = make([]responsesTool, 0, len(req.Tools))
		for _, t := range req.Tools {
			params := cloneMap(t.Parameters)
			if t.Strict {
				params = makeStrictSchema(params)
			}
			toolsList = append(toolsList, responsesTool{Type: "function", Name: t.Name, Description: t.Description, Parameters: params})
		}
	}

	var toolChoice any
	if len(toolsList) > 0 {
		choice := string(req.ToolChoice)
		// For compatibility with OpenAI-compatible gateways (e.g. Ark),
		// omit tool_choice for default "auto".
		switch choice {
		case "", "auto":
			toolChoice = nil
		case "none", "required":
			toolChoice = choice
		default:
			toolChoice = map[string]any{"type": "function", "name": choice}
		}
	}

	temp := c.Temperature
	if req.Temperature != nil {
		temp = req.Temperature
	}

	var reasoning map[string]any
	if c.ReasoningEffort != "" {
		reasoning = map[string]any{"effort": c.ReasoningEffort}
	}

	return &responsesRequest{
		Model:           c.ModelName,
		Input:           input,
		Tools:           toolsList,
		ToolChoice:      toolChoice,
		Temperature:     temp,
		TopP:            c.TopP,
		Seed:            c.Seed,
		MaxOutputTokens: c.MaxOutputTokens,
		Reasoning:       reasoning,
	}, nil
}

// ---- response parsing (best-effort) ----

func parseResponses(data []byte) (*llm.Completion, error) {
	var root map[string]any
	if err := json.Unmarshal(data, &root); err != nil {
		return nil, err
	}

	blocks := []llm.ContentBlock{}
	toolCalls := []llm.ToolCall{}

	if outArr, ok := root["output"].([]any); ok {
		for _, itemAny := range outArr {
			item, ok := itemAny.(map[string]any)
			if !ok {
				continue
			}
			typeStr, _ := item["type"].(string)
			switch typeStr {
			case "message":
				contentArr, _ := item["content"].([]any)
				for _, cAny := range contentArr {
					cm, ok := cAny.(map[string]any)
					if !ok {
						continue
					}
					ct, _ := cm["type"].(string)
					if ct == "output_text" || ct == "text" {
						if txt, ok := cm["text"].(string); ok {
							blocks = append(blocks, llm.ContentBlock{Type: "text", Text: txt})
						}
					}
				}
			case "function_call", "tool_call":
				id, _ := item["id"].(string)
				if id == "" {
					id, _ = item["call_id"].(string)
				}
				name, _ := item["name"].(string)
				args := "{}"
				if s, ok := item["arguments"].(string); ok && strings.TrimSpace(s) != "" {
					args = s
				} else if aAny, ok := item["arguments"].(map[string]any); ok {
					b, _ := json.Marshal(aAny)
					args = string(b)
				}
				if name != "" {
					toolCalls = append(toolCalls, llm.ToolCall{ID: id, Type: "function", Function: llm.FunctionCall{Name: name, Arguments: args}})
				}
			}
		}
	}

	usage := (*llm.Usage)(nil)
	if u, ok := root["usage"].(map[string]any); ok {
		pt := intFromAny(u["input_tokens"])
		ct := intFromAny(u["output_tokens"])
		tt := intFromAny(u["total_tokens"])
		usage = &llm.Usage{PromptTokens: pt, CompletionTokens: ct, TotalTokens: tt}
	}

	// Fallback: some responses variants include text at top-level "output_text".
	if len(blocks) == 0 {
		if t, ok := root["output_text"].(string); ok && strings.TrimSpace(t) != "" {
			blocks = append(blocks, llm.ContentBlock{Type: "text", Text: t})
		}
	}

	return &llm.Completion{Content: llm.Content{Blocks: blocks}, ToolCalls: toolCalls, Usage: usage, Raw: append([]byte(nil), data...)}, nil
}

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

	Temperature *float64
	TopP        *float64
	Seed        *int
	MaxOutputTokens *int

	ReasoningEffort string

	MaxRetries           int
	RetryBaseDelay       time.Duration
	RetryMaxDelay        time.Duration
	RetryableStatusCodes map[int]struct{}
}

func (c *ResponsesClient) Provider() string { return "openai" }

func (c *ResponsesClient) Model() string { return c.ModelName }

func (c *ResponsesClient) Invoke(ctx context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
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

	endpoint := baseURL + "/v1/responses"
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
				return parseResponses(data)
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

func (c *ResponsesClient) sleepBackoff(ctx context.Context, attempt int, baseDelay, maxDelay time.Duration) {
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

// ---- request mapping (best-effort) ----

type responsesMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
	ToolCallID string `json:"tool_call_id,omitempty"`
}

type responsesTool struct {
	Type        string         `json:"type"` // "function"
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
}

type responsesToolChoice struct {
	Type string `json:"type"` // "auto"|"none"|"required"|"function"
	Name string `json:"name,omitempty"`
}

type responsesRequest struct {
	Model string `json:"model"`
	Input []responsesMessage `json:"input"`

	Tools      []responsesTool      `json:"tools,omitempty"`
	ToolChoice *responsesToolChoice `json:"tool_choice,omitempty"`

	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	Seed        *int     `json:"seed,omitempty"`
	MaxOutputTokens *int `json:"max_output_tokens,omitempty"`

	Reasoning map[string]any `json:"reasoning,omitempty"`
}

func (c *ResponsesClient) buildRequest(req llm.InvokeRequest) (*responsesRequest, error) {
	if c.ModelName == "" {
		return nil, fmt.Errorf("openai responses: model is required")
	}

	input := make([]responsesMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		role := string(m.Role)
		if role != "system" && role != "user" && role != "assistant" && role != "tool" {
			continue
		}
		content := []map[string]any{}
		if txt := m.Content.PlainText(); strings.TrimSpace(txt) != "" {
			content = append(content, map[string]any{"type": "input_text", "text": txt})
		}
		msg := responsesMessage{Role: role, Content: content}
		if role == "tool" {
			msg.ToolCallID = m.ToolCallID
		}
		input = append(input, msg)
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

	var tc *responsesToolChoice
	if len(toolsList) > 0 {
		choice := string(req.ToolChoice)
		if choice == "" {
			choice = "auto"
		}
		switch choice {
		case "auto", "none", "required":
			tc = &responsesToolChoice{Type: choice}
		default:
			tc = &responsesToolChoice{Type: "function", Name: choice}
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
		Model:          c.ModelName,
		Input:          input,
		Tools:          toolsList,
		ToolChoice:     tc,
		Temperature:    temp,
		TopP:           c.TopP,
		Seed:           c.Seed,
		MaxOutputTokens: c.MaxOutputTokens,
		Reasoning:      reasoning,
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

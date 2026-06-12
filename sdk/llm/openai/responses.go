package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
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

	// ProviderLabel is returned by Provider and copied into provider errors.
	// Empty preserves the generic SDK label "openai".
	ProviderLabel string

	ModelName string

	// Extra request fields for OpenAI-compatible gateways.
	// Extra is merged at the top-level; ExtraBody is nested under "extra_body".
	Extra     map[string]any
	ExtraBody map[string]any

	Temperature     *float64
	TopP            *float64
	Seed            *int
	MaxOutputTokens *int
	ServiceTier     string

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

func (c *ResponsesClient) Provider() string { return openAIProviderLabel(c.ProviderLabel) }

func (c *ResponsesClient) Model() string { return c.ModelName }

func (c *ResponsesClient) Invoke(ctx context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	local := *c
	local.Extra = cloneMap(c.Extra)
	local.ExtraBody = cloneMap(c.ExtraBody)

	client := local.httpClient()
	baseURL := strings.TrimRight(local.baseURL(), "/")
	endpoint := openAIEndpoint(baseURL, "responses")
	lastErr := error(nil)

	retry := resolveRetryPolicy(local.MaxRetries, local.RetryBaseDelay, local.RetryMaxDelay)

	autoCompat := shouldAutoCompat(req)
	compatStage := responsesCompatFull

	for attempt := 0; attempt < retry.maxRetries; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		reqForAttempt := req
		if autoCompat {
			reqForAttempt = applyResponsesCompat(req, compatStage)
		}
		payload, err := local.buildRequest(reqForAttempt)
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
			data, readErr := readResponseBodyLimited(resp.Body, endpoint)
			if readErr != nil {
				retryAfter := parseRetryAfter(resp.Header.Get("Retry-After"))
				return nil, openAIReadBodyError(local.Provider(), resp.StatusCode, retryAfter, readErr)
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

			// Automatic downgrade: some gateways reject reasoning/settings; apply all compatible changes before retrying.
			compatChanged := false
			if resp.StatusCode == 400 || resp.StatusCode == 422 {
				if strings.TrimSpace(local.ReasoningEffort) != "" && looksLikeReasoningUnsupported(msg) {
					local.ReasoningEffort = ""
					compatChanged = true
				}
				if local.ExtraBody != nil && looksLikeExtraBodyUnsupported(msg) {
					local.ExtraBody = nil
					compatChanged = true
				}
				if hasThinkingExtra(local.Extra, local.ExtraBody) && looksLikeThinkingUnsupported(msg) && dropThinkingExtra(local.Extra, local.ExtraBody) {
					compatChanged = true
				}
				if strings.Contains(msg, "MissingParameter") && strings.Contains(msg, "input.content") {
					local.ForceStringInput = true
					compatChanged = true
				}
				if autoCompat && compatStage == responsesCompatFull && looksLikeResponsesInputUnsupported(msg) {
					compatStage = responsesCompatLegacy
					compatChanged = true
				}
			}
			if compatChanged && attempt < retry.maxRetries-1 {
				continue
			}
			if resp.StatusCode == 429 {
				lastErr = &llm.RateLimitError{Provider: local.Provider(), Message: msg, RetryAfter: retryAfter}
			} else {
				lastErr = &llm.ProviderError{Provider: local.Provider(), StatusCode: resp.StatusCode, Message: msg, RetryAfter: retryAfter}
			}
			if local.isRetryableStatus(resp.StatusCode) && attempt < retry.maxRetries-1 {
				local.sleepBackoff(ctx, attempt, retry.baseDelay, retry.maxDelay, retryAfter)
				continue
			}
			return nil, lastErr
		}

		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		lastErr = err
		if attempt < retry.maxRetries-1 && isRetryableNetErr(err) {
			local.sleepBackoff(ctx, attempt, retry.baseDelay, retry.maxDelay, 0)
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
		return defaultRetryableStatus(code)
	}
	_, ok := c.RetryableStatusCodes[code]
	return ok
}

func (c *ResponsesClient) sleepBackoff(ctx context.Context, attempt int, baseDelay, maxDelay time.Duration, retryAfter time.Duration) {
	sleepRetryBackoff(ctx, attempt, baseDelay, maxDelay, retryAfter)
}

// ---- request mapping (best-effort) ----

type responsesMessage struct {
	Role       string `json:"role"`
	Content    any    `json:"content"`
	ToolCallID string `json:"tool_call_id,omitempty"`
}

type responsesContentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
}

type responsesInputItem struct {
	Type      string `json:"type"`
	Role      string `json:"role,omitempty"`
	Content   any    `json:"content,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	CallID    string `json:"call_id,omitempty"`
	Output    any    `json:"output,omitempty"`
}

type responsesTool struct {
	Type        string         `json:"type"` // "function"
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
}

type responsesRequest struct {
	Model        string `json:"model"`
	Instructions string `json:"instructions,omitempty"`
	Input        any    `json:"input"`

	Tools []responsesTool `json:"tools,omitempty"`
	// tool_choice can be a string ("none"|"required") or an object; omit for default "auto".
	ToolChoice any `json:"tool_choice,omitempty"`

	ParallelToolCalls *bool `json:"parallel_tool_calls,omitempty"`

	Temperature     *float64 `json:"temperature,omitempty"`
	TopP            *float64 `json:"top_p,omitempty"`
	Seed            *int     `json:"seed,omitempty"`
	MaxOutputTokens *int     `json:"max_output_tokens,omitempty"`
	ServiceTier     string   `json:"service_tier,omitempty"`

	Reasoning map[string]any `json:"reasoning,omitempty"`
	Text      any            `json:"text,omitempty"`
	Include   []string       `json:"include,omitempty"`

	PromptCacheKey string `json:"prompt_cache_key,omitempty"`
	ConversationID string `json:"conversation_id,omitempty"`
	Store          *bool  `json:"store,omitempty"`

	Stream bool `json:"stream,omitempty"`

	ExtraBody map[string]any `json:"extra_body,omitempty"`
	Extra     map[string]any `json:"-"`
}

type responsesCompatStage int

const (
	responsesCompatFull responsesCompatStage = iota
	responsesCompatLegacy
)

func (r responsesRequest) MarshalJSON() ([]byte, error) {
	type alias responsesRequest
	base := map[string]any{}
	b, err := json.Marshal(alias(r))
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(b, &base); err != nil {
		return nil, err
	}
	for k, v := range r.Extra {
		if v != nil {
			base[k] = v
		}
	}
	return json.Marshal(base)
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

func shouldAutoCompat(req llm.InvokeRequest) bool {
	if req.Responses == nil {
		return true
	}
	opts := req.Responses
	return opts.UseResponseItems == nil && opts.UseInstructions == nil
}

func applyResponsesCompat(req llm.InvokeRequest, stage responsesCompatStage) llm.InvokeRequest {
	if !shouldAutoCompat(req) {
		return req
	}
	out := req
	opts := out.Responses
	if opts == nil {
		opts = &llm.ResponsesOptions{}
	} else {
		clone := *opts
		opts = &clone
	}
	switch stage {
	case responsesCompatFull:
		if opts.UseResponseItems == nil {
			v := true
			opts.UseResponseItems = &v
		}
		if opts.UseInstructions == nil {
			v := true
			opts.UseInstructions = &v
		}
	case responsesCompatLegacy:
		if opts.UseResponseItems == nil {
			v := false
			opts.UseResponseItems = &v
		}
		if opts.UseInstructions == nil {
			v := false
			opts.UseInstructions = &v
		}
	}
	out.Responses = opts
	return out
}

func looksLikeResponsesInputUnsupported(msg string) bool {
	s := strings.ToLower(msg)
	if strings.Contains(s, "instructions") || strings.Contains(s, "responseitem") || strings.Contains(s, "response_item") {
		return true
	}
	if strings.Contains(s, "unknown field") || strings.Contains(s, "unrecognized") || strings.Contains(s, "unexpected") {
		if strings.Contains(s, "instructions") || strings.Contains(s, "input") || strings.Contains(s, "text") ||
			strings.Contains(s, "include") || strings.Contains(s, "parallel_tool_calls") ||
			strings.Contains(s, "prompt_cache_key") || strings.Contains(s, "conversation_id") {
			return true
		}
	}
	if strings.Contains(s, "input") {
		if strings.Contains(s, "content") || strings.Contains(s, "item") || strings.Contains(s, "message") ||
			strings.Contains(s, "role") || strings.Contains(s, "type") || strings.Contains(s, "function_call_output") {
			return true
		}
	}
	return false
}

// InvokeStream implements true SSE streaming for OpenAI responses.
// It emits text deltas and basic tool-call deltas (best-effort).
func (c *ResponsesClient) InvokeStream(ctx context.Context, req llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	out := make(chan llm.StreamEvent, 128)
	local := *c
	local.Extra = cloneMap(c.Extra)
	local.ExtraBody = cloneMap(c.ExtraBody)
	go func() {
		defer close(out)

		client := streamHTTPClient(local.httpClient())
		baseURL := strings.TrimRight(local.baseURL(), "/")
		endpoint := openAIEndpoint(baseURL, "responses")

		retry := resolveRetryPolicy(local.MaxRetries, local.RetryBaseDelay, local.RetryMaxDelay)

		autoCompat := shouldAutoCompat(req)
		compatStage := responsesCompatFull

		for attempt := 0; attempt < retry.maxRetries; attempt++ {
			if err := ctx.Err(); err != nil {
				out <- llm.StreamErrorEvent{Err: err}
				return
			}
			reqForAttempt := req
			if autoCompat {
				reqForAttempt = applyResponsesCompat(req, compatStage)
			}
			payload, err := local.buildRequest(reqForAttempt)
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
				if ctxErr := ctx.Err(); ctxErr != nil {
					out <- llm.StreamErrorEvent{Err: ctxErr}
					return
				}
				if attempt < retry.maxRetries-1 && isRetryableNetErr(err) {
					local.sleepBackoff(ctx, attempt, retry.baseDelay, retry.maxDelay, 0)
					continue
				}
				out <- llm.StreamErrorEvent{Err: err}
				return
			}

			if resp.StatusCode < 200 || resp.StatusCode >= 300 {
				data, readErr := readResponseBodyLimited(resp.Body, endpoint)
				if readErr != nil {
					retryAfter := parseRetryAfter(resp.Header.Get("Retry-After"))
					out <- llm.StreamErrorEvent{Err: openAIReadBodyError(local.Provider(), resp.StatusCode, retryAfter, readErr)}
					return
				}
				retryAfter := parseRetryAfter(resp.Header.Get("Retry-After"))
				msg := strings.TrimSpace(string(data))
				if msg == "" {
					msg = resp.Status
				}
				msg = fmt.Sprintf("%s (POST %s)", msg, endpoint)

				// Automatic downgrade: apply all compatible request changes before retrying.
				compatChanged := false
				if resp.StatusCode == 400 || resp.StatusCode == 422 {
					if strings.TrimSpace(local.ReasoningEffort) != "" && looksLikeReasoningUnsupported(msg) {
						local.ReasoningEffort = ""
						compatChanged = true
					}
					if local.ExtraBody != nil && looksLikeExtraBodyUnsupported(msg) {
						local.ExtraBody = nil
						compatChanged = true
					}
					if hasThinkingExtra(local.Extra, local.ExtraBody) && looksLikeThinkingUnsupported(msg) && dropThinkingExtra(local.Extra, local.ExtraBody) {
						compatChanged = true
					}
					if strings.Contains(msg, "MissingParameter") && strings.Contains(msg, "input.content") {
						local.ForceStringInput = true
						compatChanged = true
					}
					if autoCompat && compatStage == responsesCompatFull && looksLikeResponsesInputUnsupported(msg) {
						compatStage = responsesCompatLegacy
						compatChanged = true
					}
				}
				if compatChanged && attempt < retry.maxRetries-1 {
					continue
				}

				var lastErr error
				if resp.StatusCode == 429 {
					lastErr = &llm.RateLimitError{Provider: local.Provider(), Message: msg, RetryAfter: retryAfter}
				} else {
					lastErr = &llm.ProviderError{Provider: local.Provider(), StatusCode: resp.StatusCode, Message: msg, RetryAfter: retryAfter}
				}
				if local.isRetryableStatus(resp.StatusCode) && attempt < retry.maxRetries-1 {
					local.sleepBackoff(ctx, attempt, retry.baseDelay, retry.maxDelay, retryAfter)
					continue
				}
				out <- llm.StreamErrorEvent{Err: lastErr}
				return
			}

			idToIndex := map[string]int{}
			autoToIndex := map[string]int{}
			usedIndices := map[int]struct{}{}
			nextIndex := 0

			indexFromAny := func(v any) (int, bool) {
				switch x := v.(type) {
				case float64:
					return int(x), true
				case int:
					return x, true
				case int64:
					return int(x), true
				case json.Number:
					i, err := x.Int64()
					if err != nil {
						return 0, false
					}
					return int(i), true
				default:
					return 0, false
				}
			}

			firstIndexHint := func(values ...any) (int, bool) {
				for _, v := range values {
					if idx, ok := indexFromAny(v); ok {
						return idx, true
					}
				}
				return 0, false
			}

			allocateIndex := func(preferred int, hasPreferred bool) int {
				if hasPreferred && preferred >= 0 {
					if _, taken := usedIndices[preferred]; !taken {
						usedIndices[preferred] = struct{}{}
						if preferred >= nextIndex {
							nextIndex = preferred + 1
						}
						return preferred
					}
				}
				for {
					if _, taken := usedIndices[nextIndex]; !taken {
						idx := nextIndex
						usedIndices[idx] = struct{}{}
						nextIndex++
						return idx
					}
					nextIndex++
				}
			}

			getIndex := func(id, autoKey string, preferred int, hasPreferred bool) int {
				id = strings.TrimSpace(id)
				autoKey = strings.TrimSpace(autoKey)
				if id != "" {
					if idx, ok := idToIndex[id]; ok {
						if autoKey != "" {
							autoToIndex[autoKey] = idx
						}
						return idx
					}
					if autoKey != "" {
						if idx, ok := autoToIndex[autoKey]; ok {
							idToIndex[id] = idx
							return idx
						}
					}
					idx := allocateIndex(preferred, hasPreferred)
					idToIndex[id] = idx
					if autoKey != "" {
						autoToIndex[autoKey] = idx
					}
					return idx
				}
				if autoKey != "" {
					if idx, ok := autoToIndex[autoKey]; ok {
						return idx
					}
					idx := allocateIndex(preferred, hasPreferred)
					autoToIndex[autoKey] = idx
					return idx
				}
				return allocateIndex(-1, false)
			}

			type streamedToolState struct {
				hasName bool
				hasID   bool
				hasArgs bool
			}
			toolState := map[int]*streamedToolState{}
			ensureToolState := func(idx int) *streamedToolState {
				st := toolState[idx]
				if st == nil {
					st = &streamedToolState{}
					toolState[idx] = st
				}
				return st
			}
			emitToolIdentity := func(idx int, id, name string) {
				if strings.TrimSpace(id) == "" && strings.TrimSpace(name) == "" {
					return
				}
				out <- llm.StreamToolCallDeltaEvent{Index: idx, ID: id, NameDelta: name}
				st := ensureToolState(idx)
				if strings.TrimSpace(id) != "" {
					st.hasID = true
				}
				if strings.TrimSpace(name) != "" {
					st.hasName = true
				}
			}
			emitToolArgs := func(idx int, id, args string) {
				if args == "" {
					return
				}
				out <- llm.StreamToolCallDeltaEvent{Index: idx, ID: id, ArgumentsDelta: args}
				st := ensureToolState(idx)
				if strings.TrimSpace(id) != "" {
					st.hasID = true
				}
				st.hasArgs = true
			}

			stopReason := ""
			thinkingEmitted := false
			textEmitted := false
			streamResponseID := ""
			emitResponseID := func(id string) {
				id = strings.TrimSpace(id)
				if id == "" || streamResponseID != "" {
					return
				}
				streamResponseID = id
				out <- llm.StreamResponseEvent{ResponseID: id}
			}
			err = consumeSSEWithBodyClose(resp.Body, func(data string) error {
				data = strings.TrimSpace(data)
				if data == "" {
					return nil
				}
				if data == "[DONE]" {
					return errSSEDone
				}
				var root map[string]any
				if err := json.Unmarshal([]byte(data), &root); err != nil {
					return fmt.Errorf("openai responses stream: decode error: %w", err)
				}
				emitResponseID(responsesStreamResponseID(root))
				if rc, ok := root["reasoning_content"].(string); ok && rc != "" {
					out <- llm.StreamThinkingDeltaEvent{Delta: rc}
					thinkingEmitted = true
				}
				typ, _ := root["type"].(string)
				switch typ {
				case "response.output_text.delta":
					// Preserve whitespace deltas to keep streaming output faithful.
					if d, ok := root["delta"].(string); ok && d != "" {
						out <- llm.StreamTextDeltaEvent{Delta: d}
						textEmitted = true
					}
				case "response.reasoning.delta", "response.reasoning_text.delta", "response.reasoning_content.delta", "response.thinking.delta":
					if d, ok := root["delta"].(string); ok && d != "" {
						out <- llm.StreamThinkingDeltaEvent{Delta: d}
						thinkingEmitted = true
					}
				case "response.output_item.added":
					item, _ := root["item"].(map[string]any)
					itType, _ := item["type"].(string)
					if itType == "reasoning" || itType == "thinking" || itType == "reasoning_text" {
						if txt, ok := item["text"].(string); ok && strings.TrimSpace(txt) != "" {
							out <- llm.StreamThinkingDeltaEvent{Delta: txt}
							thinkingEmitted = true
						} else if txt, ok := item["content"].(string); ok && strings.TrimSpace(txt) != "" {
							out <- llm.StreamThinkingDeltaEvent{Delta: txt}
							thinkingEmitted = true
						} else if summaryArr, ok := item["summary"].([]any); ok {
							// GLM format: reasoning in summary[].text
							for _, sumAny := range summaryArr {
								if sumMap, ok := sumAny.(map[string]any); ok {
									if txt, ok := sumMap["text"].(string); ok && strings.TrimSpace(txt) != "" {
										out <- llm.StreamThinkingDeltaEvent{Delta: txt}
										thinkingEmitted = true
									}
								}
							}
						}
						return nil
					}
					if itType == "message" {
						if contentArr, ok := item["content"].([]any); ok {
							for _, cAny := range contentArr {
								cm, ok := cAny.(map[string]any)
								if !ok {
									continue
								}
								ct, _ := cm["type"].(string)
								if ct == "output_text" || ct == "text" {
									if txt, ok := cm["text"].(string); ok && strings.TrimSpace(txt) != "" && !textEmitted {
										out <- llm.StreamTextDeltaEvent{Delta: txt}
										textEmitted = true
									}
								} else if ct == "reasoning" || ct == "reasoning_text" || ct == "thinking" {
									if txt, ok := cm["text"].(string); ok && strings.TrimSpace(txt) != "" {
										out <- llm.StreamThinkingDeltaEvent{Delta: txt}
										thinkingEmitted = true
									} else if txt, ok := cm["content"].(string); ok && strings.TrimSpace(txt) != "" {
										out <- llm.StreamThinkingDeltaEvent{Delta: txt}
										thinkingEmitted = true
									}
								}
							}
						}
					}
					if itType == "function_call" || itType == "tool_call" {
						id, _ := item["id"].(string)
						if id == "" {
							id, _ = item["call_id"].(string)
						}
						name, _ := item["name"].(string)
						idxHint, hasIdxHint := firstIndexHint(root["output_index"], root["item_index"], item["output_index"], item["index"])
						autoKey := ""
						if hasIdxHint {
							autoKey = fmt.Sprintf("output:%d", idxHint)
						}
						idx := getIndex(id, autoKey, idxHint, hasIdxHint)
						emitToolIdentity(idx, id, name)
						if args, ok := item["arguments"].(string); ok && strings.TrimSpace(args) != "" {
							emitToolArgs(idx, id, args)
						}
					}
				case "response.output_item.done":
					item, _ := root["item"].(map[string]any)
					itType, _ := item["type"].(string)
					if itType == "reasoning" || itType == "thinking" || itType == "reasoning_text" {
						if txt, ok := item["text"].(string); ok && strings.TrimSpace(txt) != "" {
							out <- llm.StreamThinkingDeltaEvent{Delta: txt}
							thinkingEmitted = true
						} else if txt, ok := item["content"].(string); ok && strings.TrimSpace(txt) != "" {
							out <- llm.StreamThinkingDeltaEvent{Delta: txt}
							thinkingEmitted = true
						} else if summaryArr, ok := item["summary"].([]any); ok {
							for _, sumAny := range summaryArr {
								if sumMap, ok := sumAny.(map[string]any); ok {
									if txt, ok := sumMap["text"].(string); ok && strings.TrimSpace(txt) != "" {
										out <- llm.StreamThinkingDeltaEvent{Delta: txt}
										thinkingEmitted = true
									}
								}
							}
						}
						return nil
					}
					if itType == "message" {
						if contentArr, ok := item["content"].([]any); ok {
							for _, cAny := range contentArr {
								cm, ok := cAny.(map[string]any)
								if !ok {
									continue
								}
								ct, _ := cm["type"].(string)
								if ct == "output_text" || ct == "text" {
									if txt, ok := cm["text"].(string); ok && strings.TrimSpace(txt) != "" && !textEmitted {
										out <- llm.StreamTextDeltaEvent{Delta: txt}
										textEmitted = true
									}
								} else if ct == "reasoning" || ct == "reasoning_text" || ct == "thinking" {
									if txt, ok := cm["text"].(string); ok && strings.TrimSpace(txt) != "" {
										out <- llm.StreamThinkingDeltaEvent{Delta: txt}
										thinkingEmitted = true
									} else if txt, ok := cm["content"].(string); ok && strings.TrimSpace(txt) != "" {
										out <- llm.StreamThinkingDeltaEvent{Delta: txt}
										thinkingEmitted = true
									}
								}
							}
						}
					}
					if itType != "function_call" && itType != "tool_call" {
						return nil
					}
					id, _ := item["id"].(string)
					if id == "" {
						id, _ = item["call_id"].(string)
					}
					name, _ := item["name"].(string)
					idxHint, hasIdxHint := firstIndexHint(root["output_index"], root["item_index"], item["output_index"], item["index"])
					autoKey := ""
					if hasIdxHint {
						autoKey = fmt.Sprintf("output:%d", idxHint)
					}
					idx := getIndex(id, autoKey, idxHint, hasIdxHint)
					st := ensureToolState(idx)
					needIdentity := (strings.TrimSpace(name) != "" && !st.hasName) || (strings.TrimSpace(id) != "" && !st.hasID)
					if needIdentity {
						emitToolIdentity(idx, id, name)
					}
					if args := responsesToolArguments(item["arguments"]); strings.TrimSpace(args) != "" && !st.hasArgs {
						emitToolArgs(idx, id, args)
					}
				case "response.function_call_arguments.delta":
					itemID, _ := root["item_id"].(string)
					if itemID == "" {
						itemID, _ = root["id"].(string)
					}
					if d, ok := root["delta"].(string); ok && d != "" {
						idxHint, hasIdxHint := firstIndexHint(root["output_index"], root["item_index"], root["index"])
						autoKey := ""
						if hasIdxHint {
							autoKey = fmt.Sprintf("output:%d", idxHint)
						}
						idx := getIndex(itemID, autoKey, idxHint, hasIdxHint)
						emitToolArgs(idx, itemID, d)
					}
				case "response.function_call_arguments.done":
					itemID, _ := root["item_id"].(string)
					if itemID == "" {
						itemID, _ = root["id"].(string)
					}
					args := responsesToolArguments(root["arguments"])
					if strings.TrimSpace(args) == "" {
						args = responsesToolArguments(root["delta"])
					}
					if strings.TrimSpace(args) != "" {
						idxHint, hasIdxHint := firstIndexHint(root["output_index"], root["item_index"], root["index"])
						autoKey := ""
						if hasIdxHint {
							autoKey = fmt.Sprintf("output:%d", idxHint)
						}
						idx := getIndex(itemID, autoKey, idxHint, hasIdxHint)
						if st := ensureToolState(idx); !st.hasArgs {
							emitToolArgs(idx, itemID, args)
						}
					}
				case "response.completed":
					respObj, _ := root["response"].(map[string]any)
					if respObj != nil {
						if id, ok := respObj["id"].(string); ok && strings.TrimSpace(id) != "" {
							out <- llm.StreamResponseEvent{ResponseID: id}
						}
					} else if id, ok := root["response_id"].(string); ok && strings.TrimSpace(id) != "" {
						out <- llm.StreamResponseEvent{ResponseID: id}
					}
					if u := usageFromResponses(respObj); u != nil {
						out <- llm.StreamUsageEvent{Usage: *u}
					}
					if !thinkingEmitted {
						if thinking := extractThinkingFromResponses(respObj); thinking != "" {
							out <- llm.StreamThinkingDeltaEvent{Delta: thinking}
							thinkingEmitted = true
						}
					}
					if !textEmitted {
						if text := extractTextFromResponses(respObj); strings.TrimSpace(text) != "" {
							out <- llm.StreamTextDeltaEvent{Delta: text}
							textEmitted = true
						}
					}
					// Extract stop reason from completed response
					if respObj != nil {
						if sr, ok := respObj["status"].(string); ok {
							if sr == "incomplete" {
								stopReason = "max_tokens"
							} else if sr == "completed" {
								stopReason = "end_turn"
							}
						}
					}
				case "response.error", "error":
					if streamErr := parseResponsesStreamEventError(local.Provider(), root); streamErr != nil {
						return streamErr
					}
					return errors.New("openai responses stream error")
				}
				return nil
			})
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
		out <- llm.StreamErrorEvent{Err: errors.New("openai responses stream: retry loop ended without result")}
	}()
	return out, nil
}

func parseResponsesStreamEventError(provider string, root map[string]any) error {
	errObj, _ := root["error"].(map[string]any)
	msg := firstNonEmptyString(
		stringFromAny(errObj["message"]),
		stringFromAny(errObj["detail"]),
		stringFromAny(root["message"]),
	)
	if msg == "" {
		msg = "openai responses stream error"
	}

	statusCode := firstPositiveInt(
		errObj["status_code"],
		errObj["status"],
		root["status_code"],
		root["status"],
	)
	retryAfter := firstPositiveDuration(
		retryAfterFromAny(errObj["retry_after_ms"], true),
		retryAfterFromAny(errObj["retry_after"], false),
		retryAfterFromAny(root["retry_after_ms"], true),
		retryAfterFromAny(root["retry_after"], false),
	)

	code := strings.ToLower(stringFromAny(errObj["code"]))
	errType := strings.ToLower(stringFromAny(errObj["type"]))
	rateLimited := statusCode == http.StatusTooManyRequests || looksLikeRateLimitError(code, errType, msg)
	if statusCode == 0 && rateLimited {
		statusCode = http.StatusTooManyRequests
	}
	provider = openAIProviderLabel(provider)
	if rateLimited {
		return &llm.RateLimitError{Provider: provider, Message: msg, RetryAfter: retryAfter}
	}
	return &llm.ProviderError{Provider: provider, StatusCode: statusCode, Message: msg, RetryAfter: retryAfter}
}

func stringFromAny(v any) string {
	s, _ := v.(string)
	return strings.TrimSpace(s)
}

func firstNonEmptyString(values ...string) string {
	for _, value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			return trimmed
		}
	}
	return ""
}

func firstPositiveInt(values ...any) int {
	for _, value := range values {
		if n := intFromAny(value); n > 0 {
			return n
		}
	}
	return 0
}

func firstPositiveDuration(values ...time.Duration) time.Duration {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func retryAfterFromAny(v any, milliseconds bool) time.Duration {
	if v == nil {
		return 0
	}
	unit := time.Second
	if milliseconds {
		unit = time.Millisecond
	}
	if f, ok := floatFromAny(v); ok {
		if f <= 0 {
			return 0
		}
		return time.Duration(f * float64(unit))
	}
	s := stringFromAny(v)
	if s == "" {
		return 0
	}
	if f, err := strconv.ParseFloat(s, 64); err == nil && f > 0 {
		return time.Duration(f * float64(unit))
	}
	if d, err := time.ParseDuration(s); err == nil && d > 0 {
		return d
	}
	return parseRetryAfter(s)
}

func floatFromAny(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case float32:
		return float64(x), true
	case int:
		return float64(x), true
	case int64:
		return float64(x), true
	case json.Number:
		f, err := x.Float64()
		if err != nil {
			return 0, false
		}
		return f, true
	default:
		return 0, false
	}
}

func looksLikeRateLimitError(code, errType, msg string) bool {
	lowerMsg := strings.ToLower(strings.TrimSpace(msg))
	return strings.Contains(code, "rate_limit") ||
		strings.Contains(code, "too_many_requests") ||
		strings.Contains(errType, "rate_limit") ||
		strings.Contains(errType, "too_many_requests") ||
		strings.Contains(lowerMsg, "rate limit") ||
		strings.Contains(lowerMsg, "too many requests") ||
		strings.Contains(lowerMsg, "slow down")
}

func usageTokenBreakdown(u map[string]any) (int, int) {
	if u == nil {
		return 0, 0
	}
	pt := intFromAny(u["input_tokens"])
	if pt == 0 {
		pt = intFromAny(u["prompt_tokens"])
	}
	ct := intFromAny(u["output_tokens"])
	if ct == 0 {
		ct = intFromAny(u["completion_tokens"])
	}
	return pt, ct
}

func responsesToolArguments(value any) string {
	switch v := value.(type) {
	case string:
		return v
	case nil:
		return ""
	default:
		encoded, err := json.Marshal(v)
		if err != nil {
			return ""
		}
		if string(encoded) == "null" {
			return ""
		}
		return string(encoded)
	}
}

func responsesStreamResponseID(root map[string]any) string {
	if root == nil {
		return ""
	}
	if id, ok := root["response_id"].(string); ok && strings.TrimSpace(id) != "" {
		return strings.TrimSpace(id)
	}
	if response, ok := root["response"].(map[string]any); ok {
		if id, ok := response["id"].(string); ok && strings.TrimSpace(id) != "" {
			return strings.TrimSpace(id)
		}
	}
	return ""
}

func usageFromResponses(resp map[string]any) *llm.Usage {
	if resp == nil {
		return nil
	}
	u, _ := resp["usage"].(map[string]any)
	if u == nil {
		return nil
	}
	pt, ct := usageTokenBreakdown(u)
	tt := intFromAny(u["total_tokens"])
	if tt == 0 {
		tt = pt + ct
	}
	return &llm.Usage{PromptTokens: pt, CompletionTokens: ct, TotalTokens: tt}
}

func extractThinkingFromResponses(resp map[string]any) string {
	if resp == nil {
		return ""
	}
	thinkingParts := []string{}

	if outArr, ok := resp["output"].([]any); ok {
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
					if ct == "reasoning" || ct == "reasoning_text" || ct == "thinking" {
						if txt, ok := cm["text"].(string); ok && strings.TrimSpace(txt) != "" {
							thinkingParts = append(thinkingParts, txt)
						} else if txt, ok := cm["content"].(string); ok && strings.TrimSpace(txt) != "" {
							thinkingParts = append(thinkingParts, txt)
						}
					}
				}
			case "reasoning", "reasoning_text", "thinking":
				if txt, ok := item["text"].(string); ok && strings.TrimSpace(txt) != "" {
					thinkingParts = append(thinkingParts, txt)
				} else if txt, ok := item["content"].(string); ok && strings.TrimSpace(txt) != "" {
					thinkingParts = append(thinkingParts, txt)
				} else if summaryArr, ok := item["summary"].([]any); ok {
					// GLM format: reasoning in summary[].text
					for _, sumAny := range summaryArr {
						if sumMap, ok := sumAny.(map[string]any); ok {
							if txt, ok := sumMap["text"].(string); ok && strings.TrimSpace(txt) != "" {
								thinkingParts = append(thinkingParts, txt)
							}
						}
					}
				}
			}
		}
	}
	if t, ok := resp["reasoning_content"].(string); ok && strings.TrimSpace(t) != "" {
		thinkingParts = append(thinkingParts, t)
	} else if t, ok := resp["thinking"].(string); ok && strings.TrimSpace(t) != "" {
		thinkingParts = append(thinkingParts, t)
	}

	return strings.TrimSpace(strings.Join(thinkingParts, "\n"))
}

func extractTextFromResponses(resp map[string]any) string {
	if resp == nil {
		return ""
	}
	textParts := []string{}
	if outArr, ok := resp["output"].([]any); ok {
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
						if txt, ok := cm["text"].(string); ok && strings.TrimSpace(txt) != "" {
							textParts = append(textParts, txt)
						}
					}
				}
			}
		}
	}
	if len(textParts) == 0 {
		if t, ok := resp["output_text"].(string); ok && strings.TrimSpace(t) != "" {
			textParts = append(textParts, t)
		}
	}
	return strings.TrimSpace(strings.Join(textParts, "\n"))
}

func responsesTextControlsFromOptions(opts *llm.ResponsesOptions) any {
	if opts == nil {
		return nil
	}
	if opts.Text != nil {
		return opts.Text
	}
	verbosity := strings.TrimSpace(opts.Verbosity)
	schema := opts.OutputSchema
	if verbosity == "" && schema == nil {
		return nil
	}
	text := map[string]any{}
	if verbosity != "" {
		text["verbosity"] = verbosity
	}
	if schema != nil {
		text["format"] = map[string]any{
			"type":   "json_schema",
			"strict": true,
			"schema": schema,
			"name":   "codex_output_schema",
		}
	}
	if len(text) == 0 {
		return nil
	}
	return text
}

func responsesPartsFromContent(content llm.Content, textPartType string) ([]responsesContentPart, bool) {
	if strings.TrimSpace(textPartType) == "" {
		textPartType = "input_text"
	}
	parts := []responsesContentPart{}
	hasImage := false
	if strings.TrimSpace(content.Text) != "" {
		parts = append(parts, responsesContentPart{Type: textPartType, Text: content.Text})
	}
	if len(content.Blocks) > 0 {
		for _, blk := range content.Blocks {
			switch blk.Type {
			case "text":
				if strings.TrimSpace(blk.Text) != "" {
					parts = append(parts, responsesContentPart{Type: textPartType, Text: blk.Text})
				}
			case "image_url":
				if blk.ImageURL != nil && strings.TrimSpace(blk.ImageURL.URL) != "" {
					parts = append(parts, responsesContentPart{Type: "input_image", ImageURL: blk.ImageURL.URL})
					hasImage = true
				}
			default:
				if fallback := responsesContentFallbackText(blk); fallback != "" {
					parts = append(parts, responsesContentPart{Type: textPartType, Text: fallback})
				}
			}
		}
	}
	if len(parts) == 0 {
		if txt := strings.TrimSpace(responsesContentPlainText(content)); txt != "" {
			parts = append(parts, responsesContentPart{Type: textPartType, Text: txt})
		}
	}
	return parts, hasImage
}

func responsesMessageContent(role string, content llm.Content, stringContent bool) (any, bool) {
	if stringContent {
		txt := strings.TrimSpace(responsesContentPlainText(content))
		if txt == "" {
			return nil, false
		}
		return txt, true
	}
	textPartType := "input_text"
	if strings.TrimSpace(role) == "assistant" {
		// OpenAI Responses requires assistant history items to use output_text/refusal content parts.
		textPartType = "output_text"
	}
	parts, _ := responsesPartsFromContent(content, textPartType)
	if len(parts) == 0 {
		return nil, false
	}
	return parts, true
}

func joinInputTextParts(parts []responsesContentPart) string {
	var b strings.Builder
	for _, part := range parts {
		if part.Type != "input_text" || strings.TrimSpace(part.Text) == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(part.Text)
	}
	return strings.TrimSpace(b.String())
}

func responsesContentPlainText(content llm.Content) string {
	parts := make([]string, 0, len(content.Blocks)+1)
	appendText := func(text string) {
		if strings.TrimSpace(text) == "" {
			return
		}
		parts = append(parts, text)
	}
	appendText(content.Text)
	for _, blk := range content.Blocks {
		switch blk.Type {
		case "text":
			appendText(blk.Text)
		default:
			appendText(responsesContentFallbackText(blk))
		}
	}
	return strings.TrimSpace(strings.Join(parts, "\n"))
}

func responsesContentFallbackText(block llm.ContentBlock) string {
	switch strings.TrimSpace(block.Type) {
	case "image_url":
		if block.ImageURL != nil {
			if mediaType := strings.TrimSpace(block.ImageURL.MediaType); mediaType != "" {
				return "[image: " + mediaType + "]"
			}
			if strings.TrimSpace(block.ImageURL.URL) != "" {
				return "[image]"
			}
		}
		return ""
	case "thinking", "redacted_thinking":
		if strings.TrimSpace(block.Thinking) != "" {
			return block.Thinking
		}
		if strings.TrimSpace(block.Text) != "" {
			return block.Text
		}
		return "[thinking content omitted]"
	case "document":
		if block.Source != nil && strings.TrimSpace(block.Source.MediaType) != "" {
			return "[document: " + strings.TrimSpace(block.Source.MediaType) + "]"
		}
		if strings.TrimSpace(block.Text) != "" {
			return block.Text
		}
		return "[document]"
	default:
		if strings.TrimSpace(block.Text) != "" {
			return block.Text
		}
		return ""
	}
}

func responsesToolOutput(content llm.Content, isError bool) any {
	parts, hasImage := responsesPartsFromContent(content, "input_text")
	text := strings.TrimSpace(responsesContentPlainText(content))
	if text == "" && len(parts) > 0 {
		text = joinInputTextParts(parts)
	}
	if hasImage && len(parts) > 0 {
		return parts
	}
	if isError {
		if text == "" {
			text = "(error)"
		} else {
			text = "(error) " + text
		}
	}
	return text
}

func (c *ResponsesClient) buildRequest(req llm.InvokeRequest) (*responsesRequest, error) {
	if c.ModelName == "" {
		return nil, fmt.Errorf("openai responses: model is required")
	}
	if err := validateOpenAIToolHistory(req.Messages, "openai responses"); err != nil {
		return nil, err
	}

	stringContent := c.forceStringInput()
	opts := req.Responses
	useItems := false
	useInstructions := false
	instructions := ""
	if opts != nil {
		if opts.UseResponseItems != nil {
			useItems = *opts.UseResponseItems
		}
		if opts.UseInstructions != nil {
			useInstructions = *opts.UseInstructions
		}
		if strings.TrimSpace(opts.Instructions) != "" {
			instructions = strings.TrimSpace(opts.Instructions)
			if opts.UseInstructions == nil {
				useInstructions = true
			}
		}
	}

	var systemTexts []string
	for _, m := range req.Messages {
		role := string(m.Role)
		if role != "system" && role != "user" && role != "assistant" && role != "tool" {
			continue
		}
		if role == "system" {
			if txt := strings.TrimSpace(responsesContentPlainText(m.Content)); txt != "" {
				systemTexts = append(systemTexts, txt)
			}
		}
	}
	if useInstructions && len(systemTexts) > 0 {
		joined := strings.Join(systemTexts, "\n\n")
		if instructions == "" {
			instructions = joined
		} else {
			instructions = instructions + "\n\n" + joined
		}
	}

	var input any
	if useItems {
		items := make([]responsesInputItem, 0, len(req.Messages))
		for _, m := range req.Messages {
			role := string(m.Role)
			if role != "system" && role != "user" && role != "assistant" && role != "tool" {
				continue
			}
			if role == "system" && useInstructions {
				continue
			}
			if role == "tool" {
				callID := strings.TrimSpace(m.ToolCallID)
				if callID == "" {
					// Fallback: represent as user text to avoid dropping content.
					role = "user"
				} else {
					items = append(items, responsesInputItem{
						Type:   "function_call_output",
						CallID: callID,
						Output: responsesToolOutput(m.Content, m.IsError),
					})
					continue
				}
			}
			if contentAny, ok := responsesMessageContent(role, m.Content, stringContent); ok {
				items = append(items, responsesInputItem{Type: "message", Role: role, Content: contentAny})
			}
			if role == "assistant" && len(m.ToolCalls) > 0 {
				for i, tc := range m.ToolCalls {
					name := strings.TrimSpace(tc.Function.Name)
					if name == "" {
						continue
					}
					callID := strings.TrimSpace(tc.ID)
					if callID == "" {
						callID = fmt.Sprintf("call_%d", i)
					}
					args := strings.TrimSpace(tc.Function.Arguments)
					if args == "" {
						args = "{}"
					}
					items = append(items, responsesInputItem{
						Type:      "function_call",
						CallID:    callID,
						Name:      name,
						Arguments: args,
					})
				}
			}
		}
		if len(items) == 0 {
			var content any = "(empty)"
			if !stringContent {
				content = []responsesContentPart{{Type: "input_text", Text: "(empty)"}}
			}
			items = append(items, responsesInputItem{Type: "message", Role: "user", Content: content})
		}
		input = items
	} else {
		msgs := make([]responsesMessage, 0, len(req.Messages))
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
			contentAny, ok := responsesMessageContent(role, m.Content, stringContent)
			if !ok {
				// Some providers reject empty content; skip.
				continue
			}
			msgs = append(msgs, responsesMessage{Role: role, Content: contentAny})
		}
		if len(msgs) == 0 {
			// Always send at least one user message to satisfy strict gateways.
			var content any = "(empty)"
			if !stringContent {
				content = []responsesContentPart{{Type: "input_text", Text: "(empty)"}}
			}
			msgs = append(msgs, responsesMessage{Role: "user", Content: content})
		}
		input = msgs
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
	effort := strings.TrimSpace(c.ReasoningEffort)
	if opts != nil && opts.Reasoning != nil && strings.TrimSpace(opts.Reasoning.Effort) != "" {
		effort = strings.TrimSpace(opts.Reasoning.Effort)
	}
	if effort != "" {
		reasoning = map[string]any{"effort": effort}
	}
	if opts != nil && opts.Reasoning != nil && strings.TrimSpace(opts.Reasoning.Summary) != "" {
		if reasoning == nil {
			reasoning = map[string]any{}
		}
		reasoning["summary"] = strings.TrimSpace(opts.Reasoning.Summary)
	}

	textParam := responsesTextControlsFromOptions(opts)
	include := []string(nil)
	promptCacheKey := ""
	conversationID := ""
	store := (*bool)(nil)
	parallelToolCalls := (*bool)(nil)
	if opts != nil {
		if len(opts.Include) > 0 {
			include = append([]string(nil), opts.Include...)
		}
		promptCacheKey = strings.TrimSpace(opts.PromptCacheKey)
		conversationID = strings.TrimSpace(opts.ConversationID)
		store = opts.Store
		if opts.ParallelToolCalls != nil {
			parallelToolCalls = opts.ParallelToolCalls
		}
	}

	extra := cloneMap(c.Extra)
	extraBody := cloneMap(c.ExtraBody)

	return &responsesRequest{
		Model:             c.ModelName,
		Instructions:      instructions,
		Input:             input,
		Tools:             toolsList,
		ToolChoice:        toolChoice,
		ParallelToolCalls: parallelToolCalls,
		Temperature:       temp,
		TopP:              c.TopP,
		Seed:              c.Seed,
		MaxOutputTokens:   c.MaxOutputTokens,
		ServiceTier:       strings.TrimSpace(c.ServiceTier),
		Reasoning:         reasoning,
		Text:              textParam,
		Include:           include,
		PromptCacheKey:    promptCacheKey,
		ConversationID:    conversationID,
		Store:             store,
		Extra:             extra,
		ExtraBody:         extraBody,
	}, nil
}

// ---- response parsing (best-effort) ----

func parseResponses(data []byte) (*llm.Completion, error) {
	var root map[string]any
	if err := json.Unmarshal(data, &root); err != nil {
		return nil, err
	}

	blocks := []llm.ContentBlock{}
	thinkingParts := []string{}
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
					} else if ct == "reasoning" || ct == "reasoning_text" || ct == "thinking" {
						if txt, ok := cm["text"].(string); ok && strings.TrimSpace(txt) != "" {
							thinkingParts = append(thinkingParts, txt)
						} else if txt, ok := cm["content"].(string); ok && strings.TrimSpace(txt) != "" {
							thinkingParts = append(thinkingParts, txt)
						}
					}
				}
			case "reasoning", "reasoning_text", "thinking":
				if txt, ok := item["text"].(string); ok && strings.TrimSpace(txt) != "" {
					thinkingParts = append(thinkingParts, txt)
				} else if txt, ok := item["content"].(string); ok && strings.TrimSpace(txt) != "" {
					thinkingParts = append(thinkingParts, txt)
				} else if summaryArr, ok := item["summary"].([]any); ok {
					// GLM format: reasoning in summary[].text
					for _, sumAny := range summaryArr {
						if sumMap, ok := sumAny.(map[string]any); ok {
							if txt, ok := sumMap["text"].(string); ok && strings.TrimSpace(txt) != "" {
								thinkingParts = append(thinkingParts, txt)
							}
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
			case "web_search_call":
				if b, err := json.Marshal(item); err == nil {
					blocks = append(blocks, llm.ContentBlock{Type: "web_search_call", Data: string(b)})
				}
			}
		}
	}

	usage := (*llm.Usage)(nil)
	if u, ok := root["usage"].(map[string]any); ok {
		pt, ct := usageTokenBreakdown(u)
		tt := intFromAny(u["total_tokens"])
		if tt == 0 && (pt > 0 || ct > 0) {
			tt = pt + ct
		}
		usage = &llm.Usage{PromptTokens: pt, CompletionTokens: ct, TotalTokens: tt}
	}

	// Fallback: some responses variants include text at top-level "output_text".
	if len(blocks) == 0 {
		if t, ok := root["output_text"].(string); ok && strings.TrimSpace(t) != "" {
			blocks = append(blocks, llm.ContentBlock{Type: "text", Text: t})
		}
	}
	if t, ok := root["reasoning_content"].(string); ok && strings.TrimSpace(t) != "" {
		thinkingParts = append(thinkingParts, t)
	} else if t, ok := root["thinking"].(string); ok && strings.TrimSpace(t) != "" {
		thinkingParts = append(thinkingParts, t)
	}

	// Extract stop reason from responses API (status field).
	stopReason := ""
	if sr, ok := root["status"].(string); ok {
		if sr == "incomplete" {
			stopReason = "max_tokens"
		} else if sr == "completed" {
			stopReason = "end_turn"
		}
	}

	thinking := strings.TrimSpace(strings.Join(thinkingParts, "\n"))
	responseID, _ := root["id"].(string)
	if strings.TrimSpace(responseID) == "" {
		if id, ok := root["response_id"].(string); ok {
			responseID = id
		}
	}
	return &llm.Completion{Content: llm.Content{Blocks: blocks}, Thinking: thinking, ToolCalls: toolCalls, Usage: usage, StopReason: stopReason, ResponseID: responseID, Raw: append([]byte(nil), data...)}, nil
}

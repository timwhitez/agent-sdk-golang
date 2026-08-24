package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

const defaultBaseURL = "https://api.openai.com"

type ChatClient struct {
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

	Temperature         *float64
	TopP                *float64
	Seed                *int
	MaxCompletionTokens *int
	ServiceTier         string

	// Reasoning effort for reasoning-capable models (best-effort).
	ReasoningEffort string

	MaxRetries           int
	RetryBaseDelay       time.Duration
	RetryMaxDelay        time.Duration
	RetryableStatusCodes map[int]struct{}

	// If true, include "parallel_tool_calls" when tools are provided.
	ParallelToolCalls bool

	// UseLegacyMaxTokens sends the output cap as "max_tokens" instead of
	// "max_completion_tokens" for gateways that only accept the legacy field.
	// It is also set automatically when a provider rejects
	// max_completion_tokens.
	UseLegacyMaxTokens bool

	Warningf func(format string, args ...any)
}

func (c *ChatClient) SetWarningf(warnf func(format string, args ...any)) { c.Warningf = warnf }

func (c *ChatClient) warnf(format string, args ...any) {
	if c != nil && c.Warningf != nil {
		c.Warningf(format, args...)
		return
	}
	log.Printf(format, args...)
}

// Compatibility-downgrade messages shared by the buffered and streaming paths so
// both report the same dropped setting.
const (
	downgradeReasoningEffortMessage = "OpenAI chat provider rejected reasoning_effort; retrying without reasoning_effort."
	downgradeExtraBodyMessage       = "OpenAI chat provider rejected extra request body settings; retrying without extra_body."
	downgradeThinkingMessage        = "OpenAI chat provider rejected thinking settings; retrying without thinking extras."
	downgradeMaxTokensMessage       = "OpenAI chat provider rejected max_completion_tokens; retrying with legacy max_tokens."
	downgradeStreamOptionsMessage   = "OpenAI chat provider rejected stream_options; retrying without stream usage reporting."
)

func (c *ChatClient) Provider() string { return openAIProviderLabel(c.ProviderLabel) }

func (c *ChatClient) Model() string { return c.ModelName }

func (c *ChatClient) Invoke(ctx context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	local := *c
	local.Extra = cloneMap(c.Extra)
	local.ExtraBody = cloneMap(c.ExtraBody)

	client := local.httpClient()
	baseURL := strings.TrimRight(local.baseURL(), "/")
	endpoint := openAIEndpoint(baseURL, "chat/completions")
	lastErr := error(nil)

	retry := resolveRetryPolicy(local.MaxRetries, local.RetryBaseDelay, local.RetryMaxDelay)
	diagnostics := []llm.Diagnostic{}

	for attempt := 0; attempt < retry.maxRetries; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		payload, err := local.buildRequest(req)
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
				comp, err := parseChatCompletion(data)
				if err != nil {
					return nil, err
				}
				comp.Diagnostics = append(comp.Diagnostics, diagnostics...)
				return comp, nil
			}

			retryAfter := parseRetryAfter(resp.Header.Get("Retry-After"))
			msg := strings.TrimSpace(string(data))
			if msg == "" {
				msg = resp.Status
			}
			// Include endpoint to make debugging (base-url/path) easier.
			msg = fmt.Sprintf("%s (POST %s)", msg, endpoint)

			// Automatic downgrade: some OpenAI-compatible providers/models reject reasoning or extra request settings.
			compatChanged := false
			if resp.StatusCode == 400 || resp.StatusCode == 422 {
				if strings.TrimSpace(local.ReasoningEffort) != "" && looksLikeReasoningUnsupported(msg) {
					local.ReasoningEffort = ""
					compatChanged = true
					diagnostics = append(diagnostics, llm.Diagnostic{Kind: "provider_compatibility_downgrade", Message: downgradeReasoningEffortMessage})
				}
				if local.ExtraBody != nil && looksLikeExtraBodyUnsupported(msg) {
					local.ExtraBody = nil
					compatChanged = true
					diagnostics = append(diagnostics, llm.Diagnostic{Kind: "provider_compatibility_downgrade", Message: downgradeExtraBodyMessage})
				}
				if hasThinkingExtra(local.Extra, local.ExtraBody) && looksLikeThinkingUnsupported(msg) && dropThinkingExtra(local.Extra, local.ExtraBody) {
					compatChanged = true
					diagnostics = append(diagnostics, llm.Diagnostic{Kind: "provider_compatibility_downgrade", Message: downgradeThinkingMessage})
				}
				if !local.UseLegacyMaxTokens && local.MaxCompletionTokens != nil && looksLikeMaxCompletionTokensUnsupported(msg) {
					local.UseLegacyMaxTokens = true
					compatChanged = true
					diagnostics = append(diagnostics, llm.Diagnostic{Kind: "provider_compatibility_downgrade", Message: downgradeMaxTokensMessage})
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
	return nil, errors.New("openai: retry loop ended without result")
}

// InvokeStream implements true SSE streaming for OpenAI chat/completions.
// It emits text deltas and tool_call deltas.
func (c *ChatClient) InvokeStream(ctx context.Context, req llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	out := make(chan llm.StreamEvent, 128)
	local := *c
	local.Extra = cloneMap(c.Extra)
	local.ExtraBody = cloneMap(c.ExtraBody)
	go func() {
		defer close(out)

		client := streamHTTPClient(local.httpClient())
		baseURL := strings.TrimRight(local.baseURL(), "/")
		endpoint := openAIEndpoint(baseURL, "chat/completions")

		retry := resolveRetryPolicy(local.MaxRetries, local.RetryBaseDelay, local.RetryMaxDelay)
		includeStreamOptions := true

		for attempt := 0; attempt < retry.maxRetries; attempt++ {
			if err := ctx.Err(); err != nil {
				out <- llm.StreamErrorEvent{Err: err}
				return
			}
			payload, err := local.buildRequest(req)
			if err != nil {
				out <- llm.StreamErrorEvent{Err: err}
				return
			}
			payload.Stream = true
			if includeStreamOptions {
				payload.StreamOptions = map[string]any{"include_usage": true}
			}
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

				// Automatic downgrade: disable unsupported request settings in one retry step.
				// Every downgrade is reported through the warning sink so a silently
				// dropped setting stays visible on the streaming path too.
				compatChanged := false
				if resp.StatusCode == 400 || resp.StatusCode == 422 {
					if strings.TrimSpace(local.ReasoningEffort) != "" && looksLikeReasoningUnsupported(msg) {
						local.ReasoningEffort = ""
						compatChanged = true
						local.warnf("[WARN] %s", downgradeReasoningEffortMessage)
					}
					if local.ExtraBody != nil && looksLikeExtraBodyUnsupported(msg) {
						local.ExtraBody = nil
						compatChanged = true
						local.warnf("[WARN] %s", downgradeExtraBodyMessage)
					}
					if hasThinkingExtra(local.Extra, local.ExtraBody) && looksLikeThinkingUnsupported(msg) && dropThinkingExtra(local.Extra, local.ExtraBody) {
						compatChanged = true
						local.warnf("[WARN] %s", downgradeThinkingMessage)
					}
					if !local.UseLegacyMaxTokens && local.MaxCompletionTokens != nil && looksLikeMaxCompletionTokensUnsupported(msg) {
						local.UseLegacyMaxTokens = true
						compatChanged = true
						local.warnf("[WARN] %s", downgradeMaxTokensMessage)
					}
					if includeStreamOptions && looksLikeStreamOptionsUnsupported(msg) {
						includeStreamOptions = false
						compatChanged = true
						local.warnf("[WARN] %s", downgradeStreamOptionsMessage)
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

			stopReason := ""
			responseID := ""

			// Some OpenAI-compatible gateways omit "index" on tool_call deltas.
			// Track slots by id (and by streaming order as a last resort) so
			// parallel calls do not collapse into a single accumulator slot.
			toolIndexByID := map[string]int{}
			usedToolIndices := map[int]struct{}{}
			toolSawArgs := map[int]bool{}
			nextToolIndex := 0
			lastToolIndex := -1
			allocateToolIndex := func() int {
				for {
					if _, taken := usedToolIndices[nextToolIndex]; !taken {
						idx := nextToolIndex
						usedToolIndices[idx] = struct{}{}
						nextToolIndex++
						return idx
					}
					nextToolIndex++
				}
			}
			claimToolIndex := func(idx int) int {
				usedToolIndices[idx] = struct{}{}
				if idx >= nextToolIndex {
					nextToolIndex = idx + 1
				}
				lastToolIndex = idx
				return idx
			}
			resolveToolIndex := func(rawIndex *int, id, name string) int {
				id = strings.TrimSpace(id)
				if rawIndex != nil {
					idx := *rawIndex
					if idx < 0 {
						idx = 0
					}
					if id != "" {
						toolIndexByID[id] = idx
					}
					return claimToolIndex(idx)
				}
				if id != "" {
					if idx, ok := toolIndexByID[id]; ok {
						lastToolIndex = idx
						return idx
					}
					idx := allocateToolIndex()
					toolIndexByID[id] = idx
					lastToolIndex = idx
					return idx
				}
				// Neither index nor id: continue the current slot, unless a new
				// name arrives after that slot already streamed arguments, which
				// means the gateway started another parallel call.
				if lastToolIndex >= 0 && !(name != "" && toolSawArgs[lastToolIndex]) {
					return lastToolIndex
				}
				idx := allocateToolIndex()
				lastToolIndex = idx
				return idx
			}

			err = consumeSSEWithBodyClose(resp.Body, func(data string) error {
				data = strings.TrimSpace(data)
				if data == "" {
					return nil
				}
				if data == "[DONE]" {
					return errSSEDone
				}
				var r chatCompletionStreamResponse
				if err := json.Unmarshal([]byte(data), &r); err != nil {
					return fmt.Errorf("openai chat stream: decode error (provider=openai status=%d model=%q url=%s): %w", resp.StatusCode, local.ModelName, endpoint, err)
				}
				if r.Error != nil && strings.TrimSpace(r.Error.Message) != "" {
					return fmt.Errorf("openai stream error: %s", r.Error.Message)
				}
				if id := strings.TrimSpace(r.ID); id != "" && responseID == "" {
					responseID = id
					out <- llm.StreamResponseEvent{ResponseID: id}
				}
				if u := parseUsage(r.Usage); u != nil {
					out <- llm.StreamUsageEvent{Usage: *u}
				}
				for _, ch := range r.Choices {
					if ch.FinishReason != "" {
						stopReason = normalizeOpenAIStopReason(ch.FinishReason)
					}
					// Preserve whitespace deltas to keep streaming output faithful.
					if ch.Delta.Content != "" {
						out <- llm.StreamTextDeltaEvent{Delta: ch.Delta.Content}
					}
					if ch.Delta.Refusal != "" {
						out <- llm.StreamTextDeltaEvent{Delta: ch.Delta.Refusal}
					}
					if ch.Delta.ReasoningContent != "" {
						out <- llm.StreamThinkingDeltaEvent{Delta: ch.Delta.ReasoningContent}
					}
					if ch.Delta.Thinking != "" {
						out <- llm.StreamThinkingDeltaEvent{Delta: ch.Delta.Thinking}
					}
					if fc := ch.Delta.FunctionCall; fc != nil {
						name := strings.TrimSpace(fc.Name)
						args := fc.Arguments
						if name != "" || args != "" {
							out <- llm.StreamToolCallDeltaEvent{Index: 0, NameDelta: name, ArgumentsDelta: args}
						}
					}
					for _, tc := range ch.Delta.ToolCalls {
						name := strings.TrimSpace(tc.Function.Name)
						args := tc.Function.Arguments
						if name == "" && args == "" && strings.TrimSpace(tc.ID) == "" {
							continue
						}
						idx := resolveToolIndex(tc.Index, tc.ID, name)
						if args != "" {
							toolSawArgs[idx] = true
						}
						out <- llm.StreamToolCallDeltaEvent{Index: idx, ID: tc.ID, NameDelta: name, ArgumentsDelta: args}
					}
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
		out <- llm.StreamErrorEvent{Err: errors.New("openai stream: retry loop ended without result")}
	}()
	return out, nil
}

type chatCompletionStreamResponse struct {
	ID      string `json:"id"`
	Choices []struct {
		Delta struct {
			Content          string              `json:"content"`
			Refusal          string              `json:"refusal"`
			ReasoningContent string              `json:"reasoning_content"`
			Thinking         string              `json:"thinking"`
			FunctionCall     *legacyFunctionCall `json:"function_call"`
			ToolCalls        []struct {
				// Index is a pointer so an omitted "index" (common on
				// OpenAI-compatible gateways) is distinguishable from a
				// literal 0 and does not collapse parallel calls into one slot.
				Index    *int   `json:"index"`
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
	sc.Buffer(make([]byte, 0, 64*1024), 32*1024*1024)
	dataLines := []string{}
	pending := ""
	flush := func() error {
		if len(dataLines) == 0 {
			return nil
		}
		data := strings.Join(dataLines, "\n")
		dataLines = nil
		if pending != "" {
			data = pending + "\n" + data
		}
		err := onData(data)
		if err != nil {
			if isLikelyOpenAIDecodeError(err) {
				// Some gateways emit a premature blank line in the middle of one JSON event.
				// Keep buffering and retry decode with the next data fragment.
				pending = data
				return nil
			}
			return err
		}
		pending = ""
		return nil
	}
	flushFinal := func() error {
		if err := flush(); err != nil {
			return err
		}
		if strings.TrimSpace(pending) != "" {
			err := onData(pending)
			if err == nil {
				pending = ""
			}
			return err
		}
		return nil
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
	return flushFinal()
}

func isLikelyOpenAIDecodeError(err error) bool {
	if err == nil {
		return false
	}
	return strings.Contains(strings.ToLower(err.Error()), "decode error")
}

func consumeSSEWithBodyClose(body io.ReadCloser, onData func(data string) error) error {
	defer func() {
		_ = body.Close()
	}()
	return consumeSSE(body, onData)
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
	baseURL = normalizeOpenAIBaseURL(baseURL)
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
	if hasEnterpriseVersionPath(baseURL) {
		return baseURL + "/" + suffix
	}
	return baseURL + "/v1/" + suffix
}

func normalizeOpenAIBaseURL(baseURL string) string {
	baseURL = strings.TrimSpace(baseURL)
	for len(baseURL) >= 2 {
		quote := baseURL[0]
		if (quote != '"' && quote != '\'' && quote != '`') || baseURL[len(baseURL)-1] != quote {
			break
		}
		baseURL = strings.TrimSpace(baseURL[1 : len(baseURL)-1])
	}
	return strings.TrimRight(baseURL, "/")
}

func hasEnterpriseVersionPath(baseURL string) bool {
	path := strings.TrimSpace(baseURL)
	if path == "" {
		return false
	}
	if parsed, err := url.Parse(path); err == nil && strings.TrimSpace(parsed.Path) != "" {
		path = parsed.Path
	}
	parts := strings.Split(strings.ToLower(path), "/")
	for i := 0; i < len(parts)-1; i++ {
		if parts[i] != "api" {
			continue
		}
		if isNumericVersionSegment(parts[i+1]) {
			return true
		}
	}
	return false
}

func isNumericVersionSegment(seg string) bool {
	if len(seg) < 2 || seg[0] != 'v' {
		return false
	}
	for i := 1; i < len(seg); i++ {
		if seg[i] < '0' || seg[i] > '9' {
			return false
		}
	}
	return true
}

func (c *ChatClient) isRetryableStatus(code int) bool {
	if c.RetryableStatusCodes == nil {
		return defaultRetryableStatus(code)
	}
	_, ok := c.RetryableStatusCodes[code]
	return ok
}

func defaultRetryableStatus(code int) bool {
	switch code {
	case 401, 403, 408, 409, 425, 429:
		return true
	default:
		return code >= 500 && code <= 599
	}
}

func (c *ChatClient) sleepBackoff(ctx context.Context, attempt int, baseDelay, maxDelay time.Duration, retryAfter time.Duration) {
	sleepRetryBackoff(ctx, attempt, baseDelay, maxDelay, retryAfter)
}

func exponentialBackoffDelay(attempt int, baseDelay, maxDelay time.Duration) time.Duration {
	if maxDelay <= 0 {
		maxDelay = baseDelay
	}
	if baseDelay <= 0 {
		return maxDelay
	}
	if baseDelay >= maxDelay {
		return maxDelay
	}
	if attempt <= 0 {
		return baseDelay
	}

	d := baseDelay
	for i := 0; i < attempt; i++ {
		if d >= maxDelay || d > maxDelay/2 {
			return maxDelay
		}
		d *= 2
	}
	if d > maxDelay {
		return maxDelay
	}
	return d
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

func looksLikeThinkingUnsupported(msg string) bool {
	s := strings.ToLower(msg)
	if strings.Contains(s, "enable_thinking") {
		return true
	}
	if strings.Contains(s, "thinking") && strings.Contains(s, "unknown") {
		return true
	}
	if strings.Contains(s, "thinking") && strings.Contains(s, "unsupported") {
		return true
	}
	if strings.Contains(s, "thinking") && strings.Contains(s, "unrecognized") {
		return true
	}
	return false
}

func looksLikeStreamOptionsUnsupported(msg string) bool {
	s := strings.ToLower(msg)
	if !strings.Contains(s, "stream_options") {
		return false
	}
	if strings.Contains(s, "unknown field") {
		return true
	}
	if strings.Contains(s, "unrecognized field") {
		return true
	}
	if strings.Contains(s, "invalid parameter") {
		return true
	}
	if strings.Contains(s, "unsupported") {
		return true
	}
	return false
}

func looksLikeExtraBodyUnsupported(msg string) bool {
	s := strings.ToLower(msg)
	if !strings.Contains(s, "extra_body") {
		return false
	}
	if strings.Contains(s, "unknown field") {
		return true
	}
	if strings.Contains(s, "unrecognized field") {
		return true
	}
	if strings.Contains(s, "invalid parameter") {
		return true
	}
	if strings.Contains(s, "unexpected") && strings.Contains(s, "field") {
		return true
	}
	return false
}

// looksLikeMaxCompletionTokensUnsupported reports whether the provider rejected
// max_completion_tokens, which older OpenAI-compatible gateways replace with
// max_tokens.
func looksLikeMaxCompletionTokensUnsupported(msg string) bool {
	s := strings.ToLower(msg)
	if !strings.Contains(s, "max_completion_tokens") {
		return false
	}
	if strings.Contains(s, "unknown field") {
		return true
	}
	if strings.Contains(s, "unrecognized field") {
		return true
	}
	if strings.Contains(s, "invalid parameter") {
		return true
	}
	if strings.Contains(s, "unexpected") && strings.Contains(s, "field") {
		return true
	}
	if strings.Contains(s, "unsupported") {
		return true
	}
	// Gateways that only know max_tokens often name it in the remedy text.
	return strings.Contains(s, "max_tokens")
}

func hasThinkingExtra(extra, extraBody map[string]any) bool {
	if extra != nil {
		if _, ok := extra["thinking"]; ok {
			return true
		}
		if _, ok := extra["enable_thinking"]; ok {
			return true
		}
	}
	if extraBody != nil {
		if _, ok := extraBody["enable_thinking"]; ok {
			return true
		}
		if _, ok := extraBody["thinking"]; ok {
			return true
		}
	}
	return false
}

func dropThinkingExtra(extra, extraBody map[string]any) bool {
	removed := false
	if extra != nil {
		if _, ok := extra["thinking"]; ok {
			delete(extra, "thinking")
			removed = true
		}
		if _, ok := extra["enable_thinking"]; ok {
			delete(extra, "enable_thinking")
			removed = true
		}
	}
	if extraBody != nil {
		if _, ok := extraBody["enable_thinking"]; ok {
			delete(extraBody, "enable_thinking")
			removed = true
		}
		if _, ok := extraBody["thinking"]; ok {
			delete(extraBody, "thinking")
			removed = true
		}
	}
	return removed
}

func parseRetryAfter(v string) time.Duration {
	v = strings.TrimSpace(v)
	if v == "" {
		return 0
	}
	// Retry-After is either a plain seconds count or an HTTP date. Parse the
	// seconds form ourselves: appending "s" to an arbitrary token would let a
	// unit-bearing value like "1m" become "1ms" and shrink the wait 60000x.
	if secs, ok := parseRetryAfterSeconds(v); ok {
		if secs > 0 {
			return secs
		}
		return 0
	}
	if t, err := http.ParseTime(v); err == nil {
		d := time.Until(t)
		if d > 0 {
			return d
		}
	}
	return 0
}

// parseRetryAfterSeconds parses the delay-seconds form of Retry-After (an
// optionally signed number, without a unit suffix) and reports whether v had
// that shape.
func parseRetryAfterSeconds(v string) (time.Duration, bool) {
	digits := v
	if len(digits) > 0 && (digits[0] == '+' || digits[0] == '-') {
		digits = digits[1:]
	}
	if digits == "" {
		return 0, false
	}
	dots := 0
	for i := 0; i < len(digits); i++ {
		switch {
		case digits[i] >= '0' && digits[i] <= '9':
		case digits[i] == '.':
			dots++
			if dots > 1 {
				return 0, false
			}
		default:
			return 0, false
		}
	}
	secs, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return 0, false
	}
	return time.Duration(secs * float64(time.Second)), true
}

func isRetryableNetErr(err error) bool {
	if err == nil {
		return false
	}
	var timeoutErr interface{ Timeout() bool }
	if errors.As(err, &timeoutErr) && timeoutErr.Timeout() {
		return true
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return true
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

	MaxCompletionTokens *int   `json:"max_completion_tokens,omitempty"`
	MaxTokens           *int   `json:"max_tokens,omitempty"`
	ServiceTier         string `json:"service_tier,omitempty"`

	ReasoningEffort string `json:"reasoning_effort,omitempty"`

	ParallelToolCalls *bool `json:"parallel_tool_calls,omitempty"`

	Stream        bool           `json:"stream,omitempty"`
	StreamOptions map[string]any `json:"stream_options,omitempty"`

	ExtraBody map[string]any `json:"extra_body,omitempty"`
	Extra     map[string]any `json:"-"`
}

func (r chatRequest) MarshalJSON() ([]byte, error) {
	type alias chatRequest
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

func (c *ChatClient) buildRequest(req llm.InvokeRequest) (*chatRequest, error) {
	if c.ModelName == "" {
		return nil, fmt.Errorf("openai: model is required")
	}
	if err := validateOpenAIToolHistory(req.Messages, "openai"); err != nil {
		return nil, err
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
	if len(tools) > 0 && c.ParallelToolCalls {
		v := c.ParallelToolCalls
		ptc = &v
	}

	extra := cloneMap(c.Extra)
	extraBody := cloneMap(c.ExtraBody)

	// Gateways that predate max_completion_tokens only accept max_tokens.
	maxCompletionTokens := c.MaxCompletionTokens
	var maxTokens *int
	if c.UseLegacyMaxTokens {
		maxTokens = maxCompletionTokens
		maxCompletionTokens = nil
	}

	return &chatRequest{
		Model:               c.ModelName,
		Messages:            msgs,
		Tools:               tools,
		ToolChoice:          toolChoice,
		Temperature:         temp,
		TopP:                c.TopP,
		Seed:                c.Seed,
		MaxCompletionTokens: maxCompletionTokens,
		MaxTokens:           maxTokens,
		ServiceTier:         strings.TrimSpace(c.ServiceTier),
		ReasoningEffort:     c.ReasoningEffort,
		ParallelToolCalls:   ptc,
		Extra:               extra,
		ExtraBody:           extraBody,
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
		switch b.Type {
		case "text":
			if strings.TrimSpace(b.Text) != "" {
				parts = append(parts, map[string]any{"type": "text", "text": b.Text})
			}
		case "image_url":
			if b.ImageURL != nil && strings.TrimSpace(b.ImageURL.URL) != "" {
				imagePayload := map[string]any{"url": b.ImageURL.URL}
				if detail := strings.TrimSpace(b.ImageURL.Detail); detail != "" {
					imagePayload["detail"] = detail
				}
				parts = append(parts, map[string]any{"type": "image_url", "image_url": imagePayload})
			}
		default:
			if fallback := openAIContentFallbackText(b); fallback != "" {
				parts = append(parts, map[string]any{"type": "text", "text": fallback})
			}
		}
	}
	if len(parts) == 0 {
		return c.PlainText()
	}
	return parts
}

func openAIContentFallbackText(block llm.ContentBlock) string {
	switch strings.TrimSpace(block.Type) {
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
	sort.Strings(all)
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
	if len(p) == 0 {
		p = anyJSONSchema()
	}
	// recurse nested objects
	if t, _ := p["type"].(string); t == "object" {
		p = makeStrictSchema(p)
	}
	if items, ok := p["items"].(map[string]any); ok {
		p["items"] = makeStrictArrayItemSchema(items)
	}
	if additional, ok := p["additionalProperties"].(map[string]any); ok {
		p["additionalProperties"] = makeStrictAdditionalPropertiesSchema(additional)
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
			types := append([]any(nil), arr...)
			p["type"] = append(types, "null")
			return p
		}
		// fallback
		p["nullable"] = true
	}
	return p
}

func makeStrictArrayItemSchema(item map[string]any) map[string]any {
	out := cloneMap(item)
	if len(out) == 0 {
		return anyJSONSchema()
	}
	if t, _ := out["type"].(string); t == "object" {
		out = makeStrictSchema(out)
	}
	if items, ok := out["items"].(map[string]any); ok {
		out["items"] = makeStrictArrayItemSchema(items)
	}
	if additional, ok := out["additionalProperties"].(map[string]any); ok {
		out["additionalProperties"] = makeStrictAdditionalPropertiesSchema(additional)
	}
	return out
}

func makeStrictAdditionalPropertiesSchema(schema map[string]any) map[string]any {
	out := cloneMap(schema)
	if len(out) == 0 {
		return anyJSONSchema()
	}
	if t, _ := out["type"].(string); t == "object" {
		out = makeStrictSchema(out)
	}
	if items, ok := out["items"].(map[string]any); ok {
		out["items"] = makeStrictArrayItemSchema(items)
	}
	if additional, ok := out["additionalProperties"].(map[string]any); ok {
		out["additionalProperties"] = makeStrictAdditionalPropertiesSchema(additional)
	}
	if _, ok := out["type"]; !ok {
		if _, hasProperties := out["properties"]; !hasProperties {
			if _, hasItems := out["items"]; !hasItems {
				if _, hasAdditional := out["additionalProperties"]; !hasAdditional {
					return anyJSONSchema()
				}
			}
		}
	}
	return out
}

func anyJSONSchema() map[string]any {
	return map[string]any{
		"type":                 []any{"string", "number", "integer", "boolean", "object", "array", "null"},
		"additionalProperties": false,
		"items": map[string]any{
			"type": []any{"string", "number", "integer", "boolean", "null"},
		},
	}
}

// ---- response parsing ----

type legacyFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type chatCompletionResponse struct {
	ID      string `json:"id"`
	Choices []struct {
		Message struct {
			Role             string              `json:"role"`
			Content          json.RawMessage     `json:"content"`
			Refusal          string              `json:"refusal"`
			ReasoningContent string              `json:"reasoning_content"`
			Thinking         string              `json:"thinking"`
			FunctionCall     *legacyFunctionCall `json:"function_call"`
			ToolCalls        []llm.ToolCall      `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage map[string]any `json:"usage"`
}

func parseChatMessageContent(raw json.RawMessage, refusal string) (llm.Content, error) {
	trimmed := bytes.TrimSpace(raw)
	refusal = strings.TrimSpace(refusal)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		if refusal != "" {
			return llm.TextContent(refusal), nil
		}
		return llm.Content{}, nil
	}
	var text string
	if err := json.Unmarshal(trimmed, &text); err == nil {
		return llm.TextContent(text), nil
	}
	var parts []map[string]any
	if err := json.Unmarshal(trimmed, &parts); err == nil {
		blocks := make([]llm.ContentBlock, 0, len(parts))
		for _, part := range parts {
			partType, _ := part["type"].(string)
			switch partType {
			case "", "text", "output_text":
				if txt, ok := part["text"].(string); ok && txt != "" {
					blocks = append(blocks, llm.ContentBlock{Type: "text", Text: txt})
				}
			default:
				if fallback := chatResponseContentFallbackText(partType, part); fallback != "" {
					blocks = append(blocks, llm.ContentBlock{Type: "text", Text: fallback})
				}
			}
		}
		if len(blocks) == 0 && refusal != "" {
			return llm.TextContent(refusal), nil
		}
		return llm.Content{Blocks: blocks}, nil
	}
	var obj map[string]any
	if err := json.Unmarshal(trimmed, &obj); err == nil {
		if txt, ok := obj["text"].(string); ok {
			return llm.TextContent(txt), nil
		}
		if txt, ok := obj["content"].(string); ok {
			return llm.TextContent(txt), nil
		}
	}
	if refusal != "" {
		return llm.TextContent(refusal), nil
	}
	return llm.Content{}, fmt.Errorf("openai: unsupported assistant message content shape")
}

func legacyFunctionCallToToolCalls(call *legacyFunctionCall) []llm.ToolCall {
	if call == nil {
		return nil
	}
	name := strings.TrimSpace(call.Name)
	args := call.Arguments
	if name == "" && strings.TrimSpace(args) == "" {
		return nil
	}
	return []llm.ToolCall{{
		Type: "function",
		Function: llm.FunctionCall{
			Name:      name,
			Arguments: args,
		},
	}}
}

func ensureChatToolCallIDs(toolCalls []llm.ToolCall) []llm.ToolCall {
	if len(toolCalls) == 0 {
		return toolCalls
	}
	out := append([]llm.ToolCall(nil), toolCalls...)
	used := make(map[string]struct{}, len(out))
	for _, tc := range out {
		if id := strings.TrimSpace(tc.ID); id != "" {
			used[id] = struct{}{}
		}
	}
	nextIndex := 0
	for i := range out {
		if id := strings.TrimSpace(out[i].ID); id != "" {
			out[i].ID = id
			continue
		}
		for {
			candidate := fmt.Sprintf("call_%d", nextIndex)
			nextIndex++
			if _, exists := used[candidate]; exists {
				continue
			}
			out[i].ID = candidate
			used[candidate] = struct{}{}
			break
		}
	}
	return out
}

func normalizeOpenAIStopReason(reason string) string {
	switch strings.TrimSpace(reason) {
	case "length":
		return "max_tokens"
	case "function_call":
		return "tool_calls"
	default:
		return strings.TrimSpace(reason)
	}
}

func chatResponseContentFallbackText(partType string, part map[string]any) string {
	switch strings.TrimSpace(partType) {
	case "image_url", "input_image", "image":
		return "[image]"
	case "document", "input_file", "file":
		return "[document]"
	case "reasoning", "reasoning_text", "thinking":
		if txt, ok := part["text"].(string); ok && strings.TrimSpace(txt) != "" {
			return txt
		}
		if txt, ok := part["content"].(string); ok && strings.TrimSpace(txt) != "" {
			return txt
		}
	}
	return ""
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

	content, err := parseChatMessageContent(msg.Content, msg.Refusal)
	if err != nil {
		return nil, err
	}
	toolCalls := append([]llm.ToolCall(nil), msg.ToolCalls...)
	if len(toolCalls) == 0 {
		toolCalls = legacyFunctionCallToToolCalls(msg.FunctionCall)
	}
	toolCalls = ensureChatToolCallIDs(toolCalls)

	usage := parseUsage(r.Usage)

	stopReason := normalizeOpenAIStopReason(r.Choices[0].FinishReason)

	thinking := strings.TrimSpace(msg.ReasoningContent)
	if thinking == "" {
		thinking = strings.TrimSpace(msg.Thinking)
	}

	return &llm.Completion{
		Content:    content,
		Thinking:   thinking,
		ToolCalls:  toolCalls,
		Usage:      usage,
		StopReason: stopReason,
		ResponseID: strings.TrimSpace(r.ID),
		Raw:        append([]byte(nil), data...),
	}, nil
}

func parseUsage(u map[string]any) *llm.Usage {
	if u == nil {
		return nil
	}
	pt := intFromAny(u["prompt_tokens"])
	ct := intFromAny(u["completion_tokens"])
	tt := intFromAny(u["total_tokens"])

	// Some OpenAI-compatible providers omit completion_tokens in chat responses.
	// Infer it from total-prompt when possible, but never add reasoning_tokens
	// from the breakdown because completion_tokens already includes them.
	if ct == 0 && tt >= pt {
		ct = tt - pt
	}
	if tt == 0 && (pt > 0 || ct > 0) {
		tt = pt + ct
	}

	var cached *int
	if det, ok := u["prompt_tokens_details"].(map[string]any); ok {
		v := intFromAny(det["cached_tokens"])
		if v > 0 {
			cached = &v
		}
	}
	usage := llm.NewProviderUsage(pt, ct, tt)
	usage.PromptCachedTokens = cached
	return usage
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

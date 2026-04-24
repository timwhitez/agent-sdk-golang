package anthropic

import (
	"bufio"
	"bytes"
	"context"
	cryptorand "crypto/rand"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

const defaultBaseURL = "https://api.anthropic.com"

const promptCachingBeta = "prompt-caching-2024-07-31"

var retryAfterWarningf = log.Printf
var backoffRandRead = cryptorand.Read
var toolIDNormalizationWarningf = log.Printf

type Client struct {
	HTTPClient *http.Client
	BaseURL    string

	APIKey    string
	AuthToken string

	ModelName   string
	MaxTokens   int
	Temperature *float64
	TopP        *float64
	Seed        *int

	// Thinking mode (Anthropic extended thinking). If nil or <=0, disabled.
	ThinkingBudgetTokens *int

	// Retry policy.
	MaxRetries           int
	RetryBaseDelay       time.Duration
	RetryMaxDelay        time.Duration
	RetryableStatusCodes map[int]struct{}

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
	endpoint := anthropicEndpoint(baseURL, "messages")
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
	localBeta := append([]string(nil), c.Beta...)
	localThinking := c.ThinkingBudgetTokens
	usedFinalDowngradeRetry := false

	for attempt := 0; attempt < maxRetries+1; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		payload, err := c.buildRequest(req, localThinking)
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
		httpReq.Header.Set("anthropic-version", "2023-06-01")
		betaHeader := strings.TrimSpace(strings.Join(localBeta, ", "))
		if betaHeader != "" {
			httpReq.Header.Set("anthropic-beta", betaHeader)
		}
		if c.APIKey != "" {
			httpReq.Header.Set("x-api-key", c.APIKey)
		}
		if c.AuthToken != "" {
			httpReq.Header.Set("Authorization", "Bearer "+c.AuthToken)
		}

		resp, err := client.Do(httpReq)
		if err == nil {
			data, readErr := readResponseBodyLimited(resp.Body, endpoint)
			if readErr != nil {
				retryAfter := parseRetryAfter(resp.Header.Get("Retry-After"))
				return nil, anthropicReadBodyError(resp.StatusCode, retryAfter, readErr)
			}

			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				return parseResponse(data)
			}

			retryAfter := parseRetryAfter(resp.Header.Get("Retry-After"))
			msg := strings.TrimSpace(string(data))

			didDowngrade := false
			// Automatic downgrade: some gateways reject Claude Code betas.
			if (resp.StatusCode == 400 || resp.StatusCode == 422) && betaHeader != "" && looksLikeBetaUnsupported(msg) {
				if setPromptCachingBeta(&localBeta) {
					didDowngrade = true
				}
			}
			// Automatic downgrade: disable extended thinking on models/endpoints that don't support it.
			if (resp.StatusCode == 400 || resp.StatusCode == 422) && localThinking != nil && *localThinking > 0 && looksLikeThinkingUnsupported(msg) {
				localThinking = nil
				didDowngrade = true
			}
			if didDowngrade && allowDowngradeRetry(attempt, maxRetries, &usedFinalDowngradeRetry) {
				continue
			}

			if resp.StatusCode == 429 {
				lastErr = &llm.RateLimitError{Provider: "anthropic", Message: msg, RetryAfter: retryAfter}
			} else {
				lastErr = &llm.ProviderError{Provider: "anthropic", StatusCode: resp.StatusCode, Message: msg, RetryAfter: retryAfter}
			}
			if c.isRetryableStatus(resp.StatusCode) && attempt < maxRetries-1 {
				c.sleepBackoff(ctx, attempt, baseDelay, maxDelay, retryAfter)
				continue
			}
			return nil, lastErr
		}

		// Network / timeout errors.
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
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

func (c *Client) sleepBackoff(ctx context.Context, attempt int, baseDelay, maxDelay time.Duration, retryAfter time.Duration) {
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
	// 10% jitter
	jitter := time.Duration(randomBackoffFraction() * float64(d) * 0.1)
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

func randomBackoffFraction() float64 {
	const scale = float64(1 << 53)

	var b [8]byte
	if _, err := backoffRandRead(b[:]); err == nil {
		// Keep top 53 bits so float64 conversion retains entropy exactly.
		sample := binary.BigEndian.Uint64(b[:]) >> 11
		return float64(sample) / scale
	}

	// Fallback still adds jitter if the entropy source is temporarily unavailable.
	n := uint64(time.Now().UnixNano())
	n ^= n << 13
	n ^= n >> 7
	n ^= n << 17
	return float64(n>>11) / scale
}

func allowDowngradeRetry(attempt, maxRetries int, usedFinalDowngradeRetry *bool) bool {
	if attempt < maxRetries-1 {
		return true
	}
	if attempt == maxRetries-1 && usedFinalDowngradeRetry != nil && !*usedFinalDowngradeRetry {
		*usedFinalDowngradeRetry = true
		return true
	}
	return false
}

func setPromptCachingBeta(beta *[]string) bool {
	if beta == nil {
		return false
	}
	if len(*beta) == 1 && strings.EqualFold(strings.TrimSpace((*beta)[0]), promptCachingBeta) {
		return false
	}
	*beta = []string{promptCachingBeta}
	return true
}

type compatibilityError struct {
	Message string
	Code    string
	Type    string
	Param   string
}

func parseCompatibilityError(raw string) compatibilityError {
	trimmed := strings.TrimSpace(raw)
	parsed := compatibilityError{Message: strings.ToLower(trimmed)}
	if trimmed == "" {
		return parsed
	}

	var root map[string]any
	if err := json.Unmarshal([]byte(trimmed), &root); err != nil {
		return parsed
	}

	msg := ""
	if errObj, ok := root["error"].(map[string]any); ok {
		if msg == "" {
			msg = compatibilityString(errObj["message"])
		}
		if parsed.Code == "" {
			parsed.Code = compatibilityString(errObj["code"])
		}
		if parsed.Type == "" {
			parsed.Type = compatibilityString(errObj["type"])
		}
		if parsed.Param == "" {
			parsed.Param = firstCompatibilityParam(errObj)
		}
	}
	if msg == "" {
		msg = compatibilityString(root["message"])
	}
	if parsed.Code == "" {
		parsed.Code = compatibilityString(root["code"])
	}
	if parsed.Type == "" {
		parsed.Type = compatibilityString(root["type"])
	}
	if parsed.Param == "" {
		parsed.Param = firstCompatibilityParam(root)
	}
	if msg != "" {
		parsed.Message = msg
	}
	return parsed
}

func firstCompatibilityParam(m map[string]any) string {
	if m == nil {
		return ""
	}
	keys := []string{"param", "field", "path", "pointer", "name"}
	for _, k := range keys {
		if s := compatibilityString(m[k]); s != "" {
			return s
		}
	}
	if details, ok := m["details"].([]any); ok {
		for _, d := range details {
			dm, ok := d.(map[string]any)
			if !ok {
				continue
			}
			if s := firstCompatibilityParam(dm); s != "" {
				return s
			}
		}
	}
	return ""
}

func compatibilityString(v any) string {
	switch x := v.(type) {
	case string:
		return strings.ToLower(strings.TrimSpace(x))
	case []any:
		parts := make([]string, 0, len(x))
		for _, item := range x {
			s := compatibilityString(item)
			if s != "" {
				parts = append(parts, s)
			}
		}
		return strings.Join(parts, ".")
	default:
		if x == nil {
			return ""
		}
		return strings.ToLower(strings.TrimSpace(fmt.Sprintf("%v", x)))
	}
}

func looksLikeUnsupportedError(msg string) bool {
	s := strings.ToLower(msg)
	if s == "" {
		return false
	}
	if strings.Contains(s, "unsupported") || strings.Contains(s, "not supported") {
		return true
	}
	if strings.Contains(s, "unknown") || strings.Contains(s, "unrecognized") {
		return true
	}
	if strings.Contains(s, "invalid parameter") || strings.Contains(s, "unknown field") || strings.Contains(s, "unrecognized field") {
		return true
	}
	if strings.Contains(s, "unexpected") && strings.Contains(s, "field") {
		return true
	}
	if strings.Contains(s, "extra fields not permitted") {
		return true
	}
	return false
}

func looksLikeUnsupportedCode(s string) bool {
	s = strings.ToLower(s)
	if s == "" {
		return false
	}
	if strings.Contains(s, "unsupported") || strings.Contains(s, "unknown") || strings.Contains(s, "unrecognized") {
		return true
	}
	if strings.Contains(s, "invalid_parameter") || strings.Contains(s, "invalid-parameter") || strings.Contains(s, "unknown_parameter") {
		return true
	}
	if strings.Contains(s, "invalid_request_error") {
		return true
	}
	return false
}

func containsAny(msg string, needles ...string) bool {
	for _, n := range needles {
		if strings.Contains(msg, n) {
			return true
		}
	}
	return false
}

func looksLikeThinkingUnsupported(msg string) bool {
	parsed := parseCompatibilityError(msg)
	thinkingFields := []string{"thinking", "budget_tokens", "redacted_thinking", "enable_thinking"}

	if containsAny(parsed.Param, thinkingFields...) && (looksLikeUnsupportedCode(parsed.Code) || looksLikeUnsupportedCode(parsed.Type) || looksLikeUnsupportedError(parsed.Message)) {
		return true
	}
	if (looksLikeUnsupportedCode(parsed.Code) || looksLikeUnsupportedCode(parsed.Type)) && containsAny(parsed.Message, thinkingFields...) {
		return true
	}
	return looksLikeUnsupportedError(parsed.Message) && containsAny(parsed.Message, thinkingFields...)
}

func looksLikeBetaUnsupported(msg string) bool {
	parsed := parseCompatibilityError(msg)
	betaFields := []string{"anthropic-beta", "beta", "claude-code"}

	if containsAny(parsed.Param, betaFields...) && (looksLikeUnsupportedCode(parsed.Code) || looksLikeUnsupportedCode(parsed.Type) || looksLikeUnsupportedError(parsed.Message)) {
		return true
	}
	if (looksLikeUnsupportedCode(parsed.Code) || looksLikeUnsupportedCode(parsed.Type)) && containsAny(parsed.Message, betaFields...) {
		return true
	}
	return looksLikeUnsupportedError(parsed.Message) && containsAny(parsed.Message, betaFields...)
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
		retryAfterWarningf("[WARN] Anthropic Retry-After %q is non-positive - ignoring header and using exponential backoff.", v)
		return 0
	}
	if t, err := http.ParseTime(v); err == nil {
		d := time.Until(t)
		if d > 0 {
			return d
		}
		retryAfterWarningf("[WARN] Anthropic Retry-After %q is non-positive - ignoring header and using exponential backoff.", v)
	}
	return 0
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
	Role    string              `json:"role"`
	Content []contentBlockParam `json:"content"`
}

func normalizeToolCallID(id string) string {
	return normalizeToolCallIDWithWarning(id, toolIDNormalizationWarningf)
}

func normalizeToolCallIDWithWarning(id string, warnf func(string, ...any)) string {
	original := id
	id = strings.TrimSpace(id)
	if id == "" {
		return ""
	}
	// Claude requires tool_use_id to be alphanumeric/underscore/hyphen.
	// See opencode ProviderTransform.normalizeMessages.
	out := make([]rune, 0, len(id))
	for _, r := range id {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' || r == '-' {
			out = append(out, r)
		} else {
			out = append(out, '_')
		}
	}
	normalized := string(out)
	if normalized != original && warnf != nil {
		warnf("anthropic: normalized tool call id original=%q normalized=%q", original, normalized)
	}
	return normalized
}

type thinkingParam struct {
	Type         string `json:"type"` // "enabled"
	BudgetTokens int    `json:"budget_tokens"`
}

type requestPayload struct {
	Model     string `json:"model"`
	MaxTokens int    `json:"max_tokens"`

	System   any            `json:"system,omitempty"` // string or []contentBlockParam
	Messages []messageParam `json:"messages"`

	Tools      []toolParam      `json:"tools,omitempty"`
	ToolChoice *toolChoiceParam `json:"tool_choice,omitempty"`

	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	Seed        *int     `json:"seed,omitempty"`

	Thinking *thinkingParam `json:"thinking,omitempty"`

	Stream bool `json:"stream,omitempty"`
}

// InvokeStream implements true SSE streaming for Anthropic messages.
// It emits text deltas, thinking deltas, and basic tool_use deltas (best-effort).
func (c *Client) InvokeStream(ctx context.Context, req llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	out := make(chan llm.StreamEvent, 128)
	go func() {
		defer close(out)

		client := streamHTTPClient(c.httpClient())
		baseURL := strings.TrimRight(c.baseURL(), "/")
		endpoint := anthropicEndpoint(baseURL, "messages")

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
		localBeta := append([]string(nil), c.Beta...)
		localThinking := c.ThinkingBudgetTokens
		usedFinalDowngradeRetry := false

		for attempt := 0; attempt < maxRetries+1; attempt++ {
			if err := ctx.Err(); err != nil {
				out <- llm.StreamErrorEvent{Err: err}
				return
			}
			payload, err := c.buildRequest(req, localThinking)
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
			httpReq.Header.Set("anthropic-version", "2023-06-01")
			betaHeader := strings.TrimSpace(strings.Join(localBeta, ", "))
			if betaHeader != "" {
				httpReq.Header.Set("anthropic-beta", betaHeader)
			}
			if c.APIKey != "" {
				httpReq.Header.Set("x-api-key", c.APIKey)
			}
			if c.AuthToken != "" {
				httpReq.Header.Set("Authorization", "Bearer "+c.AuthToken)
			}

			resp, err := client.Do(httpReq)
			if err != nil {
				if ctxErr := ctx.Err(); ctxErr != nil {
					out <- llm.StreamErrorEvent{Err: ctxErr}
					return
				}
				if attempt < maxRetries-1 && isRetryableNetErr(err) {
					c.sleepBackoff(ctx, attempt, baseDelay, maxDelay, 0)
					continue
				}
				out <- llm.StreamErrorEvent{Err: err}
				return
			}

			if resp.StatusCode < 200 || resp.StatusCode >= 300 {
				data, readErr := readResponseBodyLimited(resp.Body, endpoint)
				if readErr != nil {
					retryAfter := parseRetryAfter(resp.Header.Get("Retry-After"))
					out <- llm.StreamErrorEvent{Err: anthropicReadBodyError(resp.StatusCode, retryAfter, readErr)}
					return
				}
				retryAfter := parseRetryAfter(resp.Header.Get("Retry-After"))
				msg := strings.TrimSpace(string(data))

				didDowngrade := false
				if (resp.StatusCode == 400 || resp.StatusCode == 422) && betaHeader != "" && looksLikeBetaUnsupported(msg) {
					if setPromptCachingBeta(&localBeta) {
						didDowngrade = true
					}
				}
				// Automatic downgrade: disable thinking when unsupported.
				if (resp.StatusCode == 400 || resp.StatusCode == 422) && localThinking != nil && *localThinking > 0 && looksLikeThinkingUnsupported(msg) {
					localThinking = nil
					didDowngrade = true
				}
				if didDowngrade && allowDowngradeRetry(attempt, maxRetries, &usedFinalDowngradeRetry) {
					continue
				}
				var lastErr error
				if resp.StatusCode == 429 {
					lastErr = &llm.RateLimitError{Provider: "anthropic", Message: msg, RetryAfter: retryAfter}
				} else {
					lastErr = &llm.ProviderError{Provider: "anthropic", StatusCode: resp.StatusCode, Message: msg, RetryAfter: retryAfter}
				}
				if c.isRetryableStatus(resp.StatusCode) && attempt < maxRetries-1 {
					c.sleepBackoff(ctx, attempt, baseDelay, maxDelay, retryAfter)
					continue
				}
				out <- llm.StreamErrorEvent{Err: lastErr}
				return
			}

			blockToToolIndex := map[int]int{}
			inputTokens := 0
			outputTokens := 0
			var promptCachedTokens *int
			var promptCacheCreationTokens *int
			stopReason := ""
			responseID := ""
			nextTool := 0
			emitResponseID := func(id string) {
				id = strings.TrimSpace(id)
				if id == "" || id == responseID {
					return
				}
				responseID = id
				out <- llm.StreamResponseEvent{ResponseID: id}
			}
			getToolIndex := func(blockIdx int) int {
				if v, ok := blockToToolIndex[blockIdx]; ok {
					return v
				}
				idx := nextTool
				nextTool++
				blockToToolIndex[blockIdx] = idx
				return idx
			}

			err = consumeSSEWithBodyClose(resp.Body, func(data string) error {
				data = strings.TrimSpace(data)
				if data == "" {
					return nil
				}
				var root map[string]any
				if err := json.Unmarshal([]byte(data), &root); err != nil {
					return fmt.Errorf("anthropic stream: failed to decode SSE JSON event: %w", err)
				}
				if root == nil {
					return errors.New("anthropic stream: failed to decode SSE JSON event: expected object payload")
				}
				typ, _ := root["type"].(string)
				emitResponseID(streamResponseIDFromEvent(typ, root))
				switch typ {
				case "message_start":
					if msg, ok := root["message"].(map[string]any); ok {
						if u, ok := msg["usage"].(map[string]any); ok {
							inputTokens = intFromAny(u["input_tokens"])
							if raw, ok := u["cache_read_input_tokens"]; ok {
								promptCachedTokens = intPtrFromAny(raw)
							}
							if raw, ok := u["cache_creation_input_tokens"]; ok {
								promptCacheCreationTokens = intPtrFromAny(raw)
							}
						}
					}
				case "message_delta":
					if d, ok := root["delta"].(map[string]any); ok {
						if sr, ok := d["stop_reason"].(string); ok && sr != "" {
							stopReason = sr
						}
					}
					if u, ok := root["usage"].(map[string]any); ok {
						ot := intFromAny(u["output_tokens"])
						if ot > outputTokens {
							outputTokens = ot
						}
					}
				case "content_block_start":
					idx := intFromAny(root["index"])
					blk, _ := root["content_block"].(map[string]any)
					btype, _ := blk["type"].(string)
					if btype == "tool_use" {
						id, _ := blk["id"].(string)
						name, _ := blk["name"].(string)
						ti := getToolIndex(idx)
						out <- llm.StreamToolCallDeltaEvent{Index: ti, ID: id, NameDelta: name}
					}
				case "content_block_delta":
					idx := intFromAny(root["index"])
					del, _ := root["delta"].(map[string]any)
					// text delta (preserve whitespace deltas)
					if t, ok := del["text"].(string); ok && t != "" {
						out <- llm.StreamTextDeltaEvent{Delta: t}
						return nil
					}
					// thinking delta
					if t, ok := del["thinking"].(string); ok && t != "" {
						out <- llm.StreamThinkingDeltaEvent{Delta: t}
						return nil
					}
					// tool input json delta
					if pj, ok := del["partial_json"].(string); ok && pj != "" {
						ti := getToolIndex(idx)
						out <- llm.StreamToolCallDeltaEvent{Index: ti, ArgumentsDelta: pj}
						return nil
					}
				case "message_stop":
					if inputTokens > 0 || outputTokens > 0 || promptCachedTokens != nil || promptCacheCreationTokens != nil {
						out <- llm.StreamUsageEvent{Usage: llm.Usage{
							PromptTokens:              inputTokens,
							CompletionTokens:          outputTokens,
							TotalTokens:               inputTokens + outputTokens,
							PromptCachedTokens:        promptCachedTokens,
							PromptCacheCreationTokens: promptCacheCreationTokens,
						}}
					}
				}
				return nil
			})
			if err != nil {
				out <- llm.StreamErrorEvent{Err: err}
				return
			}
			out <- llm.StreamDoneEvent{StopReason: stopReason}
			return
		}
		out <- llm.StreamErrorEvent{Err: errors.New("anthropic stream: retry loop ended without result")}
	}()
	return out, nil
}

func streamResponseIDFromEvent(eventType string, root map[string]any) string {
	if root == nil {
		return ""
	}
	extractMessageID := func(m map[string]any) string {
		if m == nil {
			return ""
		}
		if id, ok := m["id"].(string); ok {
			return strings.TrimSpace(id)
		}
		return ""
	}
	switch strings.TrimSpace(eventType) {
	case "message_start":
		if msg, ok := root["message"].(map[string]any); ok {
			if id := extractMessageID(msg); id != "" {
				return id
			}
		}
		if id, ok := root["id"].(string); ok {
			return strings.TrimSpace(id)
		}
	case "message_delta", "message_stop":
		if id, ok := root["id"].(string); ok {
			return strings.TrimSpace(id)
		}
		if msg, ok := root["message"].(map[string]any); ok {
			if id := extractMessageID(msg); id != "" {
				return id
			}
		}
	}
	return ""
}

func consumeSSE(r io.Reader, onData func(data string) error) error {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	dataLines := []string{}
	pending := ""
	flush := func(final bool) error {
		if len(dataLines) == 0 {
			if final && pending != "" {
				return fmt.Errorf("anthropic stream: malformed SSE event payload")
			}
			return nil
		}
		data := strings.Join(dataLines, "\n")
		dataLines = nil
		if pending != "" {
			data = pending + "\n" + data
		}
		if !json.Valid([]byte(data)) {
			pending = data
			if final {
				return fmt.Errorf("anthropic stream: malformed SSE event payload")
			}
			return nil
		}
		pending = ""
		return onData(data)
	}
	for sc.Scan() {
		line := sc.Text()
		if line == "" {
			if err := flush(false); err != nil {
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
	return flush(true)
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

func intPtrFromAny(v any) *int {
	i := intFromAny(v)
	if i == 0 {
		switch v.(type) {
		case float64, int, int64, json.Number:
			vv := 0
			return &vv
		default:
			return nil
		}
	}
	vv := i
	return &vv
}

func (c *Client) buildRequest(req llm.InvokeRequest, thinkingBudgetTokens *int) (*requestPayload, error) {
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
	if thinkingBudgetTokens != nil && *thinkingBudgetTokens > 0 {
		thinking = &thinkingParam{Type: "enabled", BudgetTokens: *thinkingBudgetTokens}
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

	for _, m := range in {
		switch m.Role {
		case llm.RoleSystem:
			if strings.TrimSpace(m.Content.Text) != "" {
				blk := contentBlockParam{Type: "text", Text: m.Content.Text}
				if m.Cache {
					blk.CacheCtrl = &cacheControl{Type: "ephemeral"}
				}
				sysBlocks = append(sysBlocks, blk)
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
		if text, ok := joinPlainSystemText(sysBlocks); ok {
			system = text
		} else {
			// Use structured system blocks when we need cache_control or non-text blocks.
			system = sysBlocks
		}
	}
	return system, out, nil
}

func joinPlainSystemText(blocks []contentBlockParam) (string, bool) {
	parts := make([]string, 0, len(blocks))
	for _, blk := range blocks {
		if blk.Type != "text" || blk.CacheCtrl != nil {
			return "", false
		}
		parts = append(parts, blk.Text)
	}
	return strings.Join(parts, "\n\n"), true
}

func toAnthropicMessage(m llm.Message) (*messageParam, error) {
	if m.Role == llm.RoleTool {
		// Anthropic expects tool results as role=user with tool_result blocks.
		contentText := m.Content.PlainText()
		if strings.TrimSpace(contentText) == "" {
			contentText = "(no output)"
		}
		toolUseID := normalizeToolCallID(m.ToolCallID)
		if toolUseID == "" {
			toolUseID = m.ToolCallID
		}
		content := contentBlockParam{
			Type:      "tool_result",
			ToolUseID: toolUseID,
			Content:   contentText,
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
			id := normalizeToolCallID(tc.ID)
			if id == "" {
				id = tc.ID
			}
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
				ID:    id,
				Name:  tc.Function.Name,
				Input: input,
			})
		}
	}

	if len(blocks) == 0 {
		// Anthropic rejects empty messages; omit them.
		return nil, nil
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
		// unsupported blocks are ignored on wire; keep a non-empty placeholder.
		blk.Type = "text"
		blk.Text = "(unsupported content omitted)"
	}
	return blk
}

// anthropicEndpoint builds an endpoint URL for the Anthropic Messages API.
// It supports common proxy styles:
// - baseURL like "https://api.anthropic.com" => "/v1/..."
// - baseURL like "https://proxy.example.com/v1" => "/..." (avoid double v1)
// - baseURL like "https://host/api/v3" => "/..." (enterprise version path already encoded)
func anthropicEndpoint(baseURL, suffix string) string {
	baseURL = strings.TrimRight(strings.TrimSpace(baseURL), "/")
	suffix = strings.TrimLeft(strings.TrimSpace(suffix), "/")
	if baseURL == "" {
		baseURL = strings.TrimRight(defaultBaseURL, "/")
	}
	if suffix == "" {
		return baseURL
	}
	if strings.HasSuffix(baseURL, "/v1") {
		return baseURL + "/" + suffix
	}
	if strings.Contains(baseURL, "/api/v") {
		return baseURL + "/" + suffix
	}
	return baseURL + "/v1/" + suffix
}

type responsePayload struct {
	ID      string `json:"id,omitempty"`
	Content []struct {
		Type      string          `json:"type"`
		Text      string          `json:"text,omitempty"`
		ID        string          `json:"id,omitempty"`
		Name      string          `json:"name,omitempty"`
		Input     json.RawMessage `json:"input,omitempty"`
		Thinking  string          `json:"thinking,omitempty"`
		Signature string          `json:"signature,omitempty"`
		Data      string          `json:"data,omitempty"`
	} `json:"content"`
	StopReason string `json:"stop_reason,omitempty"`
	Usage      struct {
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
		case "redacted_thinking":
			blocks = append(blocks, llm.ContentBlock{Type: "redacted_thinking", Data: blk.Data})
		default:
			// ignore unknown
		}
	}

	usage := &llm.Usage{
		PromptTokens:              rp.Usage.InputTokens,
		CompletionTokens:          rp.Usage.OutputTokens,
		TotalTokens:               rp.Usage.InputTokens + rp.Usage.OutputTokens,
		PromptCachedTokens:        rp.Usage.CacheReadInputTokens,
		PromptCacheCreationTokens: rp.Usage.CacheCreationInputTokens,
	}

	return &llm.Completion{
		Content:    llm.Content{Blocks: blocks},
		Thinking:   strings.Join(thinkingParts, "\n"),
		ToolCalls:  toolCalls,
		Usage:      usage,
		StopReason: rp.StopReason,
		ResponseID: strings.TrimSpace(rp.ID),
		Raw:        append([]byte(nil), data...),
	}, nil
}

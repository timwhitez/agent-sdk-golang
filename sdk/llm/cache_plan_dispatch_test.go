package llm_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm/anthropic"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// dispatchPlanRequest has a system section, a user message with two text
// blocks and two tool definitions, so message, block and tool targets all map.
func dispatchPlanRequest() llm.InvokeRequest {
	return llm.InvokeRequest{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: llm.TextContent("system")},
			{Role: llm.RoleUser, Content: llm.Content{Blocks: []llm.ContentBlock{{Type: "text", Text: "first"}, {Type: "text", Text: "second"}}}},
		},
		Tools: []llm.ToolDefinition{
			{Name: "tool_a", Parameters: map[string]any{"type": "object"}},
			{Name: "tool_b", Parameters: map[string]any{"type": "object"}},
		},
	}
}

func dispatchDirectives() []llm.CacheDirective {
	return []llm.CacheDirective{
		{Target: llm.CacheTarget{Kind: llm.CacheAfterToolDefinition, ToolIndex: 1}, Policy: llm.CacheRequired, TTL: llm.CacheTTL1Hour},
		{Target: llm.CacheTarget{Kind: llm.CacheAfterMessageBlock, MessageIndex: 1, BlockOrdinal: 0}, Policy: llm.CacheRequired, TTL: llm.CacheTTL5Minutes},
		{Target: llm.CacheTarget{Kind: llm.CacheAfterMessage, MessageIndex: 1}, Policy: llm.CacheBestEffort},
	}
}

// wireCacheControls returns the cache_control objects in wire order
// (tools -> system -> messages) from one captured Anthropic payload.
func wireCacheControls(t *testing.T, payload []byte) []string {
	t.Helper()
	var decoded struct {
		Tools    []map[string]json.RawMessage `json:"tools"`
		System   json.RawMessage              `json:"system"`
		Messages []struct {
			Content []map[string]json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatalf("decode payload: %v", err)
	}
	var controls []string
	add := func(prefix string, object map[string]json.RawMessage) {
		if raw, ok := object["cache_control"]; ok {
			controls = append(controls, prefix+string(raw))
		}
	}
	for i, tool := range decoded.Tools {
		add(fmt.Sprintf("tool[%d]=", i), tool)
	}
	var system []map[string]json.RawMessage
	if err := json.Unmarshal(decoded.System, &system); err == nil {
		for i, block := range system {
			add(fmt.Sprintf("system[%d]=", i), block)
		}
	}
	for i, message := range decoded.Messages {
		for j, block := range message.Content {
			add(fmt.Sprintf("message[%d][%d]=", i, j), block)
		}
	}
	return controls
}

func anthropicDispatchClient(transport cacheWireTransport, warn func(string, ...any)) *anthropic.Client {
	client := &anthropic.Client{
		HTTPClient:     &http.Client{Transport: transport},
		BaseURL:        "https://fixture.invalid",
		APIKey:         "private-key",
		ModelName:      "fixture",
		MaxTokens:      64,
		MaxRetries:     2,
		RetryBaseDelay: time.Millisecond,
		RetryMaxDelay:  time.Millisecond,
	}
	client.SetWarningf(warn)
	return client
}

func fixtureResponse(status int, body string, r *http.Request) *http.Response {
	return &http.Response{StatusCode: status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body)), Request: r}
}

// A transient provider failure inside the client's retry loop re-sends the
// exact same accepted plan: admission runs once, no rebinding or reordering
// happens between attempts, and the wire carries the planned positions.
func TestCachePlanTransientRetryReusesIdenticalWire(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%v", stream), func(t *testing.T) {
			request := dispatchPlanRequest()
			bindToolCache(t, &request, dispatchDirectives())
			var payloads [][]byte
			var accepted atomic.Int32
			client := anthropicDispatchClient(func(r *http.Request) (*http.Response, error) {
				data, err := io.ReadAll(r.Body)
				if err != nil {
					return nil, err
				}
				payloads = append(payloads, data)
				if len(payloads) == 1 {
					return fixtureResponse(529, `{"error":{"type":"overloaded_error","message":"fixture overloaded"}}`, r), nil
				}
				body := admissionSuccess["anthropic"]
				if stream {
					body = admissionSSE("anthropic")
				}
				return fixtureResponse(200, body, r), nil
			}, func(format string, args ...any) {
				if strings.HasPrefix(fmt.Sprintf(format, args...), "cache_plan_accepted:") {
					accepted.Add(1)
				}
			})
			if _, err := callAdmissionModel(context.Background(), client, request, stream); err != nil {
				t.Fatalf("call: %v", err)
			}
			if len(payloads) != 2 || !bytes.Equal(payloads[0], payloads[1]) {
				t.Fatalf("attempts=%d identical=%v", len(payloads), len(payloads) == 2 && bytes.Equal(payloads[0], payloads[1]))
			}
			if got := accepted.Load(); got != 3 {
				t.Fatalf("admission diagnostics=%d, want one admission of 3 directives", got)
			}
			want := []string{
				`tool[1]={"type":"ephemeral","ttl":"1h"}`,
				`message[0][0]={"type":"ephemeral","ttl":"5m"}`,
				`message[0][1]={"type":"ephemeral"}`,
			}
			if got := wireCacheControls(t, payloads[0]); strings.Join(got, "|") != strings.Join(want, "|") {
				t.Fatalf("wire cache controls=%v want %v", got, want)
			}
		})
	}
}

// The admission diagnostics describe local acceptance only. Remote cache
// activity is visible solely through provider usage; without it the cache
// counters stay unknown (nil) instead of being reported as zero or as a hit.
func TestCachePlanDispatchStagesAreDistinct(t *testing.T) {
	for _, test := range []struct {
		name  string
		usage string
		read  *int
	}{
		{name: "no cache usage", usage: `{"input_tokens":10,"output_tokens":1}`},
		{name: "provider reported read", usage: `{"input_tokens":10,"output_tokens":1,"cache_read_input_tokens":7,"cache_creation_input_tokens":0}`, read: intPtr(7)},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := dispatchPlanRequest()
			bindToolCache(t, &request, dispatchDirectives())
			var payload []byte
			client := anthropicDispatchClient(func(r *http.Request) (*http.Response, error) {
				payload, _ = io.ReadAll(r.Body)
				return fixtureResponse(200, `{"id":"m","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":`+test.usage+`}`, r), nil
			}, nil)
			completion, err := client.Invoke(context.Background(), request)
			if err != nil {
				t.Fatal(err)
			}
			// Stage 1: locally accepted.
			accepted := 0
			for _, diagnostic := range completion.Diagnostics {
				if diagnostic.Kind == "cache_plan_accepted" {
					accepted++
				}
				lower := strings.ToLower(diagnostic.Message)
				if strings.Contains(lower, "hit") || strings.Contains(lower, "remote") {
					t.Fatalf("admission diagnostic claims remote evidence: %+v", diagnostic)
				}
			}
			if accepted != 3 {
				t.Fatalf("accepted diagnostics=%d", accepted)
			}
			// Stage 2: dispatched payload carries the accepted markers.
			if got := len(wireCacheControls(t, payload)); got != 3 {
				t.Fatalf("dispatched markers=%d", got)
			}
			// Stage 3: usage is only what the provider reported.
			got := completion.Usage.PromptCachedTokens
			if (got == nil) != (test.read == nil) || (got != nil && *got != *test.read) {
				t.Fatalf("PromptCachedTokens=%v want %v", got, test.read)
			}
		})
	}
}

func intPtr(v int) *int { return &v }

// After InvokeStream returns, caller mutation of the source request or plan
// cannot reach the payload that is dispatched asynchronously.
func TestCachePlanStreamDispatchIsolatedFromCallerMutation(t *testing.T) {
	request := dispatchPlanRequest()
	bindToolCache(t, &request, dispatchDirectives())
	release := make(chan struct{})
	var payload []byte
	client := anthropicDispatchClient(func(r *http.Request) (*http.Response, error) {
		<-release
		payload, _ = io.ReadAll(r.Body)
		return fixtureResponse(200, admissionSSE("anthropic"), r), nil
	}, nil)
	events, err := client.InvokeStream(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	request.Messages[1].Content.Blocks[0].Text = "private-mutated"
	request.Tools[1].Name = "mutated_tool"
	request.CachePlan.Directives[0].TTL = llm.CacheTTL5Minutes
	request.CachePlan.Directives = request.CachePlan.Directives[:1]
	close(release)
	for range events {
	}
	if bytes.Contains(payload, []byte("private-mutated")) || bytes.Contains(payload, []byte("mutated_tool")) {
		t.Fatal("caller mutation reached dispatched payload")
	}
	if got := wireCacheControls(t, payload); len(got) != 3 || got[0] != `tool[1]={"type":"ephemeral","ttl":"1h"}` {
		t.Fatalf("dispatched controls=%v", got)
	}
}

// frameBindingModel is a host-style wrapper that binds an explicit plan to the
// exact request it receives (mode "fresh") or keeps reusing the first bound
// plan for every request (mode "reuse").
type frameBindingModel struct {
	mu    sync.Mutex
	inner *anthropic.Client
	reuse bool
	first *llm.CachePlan
}

func (m *frameBindingModel) Provider() string { return m.inner.Provider() }
func (m *frameBindingModel) Model() string    { return m.inner.Model() }
func (m *frameBindingModel) Invoke(ctx context.Context, request llm.InvokeRequest) (*llm.Completion, error) {
	m.mu.Lock()
	if m.reuse && m.first != nil {
		request.CachePlan = m.first
	} else {
		view, err := llm.NewCacheTargetView(request)
		if err != nil {
			m.mu.Unlock()
			return nil, err
		}
		last := len(request.Messages) - 1
		plan, err := view.Bind([]llm.CacheDirective{{Target: llm.CacheTarget{Kind: llm.CacheAfterMessage, MessageIndex: last}, Policy: llm.CacheRequired}})
		if err != nil {
			m.mu.Unlock()
			return nil, err
		}
		request.CachePlan = plan
		if m.first == nil {
			m.first = plan
		}
	}
	m.mu.Unlock()
	return m.inner.Invoke(ctx, request)
}

// Through a real Agent: the framework retry of one Frame sends the identical
// planned wire, the next logical Frame carries a plan bound to its own request,
// and a plan bound to an earlier request is rejected as stale before any I/O
// instead of being rebound or silently dropped.
func TestCachePlanAgentFrameRetryAndNextFrame(t *testing.T) {
	toolUse := `{"id":"m1","type":"message","role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"echo","input":{}}],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`
	final := admissionSuccess["anthropic"]
	for _, reuse := range []bool{false, true} {
		t.Run(fmt.Sprintf("reuse=%v", reuse), func(t *testing.T) {
			var payloads [][]byte
			client := anthropicDispatchClient(func(r *http.Request) (*http.Response, error) {
				data, _ := io.ReadAll(r.Body)
				payloads = append(payloads, data)
				switch len(payloads) {
				case 1:
					return fixtureResponse(500, `{"error":{"type":"api_error","message":"fixture failure"}}`, r), nil
				case 2:
					return fixtureResponse(200, toolUse, r), nil
				default:
					return fixtureResponse(200, final, r), nil
				}
			}, nil)
			client.MaxRetries = 1 // Leave retries to the Agent's Frame-level retry.
			model := &frameBindingModel{inner: client, reuse: reuse}
			echo := tools.Func[struct{}]("echo", "fixture", func(context.Context, struct{}, *tools.Container) (any, error) { return "echoed", nil })
			ag, err := agent.New(agent.Config{LLM: model, Tools: []tools.Tool{echo}, InvokeRetryBackoff: time.Millisecond, Warningf: func(string, ...any) {}})
			if err != nil {
				t.Fatal(err)
			}
			_, err = ag.Query(context.Background(), "hello")
			if len(payloads) < 2 || !bytes.Equal(payloads[0], payloads[1]) {
				t.Fatalf("same-Frame retry changed the wire (attempts=%d)", len(payloads))
			}
			if got := wireCacheControls(t, payloads[0]); len(got) != 1 {
				t.Fatalf("first Frame controls=%v", got)
			}
			if !reuse {
				if err != nil || len(payloads) != 3 {
					t.Fatalf("fresh plans: err=%v attempts=%d", err, len(payloads))
				}
				first, next := wireCacheControls(t, payloads[1]), wireCacheControls(t, payloads[2])
				if len(next) != 1 || next[0] == first[0] {
					t.Fatalf("next Frame did not carry its own plan: first=%v next=%v", first, next)
				}
				return
			}
			// Agent.Query reports the terminal as text; the fixed reason is the
			// contract here (the typed error is asserted at the client boundary).
			if err == nil || !strings.Contains(err.Error(), "cache plan: stale_request") {
				t.Fatalf("reused plan err=%v, want stale_request", err)
			}
			stale := dispatchPlanRequest()
			bindToolCache(t, &stale, dispatchDirectives())
			stale.Messages = append(stale.Messages, llm.Message{Role: llm.RoleUser, Content: llm.TextContent("later")})
			var planErr *llm.CachePlanValidationError
			if _, clientErr := client.Invoke(context.Background(), stale); !errors.As(clientErr, &planErr) || planErr.Reason != "stale_request" {
				t.Fatalf("client err=%v, want typed stale_request", clientErr)
			}
			if len(payloads) != 2 {
				t.Fatalf("stale plan reached the network: attempts=%d", len(payloads))
			}
		})
	}
}

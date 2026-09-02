package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"reflect"
	"sync"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestNewOwnsNestedToolSchemas(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path": map[string]any{"type": "string"},
		},
	}
	agent, err := New(Config{
		LLM: historyCloneModel{},
		Tools: []tools.Tool{{
			Name:   "read",
			Schema: schema,
			Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
				return llm.TextContent("ok"), nil
			},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	schema["properties"].(map[string]any)["path"].(map[string]any)["type"] = "number"

	got := agent.toolMap["read"].Schema["properties"].(map[string]any)["path"].(map[string]any)["type"]
	if got != "string" {
		t.Fatalf("agent tool schema changed through caller-owned map: %#v", got)
	}
}

type requestMutatingRetryModel struct {
	calls int
	err   error
}

func (m *requestMutatingRetryModel) Provider() string { return "stub" }
func (m *requestMutatingRetryModel) Model() string    { return "stub" }

func (m *requestMutatingRetryModel) Invoke(_ context.Context, request llm.InvokeRequest) (*llm.Completion, error) {
	m.calls++
	if m.calls == 1 {
		request.Messages[0].Content.Text = "mutated"
		request.Tools[0].Parameters["type"] = "array"
		request.Responses.OutputSchema["type"] = "array"
		return nil, &net.DNSError{Err: "i/o timeout", IsTimeout: true}
	}
	if got := request.Messages[0].Content.Text; got != "original" {
		m.err = fmt.Errorf("retry message = %q", got)
	}
	if got := request.Tools[0].Parameters["type"]; got != "object" {
		m.err = fmt.Errorf("retry tool schema = %#v", got)
	}
	if got := request.Responses.OutputSchema["type"]; got != "object" {
		m.err = fmt.Errorf("retry response schema = %#v", got)
	}
	return &llm.Completion{Content: llm.TextContent("ok"), StopReason: "stop"}, nil
}

func TestInvokeRetryUsesFreshRequestClone(t *testing.T) {
	model := &requestMutatingRetryModel{}
	agent, err := New(Config{LLM: model, InvokeRetryMaxAttempts: 2})
	if err != nil {
		t.Fatal(err)
	}
	request := llm.InvokeRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: llm.TextContent("original")}},
		Tools:    []llm.ToolDefinition{{Name: "read", Parameters: map[string]any{"type": "object"}}},
		Responses: &llm.ResponsesOptions{
			OutputSchema: map[string]any{"type": "object"},
		},
	}
	if _, _, err := agent.invokeCompletionWithRetry(context.Background(), request, wrapLegacyEventOutput(make(chan Event, 8))); err != nil {
		t.Fatal(err)
	}
	if model.calls != 2 {
		t.Fatalf("invoke calls = %d, want 2", model.calls)
	}
	if model.err != nil {
		t.Fatal(model.err)
	}
}

type providerAdmissionModel struct {
	name       string
	retryFirst bool
	onFirst    func()

	mu       sync.Mutex
	requests []llm.InvokeRequest
}

func (m *providerAdmissionModel) Provider() string { return "fixture" }
func (m *providerAdmissionModel) Model() string    { return m.name }
func (m *providerAdmissionModel) Invoke(_ context.Context, request llm.InvokeRequest) (*llm.Completion, error) {
	snapshot, err := llm.CloneInvokeRequest(request)
	if err != nil {
		return nil, err
	}
	m.mu.Lock()
	m.requests = append(m.requests, snapshot)
	call := len(m.requests)
	m.mu.Unlock()
	if call == 1 && m.onFirst != nil {
		m.onFirst()
	}
	if call == 1 && m.retryFirst {
		request.Messages[0].Content.Blocks[0].Text = "mutated"
		request.Tools[1].Parameters["limit"] = int64(99)
		request.Tools[1].Parameters["items"].([]any)[0] = "mutated"
		*request.Temperature = 9
		*request.Responses.UseResponseItems = false
		*request.Responses.UseInstructions = true
		request.Responses.Instructions = "mutated"
		request.Responses.Include[0] = "mutated"
		*request.Responses.ParallelToolCalls = true
		*request.Responses.Store = true
		request.Responses.Text.Verbosity = "mutated"
		request.Responses.Text.Format.Name = "mutated"
		request.Responses.Text.Format.Schema["required"].([]any)[0] = "mutated"
		request.Responses.Reasoning.Effort = "mutated"
		request.Responses.Reasoning.Summary = "mutated"
		request.Responses.OutputSchema["type"] = "mutated"
		return nil, &net.DNSError{Err: "i/o timeout", IsTimeout: true}
	}
	return &llm.Completion{Content: llm.TextContent(m.name), StopReason: "stop"}, nil
}

func (m *providerAdmissionModel) snapshots() []llm.InvokeRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]llm.InvokeRequest(nil), m.requests...)
}

type dynamicProviderAdmissionModel struct {
	mu      sync.RWMutex
	current llm.ChatModel
	calls   int
}

func (m *dynamicProviderAdmissionModel) Provider() string { return "fixture" }
func (m *dynamicProviderAdmissionModel) Model() string    { return "dynamic" }
func (m *dynamicProviderAdmissionModel) Invoke(ctx context.Context, request llm.InvokeRequest) (*llm.Completion, error) {
	m.mu.Lock()
	current := m.current
	m.calls++
	m.mu.Unlock()
	return current.Invoke(ctx, request)
}
func (m *dynamicProviderAdmissionModel) swap(next llm.ChatModel) {
	m.mu.Lock()
	m.current = next
	m.mu.Unlock()
}
func (m *dynamicProviderAdmissionModel) callCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.calls
}

func TestFrameworkRetryCapturesOuterModelInterface(t *testing.T) {
	request := providerAdmissionRequest()
	primary := &providerAdmissionModel{name: "captured", retryFirst: true}
	replacement := &providerAdmissionModel{name: "replacement"}
	agent, err := New(Config{LLM: primary, InvokeRetryMaxAttempts: 2})
	if err != nil {
		t.Fatal(err)
	}
	primary.onFirst = func() { agent.llm = replacement }
	completion, _, err := agent.invokeCompletionWithRetry(context.Background(), request, wrapLegacyEventOutput(make(chan Event, 8)))
	if err != nil || completion == nil || completion.PlainText() != "captured" {
		t.Fatalf("completion=%#v err=%v", completion, err)
	}
	assertProviderAdmissionRequests(t, primary.snapshots(), providerAdmissionRequest(), 2)
	if got := len(replacement.snapshots()); got != 0 {
		t.Fatalf("replacement calls during retry=%d want 0", got)
	}
	next, _, err := agent.invokeCompletionWithRetry(context.Background(), request, wrapLegacyEventOutput(make(chan Event, 8)))
	if err != nil || next == nil || next.PlainText() != "replacement" {
		t.Fatalf("next admission completion=%#v err=%v", next, err)
	}
	assertProviderAdmissionRequests(t, replacement.snapshots(), providerAdmissionRequest(), 1)
}

func TestFrameworkRetryDoesNotSnapshotDynamicModelTarget(t *testing.T) {
	request := providerAdmissionRequest()
	oldModel := &providerAdmissionModel{name: "old", retryFirst: true}
	newModel := &providerAdmissionModel{name: "new"}
	dynamic := &dynamicProviderAdmissionModel{current: oldModel}
	oldModel.onFirst = func() { dynamic.swap(newModel) }
	agent, err := New(Config{LLM: dynamic, InvokeRetryMaxAttempts: 2})
	if err != nil {
		t.Fatal(err)
	}
	completion, _, err := agent.invokeCompletionWithRetry(context.Background(), request, wrapLegacyEventOutput(make(chan Event, 8)))
	if err != nil || completion == nil || completion.PlainText() != "new" {
		t.Fatalf("completion=%#v err=%v want new", completion, err)
	}
	assertProviderAdmissionRequests(t, oldModel.snapshots(), providerAdmissionRequest(), 1)
	assertProviderAdmissionRequests(t, newModel.snapshots(), providerAdmissionRequest(), 1)
	if got := dynamic.callCount(); got != 2 {
		t.Fatalf("outer model calls=%d want 2", got)
	}
}

func providerAdmissionRequest() llm.InvokeRequest {
	temperature := 0.25
	yes, no := true, false
	return llm.InvokeRequest{
		Messages: []llm.Message{{
			Role:    llm.RoleUser,
			Name:    "fixture",
			Cache:   true,
			Content: llm.Content{Blocks: []llm.ContentBlock{{Type: "text", Text: "original"}}},
		}},
		Tools: []llm.ToolDefinition{
			{Name: "nil-schema", Parameters: nil},
			{Name: "typed-schema", Strict: true, Parameters: map[string]any{"type": "object", "limit": int64(7), "items": []any{"stable"}, "empty": []any{}}},
		},
		ToolChoice:      llm.ToolChoice("required"),
		Temperature:     &temperature,
		DisableThinking: true,
		Responses: &llm.ResponsesOptions{
			UseResponseItems:  &yes,
			UseInstructions:   &no,
			Instructions:      "stable",
			ConversationID:    "conversation",
			PromptCacheKey:    "cache",
			Include:           []string{"trace"},
			ParallelToolCalls: &no,
			Store:             &no,
			Text: &llm.ResponsesTextControls{
				Verbosity: "low",
				Format: &llm.ResponsesTextFormat{
					Type:   "json_schema",
					Strict: true,
					Name:   "answer",
					Schema: map[string]any{"type": "object", "required": []any{"answer"}},
				},
			},
			Reasoning:    &llm.ResponsesReasoning{Effort: "low", Summary: "auto"},
			Verbosity:    "medium",
			OutputSchema: map[string]any{},
		},
	}
}

func assertProviderAdmissionRequests(t *testing.T, got []llm.InvokeRequest, want llm.InvokeRequest, count int) {
	t.Helper()
	if len(got) != count {
		t.Fatalf("provider requests=%d want %d", len(got), count)
	}
	for i, request := range got {
		if !reflect.DeepEqual(request, want) {
			t.Fatalf("request[%d]\n got: %#v\nwant: %#v", i, request, want)
		}
	}
}

type providerAdmissionNoopModel struct{}

func (providerAdmissionNoopModel) Provider() string { return "fixture" }
func (providerAdmissionNoopModel) Model() string    { return "noop" }
func (providerAdmissionNoopModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	return &llm.Completion{StopReason: "stop"}, nil
}

func BenchmarkProviderAdmissionSnapshot(b *testing.B) {
	model := providerAdmissionNoopModel{}
	agent, err := New(Config{LLM: model, InvokeRetryMaxAttempts: 1})
	if err != nil {
		b.Fatal(err)
	}
	request := providerAdmissionRequest()
	out := wrapLegacyEventOutput(make(chan Event, 1))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := agent.invokeModelCompletionWithSteering(context.Background(), model, request, out, nil); err != nil {
			b.Fatal(err)
		}
	}
}

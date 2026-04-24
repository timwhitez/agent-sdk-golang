package agent

import (
	"context"
	"encoding/json"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type stubModel struct {
	calls    int
	toolName string
	toolArgs string
	toolID   string
}

func (s *stubModel) Provider() string { return "stub" }
func (s *stubModel) Model() string    { return "stub" }

func (s *stubModel) Invoke(_ context.Context, _ llm.InvokeRequest) (*llm.Completion, error) {
	s.calls++
	if s.calls == 1 {
		return &llm.Completion{
			Content: llm.TextContent(""),
			ToolCalls: []llm.ToolCall{
				{
					ID:   s.toolID,
					Type: "function",
					Function: llm.FunctionCall{
						Name:      s.toolName,
						Arguments: s.toolArgs,
					},
				},
			},
		}, nil
	}
	return &llm.Completion{Content: llm.TextContent("done")}, nil
}

type captureModel struct {
	calls    int
	toolName string
	toolArgs string
	toolID   string
	toolDefs [][]string
}

func (m *captureModel) Provider() string { return "stub" }
func (m *captureModel) Model() string    { return "stub" }

func (m *captureModel) Invoke(_ context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	names := make([]string, 0, len(req.Tools))
	for _, t := range req.Tools {
		names = append(names, t.Name)
	}
	m.toolDefs = append(m.toolDefs, names)

	m.calls++
	if m.calls == 1 {
		return &llm.Completion{
			Content: llm.TextContent(""),
			ToolCalls: []llm.ToolCall{
				{
					ID:   m.toolID,
					Type: "function",
					Function: llm.FunctionCall{
						Name:      m.toolName,
						Arguments: m.toolArgs,
					},
				},
			},
		}, nil
	}
	return &llm.Completion{Content: llm.TextContent("done")}, nil
}

func collectEvents(ch <-chan Event) []Event {
	events := []Event{}
	for ev := range ch {
		events = append(events, ev)
	}
	return events
}

func findToolCall(events []Event, name string) (ToolCallEvent, bool) {
	for _, ev := range events {
		if tc, ok := ev.(ToolCallEvent); ok {
			if tc.Tool == name {
				return tc, true
			}
		}
	}
	return ToolCallEvent{}, false
}

func findToolResult(events []Event, name string) (ToolResultEvent, bool) {
	for _, ev := range events {
		if tr, ok := ev.(ToolResultEvent); ok {
			if tr.Tool == name {
				return tr, true
			}
		}
	}
	return ToolResultEvent{}, false
}

func containsString(list []string, target string) bool {
	for _, item := range list {
		if item == target {
			return true
		}
	}
	return false
}

func assertSeverityActionDiagnostic(t *testing.T, text string) {
	t.Helper()
	trimmed := strings.TrimSpace(text)
	if !strings.HasPrefix(trimmed, "[ERROR] ") {
		t.Fatalf("expected [ERROR] diagnostic prefix, got %q", text)
	}
	if !strings.Contains(trimmed, " - ") {
		t.Fatalf("expected actionable summary delimiter, got %q", text)
	}
}

func TestAutoInvalidTool_UsesSeverityActionDiagnostic(t *testing.T) {
	ctx := tools.WithToolResultMetadata(context.Background())
	out, err := autoInvalidTool().Execute(ctx, `{"tool":"mystery"}`, tools.NewContainer())
	if err == nil {
		t.Fatalf("expected unknown-tool error")
	}
	assertSeverityActionDiagnostic(t, out.PlainText())
	meta := tools.ToolResultMetadataSnapshot(ctx)
	if meta == nil {
		t.Fatalf("expected metadata on unknown tool")
	}
	if kind, _ := meta["error_kind"].(string); kind != "tool_not_found" {
		t.Fatalf("expected error_kind=tool_not_found, got %#v", meta["error_kind"])
	}
	if got, _ := meta["tool"].(string); got != "mystery" {
		t.Fatalf("expected tool metadata mystery, got %#v", meta["tool"])
	}
}

func TestAgentResolvesAliasToolName(t *testing.T) {
	called := atomic.Bool{}
	lsTool := tools.Tool{
		Name: "ls",
		Handler: func(_ context.Context, _ json.RawMessage, _ *tools.Container) (llm.Content, error) {
			called.Store(true)
			return llm.TextContent("ok"), nil
		},
	}
	model := &stubModel{toolName: "list", toolArgs: `{}`}
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{lsTool}})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hi")))
	if !called.Load() {
		t.Fatalf("expected ls tool to be called")
	}
	if _, ok := findToolCall(events, "ls"); !ok {
		t.Fatalf("expected tool call event for ls")
	}
	if tr, ok := findToolResult(events, "ls"); !ok || tr.IsError {
		t.Fatalf("expected successful tool result for ls")
	}
}

func TestAgentUnknownToolFallsBackToInvalid(t *testing.T) {
	invalidArgs := make(chan map[string]any, 1)
	invalidTool := tools.Tool{
		Name: "invalid",
		Handler: func(_ context.Context, raw json.RawMessage, _ *tools.Container) (llm.Content, error) {
			m := map[string]any{}
			_ = json.Unmarshal(raw, &m)
			invalidArgs <- m
			return llm.TextContent("invalid"), nil
		},
	}
	model := &stubModel{toolName: "mystery", toolArgs: `{"x":1}`}
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{invalidTool}})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hi")))
	if _, ok := findToolCall(events, "invalid"); !ok {
		t.Fatalf("expected tool call event for invalid")
	}
	tr, ok := findToolResult(events, "invalid")
	if !ok {
		t.Fatalf("expected tool result event for invalid")
	}
	if !tr.IsError {
		t.Fatalf("expected invalid tool result to be marked error")
	}
	if tr.Metadata == nil || tr.Metadata["error_kind"] != "tool_not_found" {
		t.Fatalf("expected error_kind metadata for unknown tool")
	}
	if tr.Metadata["tool"] != "mystery" {
		t.Fatalf("expected metadata tool to be mystery, got %v", tr.Metadata["tool"])
	}
	select {
	case m := <-invalidArgs:
		if m["tool"] != "mystery" {
			t.Fatalf("expected invalid tool args to include tool=mystery")
		}
	default:
		t.Fatalf("expected invalid tool to be called")
	}
}

func TestAgentUnknownToolAutoInvalidHidden(t *testing.T) {
	visibleTool := tools.Tool{
		Name: "echo",
		Handler: func(_ context.Context, _ json.RawMessage, _ *tools.Container) (llm.Content, error) {
			return llm.TextContent("ok"), nil
		},
	}
	model := &captureModel{toolName: "mystery", toolArgs: `{"x":1}`}

	ag, err := New(Config{LLM: model, Tools: []tools.Tool{visibleTool}})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hi")))

	if len(model.toolDefs) == 0 {
		t.Fatalf("expected tool definitions to be captured")
	}
	if containsString(model.toolDefs[0], "invalid") {
		t.Fatalf("expected invalid tool to be hidden from tool definitions")
	}
	if !containsString(model.toolDefs[0], "echo") {
		t.Fatalf("expected visible tool to be exposed to model, got %v", model.toolDefs[0])
	}

	if _, ok := findToolCall(events, "invalid"); !ok {
		t.Fatalf("expected tool call event for invalid")
	}
	tr, ok := findToolResult(events, "invalid")
	if !ok {
		t.Fatalf("expected tool result event for invalid")
	}
	if !tr.IsError {
		t.Fatalf("expected invalid tool result to be marked error")
	}
	if tr.Metadata == nil || tr.Metadata["error_kind"] != "tool_not_found" {
		t.Fatalf("expected error_kind metadata for unknown tool")
	}
	if tr.Metadata["tool"] != "mystery" {
		t.Fatalf("expected metadata tool to be mystery, got %v", tr.Metadata["tool"])
	}
	assertSeverityActionDiagnostic(t, tr.Result)
}

func TestAgentExplicitHiddenToolExcludedFromToolDefinitions(t *testing.T) {
	visibleTool := tools.Tool{
		Name: "echo",
		Handler: func(_ context.Context, _ json.RawMessage, _ *tools.Container) (llm.Content, error) {
			return llm.TextContent("ok"), nil
		},
	}
	hiddenTool := tools.InvalidTool()
	model := &captureModel{calls: 1}

	ag, err := New(Config{LLM: model, Tools: []tools.Tool{visibleTool, hiddenTool}})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	_ = collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hi")))

	if len(model.toolDefs) == 0 {
		t.Fatalf("expected tool definitions to be captured")
	}
	if containsString(model.toolDefs[0], "invalid") {
		t.Fatalf("expected explicitly configured hidden tool to be excluded from model-visible definitions, got %v", model.toolDefs[0])
	}
	if !containsString(model.toolDefs[0], "echo") {
		t.Fatalf("expected visible tool to be exposed to model, got %v", model.toolDefs[0])
	}
}

func TestQueryStream_UnknownToolResultUsesSeverityActionDiagnostic(t *testing.T) {
	model := &stubModel{toolName: "mystery", toolArgs: `{"x":1}`}
	ag, err := New(Config{LLM: model, Tools: nil})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hi")))
	tr, ok := findToolResult(events, "invalid")
	if !ok {
		t.Fatalf("expected invalid tool result event")
	}
	if !tr.IsError {
		t.Fatalf("expected unknown tool to be marked error")
	}
	if tr.Metadata == nil || tr.Metadata["error_kind"] != "tool_not_found" {
		t.Fatalf("expected error_kind metadata for unknown tool")
	}
	if tr.Metadata["tool"] != "mystery" {
		t.Fatalf("expected metadata tool to be mystery, got %v", tr.Metadata["tool"])
	}
	assertSeverityActionDiagnostic(t, tr.Result)
}

func TestBuildNormalizedToolMapStableCollision(t *testing.T) {
	first := tools.Tool{Name: "web_search"}
	second := tools.Tool{Name: "websearch"}
	toolMap := map[string]tools.Tool{
		first.Name:  first,
		second.Name: second,
	}
	norm := tools.NormalizeToolName(first.Name)
	if norm == "" || norm != tools.NormalizeToolName(second.Name) {
		t.Fatalf("expected normalization collision")
	}

	ordered := []tools.Tool{first, second}
	normMap := buildNormalizedToolMap(toolMap, ordered)
	if got := normMap[norm].Name; got != first.Name {
		t.Fatalf("expected first tool to win, got %q", got)
	}

	ordered = []tools.Tool{second, first}
	normMap = buildNormalizedToolMap(toolMap, ordered)
	if got := normMap[norm].Name; got != second.Name {
		t.Fatalf("expected second tool to win, got %q", got)
	}
}

func TestAgentSynthesizesToolCallIDWhenMissing(t *testing.T) {
	called := atomic.Bool{}
	writeTool := tools.Tool{
		Name: "write",
		Handler: func(_ context.Context, _ json.RawMessage, _ *tools.Container) (llm.Content, error) {
			called.Store(true)
			return llm.TextContent("ok"), nil
		},
	}
	model := &stubModel{toolName: "write", toolArgs: `{}`, toolID: ""}
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{writeTool}})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("hi")))
	if !called.Load() {
		t.Fatalf("expected tool handler to be invoked")
	}
	call, ok := findToolCall(events, "write")
	if !ok {
		t.Fatalf("expected tool call event")
	}
	if !strings.HasPrefix(call.ToolCallID, syntheticToolCallIDPrefix) {
		t.Fatalf("tool call id = %q, want synthetic prefix %q", call.ToolCallID, syntheticToolCallIDPrefix)
	}
	result, ok := findToolResult(events, "write")
	if !ok {
		t.Fatalf("expected tool result event")
	}
	if result.ToolCallID != call.ToolCallID {
		t.Fatalf("tool result id = %q, want %q", result.ToolCallID, call.ToolCallID)
	}
	msgs := ag.Messages()
	if len(msgs) < 3 {
		t.Fatalf("expected assistant + tool history, got %d messages", len(msgs))
	}
	toolMsg := msgs[len(msgs)-2]
	if toolMsg.Role != llm.RoleTool {
		t.Fatalf("expected penultimate message to be tool, got %q", toolMsg.Role)
	}
	if toolMsg.ToolCallID != call.ToolCallID {
		t.Fatalf("tool history id = %q, want %q", toolMsg.ToolCallID, call.ToolCallID)
	}
}

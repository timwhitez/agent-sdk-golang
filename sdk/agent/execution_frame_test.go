package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type frameScriptModel struct {
	invoke func(llm.InvokeRequest) (*llm.Completion, error)
}

func (*frameScriptModel) Provider() string { return "fixture" }
func (*frameScriptModel) Model() string    { return "frame" }
func (m *frameScriptModel) Invoke(_ context.Context, req llm.InvokeRequest) (*llm.Completion, error) {
	return m.invoke(req)
}

func TestExecutionFrameOwnsRequestAndResolverSchemas(t *testing.T) {
	request := providerAdmissionRequest()
	want, err := llm.CloneInvokeRequest(request)
	if err != nil {
		t.Fatal(err)
	}
	model := &providerAdmissionModel{name: "outer"}
	handlerCalls := 0
	tool := tools.Tool{Name: "read_file", Schema: map[string]any{"nested": map[string]any{"value": int64(7)}}, Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
		handlerCalls++
		return llm.TextContent("ok"), nil
	}}
	exact := map[string]tools.Tool{tool.Name: tool}
	normalized := buildNormalizedToolMap(exact, []tools.Tool{tool})
	frame, err := newExecutionFrame(model, request, exact, normalized)
	if err != nil {
		t.Fatal(err)
	}
	request.Messages[0].Content.Blocks[0].Text = "changed"
	request.Tools[1].Parameters["limit"] = int64(999)
	tool.Schema["nested"].(map[string]any)["value"] = int64(999)
	delete(exact, tool.Name)
	delete(normalized, "read")
	if frame.model != model || !reflect.DeepEqual(frame.request, want) {
		t.Fatal("frame lost model/request ownership")
	}
	for _, registry := range []map[string]tools.Tool{frame.exact, frame.normalized} {
		for _, owned := range registry {
			if owned.Schema["nested"].(map[string]any)["value"] != int64(7) {
				t.Fatal("schema aliased source")
			}
		}
	}
	resolved, name, found, alias := resolveToolByName("read", frame.exact, frame.normalized)
	if !found || !alias || name != "read_file" {
		t.Fatalf("alias result=%q/%v/%v", name, found, alias)
	}
	if _, err := resolved.Handler(context.Background(), nil, nil); err != nil || handlerCalls != 1 {
		t.Fatal("runtime handle not retained")
	}
	frame.exact[tool.Name].Schema["nested"].(map[string]any)["value"] = int64(8)
	if frame.normalized["read"].Schema["nested"].(map[string]any)["value"] != int64(7) {
		t.Fatal("resolver maps share schema ownership")
	}
}

func TestExecutionFrameResolutionCompatibilityAndSafeDiagnostics(t *testing.T) {
	var warnings []string
	a, err := New(Config{LLM: historyCloneModel{}, Tools: []tools.Tool{
		{Name: "read_file", Description: "secret-schema-marker", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
			return llm.TextContent("one"), nil
		}},
		{Name: "ReadFile", Hidden: true},
	}, Warningf: func(format string, args ...any) { warnings = append(warnings, fmt.Sprintf(format, args...)) }})
	if err != nil {
		t.Fatal(err)
	}
	request := llm.InvokeRequest{Tools: []llm.ToolDefinition{a.tools[0].Definition()}}
	frame, err := newExecutionFrame(a.llm, request, a.toolMap, a.toolMapNormalized)
	if err != nil {
		t.Fatal(err)
	}
	a.observeFrameAdvertisement(frame)
	for i, name := range []string{"read_file", "read", " READ_FILE ", "ReadFile", "invalid", "unknown-secret-name"} {
		tool, resolved, found, alias := a.resolveToolByName(name)
		if !found {
			var ok bool
			tool, ok = a.toolMap["invalid"]
			if !ok {
				tool = autoInvalidTool()
			}
			resolved = "invalid"
		}
		a.observeFrameResolution(frame, i, name, tool, resolved, found, alias)
	}
	// A distinct closure is not comparable identity evidence. Equal metadata
	// must not generate a false mismatch merely because Handler is non-nil.
	actual := a.toolMap["read_file"]
	actual.Handler = func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
		return llm.TextContent("two"), nil
	}
	a.observeFrameResolution(frame, 0, "read_file", actual, "read_file", true, false)
	registeredFallback := autoInvalidTool()
	registeredFallback.Description = "registered private fallback"
	a.toolMap["invalid"] = registeredFallback
	registeredFrame, err := newExecutionFrame(a.llm, request, a.toolMap, a.toolMapNormalized)
	if err != nil {
		t.Fatal(err)
	}
	a.observeFrameResolution(registeredFrame, 0, "unknown", registeredFallback, "invalid", false, false)
	a.observeFrameResolution(registeredFrame, 0, "invalid", registeredFallback, "invalid", true, false)
	if len(warnings) != 0 {
		t.Fatalf("false warnings: %v", warnings)
	}
	actual.Description = "other-secret-marker"
	a.observeFrameResolution(frame, 3, "read_file", actual, "read_file", true, false)
	frame.request.Tools[0].Description = "private-prompt-marker"
	a.observeFrameAdvertisement(frame)
	want := []string{"warning: execution frame shadow mismatch: resolved_tool[3]", "warning: execution frame shadow mismatch: advertised_tool[0]"}
	if !reflect.DeepEqual(warnings, want) {
		t.Fatalf("unsafe/unexpected diagnostics: %v", warnings)
	}
}

func TestExecutionFrameShadowFailureDoesNotTakeAuthority(t *testing.T) {
	for _, cancelOnWarning := range []bool{false, true} {
		t.Run(fmt.Sprint(cancelOnWarning), func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			calls, handled := 0, 0
			model := &frameScriptModel{invoke: func(req llm.InvokeRequest) (*llm.Completion, error) {
				calls++
				if len(req.Tools) != 1 || req.Tools[0].Name != "work" {
					t.Error("shadow altered provider tools")
				}
				return &llm.Completion{ToolCalls: []llm.ToolCall{{ID: "a", Function: llm.FunctionCall{Name: "work", Arguments: "{}"}}}}, nil
			}}
			var warnings []string
			a, err := New(Config{LLM: model, Tools: []tools.Tool{{Name: "work", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
				handled++
				return llm.Content{}, tools.TaskComplete("legacy result")
			}}}, Warningf: func(format string, args ...any) {
				warnings = append(warnings, fmt.Sprintf(format, args...))
				if cancelOnWarning {
					cancel()
				}
			}})
			if err != nil {
				t.Fatal(err)
			}
			// Hidden runtime-only corruption cannot affect the old request clone.
			a.toolMap["hidden"] = tools.Tool{Hidden: true, Schema: map[string]any{"private-marker": func() {}}}
			for range a.QueryStream(ctx, llm.TextContent("private prompt")) {
			}
			wantCalls := 1
			if cancelOnWarning {
				wantCalls = 0
			}
			if calls != wantCalls || handled != wantCalls {
				t.Fatalf("provider/handler=%d/%d want %d", calls, handled, wantCalls)
			}
			if !reflect.DeepEqual(warnings, []string{"warning: execution frame shadow snapshot unavailable"}) {
				t.Fatalf("warnings=%v", warnings)
			}
		})
	}
}

func TestExecutionFrameContinuationUsesFinalizingRequest(t *testing.T) {
	var a *Agent
	var warnings []string
	calls, handled := 0, 0
	setDescription := func(description string, advertised bool) {
		tool := a.toolMap["work"]
		tool.Description = description
		a.toolMap["work"] = tool
		a.toolMapNormalized = buildNormalizedToolMap(a.toolMap, a.tools)
		if advertised {
			a.tools[0] = tool
		}
	}
	model := &frameScriptModel{invoke: func(req llm.InvokeRequest) (*llm.Completion, error) {
		calls++
		if calls == 1 {
			setDescription("finalizing", true)
			a.toolChoice = "required"
			return &llm.Completion{StopReason: "max_tokens", ToolCalls: []llm.ToolCall{{ID: "a", Function: llm.FunctionCall{Name: "work", Arguments: `{"text":`}}}}, nil
		}
		if calls != 2 {
			t.Error("unexpected extra admission")
		}
		if req.Tools[0].Description != "finalizing" || req.ToolChoice != "required" {
			t.Error("finalizing request not refreshed")
		}
		// Restore the first frame's metadata. Only the finalizing frame can
		// detect this drift; dispatch deliberately remains the legacy path.
		setDescription("initial", false)
		return &llm.Completion{StopReason: "tool_calls", ToolCalls: []llm.ToolCall{{ID: "a", Function: llm.FunctionCall{Name: "work", Arguments: `"ok"}`}}}}, nil
	}}
	var err error
	a, err = New(Config{LLM: model, ToolChoice: "auto", Tools: []tools.Tool{{Name: "work", Description: "initial", Handler: func(_ context.Context, args json.RawMessage, _ *tools.Container) (llm.Content, error) {
		handled++
		if !strings.Contains(string(args), "ok") {
			t.Errorf("merged args=%s", args)
		}
		return llm.Content{}, tools.TaskComplete("legacy result")
	}}}, Warningf: func(format string, args ...any) { warnings = append(warnings, fmt.Sprintf(format, args...)) }})
	if err != nil {
		t.Fatal(err)
	}
	for range a.QueryStream(context.Background(), llm.TextContent("continue")) {
	}
	if calls != 2 || handled != 1 {
		t.Fatalf("calls/handled=%d/%d", calls, handled)
	}
	var frameWarnings []string
	for _, warning := range warnings {
		if strings.Contains(warning, "execution frame") {
			frameWarnings = append(frameWarnings, warning)
		}
	}
	if !reflect.DeepEqual(frameWarnings, []string{"warning: execution frame shadow mismatch: resolved_tool[0]"}) {
		t.Fatalf("frame warnings=%v", frameWarnings)
	}
	if _, changed, _ := repairToolCallPairsDetailed(a.Messages()); changed {
		t.Fatal("shadow changed tool pair topology")
	}
}

func TestExecutionFrameResolutionDriftKeepsLegacyAuthorityAndCancellation(t *testing.T) {
	for _, cancelOnWarning := range []bool{false, true} {
		t.Run(fmt.Sprint(cancelOnWarning), func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var a *Agent
			providerCalls, oldCalls, newCalls := 0, 0, 0
			var warnings []string
			model := &frameScriptModel{invoke: func(llm.InvokeRequest) (*llm.Completion, error) {
				providerCalls++
				changed := a.toolMap["work"]
				changed.Description = "private changed definition"
				changed.Handler = func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
					newCalls++
					return llm.Content{}, tools.TaskComplete("new legacy result")
				}
				a.toolMap["work"] = changed
				return &llm.Completion{ToolCalls: []llm.ToolCall{{ID: "a", Function: llm.FunctionCall{Name: "work", Arguments: "{}"}}}}, nil
			}}
			var err error
			a, err = New(Config{LLM: model, Tools: []tools.Tool{{Name: "work", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
				oldCalls++
				return llm.Content{}, tools.TaskComplete("old shadow result")
			}}}, Warningf: func(format string, args ...any) {
				warnings = append(warnings, fmt.Sprintf(format, args...))
				if cancelOnWarning {
					cancel()
				}
			}})
			if err != nil {
				t.Fatal(err)
			}
			for range a.QueryStream(ctx, llm.TextContent("run")) {
			}
			wantNew := 1
			if cancelOnWarning {
				wantNew = 0
			}
			if providerCalls != 1 || oldCalls != 0 || newCalls != wantNew {
				t.Fatalf("provider/old/new calls=%d/%d/%d", providerCalls, oldCalls, newCalls)
			}
			if !reflect.DeepEqual(warnings, []string{"warning: execution frame shadow mismatch: resolved_tool[0]"}) {
				t.Fatalf("warnings=%v", warnings)
			}
			results := 0
			for _, message := range a.Messages() {
				if message.Role != llm.RoleTool {
					continue
				}
				results++
				if message.ToolCallID != "a" {
					t.Error("tool identity changed")
				}
				if !cancelOnWarning && !strings.Contains(message.Content.PlainText(), "new legacy result") {
					t.Error("shadow changed result")
				}
				if cancelOnWarning && message.Content.PlainText() != toolSkippedByCancellationText {
					t.Error("unstarted call not closed as cancellation")
				}
			}
			if results != 1 {
				t.Fatalf("terminal results=%d", results)
			}
		})
	}
}

func BenchmarkExecutionFrameSnapshot(b *testing.B) {
	a, err := New(Config{LLM: historyCloneModel{}, Tools: []tools.Tool{{Name: "read", Schema: map[string]any{"type": "object"}}}})
	if err != nil {
		b.Fatal(err)
	}
	request := providerAdmissionRequest()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := newExecutionFrame(a.llm, request, a.toolMap, a.toolMapNormalized); err != nil {
			b.Fatal(err)
		}
	}
}

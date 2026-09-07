package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
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
	request.CachePlan.RequestFingerprint = "changed"
	request.CachePlan.Directives[0].Target.MessageIndex = 999
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

func TestExecutionFrameBindings(t *testing.T) {
	a, err := New(Config{LLM: historyCloneModel{}, Tools: []tools.Tool{
		{Name: "read_file", Description: "private metadata"},
		{Name: "ReadFile", Hidden: true},
	}})
	if err != nil {
		t.Fatal(err)
	}
	request := llm.InvokeRequest{Tools: []llm.ToolDefinition{a.tools[0].Definition()}}
	frame, err := newExecutionFrame(a.llm, request, a.toolMap, a.toolMapNormalized)
	if err != nil {
		t.Fatal(err)
	}
	if !frame.validBindings() {
		t.Fatal("valid hidden/collision/alias binding rejected")
	}
	for _, name := range []string{"read_file", "read", " READ_FILE ", "ReadFile", "unknown", "invalid"} {
		want, wn, wf, wa := resolveToolByName(name, a.toolMap, a.toolMapNormalized)
		got, gn, gf, ga := resolveToolByName(name, frame.exact, frame.normalized)
		if !reflect.DeepEqual(want.Definition(), got.Definition()) || wn != gn || wf != gf || wa != ga {
			t.Errorf("resolution changed for %q", name)
		}
	}
	frame.request.Tools[0].Description = "different"
	if frame.validBindings() {
		t.Fatal("advertisement mismatch accepted")
	}
	frame.request.Tools[0] = frame.exact["read_file"].Definition()
	alias := frame.normalized["read"]
	alias.Description = "different"
	frame.normalized["read"] = alias
	if frame.validBindings() {
		t.Fatal("normalized binding mismatch accepted")
	}
}

func TestExecutionFrameFailureStopsBeforeAdmission(t *testing.T) {
	for _, corrupt := range []string{"request", "registry", "advertisement", "normalized"} {
		for _, cancelOnWarning := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/cancel=%v", corrupt, cancelOnWarning), func(t *testing.T) {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				calls, handled, errorsSeen := 0, 0, 0
				var warnings []string
				model := &frameScriptModel{invoke: func(llm.InvokeRequest) (*llm.Completion, error) {
					calls++
					return &llm.Completion{Content: llm.TextContent("unexpected")}, nil
				}}
				a, err := New(Config{LLM: model, Tools: []tools.Tool{{Name: "work", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
					handled++
					return llm.Content{}, nil
				}}}, Warningf: func(format string, args ...any) {
					warnings = append(warnings, fmt.Sprintf(format, args...))
					if cancelOnWarning {
						cancel()
					}
				}})
				if err != nil {
					t.Fatal(err)
				}
				switch corrupt {
				case "request":
					a.tools[0].Schema = map[string]any{"secret-marker": func() {}}
				case "registry":
					a.toolMap["hidden"] = tools.Tool{Name: "hidden", Hidden: true, Schema: map[string]any{"secret-marker": func() {}}}
				case "advertisement":
					a.tools[0].Description = "secret-marker"
				case "normalized":
					tool := a.toolMapNormalized["work"]
					tool.Description = "secret-marker"
					a.toolMapNormalized["work"] = tool
				}
				for envelope := range a.QueryStreamEnveloped(ctx, llm.TextContent("secret-marker")) {
					switch event := envelope.Event.(type) {
					case ErrorEvent:
						errorsSeen++
						want := "invalid_request"
						if cancelOnWarning {
							want = "canceled"
						}
						if event.Kind != want || envelope.Origin != EventOriginSDKDriver || strings.Contains(event.Message, "secret-marker") {
							t.Errorf("unsafe error or provenance: %#v / %s", event, envelope.Origin)
						}
					case ToolCallEvent, ToolResultEvent, StepStartEvent, StepCompleteEvent:
						t.Errorf("unexpected tool event %T", event)
					}
				}
				if calls != 0 || handled != 0 || errorsSeen != 1 {
					t.Fatalf("calls/handled/errors=%d/%d/%d", calls, handled, errorsSeen)
				}
				wantWarning := "warning: execution frame tool bindings inconsistent"
				if corrupt == "request" || corrupt == "registry" {
					wantWarning = "warning: execution frame snapshot unavailable"
				}
				if !reflect.DeepEqual(warnings, []string{wantWarning}) {
					t.Fatalf("warnings=%v", warnings)
				}
			})
		}
	}
}

func TestExecutionFrameDispatchUsesCapturedHandlers(t *testing.T) {
	for _, name := range []string{"work", " WORK ", "read", "hidden", "invalid", "unknown"} {
		for _, canceled := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/cancel=%v", name, canceled), func(t *testing.T) {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				var a *Agent
				providerCalls, capturedCalls, replacementCalls := 0, 0, 0
				captured := func(_ context.Context, args json.RawMessage, _ *tools.Container) (llm.Content, error) {
					capturedCalls++
					if name == "unknown" && !strings.Contains(string(args), "unknown") {
						t.Error("fallback args not wrapped")
					}
					return llm.Content{}, tools.TaskComplete("captured result")
				}
				model := &frameScriptModel{invoke: func(llm.InvokeRequest) (*llm.Completion, error) {
					providerCalls++
					for key, tool := range a.toolMap {
						tool.Description = "changed"
						tool.Handler = func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
							replacementCalls++
							return llm.Content{}, tools.TaskComplete("replacement result")
						}
						a.toolMap[key] = tool
					}
					a.toolMapNormalized = buildNormalizedToolMap(a.toolMap, a.tools)
					if canceled {
						cancel()
					}
					return &llm.Completion{ToolCalls: []llm.ToolCall{{ID: "a", Function: llm.FunctionCall{Name: name, Arguments: "{}"}}}}, nil
				}}
				var err error
				a, err = New(Config{LLM: model, Tools: []tools.Tool{
					{Name: "work", Handler: captured}, {Name: "read_file", Handler: captured},
					{Name: "hidden", Hidden: true, Handler: captured}, {Name: "invalid", Hidden: true, Handler: captured},
				}})
				if err != nil {
					t.Fatal(err)
				}
				for range a.QueryStream(ctx, llm.TextContent("run")) {
				}
				wantCalls := 1
				if canceled {
					wantCalls = 0
				}
				if providerCalls != 1 || capturedCalls != wantCalls || replacementCalls != 0 {
					t.Fatalf("provider/captured/replacement=%d/%d/%d", providerCalls, capturedCalls, replacementCalls)
				}
				results := 0
				for _, message := range a.Messages() {
					if message.Role != llm.RoleTool {
						continue
					}
					results++
					want := "captured result"
					if canceled {
						want = toolSkippedByCancellationText
					}
					if message.ToolCallID != "a" || !strings.Contains(message.Content.PlainText(), want) {
						t.Error("wrong terminal result")
					}
				}
				if results != wantCalls {
					t.Fatalf("results=%d", results)
				}
			})
		}
	}
}

func TestExecutionFrameRetryOwnsLogicalRequest(t *testing.T) {
	var a *Agent
	calls, replacementCalls := 0, 0
	replacement := &frameScriptModel{invoke: func(llm.InvokeRequest) (*llm.Completion, error) {
		replacementCalls++
		return &llm.Completion{}, nil
	}}
	model := &frameScriptModel{invoke: func(req llm.InvokeRequest) (*llm.Completion, error) {
		calls++
		if req.Tools[0].Parameters["value"] != int64(7) {
			t.Error("logical request changed across attempts")
		}
		if calls == 1 {
			a.tools[0].Schema["value"] = int64(99)
			req.Tools[0].Parameters["value"] = int64(88)
			a.llm = replacement
			return nil, &net.DNSError{Err: "fixture timeout", IsTimeout: true}
		}
		return &llm.Completion{Content: llm.TextContent("done")}, nil
	}}
	var err error
	a, err = New(Config{LLM: model, InvokeRetryMaxAttempts: 2, Tools: []tools.Tool{{Name: "work", Schema: map[string]any{"value": int64(7)}}}})
	if err != nil {
		t.Fatal(err)
	}
	for range a.QueryStream(context.Background(), llm.TextContent("run")) {
	}
	if calls != 2 || replacementCalls != 0 {
		t.Fatalf("calls/replacement=%d/%d", calls, replacementCalls)
	}
}

func TestExecutionFrameUnregisteredInvalidStaysInternal(t *testing.T) {
	var a *Agent
	calls, injected := 0, 0
	model := &frameScriptModel{invoke: func(llm.InvokeRequest) (*llm.Completion, error) {
		calls++
		if calls == 1 {
			a.toolMap["invalid"] = tools.Tool{Name: "invalid", Hidden: true, Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
				injected++
				return llm.Content{}, tools.TaskComplete("wrong fallback")
			}}
			return &llm.Completion{ToolCalls: []llm.ToolCall{{ID: "a", Function: llm.FunctionCall{Name: "unknown", Arguments: "{}"}}}}, nil
		}
		return &llm.Completion{Content: llm.TextContent("done")}, nil
	}}
	var err error
	a, err = New(Config{LLM: model})
	if err != nil {
		t.Fatal(err)
	}
	for range a.QueryStream(context.Background(), llm.TextContent("run")) {
	}
	if calls != 2 || injected != 0 {
		t.Fatalf("provider/injected=%d/%d", calls, injected)
	}
	results := 0
	for _, message := range a.Messages() {
		if message.Role == llm.RoleTool {
			results++
			if message.ToolCallID != "a" || !message.IsError {
				t.Error("internal fallback result changed")
			}
		}
	}
	if results != 1 {
		t.Fatalf("results=%d", results)
	}
}

func TestExecutionFrameContinuationFinalizingAuthority(t *testing.T) {
	for _, failure := range []string{"none", "snapshot", "binding"} {
		t.Run(failure, func(t *testing.T) {
			var a *Agent
			calls, first, last, errorsSeen := 0, 0, 0, 0
			model := &frameScriptModel{invoke: func(req llm.InvokeRequest) (*llm.Completion, error) {
				calls++
				if calls == 1 {
					tool := a.toolMap["work"]
					tool.Description = "finalizing"
					tool.Handler = func(_ context.Context, args json.RawMessage, _ *tools.Container) (llm.Content, error) {
						last++
						if !strings.Contains(string(args), "ok") {
							t.Error("merged args lost")
						}
						return llm.Content{}, tools.TaskComplete("finalizing")
					}
					a.tools[0] = tool
					a.toolMap["work"] = tool
					a.toolMapNormalized = buildNormalizedToolMap(a.toolMap, a.tools)
					a.toolChoice = "required"
					if failure == "snapshot" {
						a.tools[0].Schema = map[string]any{"private": func() {}}
					}
					if failure == "binding" {
						a.tools[0].Description = "private"
					}
					return &llm.Completion{Content: llm.TextContent("partial visible"), StopReason: "max_tokens", ToolCalls: []llm.ToolCall{{ID: "a", Function: llm.FunctionCall{Name: "work", Arguments: `{"text":`}}}}, nil
				}
				if req.Tools[0].Description != "finalizing" || req.ToolChoice != "required" {
					t.Error("wrong finalizing request")
				}
				tool := a.toolMap["work"]
				tool.Handler = func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
					first++
					return llm.Content{}, tools.TaskComplete("wrong")
				}
				a.toolMap["work"] = tool
				return &llm.Completion{ToolCalls: []llm.ToolCall{{ID: "a", Function: llm.FunctionCall{Name: "work", Arguments: `"ok"}`}}}}, nil
			}}
			var err error
			a, err = New(Config{LLM: model, Tools: []tools.Tool{{Name: "work", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
				first++
				return llm.Content{}, nil
			}}}})
			if err != nil {
				t.Fatal(err)
			}
			for event := range a.QueryStream(context.Background(), llm.TextContent("run")) {
				if _, ok := event.(ErrorEvent); ok {
					errorsSeen++
				}
			}
			if failure == "none" {
				if calls != 2 || first != 0 || last != 1 || errorsSeen != 0 {
					t.Fatalf("calls/first/last/errors=%d/%d/%d/%d", calls, first, last, errorsSeen)
				}
			} else {
				if calls != 1 || first != 0 || last != 0 || errorsSeen != 1 {
					t.Fatalf("calls/first/last/errors=%d/%d/%d/%d", calls, first, last, errorsSeen)
				}
				for _, m := range a.Messages() {
					if len(m.ToolCalls) != 0 || m.Role == llm.RoleTool {
						t.Fatal("unaccepted continuation survived failure")
					}
				}
			}
			if _, changed, _ := repairToolCallPairsDetailed(a.Messages()); changed {
				t.Fatal("history requires repair")
			}
			if failure != "none" {
				// Repair only the injected private registry corruption, not history.
				a.tools[0] = a.toolMap["work"]
				nextCalls := 0
				a.llm = &frameScriptModel{invoke: func(llm.InvokeRequest) (*llm.Completion, error) {
					nextCalls++
					return &llm.Completion{Content: llm.TextContent("next")}, nil
				}}
				for event := range a.QueryStream(context.Background(), llm.TextContent("next")) {
					if warning, ok := event.(WarnEvent); ok && warning.Kind == "tool_pairing_repaired" {
						t.Error("next query needed history repair")
					}
				}
				if nextCalls != 1 {
					t.Fatalf("next provider calls=%d", nextCalls)
				}
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

func BenchmarkExecutionFrameAdmissionScaling(b *testing.B) {
	for _, size := range []struct{ tools, messages int }{{1, 1}, {32, 32}, {128, 256}} {
		b.Run(fmt.Sprintf("tools%d/messages%d", size.tools, size.messages), func(b *testing.B) {
			registry := make([]tools.Tool, size.tools)
			request := llm.InvokeRequest{}
			for i := range registry {
				registry[i] = tools.Tool{Name: fmt.Sprintf("tool_%d", i), Schema: map[string]any{"type": "object", "properties": map[string]any{"value": map[string]any{"type": "string"}}}}
				request.Tools = append(request.Tools, registry[i].Definition())
			}
			for i := 0; i < size.messages; i++ {
				request.Messages = append(request.Messages, llm.NewUserMessage(strings.Repeat("fixture ", 128)))
			}
			a, err := New(Config{LLM: historyCloneModel{}, Tools: registry})
			if err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				frame, err := newExecutionFrame(a.llm, request, a.toolMap, a.toolMapNormalized)
				if err != nil || !frame.validBindings() {
					b.Fatal("invalid fixture frame")
				}
			}
		})
	}
}

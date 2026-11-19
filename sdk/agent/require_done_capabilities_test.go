package agent

import (
	"context"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type capabilityFixtureModel struct {
	caps       llm.ThinkingCapabilities
	reportCaps bool
	calls      int
	safetyOnly bool
	requests   []llm.InvokeRequest
}

func (*capabilityFixtureModel) Provider() string { return "fixture" }
func (*capabilityFixtureModel) Model() string    { return "fixture-cap" }

func (m *capabilityFixtureModel) ThinkingCapabilities() (llm.ThinkingCapabilities, bool) {
	return m.caps, m.reportCaps
}

func (m *capabilityFixtureModel) Invoke(_ context.Context, r llm.InvokeRequest) (*llm.Completion, error) {
	copyReq, err := llm.CloneInvokeRequest(r)
	if err != nil {
		return nil, err
	}
	m.requests = append(m.requests, copyReq)
	m.calls++
	if m.calls == 1 {
		return &llm.Completion{
			ToolCalls: []llm.ToolCall{{ID: "work-1", Function: llm.FunctionCall{Name: "work", Arguments: "{}"}}},
		}, nil
	}
	if m.calls == 2 || m.safetyOnly {
		return &llm.Completion{Content: llm.TextContent("text answer"), StopReason: "stop"}, nil
	}
	return &llm.Completion{
		ToolCalls: []llm.ToolCall{{ID: "done-1", Function: llm.FunctionCall{Name: "done", Arguments: `{"message":"completed"}`}}},
	}, nil
}

func TestRequireDoneCapabilityDegradedSafetyFallbackHasTruthfulProvenance(t *testing.T) {
	model := &capabilityFixtureModel{
		reportCaps: true,
		safetyOnly: true,
		caps: llm.ThinkingCapabilities{
			SupportsDisableThinking:  false,
			SupportsForcedToolChoice: false,
			DisableThinkingReason:    "model does not support disabling extended thinking",
		},
	}
	work := tools.Func[struct{}]("work", "work", func(context.Context, struct{}, *tools.Container) (any, error) {
		return "ok", nil
	})
	done := tools.Func[struct {
		Message string `json:"message"`
	}]("done", "finish", func(_ context.Context, args struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) {
		return nil, tools.TaskComplete(args.Message)
	})
	ag, err := New(Config{
		LLM:              model,
		Tools:            []tools.Tool{work, done},
		RequireDoneTool:  true,
		Warningf:         func(string, ...any) {},
		QueryIDGenerator: func() string { return "cap-safety" },
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	var degradedWarns int
	var gotSafetyFinal bool
	for env := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("run")) {
		if env.RequestControlRelation != "" || env.RequestControlSourceFrameID != "" {
			t.Fatalf("non-disableable model must not receive RequestControl provenance: rel=%q source=%q frame=%q",
				env.RequestControlRelation, env.RequestControlSourceFrameID, env.FrameID)
		}
		switch ev := env.Event.(type) {
		case WarnEvent:
			if ev.Kind == "require_done_capability_degraded" {
				degradedWarns++
			}
		case FinalResponseEvent:
			if ev.Reason == "require_done_safety" && ev.Content == "text answer" {
				gotSafetyFinal = true
			}
		case ErrorEvent:
			t.Fatalf("unexpected error event: %+v", ev)
		}
	}
	if !gotSafetyFinal {
		t.Fatalf("expected require_done_safety final response")
	}
	if degradedWarns != 1 {
		t.Fatalf("degraded warnings = %d, want 1", degradedWarns)
	}
	for i, req := range model.requests {
		if req.DisableThinking {
			t.Fatalf("request[%d].DisableThinking = true, want false", i)
		}
		if req.ToolChoice == "required" {
			t.Fatalf("request[%d].ToolChoice = required, want non-forced", i)
		}
	}
}

func TestRequireDoneForcedToolWithoutDisableThinkingSkipsDisableProvenance(t *testing.T) {
	model := &capabilityFixtureModel{
		reportCaps: true,
		caps: llm.ThinkingCapabilities{
			SupportsDisableThinking:  false,
			SupportsForcedToolChoice: true,
			DisableThinkingReason:    "thinking stays enabled",
		},
	}
	work := tools.Func[struct{}]("work", "work", func(context.Context, struct{}, *tools.Container) (any, error) {
		return "ok", nil
	})
	done := tools.Func[struct {
		Message string `json:"message"`
	}]("done", "finish", func(_ context.Context, args struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) {
		return nil, tools.TaskComplete(args.Message)
	})
	ag, err := New(Config{
		LLM:              model,
		Tools:            []tools.Tool{work, done},
		RequireDoneTool:  true,
		Warningf:         func(string, ...any) {},
		QueryIDGenerator: func() string { return "cap-force" },
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	var degradedWarns int
	var final string
	for env := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("run")) {
		if env.RequestControlRelation != "" || env.RequestControlSourceFrameID != "" {
			t.Fatalf("unexpected disable-thinking provenance when DisableThinking was not active: rel=%q source=%q",
				env.RequestControlRelation, env.RequestControlSourceFrameID)
		}
		switch ev := env.Event.(type) {
		case WarnEvent:
			if ev.Kind == "require_done_capability_degraded" {
				degradedWarns++
			}
		case FinalResponseEvent:
			final = ev.Content
		case ErrorEvent:
			t.Fatalf("unexpected error event: %+v", ev)
		}
	}
	if final != "text answer\n\ncompleted" {
		t.Fatalf("final = %q, want %q", final, "text answer\n\ncompleted")
	}
	if degradedWarns != 1 {
		t.Fatalf("degraded warnings = %d, want 1", degradedWarns)
	}
	if len(model.requests) != 3 {
		t.Fatalf("request count = %d, want 3", len(model.requests))
	}
	if model.requests[2].DisableThinking {
		t.Fatalf("recovery request must not set DisableThinking")
	}
	if model.requests[2].ToolChoice != "required" {
		t.Fatalf("recovery request ToolChoice = %q, want required", model.requests[2].ToolChoice)
	}
}

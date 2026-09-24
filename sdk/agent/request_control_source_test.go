package agent

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type controlSourceModel struct {
	calls    int
	safety   bool
	requests []llm.InvokeRequest
}

func (*controlSourceModel) Provider() string { return "fixture" }
func (*controlSourceModel) Model() string    { return "fixture" }
func (m *controlSourceModel) Invoke(_ context.Context, r llm.InvokeRequest) (*llm.Completion, error) {
	copy, err := llm.CloneInvokeRequest(r)
	if err != nil {
		return nil, err
	}
	m.requests = append(m.requests, copy)
	m.calls++
	call := func(id, name string) *llm.Completion {
		return &llm.Completion{ToolCalls: []llm.ToolCall{{ID: id, Function: llm.FunctionCall{Name: name, Arguments: "{}"}}}}
	}
	if m.calls == 1 {
		return call("work-1", "work"), nil
	}
	if m.calls == 2 || m.safety {
		return &llm.Completion{Content: llm.TextContent("answer"), StopReason: "stop"}, nil
	}
	if m.calls == 3 {
		return nil, &llm.ProviderError{Provider: "fixture", StatusCode: 500, Message: "retry fixture"}
	}
	if m.calls == 4 {
		return call("work-2", "work"), nil
	}
	if m.calls == 5 {
		return call("done-1", "done"), nil
	}
	return &llm.Completion{Content: llm.TextContent("fresh"), StopReason: "stop"}, nil
}

func TestRequireDoneControlSourceRetryWorkSteeringAndLegacyParity(t *testing.T) {
	for _, steer := range []bool{false, true} {
		t.Run(map[bool]string{false: "work_retains", true: "steering_clears"}[steer], func(t *testing.T) {
			var baselineRequests []llm.InvokeRequest
			var baselineHistory []llm.Message
			var baselineKinds []EventKind
			for _, enveloped := range []bool{false, true} {
				model := &controlSourceModel{}
				steering := make(chan SteeringMsg, 1)
				workCalls := 0
				work := tools.Func[struct{}]("work", "fixture", func(context.Context, struct{}, *tools.Container) (any, error) {
					workCalls++
					if steer && workCalls == 2 {
						steering <- SteeringMsg{Content: "new instruction"}
					}
					return "work result", nil
				})
				done := tools.Func[struct{}]("done", "fixture", func(context.Context, struct{}, *tools.Container) (any, error) { return nil, tools.TaskComplete("done") })
				ag, err := New(Config{LLM: model, Tools: []tools.Tool{work, done}, RequireDoneTool: true, InvokeRetryMaxAttempts: 2, InvokeRetryBackoff: time.Nanosecond, Warningf: func(string, ...any) {}, QueryIDGenerator: func() string { return "control-query" }})
				if err != nil {
					t.Fatal(err)
				}
				var kinds []EventKind
				observed := map[string]bool{}
				if enveloped {
					for e := range ag.QueryStreamEnvelopedWithSteering(context.Background(), llm.TextContent("run"), steering) {
						kinds = append(kinds, e.Kind)
						want := ""
						if strings.HasSuffix(e.FrameID, "/frame/3") || !steer && strings.HasSuffix(e.FrameID, "/frame/4") {
							want = "control-query/frame/2"
						}
						if e.RequestControlSourceFrameID != want || (e.RequestControlRelation != "") != (want != "") {
							t.Errorf("wrong control metadata kind=%s frame=%s source=%s", e.Kind, e.FrameID, e.RequestControlSourceFrameID)
						}
						if want != "" {
							if e.RequestControlRelation != RequestControlRequireDoneDisableThinking {
								t.Error("wrong relation")
							}
							observed[e.FrameID] = true
						}
						// Steering accepted in Frame 3's block relates only
						// Frame 4 and never replaces the control relation.
						wantSteering := ""
						if steer && strings.HasSuffix(e.FrameID, "/frame/4") {
							wantSteering = "control-query/frame/3"
						}
						if e.RequestSteeringSourceFrameID != wantSteering || (e.RequestSteeringRelation != "") != (wantSteering != "") {
							t.Errorf("wrong steering metadata kind=%s frame=%s source=%s", e.Kind, e.FrameID, e.RequestSteeringSourceFrameID)
						}
						if e.Kind == EventKindSteeringReceived && (e.FrameID != "" || e.RequestControlRelation != "") {
							t.Error("steering inherited control source")
						}
					}
					if !observed["control-query/frame/3"] || !steer && !observed["control-query/frame/4"] {
						t.Fatalf("missing actual consumer observations: %v", observed)
					}
				} else {
					for e := range ag.QueryStreamWithSteering(context.Background(), llm.TextContent("run"), steering) {
						kind, _ := classifyEvent(e)
						kinds = append(kinds, kind)
					}
				}
				if len(model.requests) != 5 {
					t.Fatalf("requests=%d", len(model.requests))
				}
				for i, r := range model.requests {
					want := i >= 2
					if steer && i == 4 {
						want = false
					}
					if r.DisableThinking != want {
						t.Fatalf("request[%d].DisableThinking=%v", i, r.DisableThinking)
					}
				}
				if model.requests[4].ToolChoice == "required" {
					t.Fatal("ordinary work did not reset forced choice independently")
				}
				if !enveloped {
					baselineRequests = model.requests
					baselineHistory = ag.Messages()
					baselineKinds = kinds
				} else if !reflect.DeepEqual(baselineRequests, model.requests) || !reflect.DeepEqual(baselineHistory, ag.Messages()) || !reflect.DeepEqual(baselineKinds, kinds) {
					t.Fatal("envelope observation changed request/history/events")
				}
				for e := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("fresh turn")) {
					if e.RequestControlRelation != "" || e.RequestControlSourceFrameID != "" {
						t.Fatal("terminated control leaked into next Query")
					}
				}
				if model.requests[len(model.requests)-1].DisableThinking {
					t.Fatal("next Query kept thinking override")
				}
			}
		})
	}
}

func TestRequireDoneControlSourceSafetyRetainsOldFrameAndClearsNextQuery(t *testing.T) {
	model := &controlSourceModel{safety: true}
	work := tools.Func[struct{}]("work", "fixture", func(context.Context, struct{}, *tools.Container) (any, error) { return "ok", nil })
	done := tools.Func[struct{}]("done", "fixture", func(context.Context, struct{}, *tools.Container) (any, error) { return nil, tools.TaskComplete("done") })
	ag, err := New(Config{LLM: model, Tools: []tools.Tool{work, done}, RequireDoneTool: true, Warningf: func(string, ...any) {}, QueryIDGenerator: func() string { return "control-query" }})
	if err != nil {
		t.Fatal(err)
	}
	final := false
	for e := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("run")) {
		if ev, ok := e.Event.(FinalResponseEvent); ok {
			final = true
			if ev.Reason != "require_done_safety" || e.FrameID != "control-query/frame/4" || e.RequestControlSourceFrameID != "control-query/frame/3" {
				b, _ := json.Marshal(e)
				t.Fatalf("safety frame relabeled: %s", b)
			}
		}
	}
	if !final {
		t.Fatal("missing safety terminal")
	}
	for e := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("fresh")) {
		if e.RequestControlSourceFrameID != "" {
			t.Fatal("safety source leaked into new Query")
		}
	}
	if model.requests[len(model.requests)-1].DisableThinking {
		t.Fatal("safety thinking state persisted")
	}
}

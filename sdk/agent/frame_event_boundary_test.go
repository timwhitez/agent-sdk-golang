package agent

import (
	"context"
	"encoding/json"
	"net"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// A transport retry is not another logical request; continuation and the next
// post-tool request are. Existing envelope sequence counts events, not attempts.
// Freeze that distinction before adding Frame/attempt identity metadata.
func TestFrameEventBoundaryRetryContinuationAndLegacyParity(t *testing.T) {
	var baselineRequests [][]byte
	var baselineHistory []llm.Message
	var baselineKinds []EventKind
	wantKinds := []EventKind{EventKindWarning, EventKindAutoContinue, EventKindStepStart, EventKindToolCall, EventKindToolResult, EventKindAccounting, EventKindStepComplete, EventKindStepStart, EventKindToolCall, EventKindToolResult, EventKindAccounting, EventKindStepComplete, EventKindText, EventKindFinalResponse}
	wantOrigins := []EventOrigin{EventOriginSDKDriver, EventOriginSDKDriver, EventOriginToolRuntime, EventOriginToolRuntime, EventOriginToolRuntime, EventOriginSDKDriver, EventOriginToolRuntime, EventOriginToolRuntime, EventOriginToolRuntime, EventOriginToolRuntime, EventOriginSDKDriver, EventOriginToolRuntime, EventOriginModel, EventOriginSDKDriver}
	for _, enveloped := range []bool{false, true} {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		var requests [][]byte
		handlers, warnings := 0, 0
		model := &frameScriptModel{invoke: func(req llm.InvokeRequest) (*llm.Completion, error) {
			encoded, err := json.Marshal(req)
			if err != nil {
				t.Error(err)
				return nil, err
			}
			requests = append(requests, encoded)
			switch len(requests) {
			case 1:
				return nil, &net.DNSError{Err: "fixture timeout", IsTimeout: true}
			case 2:
				return &llm.Completion{StopReason: "max_tokens", ToolCalls: []llm.ToolCall{{ID: "a", Function: llm.FunctionCall{Name: "work", Arguments: `{"text":`}}}}, nil
			case 3:
				return &llm.Completion{ToolCalls: []llm.ToolCall{{ID: "a", Function: llm.FunctionCall{Name: "work", Arguments: `"ok"}`}}}}, nil
			case 4:
				// Provider Call ID reuse across completed blocks is legal and
				// cannot serve as a logical-request identity.
				return &llm.Completion{ToolCalls: []llm.ToolCall{{ID: "a", Function: llm.FunctionCall{Name: "work", Arguments: `{"text":"ok"}`}}}}, nil
			default:
				return &llm.Completion{Content: llm.TextContent("done")}, nil
			}
		}}
		fixed := time.Unix(100, 0)
		ids := 0
		ag, err := New(Config{LLM: model, InvokeRetryMaxAttempts: 2, Warningf: func(string, ...any) { warnings++ }, QueryIDGenerator: func() string { ids++; return "boundary-query" }, EventClock: func() time.Time { return fixed }, Tools: []tools.Tool{{Name: "work", Description: "PRIVATE_SCHEMA_FIXTURE", Schema: map[string]any{"type": "object", "properties": map[string]any{"text": map[string]any{"type": "string"}}}, Handler: func(_ context.Context, args json.RawMessage, _ *tools.Container) (llm.Content, error) {
			handlers++
			if string(args) != `{"text":"ok"}` {
				t.Errorf("merged args=%s", args)
			}
			return llm.TextContent("PRIVATE_RESULT_FIXTURE"), nil
		}}}})
		if err != nil {
			t.Fatal(err)
		}
		var kinds []EventKind
		if enveloped {
			for e := range ag.QueryStreamEnveloped(ctx, llm.TextContent("PRIVATE_PROMPT_FIXTURE")) {
				kinds = append(kinds, e.Kind)
				if e.QueryID != "boundary-query" || e.Sequence != uint64(len(kinds)) || !e.Timestamp.Equal(fixed) {
					t.Errorf("envelope identity/order=%+v", e)
				}
				kind, origin := classifyEvent(e.Event)
				if e.Kind != kind || e.Origin != origin {
					t.Errorf("kind/origin=%s/%s", e.Kind, e.Origin)
				}
				if len(kinds) <= len(wantOrigins) && e.Origin != wantOrigins[len(kinds)-1] {
					t.Errorf("origin golden at %d=%s", len(kinds), e.Origin)
				}
				metadata := e
				metadata.Event = nil
				encoded, err := json.Marshal(metadata)
				if err != nil {
					t.Fatal(err)
				}
				if strings.Contains(string(encoded), "PRIVATE_") {
					t.Fatal("envelope metadata duplicated private request/result content")
				}
			}
		} else {
			for e := range ag.QueryStream(ctx, llm.TextContent("PRIVATE_PROMPT_FIXTURE")) {
				kind, _ := classifyEvent(e)
				kinds = append(kinds, kind)
			}
		}
		if !reflect.DeepEqual(kinds, wantKinds) {
			t.Fatalf("event golden=%v", kinds)
		}
		if len(requests) != 5 || handlers != 2 || warnings != 1 || ids != 1 {
			t.Fatalf("enveloped=%v requests=%d handlers=%d warnings=%d ids=%d", enveloped, len(requests), handlers, warnings, ids)
		}
		if !reflect.DeepEqual(requests[0], requests[1]) || reflect.DeepEqual(requests[1], requests[2]) || reflect.DeepEqual(requests[2], requests[3]) || reflect.DeepEqual(requests[3], requests[4]) {
			t.Fatal("retry/logical-request boundaries changed")
		}
		if !enveloped {
			baselineRequests, baselineHistory, baselineKinds = requests, ag.Messages(), kinds
		} else if !reflect.DeepEqual(requests, baselineRequests) || !reflect.DeepEqual(ag.Messages(), baselineHistory) || !reflect.DeepEqual(kinds, baselineKinds) {
			t.Fatal("envelope observation changed the logical execution")
		}
		if _, changed, _ := repairToolCallPairsDetailed(ag.Messages()); changed {
			t.Fatal("completed continuation history still needs repair")
		}
	}
}

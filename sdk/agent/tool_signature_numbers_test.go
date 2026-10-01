package agent

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestToolSignaturePreservesJSONNumbers(t *testing.T) {
	for _, number := range []string{"9007199254740991", "9007199254740992", "9007199254740993", "-9007199254740993", "9223372036854775807", "18446744073709551615", "1.00000000000000001", "1e1000"} {
		raw := `{"z":[` + number + `],"a":true}`
		want := `{"a":true,"z":[` + number + `]}`
		if got := canonicalJSON(nil, raw); got != want {
			t.Errorf("canonical %s = %s, want %s", raw, got, want)
		}
		if got := normalizeToolSignature("lookup", nil, raw); got != "lookup|"+want {
			t.Errorf("signature = %s, want %s", got, "lookup|"+want)
		}
		if got := canonicalJSON([]byte(raw), "ignored"); got != want {
			t.Errorf("normalized canonical = %s", got)
		}
	}
	for _, raw := range []string{` {"id":1} {"id":2} `, ` {invalid `, ` true `, ` null `, ` "string" `} {
		if got := canonicalJSON(nil, raw); got != strings.TrimSpace(raw) {
			t.Errorf("fallback %q = %q", raw, got)
		}
	}
	// Numeric spelling is retained: safe false-negative deduplication beats merging distinct targets.
	if normalizeToolSignature("lookup", nil, `{"id":1}`) == normalizeToolSignature("lookup", nil, `{"id":1.0}`) {
		t.Fatal("numeric spellings must be retained")
	}
	if normalizeToolSignature("lookup", nil, ` {"b":2,"a":1} `) != normalizeToolSignature("lookup", nil, `{"a":1,"b":2}`) {
		t.Fatal("object order/whitespace changed identity")
	}
}

func TestAgentLargeIntegerToolIdentity(t *testing.T) {
	for _, tc := range []struct {
		name     string
		second   int64
		guard    int
		executed []int64
		released bool
	}{
		{"distinct_guard", 9007199254740993, 2, []int64{9007199254740992, 9007199254740993}, false},
		{"same_guard", 9007199254740992, 2, []int64{9007199254740992}, false},
		{"distinct_retention", 9007199254740993, 0, []int64{9007199254740992, 9007199254740993}, false},
		{"same_retention", 9007199254740992, 0, []int64{9007199254740992, 9007199254740992}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ids := []int64{9007199254740992, tc.second}
			var executed []int64
			var results []llm.Message
			calls := 0
			model := &frameScriptModel{invoke: func(req llm.InvokeRequest) (*llm.Completion, error) {
				calls++
				if calls <= 2 {
					return &llm.Completion{ToolCalls: []llm.ToolCall{{ID: fmt.Sprintf("lookup-%d", calls), Type: "function", Function: llm.FunctionCall{Name: "lookup", Arguments: fmt.Sprintf(`{"id":%d}`, ids[calls-1])}}}}, nil
				}
				for _, m := range req.Messages {
					if m.Role == llm.RoleTool {
						results = append(results, m)
					}
				}
				return &llm.Completion{Content: llm.TextContent("done")}, nil
			}}
			type args struct {
				ID int64 `json:"id"`
			}
			tool := tools.Func[args]("lookup", "lookup", func(_ context.Context, a args, _ *tools.Container) (any, error) {
				executed = append(executed, a.ID)
				return fmt.Sprint(a.ID), nil
			}).WithEphemeralKeep(1)
			ag, err := New(Config{LLM: model, Tools: []tools.Tool{tool}, MaxIterations: 5, RepeatToolSignatureThreshold: tc.guard})
			if err != nil {
				t.Fatal(err)
			}
			events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("lookup two IDs")))
			for _, e := range events {
				if e, ok := e.(ErrorEvent); ok {
					t.Fatalf("agent error: %#v", e)
				}
			}
			if !reflect.DeepEqual(executed, tc.executed) {
				t.Fatalf("handler IDs = %v, want %v", executed, tc.executed)
			}
			if len(results) != 2 {
				t.Fatalf("tool results = %d, want 2", len(results))
			}
			if tc.guard > 0 && tc.second == ids[0] {
				if !results[1].IsError || !strings.Contains(results[1].PlainText(), "loop guard") {
					t.Fatalf("same target not suppressed: %#v", results[1])
				}
			} else {
				if results[0].Destroyed != tc.released {
					t.Fatalf("first result released=%v, want %v", results[0].Destroyed, tc.released)
				}
				if !tc.released && results[0].PlainText() != fmt.Sprint(ids[0]) {
					t.Fatalf("first target result lost: %q", results[0].PlainText())
				}
				if results[1].Destroyed || results[1].IsError || results[1].PlainText() != fmt.Sprint(tc.second) {
					t.Fatalf("second target result: %#v", results[1])
				}
			}
		})
	}
}

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

func TestContinuationNumericMerge(t *testing.T) {
	for _, n := range []string{"9007199254740991", "9007199254740992", "9007199254740993", "-9007199254740993", "-9223372036854775808", "9223372036854775807", "18446744073709551615", "1", "1.250", "1e20"} {
		for _, pair := range [][2]string{{`{"id":` + n + `}`, `{"extra":true}`}, {`{"extra":true}`, `{"id":` + n + `}`}} {
			t.Run(n+pair[0], func(t *testing.T) {
				got := mergeToolArgs(pair[0], pair[1])
				var decoded struct {
					ID    json.Number `json:"id"`
					Extra bool        `json:"extra"`
				}
				if err := json.Unmarshal([]byte(got), &decoded); err != nil || decoded.ID.String() != n || !decoded.Extra {
					t.Fatalf("merge=%s decoded=%+v err=%v, want number %s", got, decoded, err, n)
				}
			})
		}
	}
	cases := []struct{ name, old, next, want string }{
		{"nested arrays and maps", `{"map":{"old":9007199254740993},"items":[{"id":9223372036854775807},18446744073709551615]}`, `{"map":{"new":-9007199254740993},"items":[{"extra":true}]}`, `{"map":{"old":9007199254740993,"new":-9007199254740993},"items":[{"id":9223372036854775807,"extra":true},18446744073709551615]}`},
		{"new array tail", `{"items":[1]}`, `{"items":[2,9007199254740993]}`, `{"items":[2,9007199254740993]}`},
		{"scalars", `{"id":9007199254740993,"b":true,"n":null,"s":"text"}`, `{"extra":false}`, `{"id":9007199254740993,"b":true,"n":null,"s":"text","extra":false}`},
		{"whitespace", " {\"id\":9007199254740993}\n", "\t{\"extra\":true} ", `{"id":9007199254740993,"extra":true}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			decode := func(s string) any {
				d := json.NewDecoder(strings.NewReader(s))
				d.UseNumber()
				var v any
				if err := d.Decode(&v); err != nil {
					t.Fatal(err)
				}
				return v
			}
			got := mergeToolArgs(tc.old, tc.next)
			if !reflect.DeepEqual(decode(got), decode(tc.want)) {
				t.Fatalf("got %s, want %s", got, tc.want)
			}
		})
	}
	conflict := mergeToolArgsWithDiagnostics(`{"id":{}}`, `{"id":9007199254740993}`)
	if len(conflict.diagnostics) != 1 || conflict.diagnostics[0] != "$.id changed shape from object to number" {
		t.Fatalf("diagnostics=%v", conflict.diagnostics)
	}
}

func TestContinuationNumericFragmentBoundaries(t *testing.T) {
	for _, tc := range []struct{ old, next, want string }{
		{`{"id":900719925474`, `0993}`, `{"id":9007199254740993}`},
		{"", `{"id":9007199254740993}`, `{"id":9007199254740993}`},
		{`{"id":9007199254740993}`, " ", `{"id":9007199254740993}`},
		{`{"id":9007199254740993} {}`, `{"extra":true}`, `{"id":9007199254740993} {}{"extra":true}`},
		{`{"id":9007199254740993}`, `{"extra":true} {}`, `{"id":9007199254740993}{"extra":true} {}`},
		{`{"id":?}`, `{"extra":true}`, `{"id":?}{"extra":true}`},
	} {
		if got := mergeToolArgs(tc.old, tc.next); got != tc.want {
			t.Fatalf("merge(%q,%q)=%q want %q", tc.old, tc.next, got, tc.want)
		}
	}
}

func TestContinuationNumericAgent(t *testing.T) {
	type args struct {
		ID       int64   `json:"id"`
		Unsigned uint64  `json:"unsigned"`
		Extra    bool    `json:"extra"`
		Float    float64 `json:"float"`
	}
	for _, tc := range []struct {
		name, partial string
		want          args
		overflow      bool
	}{
		{"below float boundary", `{"id":9007199254740991}`, args{ID: 9007199254740991, Extra: true}, false},
		{"at float boundary", `{"id":9007199254740992}`, args{ID: 9007199254740992, Extra: true}, false},
		{"fractional", `{"float":1.25}`, args{Float: 1.25, Extra: true}, false},
		{"exponent", `{"float":1e20}`, args{Float: 1e20, Extra: true}, false},
		{"large signed", `{"id":9007199254740993}`, args{ID: 9007199254740993, Extra: true}, false},
		{"negative", `{"id":-9007199254740993}`, args{ID: -9007199254740993, Extra: true}, false},
		{"signed min", `{"id":-9223372036854775808}`, args{ID: -9223372036854775808, Extra: true}, false},
		{"signed max", `{"id":9223372036854775807}`, args{ID: 9223372036854775807, Extra: true}, false},
		{"unsigned max", `{"unsigned":18446744073709551615}`, args{Unsigned: 18446744073709551615, Extra: true}, false},
		{"signed overflow", `{"id":9223372036854775808}`, args{}, true},
		{"unsigned overflow", `{"unsigned":18446744073709551616}`, args{}, true},
	} {
		for _, newSource := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/new=%v", tc.name, newSource), func(t *testing.T) {
				old, next := tc.partial, `{"extra":true}`
				if newSource {
					old, next = next, old
				}
				model := &steeringRelationModel{steps: []func() (*llm.Completion, error){lineageCall("numeric-1", "lookup", old, "max_tokens"), lineageCall("numeric-1", "lookup", next, "tool_calls"), steeringFinal}}
				var received []args
				lookup := tools.Func("lookup", "fixture", func(_ context.Context, a args, _ *tools.Container) (any, error) {
					received = append(received, a)
					return fmt.Sprintf("%d/%d/%g", a.ID, a.Unsigned, a.Float), nil
				})
				ag, err := New(Config{LLM: model, Tools: []tools.Tool{lookup}, Warningf: func(string, ...any) {}, QueryIDGenerator: func() string { return "lineage-query" }})
				if err != nil {
					t.Fatal(err)
				}
				var calls, results int
				var eventArgs string
				var result ToolResultEvent
				for env := range ag.QueryStreamEnveloped(context.Background(), llm.TextContent("lookup")) {
					switch e := env.Event.(type) {
					case ToolCallEvent:
						calls++
						eventArgs = string(e.ArgsJSON)
					case ToolResultEvent:
						results++
						result = e
					}
					if _, ok := env.Event.(ToolResultEvent); ok && env.FrameID != "lineage-query/frame/2" {
						t.Fatalf("result frame=%s", env.FrameID)
					}
					if env.FrameID == "lineage-query/frame/3" && strings.Join(env.RequestContinuationSourceFrameIDs, ",") != "lineage-query/frame/1,lineage-query/frame/2" {
						t.Fatalf("sources=%v", env.RequestContinuationSourceFrameIDs)
					}
				}
				if results != 1 || result.IsError != tc.overflow {
					t.Fatalf("results=%d result=%+v", results, result)
				}
				if tc.overflow {
					if len(received) != 0 {
						t.Fatalf("overflow executed: %v", received)
					}
				} else {
					if len(received) != 1 || received[0] != tc.want || calls != 1 {
						t.Fatalf("received=%v want=%v calls=%d", received, tc.want, calls)
					}
					var final args
					if err := json.Unmarshal([]byte(eventArgs), &final); err != nil || final != tc.want {
						t.Fatalf("prepared arguments=%s err=%v", eventArgs, err)
					}
				}
				requests := model.recorded()
				if len(requests) != 3 {
					t.Fatalf("requests=%d", len(requests))
				}
				for _, history := range [][]llm.Message{ag.Messages(), requests[2].Messages} {
					var accepted, terminal int
					for _, m := range history {
						if m.Role == llm.RoleAssistant {
							for _, call := range m.ToolCalls {
								if call.ID == "numeric-1" {
									accepted++
									if !tc.overflow && call.Function.Arguments != eventArgs {
										t.Fatalf("history args=%s prepared=%s", call.Function.Arguments, eventArgs)
									}
								}
							}
						}
						if m.Role == llm.RoleTool && m.ToolCallID == "numeric-1" {
							terminal++
							if m.Content.PlainText() != result.Result {
								t.Fatalf("history result=%q event=%q", m.Content.PlainText(), result.Result)
							}
						}
					}
					if accepted != 1 || terminal != 1 {
						t.Fatalf("accepted=%d terminal=%d", accepted, terminal)
					}
				}
			})
		}
	}
}

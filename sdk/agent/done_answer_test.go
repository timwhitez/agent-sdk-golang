package agent

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type doneAnswerModel struct{ text, done string }

func (m doneAnswerModel) Provider() string { return "fixture" }
func (m doneAnswerModel) Model() string    { return "fixture" }
func (m doneAnswerModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	args, _ := json.Marshal(map[string]string{"message": m.done})
	return &llm.Completion{Content: llm.TextContent(m.text), ResponseID: "resp-answer", StopReason: "tool_calls", ToolCalls: []llm.ToolCall{{ID: "done-1", Type: "function", Function: llm.FunctionCall{Name: "done", Arguments: string(args)}}}}, nil
}

type streamingDoneAnswerModel struct{ doneAnswerModel }

func (m streamingDoneAnswerModel) InvokeStream(ctx context.Context, req llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	comp, _ := m.Invoke(ctx, req)
	ch := make(chan llm.StreamEvent, 4)
	ch <- llm.StreamTextDeltaEvent{Delta: m.text}
	ch <- llm.StreamToolCallDeltaEvent{Index: 0, ID: "done-1", NameDelta: "done", ArgumentsDelta: comp.ToolCalls[0].Function.Arguments}
	ch <- llm.StreamResponseEvent{ResponseID: "resp-answer"}
	ch <- llm.StreamDoneEvent{StopReason: "tool_calls"}
	close(ch)
	return ch, nil
}

func TestDoneAnswerRetainsTextAndDistinctPayload(t *testing.T) {
	for _, tc := range []struct{ name, text, done, want string }{
		{"distinct", "The answer is 42.", "Report saved.", "The answer is 42.\n\nReport saved."},
		{"identical", "The answer is 42.", " The answer is 42. ", "The answer is 42."},
		{"payload only", "", "Report saved.", "Report saved."},
		{"text only", "The answer is 42.", "", "The answer is 42."},
	} {
		for _, streaming := range []bool{false, true} {
			for _, query := range []bool{false, true} {
				for _, requireDone := range []bool{false, true} {
					t.Run(tc.name+testBoolName(streaming, "/stream")+testBoolName(query, "/query")+testBoolName(requireDone, "/required"), func(t *testing.T) {
						var model llm.ChatModel = doneAnswerModel{text: tc.text, done: tc.done}
						if streaming {
							model = streamingDoneAnswerModel{model.(doneAnswerModel)}
						}
						done := tools.Func[struct {
							Message string `json:"message"`
						}]("done", "complete", func(_ context.Context, args struct {
							Message string `json:"message"`
						}, _ *tools.Container) (any, error) {
							return nil, tools.TaskComplete(args.Message)
						})
						ag, err := New(Config{LLM: model, Tools: []tools.Tool{done}, RequireDoneTool: requireDone})
						if err != nil {
							t.Fatal(err)
						}
						if query {
							got, err := ag.Query(context.Background(), "answer")
							if err != nil || got != tc.want {
								t.Fatalf("Query = %q, %v; want %q", got, err, tc.want)
							}
						} else {
							var final FinalResponseEvent
							finals, results, deltas := 0, 0, ""
							for _, ev := range collectEvents(ag.QueryStream(context.Background(), llm.TextContent("answer"))) {
								switch e := ev.(type) {
								case ErrorEvent:
									t.Fatalf("error: %#v", e)
								case FinalResponseEvent:
									final = e
									finals++
								case TextDeltaEvent:
									deltas += e.Delta
								case ToolResultEvent:
									results++
									if e.IsError || e.HandlerFailed {
										t.Fatalf("completion became failure: %#v", e)
									}
								}
							}
							if finals != 1 || results != 1 || final.Content != tc.want || final.ResponseID != "resp-answer" {
								t.Fatalf("final=%#v finals=%d results=%d", final, finals, results)
							}
							if streaming && deltas != tc.text {
								t.Fatalf("stream replayed final: %q", deltas)
							}
						}
						msgs := ag.Messages()
						if len(msgs) < 3 || msgs[len(msgs)-2].Content.PlainText() != tc.text || msgs[len(msgs)-1].Role != llm.RoleTool {
							t.Fatalf("history changed: %#v", msgs)
						}
					})
				}
			}
		}
	}
}

func testBoolName(value bool, name string) string {
	if value {
		return name
	}
	return ""
}

func TestDoneAnswerUsesCurrentTextAndRetainsContinuation(t *testing.T) {
	echo := tools.Tool{Name: "echo", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
		return llm.TextContent("ok"), nil
	}}
	done := tools.Func[struct {
		Message string `json:"message"`
	}]("done", "complete", func(_ context.Context, args struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) {
		return nil, tools.TaskComplete(args.Message)
	})
	for _, tc := range []struct {
		name     string
		turns    []*llm.Completion
		want, id string
	}{
		{
			name: "current response beats reminder text",
			turns: []*llm.Completion{
				{ToolCalls: []llm.ToolCall{{ID: "e1", Function: llm.FunctionCall{Name: "echo", Arguments: `{}`}}}},
				{Content: llm.TextContent("Earlier answer."), ResponseID: "resp-earlier"},
				{Content: llm.TextContent("Revised answer."), ResponseID: "resp-current", ToolCalls: []llm.ToolCall{{ID: "d1", Function: llm.FunctionCall{Name: "done", Arguments: `{"message":"Report saved."}`}}}},
			},
			want: "Revised answer.\n\nReport saved.", id: "resp-current",
		},
		{
			name: "max tokens text followed by done",
			turns: []*llm.Completion{
				{Content: llm.TextContent("The answer is "), StopReason: "max_tokens", ResponseID: "resp-part"},
				{Content: llm.TextContent("42."), ResponseID: "resp-current", ToolCalls: []llm.ToolCall{{ID: "d1", Function: llm.FunctionCall{Name: "done", Arguments: `{"message":"Report saved."}`}}}},
			},
			want: "The answer is 42.\n\nReport saved.", id: "resp-current",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := &turnModel{turns: tc.turns}
			ag, err := New(Config{LLM: model, Tools: []tools.Tool{echo, done}, RequireDoneTool: true})
			if err != nil {
				t.Fatal(err)
			}
			for _, ev := range collectEvents(ag.QueryStream(context.Background(), llm.TextContent("answer"))) {
				if final, ok := ev.(FinalResponseEvent); ok {
					if final.Content != tc.want || final.ResponseID != tc.id {
						t.Fatalf("final=%#v", final)
					}
					return
				}
				if failure, ok := ev.(ErrorEvent); ok {
					t.Fatalf("failure=%#v", failure)
				}
			}
			t.Fatal("no final event")
		})
	}
}

func TestDoneReminderConsumesCompletedTextContinuation(t *testing.T) {
	echo := tools.Tool{Name: "echo", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
		return llm.TextContent("ok"), nil
	}}
	done := tools.Func[struct {
		Message string `json:"message"`
	}]("done", "complete", func(_ context.Context, args struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) { return nil, tools.TaskComplete(args.Message) })
	for _, required := range []bool{false, true} {
		t.Run(testBoolName(required, "required"), func(t *testing.T) {
			model := &turnModel{turns: []*llm.Completion{
				{ToolCalls: []llm.ToolCall{{ID: "e1", Function: llm.FunctionCall{Name: "echo", Arguments: `{}`}}}},
				{Content: llm.TextContent("The answer is "), StopReason: "max_tokens", ResponseID: "resp-part"},
				{Content: llm.TextContent("42."), ResponseID: "resp-full-answer"},
				{ResponseID: "resp-done-only", ToolCalls: []llm.ToolCall{{ID: "d1", Function: llm.FunctionCall{Name: "done", Arguments: `{"message":"Report saved."}`}}}},
			}}
			ag, err := New(Config{LLM: model, Tools: []tools.Tool{echo, done}, RequireDoneTool: required})
			if err != nil {
				t.Fatal(err)
			}
			for _, ev := range collectEvents(ag.QueryStream(context.Background(), llm.TextContent("answer"))) {
				if final, ok := ev.(FinalResponseEvent); ok {
					if final.Content != "The answer is 42.\n\nReport saved." || final.ResponseID != "resp-full-answer" {
						t.Fatalf("final=%#v", final)
					}
					return
				}
				if failure, ok := ev.(ErrorEvent); ok {
					t.Fatalf("failure=%#v", failure)
				}
			}
			t.Fatal("no final event")
		})
	}
}

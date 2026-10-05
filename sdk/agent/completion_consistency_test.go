package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type completionTurnModel struct {
	turns []*llm.Completion
	next  int
	fail  error
}

func (*completionTurnModel) Provider() string { return "offline" }
func (*completionTurnModel) Model() string    { return "fresh" }
func (m *completionTurnModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	if m.next >= len(m.turns) {
		if m.fail != nil {
			return nil, m.fail
		}
		return nil, errors.New("fixture exhausted")
	}
	c := m.turns[m.next]
	m.next++
	return c, nil
}

type completionTurnStreamer struct{ *completionTurnModel }

func (m completionTurnStreamer) InvokeStream(ctx context.Context, r llm.InvokeRequest) (<-chan llm.StreamEvent, error) {
	c, err := m.Invoke(ctx, r)
	if err != nil {
		return nil, err
	}
	ch := make(chan llm.StreamEvent, 20)
	for _, s := range []string{"", c.Content.Text} {
		ch <- llm.StreamTextDeltaEvent{Delta: s}
	}
	for _, block := range c.Content.Blocks {
		if block.Type == "text" {
			ch <- llm.StreamTextDeltaEvent{Delta: block.Text}
		}
	}
	for i, tc := range c.ToolCalls {
		ch <- llm.StreamToolCallDeltaEvent{Index: i, ID: tc.ID, NameDelta: tc.Function.Name, ArgumentsDelta: tc.Function.Arguments}
	}
	ch <- llm.StreamResponseEvent{ResponseID: c.ResponseID}
	ch <- llm.StreamDoneEvent{StopReason: c.StopReason}
	close(ch)
	return ch, nil
}
func completionCall(name, id, args string) llm.ToolCall {
	return llm.ToolCall{ID: id, Type: "function", Function: llm.FunctionCall{Name: name, Arguments: args}}
}
func TestCompletionSnapshotContinuationMatrix(t *testing.T) {
	work := tools.Tool{Name: "work", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
		return llm.TextContent("worked"), nil
	}}
	done := tools.Func[struct {
		Message string `json:"message"`
	}]("done", "finish", func(_ context.Context, x struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) {
		return nil, tools.TaskComplete(x.Message)
	})
	for _, stream := range []bool{false, true} {
		for _, query := range []bool{false, true} {
			for _, required := range []bool{false, true} {
				for _, tc := range []struct {
					name     string
					turns    []*llm.Completion
					want, id string
				}{
					{"doneonly", []*llm.Completion{{ResponseID: "done", ToolCalls: []llm.ToolCall{completionCall("done", "d", `{"message":"Saved."}`)}}}, "Saved.", "done"},
					{"same", []*llm.Completion{{Content: llm.TextContent("Answer."), ResponseID: "done", ToolCalls: []llm.ToolCall{completionCall("done", "d", `{"message":"Answer."}`)}}}, "Answer.", "done"},
					{"distinct", []*llm.Completion{{Content: llm.TextContent("Answer."), ResponseID: "done", ToolCalls: []llm.ToolCall{completionCall("done", "d", `{"message":"Saved."}`)}}}, "Answer.\n\nSaved.", "done"},
					{"reminderwhitespace", []*llm.Completion{{ToolCalls: []llm.ToolCall{completionCall("work", "w", `{}`)}}, {Content: llm.TextContent("Answer."), ResponseID: "answer"}, {Content: llm.TextContent(" \n"), ResponseID: "done", ToolCalls: []llm.ToolCall{completionCall("done", "d", `{"message":"Saved."}`)}}}, "Answer.\n\nSaved.", "answer"},
					{"continuationwhitespace", []*llm.Completion{{Content: llm.TextContent("First"), StopReason: "max_tokens", ResponseID: "part1"}, {Content: llm.TextContent(" \n"), StopReason: "max_tokens", ResponseID: "part2"}, {Content: llm.TextContent("Second"), ResponseID: "done", ToolCalls: []llm.ToolCall{completionCall("done", "d", `{"message":"Saved."}`)}}}, "First \nSecond\n\nSaved.", "done"},
					{"continuationwhitespaceblock", []*llm.Completion{{Content: llm.TextContent("First"), StopReason: "max_tokens"}, {Content: llm.Content{Blocks: []llm.ContentBlock{{Type: "text", Text: "\n"}}}, StopReason: "max_tokens"}, {Content: llm.TextContent("Second"), ResponseID: "done", ToolCalls: []llm.ToolCall{completionCall("done", "d", `{"message":"Saved."}`)}}}, "First\nSecond\n\nSaved.", "done"},
					{"abandonedarguments", []*llm.Completion{{Content: llm.TextContent("Abandoned."), StopReason: "max_tokens", ToolCalls: []llm.ToolCall{completionCall("done", "partial", `{"message":"`)}}, {Content: llm.TextContent("Replacement."), ResponseID: "replacement"}, {ResponseID: "done", ToolCalls: []llm.ToolCall{completionCall("done", "d", `{"message":"Saved."}`)}}}, "Replacement.\n\nSaved.", "replacement"},
					{"invalidmergethencompletion", []*llm.Completion{{Content: llm.TextContent("Answer."), StopReason: "max_tokens", ToolCalls: []llm.ToolCall{completionCall("done", "d", `{"message":"Sa`)}}, {Content: llm.TextContent(" Details."), ToolCalls: []llm.ToolCall{completionCall("done", "d", `v`)}}, {ResponseID: "done", ToolCalls: []llm.ToolCall{completionCall("done", "d", `ed."}`)}}}, "Answer. Details.\n\nSaved.", "done"},
				} {
					t.Run(fmt.Sprintf("%s/stream%t/query%t/require%t", tc.name, stream, query, required), func(t *testing.T) {
						base := &completionTurnModel{turns: tc.turns}
						var model llm.ChatModel = base
						if stream {
							model = completionTurnStreamer{base}
						}
						a, err := New(Config{LLM: model, Tools: []tools.Tool{work, done}, RequireDoneTool: required, Warningf: func(string, ...any) {}})
						if err != nil {
							t.Fatal(err)
						}
						if query {
							got, e := a.Query(context.Background(), "test")
							if e != nil || got != tc.want {
								t.Fatalf("got %q %v want %q", got, e, tc.want)
							}
						} else {
							finals := 0
							for env := range a.QueryStreamEnveloped(context.Background(), llm.TextContent("test")) {
								switch e := env.Event.(type) {
								case ErrorEvent:
									t.Fatal(e)
								case FinalResponseEvent:
									finals++
									if e.Content != tc.want || e.ResponseID != tc.id {
										t.Fatalf("final %+v want %q id %q", e, tc.want, tc.id)
									}
								}
							}
							if finals != 1 {
								t.Fatalf("finals=%d", finals)
							}
						}
						history := a.Messages()
						calls, results := 0, 0
						for _, m := range history {
							calls += len(m.ToolCalls)
							if m.Role == llm.RoleTool {
								results++
							}
						}
						if calls != results {
							t.Fatalf("unpaired history %d %d", calls, results)
						}
					})
				}
			}
		}
	}
}
func TestCompletionRetainsTruncatedToolArgumentText(t *testing.T) {
	done := tools.Func[struct {
		Message string `json:"message"`
	}]("done", "finish", func(_ context.Context, x struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) {
		return nil, tools.TaskComplete(x.Message)
	})
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprint(stream), func(t *testing.T) {
			base := &completionTurnModel{turns: []*llm.Completion{{Content: llm.TextContent("Answer."), ResponseID: "part", StopReason: "max_tokens", ToolCalls: []llm.ToolCall{completionCall("done", "d", `{"message":"Sa`)}}, {ResponseID: "done", ToolCalls: []llm.ToolCall{completionCall("done", "d", `ved."}`)}}}}
			var model llm.ChatModel = base
			if stream {
				model = completionTurnStreamer{base}
			}
			a, e := New(Config{LLM: model, Tools: []tools.Tool{done}, Warningf: func(string, ...any) {}})
			if e != nil {
				t.Fatal(e)
			}
			got, e := a.Query(context.Background(), "test")
			t.Logf("actual snapshot %q err=%v", got, e)
			if got != "Answer.\n\nSaved." || e != nil {
				t.Fatalf("lost retained answer: %q %v", got, e)
			}
		})
	}
}
func TestCompletionTextThenTruncatedDoneArguments(t *testing.T) {
	done := tools.Func[struct {
		Message string `json:"message"`
	}]("done", "finish", func(_ context.Context, x struct {
		Message string `json:"message"`
	}, _ *tools.Container) (any, error) {
		return nil, tools.TaskComplete(x.Message)
	})
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprint(stream), func(t *testing.T) {
			base := &completionTurnModel{turns: []*llm.Completion{{Content: llm.TextContent("Answer."), ResponseID: "text", StopReason: "max_tokens"}, {ResponseID: "part", StopReason: "max_tokens", ToolCalls: []llm.ToolCall{completionCall("done", "d", `{"message":"Sa`)}}, {ResponseID: "done", ToolCalls: []llm.ToolCall{completionCall("done", "d", `ved."}`)}}}}
			var model llm.ChatModel = base
			if stream {
				model = completionTurnStreamer{base}
			}
			a, e := New(Config{LLM: model, Tools: []tools.Tool{done}, Warningf: func(string, ...any) {}})
			if e != nil {
				t.Fatal(e)
			}
			got, e := a.Query(context.Background(), "test")
			t.Logf("actual snapshot %q err=%v", got, e)
			if got != "Answer.\n\nSaved." || e != nil {
				t.Fatalf("lost text continuation: %q %v", got, e)
			}
		})
	}
}

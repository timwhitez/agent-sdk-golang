package agent

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type mixedToolBlockModel struct{}

func (mixedToolBlockModel) Provider() string { return "fixture" }
func (mixedToolBlockModel) Model() string    { return "mixed-tool-block" }
func (mixedToolBlockModel) Invoke(context.Context, llm.InvokeRequest) (*llm.Completion, error) {
	call := func(id, name string) llm.ToolCall {
		return llm.ToolCall{
			ID:       id,
			Type:     "function",
			Function: llm.FunctionCall{Name: name, Arguments: `{}`},
		}
	}
	return &llm.Completion{
		ToolCalls: []llm.ToolCall{
			call("unknown-1", "mystery"),
			call("panic-1", "panic_tool"),
			call("error-1", "error_tool"),
			call("done-1", "done"),
			call("tail-1", "tail_tool"),
		},
		StopReason: "tool_calls",
	}, nil
}

func TestToolBlockSequentialClosureCharacterization(t *testing.T) {
	starts := map[string]int{}
	tool := func(name string, handler func() (llm.Content, error)) tools.Tool {
		return tools.Tool{
			Name: name,
			Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
				starts[name]++
				return handler()
			},
		}
	}
	invalid := tool("invalid", func() (llm.Content, error) {
		return llm.TextContent("unknown result"), nil
	})
	invalid.Hidden = true
	panicTool := tool("panic_tool", func() (llm.Content, error) {
		panic("boom")
	})
	errorTool := tool("error_tool", func() (llm.Content, error) {
		return llm.TextContent("ordinary failure result"), errors.New("ordinary failure")
	})
	done := tool("done", func() (llm.Content, error) {
		return llm.Content{}, &tools.TaskCompleteError{Message: "finished"}
	})
	tail := tool("tail_tool", func() (llm.Content, error) {
		return llm.TextContent("must not run"), nil
	})

	agent, err := New(Config{
		LLM:   mixedToolBlockModel{},
		Tools: []tools.Tool{invalid, panicTool, errorTool, done, tail},
	})
	if err != nil {
		t.Fatal(err)
	}
	events := collectEvents(agent.QueryStream(context.Background(), llm.TextContent("run")))

	wantStarts := map[string]int{
		"invalid": 1, "panic_tool": 1, "error_tool": 1, "done": 1, "tail_tool": 0,
	}
	for name, want := range wantStarts {
		if got := starts[name]; got != want {
			t.Fatalf("%s starts=%d want %d", name, got, want)
		}
	}

	messages := agent.Messages()
	assistantIndex := -1
	for i, message := range messages {
		if message.Role == llm.RoleAssistant && len(message.ToolCalls) == 5 {
			assistantIndex = i
			break
		}
	}
	if assistantIndex < 0 || assistantIndex+5 >= len(messages) {
		t.Fatalf("missing complete mixed tool block: %#v", messages)
	}

	wantIDs := []string{"unknown-1", "panic-1", "error-1", "done-1", "tail-1"}
	wantNames := []string{"mystery", "panic_tool", "error_tool", "done", "tail_tool"}
	wantResults := []struct {
		text    string
		isError bool
	}{
		{text: "unknown result", isError: true},
		{text: `Error: tool "panic_tool" panicked: boom`, isError: true},
		{text: "ordinary failure result", isError: true},
		{text: "Task completed: finished", isError: false},
		{text: toolSkippedByTurnEndText, isError: true},
	}
	for i, call := range messages[assistantIndex].ToolCalls {
		if call.ID != wantIDs[i] || call.Function.Name != wantNames[i] {
			t.Fatalf("assistant call[%d]=%#v want id=%q name=%q", i, call, wantIDs[i], wantNames[i])
		}
		result := messages[assistantIndex+1+i]
		if result.Role != llm.RoleTool || result.ToolCallID != wantIDs[i] {
			t.Fatalf("result[%d]=%#v want tool id=%q", i, result, wantIDs[i])
		}
		if result.IsError != wantResults[i].isError || result.Content.PlainText() != wantResults[i].text {
			t.Fatalf("result[%d]=%#v want text=%q is_error=%v", i, result, wantResults[i].text, wantResults[i].isError)
		}
	}

	if _, changed, unexpected := repairToolCallPairsDetailed(messages); changed || unexpected {
		t.Fatalf("closed block required outbound repair: changed=%v unexpected=%v", changed, unexpected)
	}
	var final FinalResponseEvent
	for _, event := range events {
		if candidate, ok := event.(FinalResponseEvent); ok {
			final = candidate
		}
	}
	if final.Content != "finished" {
		t.Fatalf("final=%#v want finished", final)
	}
}

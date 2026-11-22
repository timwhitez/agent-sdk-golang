package agent

import (
	"context"
	"fmt"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools/sandbox"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func issue282Call(id, name, args string) llm.ToolCall {
	return llm.ToolCall{ID: id, Type: "function", Function: llm.FunctionCall{Name: name, Arguments: args}}
}
func TestD1FailedReadDoesNotCoverCorrectedRange(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "report.txt"), []byte("first line\nsecond line\n"), 0600); err != nil {
		t.Fatal(err)
	}
	sb, err := sandbox.New(dir)
	if err != nil {
		t.Fatal(err)
	}
	deps := tools.NewContainer()
	tools.Provide(deps, sandbox.Key, func(context.Context) (*sandbox.Sandbox, error) { return sb, nil })
	model := &turnModel{}
	for i, limit := range []int{10001, 10001, 1} {
		model.turns = append(model.turns, &llm.Completion{ToolCalls: []llm.ToolCall{issue282Call(fmt.Sprint(i), "read", fmt.Sprintf(`{"file_path":"report.txt","offset":1,"limit":%d}`, limit))}})
	}
	var read tools.Tool
	for _, tool := range sandbox.Tools() {
		if tool.Name == "read" {
			read = tool
		}
	}
	a, err := New(Config{LLM: model, Tools: []tools.Tool{read}, Deps: deps})
	if err != nil {
		t.Fatal(err)
	}
	var results []ToolResultEvent
	for ev := range a.QueryStream(t.Context(), llm.TextContent("read the report")) {
		if r, ok := ev.(ToolResultEvent); ok {
			results = append(results, r)
		}
	}
	if len(results) != 3 || !results[0].IsError || !results[1].IsError {
		t.Fatalf("unexpected prior results: %+v", results)
	}
	if !strings.Contains(results[2].Result, "first line") {
		t.Fatalf("two rejected reads followed by corrected limit=1 yielded %q; want first line", results[2].Result)
	}
}

package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestIndependentArtifactReferencesAreNotEvidenceProgress(t *testing.T) {
	handled := 0
	tool := tools.Tool{Name: "read", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
		handled++
		return llm.TextContent(strings.Repeat("identical content\n", 10000)), nil
	}}
	model := &turnModel{}
	for i := 0; i < 3; i++ {
		model.turns = append(model.turns, &llm.Completion{ToolCalls: []llm.ToolCall{issue282Call(fmt.Sprint(i), "read", `{"file_path":"report.txt","offset":1,"limit":2000}`)}})
	}
	sink := &artifactBoundarySink{}
	a, err := New(Config{LLM: model, Tools: []tools.Tool{tool}, MaxToolResultBytes: 4096, MaxToolResultTokens: 2000, ArtifactSink: sink, ArtifactOwner: artifactBoundaryOwner(), ArtifactResolverCapability: artifactBoundaryCapability("Call artifact_read with object_ref and byte range."), Compaction: &compaction.Config{Enabled: false, ContextWindow: 10_000_000}})
	if err != nil {
		t.Fatal(err)
	}
	var results []ToolResultEvent
	for ev := range a.QueryStream(t.Context(), llm.TextContent("read")) {
		if r, ok := ev.(ToolResultEvent); ok {
			results = append(results, r)
		}
	}
	if len(results) != 3 {
		t.Fatalf("results=%d", len(results))
	}
	if handled != 2 || !strings.Contains(results[2].Result, "[already_observed]") {
		t.Fatalf("identical read executed %d times, final suppressed=%v; expected 2 executions and third suppressed", handled, strings.Contains(results[2].Result, "[already_observed]"))
	}
}

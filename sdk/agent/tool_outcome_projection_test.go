package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	sdkaccounting "github.com/timwhitez/agent-sdk-golang/sdk/accounting"
	"github.com/timwhitez/agent-sdk-golang/sdk/artifact"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// Freeze one real trajectory across execution knowledge and all result views.
// Artifact failure is not evidence that a completed handler never executed.
func TestToolOutcomeProjectionCharacterization(t *testing.T) {
	for _, complete := range []bool{false, true} {
		for _, failureStage := range []string{"none", "artifact_sink", "artifact_codec"} {
			t.Run(fmt.Sprintf("task_complete=%v/projection=%s", complete, failureStage), func(t *testing.T) {
				sink := &artifactBoundarySink{}
				if failureStage == "artifact_sink" {
					sink.err = errors.New("fixture storage unavailable")
				}
				maxBytes, maxTokens := 4096, 2500
				if failureStage == "artifact_codec" {
					maxBytes, maxTokens = 1024, 300
				}
				raw := strings.Repeat("private-result-marker ", 1000)
				original := raw
				closure := "handler_return"
				if complete {
					original = "Task completed: " + raw
					closure = "task_complete"
				}
				effects := 0
				ag, err := New(Config{
					LLM: &stubModel{toolName: "large_result", toolArgs: `{}`, toolID: "call-projection"},
					Tools: []tools.Tool{{Name: "large_result", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) {
						effects++
						if complete {
							return llm.Content{}, tools.TaskComplete(raw)
						}
						return llm.TextContent(raw), nil
					}}},
					MaxToolResultBytes: maxBytes, MaxToolResultTokens: maxTokens,
					ArtifactOwner: artifactBoundaryOwner(), ArtifactSink: sink,
					ArtifactResolverCapability: artifactBoundaryCapability("Call artifact_read with object_ref and byte range."),
					ArtifactEnvelopeCodec:      artifact.JSONEnvelopeCodec{},
					Warningf:                   func(string, ...any) {},
				})
				if err != nil {
					t.Fatal(err)
				}
				var states []toolCallState
				ag.toolBlockStateObserved = func(block *toolBlockState) { states = append(states, block.calls...) }
				events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("run")))
				if effects != 1 || len(sink.requests) != 1 {
					t.Fatalf("handler/sink=%d/%d", effects, len(sink.requests))
				}
				wantObjects := 1
				if failureStage == "artifact_sink" {
					wantObjects = 0
				}
				if len(sink.objects) != wantObjects {
					t.Fatal("projection failure changed actual stored-object evidence")
				}
				if len(states) != 1 || states[0].phase != toolCallTerminal || states[0].terminalCount != 1 || states[0].executionKnowledge != toolExecutionOutcomeObserved || states[0].closure != closure {
					t.Fatalf("terminal knowledge=%+v", states)
				}
				var result ToolResultEvent
				var accounting AccountingEvent
				results, accounts, resultIndex, accountingIndex := 0, 0, -1, -1
				for i, event := range events {
					switch e := event.(type) {
					case ToolResultEvent:
						result = e
						results++
						resultIndex = i
					case AccountingEvent:
						if e.Payload.EventKind == sdkaccounting.EventKindToolResult {
							accounting = e
							accounts++
							accountingIndex = i
						}
					}
				}
				if results != 1 || accounts != 1 || accountingIndex != resultIndex+1 {
					t.Fatalf("results/accounts/index=%d/%d/%d/%d", results, accounts, resultIndex, accountingIndex)
				}
				if result.IsError || result.ToolCallID != "call-projection" || accounting.ToolCallID != result.ToolCallID || accounting.Payload.Status != sdkaccounting.StatusSuccess {
					t.Fatal("projection failure changed successful handler status/correlation")
				}
				historyResults := 0
				for _, message := range ag.Messages() {
					if message.Role != llm.RoleTool {
						continue
					}
					historyResults++
					if message.ToolCallID != result.ToolCallID || message.ToolName != result.Tool || message.IsError != result.IsError || message.Content.PlainText() != result.Result {
						t.Fatal("history and result event diverged")
					}
				}
				if historyResults != 1 {
					t.Fatalf("history results=%d", historyResults)
				}
				if _, changed, _ := repairToolCallPairsDetailed(ag.Messages()); changed {
					t.Fatal("completed block needs repair")
				}
				measurement := accounting.Payload.Measurements
				if measurement.OriginalBytes == nil || *measurement.OriginalBytes != int64(len(original)) || measurement.VisibleBytes == nil || *measurement.VisibleBytes != int64(len(result.Result)) {
					t.Fatal("original and projected measurements conflated")
				}
				disposition := accounting.Payload.Artifact
				projectionFailed := failureStage != "none"
				if disposition == nil || disposition.Complete == nil || disposition.Recoverable == nil || *disposition.Complete == projectionFailed || *disposition.Recoverable == projectionFailed {
					t.Fatal("artifact completeness lost")
				}
				if projectionFailed {
					if disposition.ObjectRef != "" || result.Metadata["artifact_manifest"] != nil || !strings.Contains(result.Result, "stage="+failureStage) {
						t.Fatal("failed projection invented recovery or hid failure")
					}
				} else if disposition.ObjectRef == "" || result.Metadata["artifact_manifest"] == nil {
					t.Fatal("successful projection lost artifact")
				}
				encoded, err := json.Marshal(accounting.Payload)
				if err != nil {
					t.Fatal(err)
				}
				if strings.Contains(string(encoded), "private-result-marker") || strings.Contains(string(encoded), "fixture storage unavailable") {
					t.Fatal("raw data leaked into accounting")
				}
				if err := accounting.Payload.Validate(); err != nil {
					t.Fatal(err)
				}
			})
		}
	}
}

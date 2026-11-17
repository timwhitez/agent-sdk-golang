package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/artifact"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestAgentArtifactBoundaryTreatsMarkerInPlainOutputAsPlain(t *testing.T) {
	small := "const EnvelopeStartMarker = \"" + artifact.EnvelopeStartMarker + "\"\nconst EnvelopeEndMarker = \"" + artifact.EnvelopeEndMarker + "\"\n"
	large := small + strings.Repeat("plain-output-", 20_000)
	for name, tc := range map[string]struct {
		output     string
		wantStored bool
	}{
		"within budget":  {small, false},
		"exceeds budget": {large, true},
	} {
		t.Run(name, func(t *testing.T) {
			sink := &artifactBoundarySink{}
			ag, err := New(Config{
				LLM:                        &stubModel{toolName: "large_result", toolArgs: `{}`, toolID: "call-marker"},
				Tools:                      []tools.Tool{artifactBoundaryTool(tc.output)},
				MaxToolResultBytes:         4_096,
				MaxToolResultTokens:        2_500,
				ArtifactOwner:              artifactBoundaryOwner(),
				ArtifactSink:               sink,
				ArtifactResolverCapability: artifactBoundaryCapability("Call artifact_read with object_ref and byte range."),
				ArtifactEnvelopeCodec:      artifact.JSONEnvelopeCodec{},
			})
			if err != nil {
				t.Fatal(err)
			}
			result, ok := findToolResult(collectEvents(ag.QueryStream(context.Background(), llm.TextContent("run"))), "large_result")
			if !ok {
				t.Fatal("missing tool result event")
			}
			if strings.Contains(result.Result, "stage=artifact_decode") || result.Metadata["artifact_recoverable"] == false {
				t.Fatalf("plain output was flagged as a corrupted envelope: %q %#v", result.Result, result.Metadata)
			}
			if !tc.wantStored {
				if result.Result != tc.output {
					t.Fatalf("plain output was rewritten: %q", result.Result)
				}
				return
			}
			if len(sink.requests) != 1 || string(sink.requests[0].Content) != tc.output {
				t.Fatalf("oversized plain output did not reach the sink: requests=%d", len(sink.requests))
			}
		})
	}
}

func TestIsFramedEnvelope(t *testing.T) {
	framed := artifact.EnvelopeStartMarker + "\n{}\n" + artifact.EnvelopeEndMarker
	for text, want := range map[string]bool{
		framed:                       true,
		"\n " + framed + "\n":        true,
		"see " + framed:              false,
		framed + " trailing":         false,
		artifact.EnvelopeStartMarker: false,
		"plain":                      false,
	} {
		if got := artifact.IsFramedEnvelope(text); got != want {
			t.Errorf("IsFramedEnvelope(%q) = %v, want %v", text, got, want)
		}
	}
}

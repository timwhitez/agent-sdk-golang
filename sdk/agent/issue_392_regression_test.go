package agent

import (
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/artifact"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestR2DefaultCodecFormsTreatQuotedMarkersAsPlainThroughQuery(t *testing.T) {
	const plain = `const marker = "` + artifact.EnvelopeStartMarker + `";`
	for _, tc := range []struct {
		name  string
		codec artifact.EnvelopeCodec
	}{
		{"nil", nil},
		{"value", artifact.JSONEnvelopeCodec{}},
		{"pointer", &artifact.JSONEnvelopeCodec{}},
	} {
		for _, large := range []bool{false, true} {
			name := tc.name + "/small"
			output := plain
			if large {
				name = tc.name + "/large"
				output += strings.Repeat("regular source\n", 1000)
			}
			t.Run(name, func(t *testing.T) {
				sink := &artifactBoundarySink{}
				ag, err := New(Config{
					LLM:                &stubModel{toolName: "large_result", toolArgs: `{}`, toolID: "call-pointer-marker"},
					Tools:              []tools.Tool{artifactBoundaryTool(output)},
					MaxToolResultBytes: 4096, MaxToolResultTokens: 2500,
					ArtifactOwner: artifactBoundaryOwner(), ArtifactSink: sink,
					ArtifactResolverCapability: artifactBoundaryCapability("Call artifact_read with object_ref and byte range."),
					ArtifactEnvelopeCodec:      tc.codec,
				})
				if err != nil {
					t.Fatal(err)
				}
				result, ok := findToolResult(collectEvents(ag.QueryStream(t.Context(), llm.TextContent("run"))), "large_result")
				if !ok {
					t.Fatal("missing tool result event")
				}
				if strings.Contains(result.Result, "stage=artifact_decode") || result.Metadata["artifact_recoverable"] == false {
					t.Fatalf("default %s codec corrupted ordinary source: result_bytes=%d stage=%v recoverable=%v puts=%d", tc.name, len(result.Result), result.Metadata["artifact_stage"], result.Metadata["artifact_recoverable"], len(sink.requests))
				}
				if !large && result.Result != output {
					t.Fatalf("small source changed: %q", result.Result)
				}
				if large && (len(sink.requests) != 1 || string(sink.requests[0].Content) != output) {
					t.Fatalf("large source not archived exactly: puts=%d", len(sink.requests))
				}
			})
		}
	}
}

type r2CustomRecognitionCodec struct {
	artifact.JSONEnvelopeCodec
	calls  atomic.Int32
	reject bool
}

func (codec *r2CustomRecognitionCodec) Decode(string) (artifact.Envelope, bool, error) {
	codec.calls.Add(1)
	if codec.reject {
		return artifact.Envelope{}, true, errors.New("custom codec recognizes and rejects this source")
	}
	return artifact.Envelope{}, false, nil
}

func TestR2CustomCodecStillControlsUnframedRecognition(t *testing.T) {
	const output = "ordinary source with " + artifact.EnvelopeStartMarker
	for _, reject := range []bool{false, true} {
		codec := &r2CustomRecognitionCodec{reject: reject}
		ag, err := New(Config{
			LLM:                &stubModel{toolName: "large_result", toolArgs: `{}`, toolID: "call-custom-marker"},
			Tools:              []tools.Tool{artifactBoundaryTool(output)},
			MaxToolResultBytes: 4096, MaxToolResultTokens: 2500,
			ArtifactEnvelopeCodec: codec,
		})
		if err != nil {
			t.Fatal(err)
		}
		result, ok := findToolResult(collectEvents(ag.QueryStream(t.Context(), llm.TextContent("run"))), "large_result")
		if !ok {
			t.Fatal("missing tool result event")
		}
		if codec.calls.Load() != 1 || strings.Contains(result.Result, "stage=artifact_decode") != reject {
			t.Fatalf("custom recognition changed: reject=%v calls=%d result=%q", reject, codec.calls.Load(), result.Result)
		}
		if !reject && result.Result != output {
			t.Fatalf("custom codec declined recognition but source changed: %q", result.Result)
		}
	}
}

func TestDefaultCodecFormsRejectMalformedFramedEnvelope(t *testing.T) {
	const malformed = artifact.EnvelopeStartMarker + "\n{}\n" + artifact.EnvelopeEndMarker
	for name, codec := range map[string]artifact.EnvelopeCodec{
		"nil": nil, "value": artifact.JSONEnvelopeCodec{}, "pointer": &artifact.JSONEnvelopeCodec{},
	} {
		t.Run(name, func(t *testing.T) {
			ag, err := New(Config{
				LLM:   &stubModel{toolName: "large_result", toolArgs: `{}`, toolID: "call-malformed"},
				Tools: []tools.Tool{artifactBoundaryTool(malformed)}, ArtifactEnvelopeCodec: codec,
			})
			if err != nil {
				t.Fatal(err)
			}
			result, ok := findToolResult(collectEvents(ag.QueryStream(t.Context(), llm.TextContent("run"))), "large_result")
			if !ok || result.Metadata["artifact_stage"] != artifactDecodeFailureStage || result.Metadata["artifact_recoverable"] != false {
				t.Fatalf("malformed frame bypassed validation: found=%v metadata=%#v", ok, result.Metadata)
			}
		})
	}
}

package agent

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/artifact"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

func TestRound3AgentArtifactBoundaryPreservesToolAttachments(t *testing.T) {
	for _, mode := range []string{"small_control", "canonical", "fallback"} {
		t.Run(mode, func(t *testing.T) {
			text := "short explanation"
			if mode != "small_control" {
				text = strings.Repeat("The attached diagram and report provide the evidence.\n", 400)
			}
			original := llm.Content{Blocks: []llm.ContentBlock{
				{Type: "image_url", ImageURL: &llm.ImageURL{URL: "https://example.test/evidence.png"}},
				{Type: "text", Text: text},
				{Type: "document", Source: &llm.DocSrc{MediaType: "application/pdf", Data: "c3VwcG9ydGluZyBldmlkZW5jZQ=="}},
			}}
			calls := 0
			var delivered llm.Content
			model := &frameScriptModel{invoke: func(req llm.InvokeRequest) (*llm.Completion, error) {
				calls++
				if calls == 1 {
					return &llm.Completion{ToolCalls: []llm.ToolCall{{ID: "call-attachments", Function: llm.FunctionCall{Name: "multimodal", Arguments: `{}`}}}}, nil
				}
				for _, message := range req.Messages {
					if message.Role == llm.RoleTool {
						delivered = llm.CloneContent(message.Content)
					}
				}
				return &llm.Completion{Content: llm.TextContent("done")}, nil
			}}
			cfg := Config{
				LLM: model, MaxIterations: 3, MaxToolResultBytes: 4096, MaxToolResultTokens: 2000,
				Tools: []tools.Tool{{Name: "multimodal", Handler: func(context.Context, json.RawMessage, *tools.Container) (llm.Content, error) { return original, nil }}},
			}
			if mode == "canonical" {
				cfg.ArtifactOwner = artifactBoundaryOwner()
				cfg.ArtifactSink = &artifactBoundarySink{}
				cfg.ArtifactResolverCapability = artifactBoundaryCapability("Call artifact_read with object_ref and byte range.")
			}
			ag, err := New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			if result, err := ag.Query(t.Context(), "inspect the attachments"); err != nil || result != "done" {
				t.Fatalf("query=%q err=%v", result, err)
			}
			if calls != 2 {
				t.Fatalf("provider calls=%d", calls)
			}
			for label, content := range map[string]llm.Content{"provider request": delivered, "history": round3ToolContent(ag.Messages())} {
				if len(content.Blocks) != 3 || !reflect.DeepEqual(content.Blocks[0], original.Blocks[0]) || content.Blocks[1].Type != "text" || !reflect.DeepEqual(content.Blocks[2], original.Blocks[2]) {
					t.Errorf("%s lost or reordered attachments at %s boundary: blocks=%d", label, mode, len(content.Blocks))
				}
				if mode != "small_control" && content.PlainText() == text {
					t.Errorf("%s did not exercise text bounding", label)
				}
			}
		})
	}
}

func round3ToolContent(messages []llm.Message) llm.Content {
	for _, message := range messages {
		if message.Role == llm.RoleTool {
			return message.Content
		}
	}
	return llm.Content{}
}

type attachmentRejectEncodeCodec struct{ artifact.JSONEnvelopeCodec }

func (attachmentRejectEncodeCodec) Encode(artifact.Envelope, artifact.Budget) (string, artifact.Envelope, error) {
	return "", artifact.Envelope{}, errors.New("fixture encode failure")
}

func TestToolArtifactProjectionPreservesEveryBoundaryAndOwnership(t *testing.T) {
	for _, mode := range []string{"plain", "canonical", "existing", "decode_failure", "owner_failure", "capability_failure", "sink_failure", "manifest_failure", "codec_failure", "lineage_failure"} {
		t.Run(mode, func(t *testing.T) {
			sink := &artifactBoundarySink{}
			cfg := Config{
				LLM: &stubModel{}, MaxToolResultBytes: 4096, MaxToolResultTokens: 2000,
				ArtifactOwner: artifactBoundaryOwner(), ArtifactSink: sink,
				ArtifactResolverCapability: artifactBoundaryCapability("Call artifact_read with object_ref and byte range."),
			}
			text := strings.Repeat("source text with evidence\n", 2000)
			var meta map[string]any
			switch mode {
			case "plain":
				text = "short evidence"
			case "decode_failure":
				text = artifact.EnvelopeStartMarker + "\n{}\n" + artifact.EnvelopeEndMarker
			case "owner_failure":
				cfg.ArtifactOwner = artifact.Owner{}
			case "capability_failure":
				cfg.ArtifactResolverCapability.Registered = false
			case "sink_failure":
				sink.err = errors.New("fixture sink failure")
			case "manifest_failure":
				sink.mutate = func(m *artifact.Manifest) { m.Owner.SubjectID = "wrong-owner" }
			case "codec_failure":
				cfg.ArtifactEnvelopeCodec = attachmentRejectEncodeCodec{}
			case "lineage_failure":
				meta = map[string]any{artifact.SourceManifestsMetadataKey: "invalid manifest type"}
			}
			ag, err := New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			if mode == "existing" {
				ag.maxToolResultBytes, ag.maxToolResultTokens = 20000, 10000
				encoded, _ := ag.applyToolResultBoundary(t.Context(), llm.TextContent(text), nil, "evidence", "call-evidence")
				text = encoded.PlainText()
				if len(text) <= 4096 || len(sink.requests) != 1 {
					t.Fatal("fixture did not create a large canonical envelope")
				}
				ag.maxToolResultBytes, ag.maxToolResultTokens = 4096, 2000
			}
			input := llm.Content{Blocks: []llm.ContentBlock{
				{Type: "image_url", ImageURL: &llm.ImageURL{URL: "https://example.test/chart.png", Detail: "high"}},
				{Type: "text", Text: text},
				{Type: "document", Source: &llm.DocSrc{MediaType: "application/pdf", Data: "cmVwb3J0"}},
				{Type: "future_attachment", Data: "preserve opaque attachment"},
			}}
			input, err = llm.WithProviderState(input, []llm.ProviderState{{Provider: "fixture", Kind: "state", Data: json.RawMessage(`{"text":"original"}`)}})
			if err != nil {
				t.Fatal(err)
			}
			snapshot := llm.CloneContent(input)
			out, metadata := ag.applyToolResultBoundary(t.Context(), input, meta, "evidence", "call-evidence")
			wantBlocks := 4
			if mode == "plain" {
				wantBlocks++
			}
			if len(out.Blocks) != wantBlocks || !reflect.DeepEqual(out.Blocks[0], input.Blocks[0]) || !reflect.DeepEqual(out.Blocks[2:4], input.Blocks[2:4]) {
				t.Fatalf("attachments changed: %#v", out.Blocks)
			}
			if mode == "plain" {
				if !reflect.DeepEqual(out, snapshot) {
					t.Fatal("within-budget content was rewritten")
				}
			} else {
				if out.PlainText() == text || ag.toolResultExceedsBudget(out.PlainText()) || llm.HasProviderState(out) {
					t.Fatal("text was not bounded or stale provider state was retained")
				}
				if mode == "canonical" || mode == "existing" {
					if _, found, err := (artifact.JSONEnvelopeCodec{}).Decode(out.PlainText()); !found || err != nil {
						t.Fatalf("canonical envelope lost: found=%v err=%v", found, err)
					}
					if len(sink.requests) != 1 {
						t.Fatal("canonical source was not archived exactly once")
					}
				} else if metadata["artifact_recoverable"] != false {
					t.Fatalf("fallback claimed recovery: %#v", metadata)
				}
			}
			out.Blocks[0].ImageURL.URL = "changed"
			out.Blocks[2].Source.Data = "changed"
			out.Blocks[3].Data = "changed"
			if !reflect.DeepEqual(input, snapshot) {
				t.Fatal("projected result aliases handler-owned content")
			}
		})
	}
}

func TestToolArtifactMultipleTextBlocksKeepAttachmentAssociations(t *testing.T) {
	for _, mode := range []string{"small", "large_second", "large_first", "whitespace", "mixed", "tiny_budget", "source_metadata", "framed"} {
		t.Run(mode, func(t *testing.T) {
			sink := &artifactBoundarySink{}
			ag, err := New(Config{
				LLM: &stubModel{}, MaxToolResultBytes: 4096, MaxToolResultTokens: 2000,
				ArtifactOwner: artifactBoundaryOwner(), ArtifactSink: sink,
				ArtifactResolverCapability: artifactBoundaryCapability("Call artifact_read with object_ref and byte range."),
			})
			if err != nil {
				t.Fatal(err)
			}
			before, after := "before image", "after image"
			var meta map[string]any
			switch mode {
			case "large_second", "tiny_budget":
				after = strings.Repeat(after+"\n", 1000)
			case "large_first":
				before = strings.Repeat(before+"\n", 1000)
			case "whitespace":
				before, after = " \n\t", strings.Repeat(after+"\n", 1000)
			case "source_metadata":
				meta = map[string]any{artifact.SourceManifestsMetadataKey: "cannot authenticate a multi-part projection"}
			case "framed":
				before, after = artifact.EnvelopeStartMarker+"\n{}", artifact.EnvelopeEndMarker
			}
			if mode == "tiny_budget" {
				ag.maxToolResultBytes, ag.maxToolResultTokens = 32, 8
			}
			input := llm.Content{Blocks: []llm.ContentBlock{
				{Type: "text", Text: before},
				{Type: "image_url", ImageURL: &llm.ImageURL{URL: "https://example.test/chart.png"}},
				{Type: "text", Text: after},
				{Type: "document", Source: &llm.DocSrc{MediaType: "application/pdf", Data: "cmVwb3J0"}},
			}}
			if mode == "mixed" {
				input.Text = "additional Text field must not be ignored"
			}
			snapshot := llm.CloneContent(input)
			layout := llm.CloneContent(input)
			if mode == "mixed" {
				layout.Blocks = append([]llm.ContentBlock{{Type: "text", Text: input.Text}}, layout.Blocks...)
				layout.Text = ""
			}
			out, metadata := ag.applyToolResultBoundary(t.Context(), input, meta, "evidence", "call-evidence")
			if len(out.Blocks) != len(layout.Blocks) || len(sink.requests) != 0 || out.Text != "" {
				t.Fatalf("ambiguous layout was flattened or archived: blocks=%d puts=%d", len(out.Blocks), len(sink.requests))
			}
			if mode == "small" {
				if !reflect.DeepEqual(out, input) {
					t.Fatal("small multi-part result changed")
				}
			} else {
				if metadata["artifact_stage"] != artifactProjectionFailureStage || metadata["artifact_recoverable"] != false || metadata["artifact_complete"] != false {
					t.Fatalf("missing explicit projection fallback: %#v", metadata)
				}
				if _, retained := metadata[artifact.SourceManifestsMetadataKey]; retained {
					t.Fatal("unsupported projection retained source recovery claims")
				}
				if metadata["result_original_bytes"] != len(layout.PlainText()) || metadata["result_bytes"] != len(out.PlainText()) || ag.toolResultExceedsBudget(out.PlainText()) {
					t.Fatal("projection ignored source text or exceeded the aggregate budget")
				}
			}
			for i, block := range layout.Blocks {
				if block.Type != "text" {
					if !reflect.DeepEqual(out.Blocks[i], block) {
						t.Fatalf("attachment %d moved or changed", i)
					}
					continue
				}
				visible := out.Blocks[i].Text
				if pos := strings.Index(visible, "[WARN]"); pos >= 0 {
					visible = strings.TrimSuffix(visible[:pos], "\n")
				}
				if !strings.HasPrefix(block.Text, visible) {
					t.Fatalf("text moved between slots: slot %d output=%q original=%q", i, visible, block.Text)
				}
			}
			for i := range out.Blocks {
				if out.Blocks[i].ImageURL != nil {
					out.Blocks[i].ImageURL.URL = "changed"
				}
			}
			if !reflect.DeepEqual(input, snapshot) {
				t.Fatal("multi-part projection aliases input attachments")
			}
		})
	}
}

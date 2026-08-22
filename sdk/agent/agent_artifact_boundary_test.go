package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/timwhitez/agent-sdk-golang/sdk/artifact"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

type artifactBoundarySink struct {
	mu       sync.Mutex
	requests []artifact.PutRequest
	objects  map[string]artifact.ResolveResult
	err      error
	mutate   func(*artifact.Manifest)
}

func (s *artifactBoundarySink) Put(_ context.Context, request artifact.PutRequest) (artifact.Manifest, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.requests = append(s.requests, cloneArtifactPutRequest(request))
	if s.err != nil {
		return artifact.Manifest{}, s.err
	}
	if s.objects == nil {
		s.objects = make(map[string]artifact.ResolveResult)
	}
	ref := fmt.Sprintf("obj:v1:agent-result-%03d", len(s.requests))
	bytesCount := int64(len(request.Content))
	linesCount := int64(0)
	if len(request.Content) > 0 {
		linesCount = int64(strings.Count(string(request.Content), "\n") + 1)
	}
	complete := true
	manifest := artifact.Manifest{
		SchemaVersion: artifact.SchemaVersion,
		ObjectRef:     ref,
		ObjectKind:    request.ObjectKind,
		Owner:         request.Owner,
		Lineage:       request.Lineage,
		Complete:      true,
		Recoverable:   true,
		ObjectMeasurement: artifact.Measurement{
			Bytes:             &bytesCount,
			Lines:             &linesCount,
			SHA256:            artifact.DigestSHA256(request.Content),
			MeasurementSource: "test_host_sink",
			Complete:          &complete,
		},
		Preview: artifact.Preview{
			Kind: artifact.PreviewKindNone,
		},
		Retention:   request.Retention,
		ContentType: request.ContentType,
		Encoding:    request.Encoding,
		Recovery:    request.Recovery,
	}
	if s.mutate != nil {
		s.mutate(&manifest)
	}
	if err := manifest.Validate(); err != nil {
		return artifact.Manifest{}, fmt.Errorf("test sink produced invalid manifest: %w", err)
	}
	s.objects[ref] = artifact.ResolveResult{
		Manifest: manifest.Clone(),
		Content:  append([]byte(nil), request.Content...),
	}
	return manifest, nil
}

func (s *artifactBoundarySink) Resolve(_ context.Context, request artifact.ResolveRequest) (artifact.ResolveResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	result, ok := s.objects[request.ObjectRef]
	if !ok {
		return artifact.ResolveResult{}, errors.New("artifact not found")
	}
	if result.Manifest.Owner.WorkspaceID != request.Owner.WorkspaceID ||
		result.Manifest.Owner.SubjectKind != request.Owner.SubjectKind ||
		result.Manifest.Owner.SubjectID != request.Owner.SubjectID {
		return artifact.ResolveResult{}, errors.New("artifact owner mismatch")
	}
	result.Manifest = result.Manifest.Clone()
	result.Content = append([]byte(nil), result.Content...)
	return result, nil
}

func cloneArtifactPutRequest(request artifact.PutRequest) artifact.PutRequest {
	out := request
	out.Content = append([]byte(nil), request.Content...)
	if request.Lineage != nil {
		lineage := *request.Lineage
		lineage.DerivedFrom = append([]string(nil), request.Lineage.DerivedFrom...)
		out.Lineage = &lineage
	}
	out.Recovery.AllowedRangeUnits = append([]artifact.RangeUnit(nil), request.Recovery.AllowedRangeUnits...)
	return out
}

func artifactBoundaryOwner() artifact.Owner {
	return artifact.Owner{
		WorkspaceID: "workspace-boundary",
		SubjectKind: artifact.SubjectKindSession,
		SubjectID:   "session-boundary",
	}
}

func artifactBoundaryCapability(instruction string) artifact.ResolverCapability {
	return artifact.ResolverCapability{
		Registered: true,
		Recovery: artifact.Recovery{
			Capability:        "goode.artifact.resolve.v1",
			Tool:              "artifact_read",
			AllowedRangeUnits: []artifact.RangeUnit{artifact.RangeUnitBytes},
			Instruction:       instruction,
		},
	}
}

func artifactBoundaryTool(content string) tools.Tool {
	return tools.Tool{
		Name: "large_result",
		Handler: func(_ context.Context, _ json.RawMessage, _ *tools.Container) (llm.Content, error) {
			return llm.TextContent(content), nil
		},
	}
}

func TestAgentArtifactBoundaryReencodesCanonicalEnvelopeWithoutLosingRecovery(t *testing.T) {
	owner := artifactBoundaryOwner()
	owner.ToolCallID = "call-canonical"
	owner.ToolName = "large_result"
	source := []byte(strings.Repeat("source-bytes-", 12_000))
	bytesCount := int64(len(source))
	complete := true
	now := time.Date(2026, 7, 20, 12, 0, 0, 0, time.UTC)
	manifest := artifact.Manifest{
		SchemaVersion: artifact.SchemaVersion,
		ObjectRef:     "obj:v1:existing-canonical-result",
		ObjectKind:    artifact.ObjectKindLogicalToolResult,
		Owner:         owner,
		Complete:      true,
		Recoverable:   true,
		ObjectMeasurement: artifact.Measurement{
			Bytes:             &bytesCount,
			SHA256:            artifact.DigestSHA256(source),
			MeasurementSource: "test_host_sink",
			Complete:          &complete,
		},
		Preview: artifact.Preview{Kind: artifact.PreviewKindPrefix, Truncated: true},
		Retention: artifact.Retention{
			Class:     artifact.RetentionDurable,
			CreatedAt: now,
		},
		ContentType: "text/plain",
		Encoding:    artifact.EncodingUTF8,
		Recovery:    artifactBoundaryCapability("Call artifact_read with object_ref and byte range.").Recovery,
	}
	preview := strings.Repeat("\"escaped\\界🙂\n", 5_000)
	codec := artifact.JSONEnvelopeCodec{}
	encoded, _, err := codec.Encode(artifact.Envelope{
		Manifest: manifest,
		Preview:  preview,
		Continuation: &artifact.Continuation{
			ObjectRef: manifest.ObjectRef,
			RangeUnit: artifact.RangeUnitBytes,
		},
	}, artifact.Budget{MaxBytes: 256 * 1024, MaxTokens: 256 * 1024})
	if err != nil {
		t.Fatalf("encode source envelope: %v", err)
	}
	if len(encoded) < 50*1024 {
		t.Fatalf("fixture envelope is only %d bytes; want a near/over-50 KiB escaped envelope", len(encoded))
	}

	model := &stubModel{toolName: "large_result", toolArgs: `{}`, toolID: owner.ToolCallID}
	ag, err := New(Config{
		LLM:                        model,
		Tools:                      []tools.Tool{artifactBoundaryTool(encoded)},
		MaxToolResultBytes:         4_096,
		MaxToolResultTokens:        1_500,
		ArtifactOwner:              artifactBoundaryOwner(),
		ArtifactResolverCapability: artifactBoundaryCapability("Call artifact_read with object_ref and byte range."),
		ArtifactEnvelopeCodec:      codec,
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("run")))
	result, ok := findToolResult(events, "large_result")
	if !ok {
		t.Fatal("missing tool result event")
	}
	if len(result.Result) > 4_096 {
		t.Fatalf("provider envelope bytes = %d, want <= 4096", len(result.Result))
	}
	decoded, found, err := codec.Decode(result.Result)
	if err != nil || !found {
		t.Fatalf("decode bounded canonical envelope: found=%v err=%v\n%s", found, err, result.Result)
	}
	if decoded.Manifest.ObjectRef != manifest.ObjectRef || decoded.Continuation == nil || decoded.Continuation.ObjectRef != manifest.ObjectRef {
		t.Fatalf("recovery identity/continuation lost: %#v", decoded)
	}
	if decoded.Manifest.ObjectMeasurement.SHA256 != manifest.ObjectMeasurement.SHA256 || !decoded.Manifest.Complete || !decoded.Manifest.Recoverable {
		t.Fatalf("fixed integrity fields changed: %#v", decoded.Manifest)
	}
}

func TestAgentArtifactBoundaryPersistsPlainResultAndProjectsOneManifest(t *testing.T) {
	const maxBytes = 4_096
	const maxTokens = 2_500
	large := strings.Repeat("界🙂no-newline-", 12_000) + "TAIL-SENTINEL"
	sink := &artifactBoundarySink{}
	capability := artifactBoundaryCapability(strings.Repeat("Use artifact_read with object_ref and byte range. ", 12))
	model := &stubModel{toolName: "large_result", toolArgs: `{}`, toolID: "call-sink-success"}
	ag, err := New(Config{
		LLM:                        model,
		Tools:                      []tools.Tool{artifactBoundaryTool(large)},
		MaxToolResultBytes:         maxBytes,
		MaxToolResultTokens:        maxTokens,
		ToolResultTokenEstimator:   func(text string) int { return len([]rune(text)) },
		ArtifactOwner:              artifactBoundaryOwner(),
		ArtifactSink:               sink,
		ArtifactResolverCapability: capability,
		ArtifactEnvelopeCodec:      artifact.JSONEnvelopeCodec{},
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("run")))
	result, ok := findToolResult(events, "large_result")
	if !ok {
		t.Fatal("missing tool result event")
	}
	if len(result.Result) > maxBytes || len([]rune(result.Result)) > maxTokens {
		t.Fatalf("bounded envelope exceeded budget: bytes=%d tokens=%d", len(result.Result), len([]rune(result.Result)))
	}
	if !utf8.ValidString(result.Result) {
		t.Fatal("provider envelope is not valid UTF-8")
	}
	decoded, found, err := (artifact.JSONEnvelopeCodec{}).Decode(result.Result)
	if err != nil || !found {
		t.Fatalf("decode provider envelope: found=%v err=%v\n%s", found, err, result.Result)
	}
	metadataManifest, ok := result.Metadata["artifact_manifest"].(artifact.Manifest)
	if !ok {
		t.Fatalf("artifact_manifest metadata type = %T; metadata=%#v", result.Metadata["artifact_manifest"], result.Metadata)
	}
	if !reflect.DeepEqual(metadataManifest, decoded.Manifest) {
		t.Fatalf("event metadata diverged from provider manifest\nmetadata=%#v\nprovider=%#v", metadataManifest, decoded.Manifest)
	}
	if decoded.Continuation == nil || decoded.Continuation.ObjectRef != decoded.Manifest.ObjectRef {
		t.Fatalf("missing ref-based continuation: %#v", decoded.Continuation)
	}
	if decoded.Manifest.Preview.Kind != artifact.PreviewKindPrefix || !decoded.Manifest.Preview.Truncated {
		t.Fatalf("oversized logical result view state = %#v", decoded.Manifest.Preview)
	}
	if len(sink.requests) != 1 || string(sink.requests[0].Content) != large {
		t.Fatalf("sink did not receive exactly one complete logical result: requests=%d", len(sink.requests))
	}
	if sink.requests[0].Owner.ToolCallID != "call-sink-success" || sink.requests[0].Owner.ToolName != "large_result" {
		t.Fatalf("sink owner not bound to tool call: %#v", sink.requests[0].Owner)
	}
	resolved, err := sink.Resolve(context.Background(), artifact.ResolveRequest{
		ObjectRef: decoded.Manifest.ObjectRef,
		Owner:     sink.requests[0].Owner,
	})
	if err != nil {
		t.Fatalf("resolve by provider-visible object_ref: %v", err)
	}
	if !strings.HasSuffix(string(resolved.Content), "TAIL-SENTINEL") || artifact.DigestSHA256(resolved.Content) != decoded.Manifest.ObjectMeasurement.SHA256 {
		t.Fatal("resolved object does not recover the complete sink bytes")
	}

	ag.ClearHistory()
	ag.ReplaceHistory([]llm.Message{llm.NewUserMessage("replacement")})
	resolvedAfterReplacement, err := sink.Resolve(context.Background(), artifact.ResolveRequest{
		ObjectRef: decoded.Manifest.ObjectRef,
		Owner:     sink.requests[0].Owner,
	})
	if err != nil || string(resolvedAfterReplacement.Content) != large {
		t.Fatalf("history replacement touched host durable object: err=%v", err)
	}
}

func TestAgentArtifactBoundaryPersistsBoundedDerivedResultWithOrderedSourceLineage(t *testing.T) {
	const (
		toolName   = "shell"
		toolCallID = "call-shell-lineage"
	)
	capability := artifactBoundaryCapability("Call artifact_read with object_ref and byte range.")
	sink := &artifactBoundarySink{}
	baseOwner := artifactBoundaryOwner()
	baseOwner.ToolName = toolName
	baseOwner.ToolCallID = toolCallID
	createdAt := time.Date(2026, 7, 20, 13, 0, 0, 0, time.UTC)

	sourceContents := [][]byte{
		[]byte(strings.Repeat("stdout-source-", 128) + "STDOUT-TAIL"),
		[]byte(strings.Repeat("stderr-source-", 64) + "STDERR-TAIL"),
	}
	sourceManifests := make([]artifact.Manifest, 0, len(sourceContents))
	for i, content := range sourceContents {
		owner := baseOwner
		if i == 0 {
			owner.Stream = "stdout"
			owner.Part = "process_stdout"
		} else {
			owner.Stream = "stderr"
			owner.Part = "process_stderr"
		}
		manifest, err := sink.Put(context.Background(), artifact.PutRequest{
			ObjectKind: artifact.ObjectKindRawStream,
			Owner:      owner,
			Content:    content,
			Retention: artifact.Retention{
				Class:     artifact.RetentionDurable,
				CreatedAt: createdAt,
			},
			ContentType: "text/plain",
			Encoding:    artifact.EncodingUTF8,
			Recovery:    capability.Recovery,
		})
		if err != nil {
			t.Fatalf("persist source %d: %v", i, err)
		}
		sourceManifests = append(sourceManifests, manifest)
	}

	plain := "bounded shell preview\n<raw_stream_artifacts_v1>verified source manifests</raw_stream_artifacts_v1>"
	if len(plain) >= 8_192 {
		t.Fatalf("fixture plain result bytes = %d, want below boundary budget", len(plain))
	}
	tool := tools.Tool{
		Name: toolName,
		Handler: func(ctx context.Context, _ json.RawMessage, _ *tools.Container) (llm.Content, error) {
			tools.UpsertToolResultMetadata(ctx, map[string]any{
				artifact.SourceManifestsMetadataKey: sourceManifests,
			})
			return llm.TextContent(plain), nil
		},
	}
	model := &stubModel{toolName: toolName, toolArgs: `{}`, toolID: toolCallID}
	ag, err := New(Config{
		LLM:                        model,
		Tools:                      []tools.Tool{tool},
		MaxToolResultBytes:         8_192,
		MaxToolResultTokens:        4_096,
		ArtifactOwner:              artifactBoundaryOwner(),
		ArtifactSink:               sink,
		ArtifactResolver:           sink,
		ArtifactResolverCapability: capability,
		ArtifactEnvelopeCodec:      artifact.JSONEnvelopeCodec{},
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("run shell")))
	result, ok := findToolResult(events, toolName)
	if !ok {
		t.Fatal("missing shell tool result event")
	}
	decoded, found, err := (artifact.JSONEnvelopeCodec{}).Decode(result.Result)
	if err != nil || !found {
		t.Fatalf("bounded derived result was not canonicalized: found=%v err=%v result=%q", found, err, result.Result)
	}
	if len(sink.requests) != len(sourceContents)+1 {
		t.Fatalf("sink requests = %d, want two raw sources plus one logical result", len(sink.requests))
	}
	logicalRequest := sink.requests[len(sink.requests)-1]
	if logicalRequest.ObjectKind != artifact.ObjectKindLogicalToolResult || string(logicalRequest.Content) != plain {
		t.Fatalf("logical sink request = %#v", logicalRequest)
	}
	wantRefs := []string{sourceManifests[0].ObjectRef, sourceManifests[1].ObjectRef}
	if logicalRequest.Lineage == nil || logicalRequest.Lineage.Transformation != artifact.TransformationToolSerializeV1 ||
		!reflect.DeepEqual(logicalRequest.Lineage.DerivedFrom, wantRefs) {
		t.Fatalf("logical source lineage = %#v, want ordered refs %#v", logicalRequest.Lineage, wantRefs)
	}
	if !reflect.DeepEqual(decoded.Manifest.Lineage, logicalRequest.Lineage) {
		t.Fatalf("provider lineage diverged from sink request: manifest=%#v request=%#v", decoded.Manifest.Lineage, logicalRequest.Lineage)
	}
	if decoded.Preview != plain || decoded.Manifest.Preview.Kind != artifact.PreviewKindFull || decoded.Manifest.Preview.Truncated {
		t.Fatalf("complete bounded provider view was mislabeled: preview=%q state=%#v", decoded.Preview, decoded.Manifest.Preview)
	}
	if decoded.Manifest.ObjectRef == sourceManifests[0].ObjectRef || decoded.Manifest.ObjectRef == sourceManifests[1].ObjectRef {
		t.Fatalf("logical result reused a raw stream identity: %#v", decoded.Manifest)
	}
	for i, source := range sourceManifests {
		resolved, err := sink.Resolve(context.Background(), artifact.ResolveRequest{ObjectRef: source.ObjectRef, Owner: source.Owner})
		if err != nil || !bytes.Equal(resolved.Content, sourceContents[i]) {
			t.Fatalf("resolve source %d: err=%v bytes=%d", i, err, len(resolved.Content))
		}
	}
	resolvedLogical, err := sink.Resolve(context.Background(), artifact.ResolveRequest{
		ObjectRef: decoded.Manifest.ObjectRef,
		Owner:     logicalRequest.Owner,
	})
	if err != nil || string(resolvedLogical.Content) != plain {
		t.Fatalf("resolve logical result: err=%v bytes=%q", err, resolvedLogical.Content)
	}
}

func TestAgentArtifactBoundaryRejectsInvalidSourceLineageBeforeDerivedWrite(t *testing.T) {
	const (
		toolName   = "shell"
		toolCallID = "call-invalid-source-lineage"
	)
	capability := artifactBoundaryCapability("Call artifact_read with object_ref and byte range.")
	baseOwner := artifactBoundaryOwner()
	baseOwner.ToolName = toolName
	baseOwner.ToolCallID = toolCallID
	baseOwner.Stream = "stdout"
	baseOwner.Part = "process_stdout"
	sink := &artifactBoundarySink{}
	source, err := sink.Put(context.Background(), artifact.PutRequest{
		ObjectKind: artifact.ObjectKindRawStream,
		Owner:      baseOwner,
		Content:    []byte("verified source"),
		Retention: artifact.Retention{
			Class:     artifact.RetentionDurable,
			CreatedAt: time.Date(2026, 7, 20, 13, 30, 0, 0, time.UTC),
		},
		ContentType: "text/plain",
		Encoding:    artifact.EncodingUTF8,
		Recovery:    capability.Recovery,
	})
	if err != nil {
		t.Fatalf("persist source: %v", err)
	}
	beforeRequests := len(sink.requests)

	tests := []struct {
		name  string
		value any
	}{
		{name: "wrong_metadata_type", value: []string{source.ObjectRef}},
		{name: "duplicate_ref", value: []artifact.Manifest{source, source.Clone()}},
		{name: "owner_mismatch", value: func() []artifact.Manifest {
			mismatched := source.Clone()
			mismatched.Owner.SubjectID = "different-session"
			return []artifact.Manifest{mismatched}
		}()},
		{name: "ephemeral_source", value: func() []artifact.Manifest {
			ephemeral := source.Clone()
			expires := ephemeral.Retention.CreatedAt.Add(time.Hour)
			ephemeral.Retention.Class = artifact.RetentionEphemeral
			ephemeral.Retention.ExpiresAt = &expires
			return []artifact.Manifest{ephemeral}
		}()},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ag, err := New(Config{
				LLM:                        &stubModel{},
				MaxToolResultBytes:         4_096,
				MaxToolResultTokens:        2_048,
				ArtifactOwner:              artifactBoundaryOwner(),
				ArtifactSink:               sink,
				ArtifactResolver:           sink,
				ArtifactResolverCapability: capability,
				ArtifactEnvelopeCodec:      artifact.JSONEnvelopeCodec{},
				Warningf:                   func(string, ...any) {},
			})
			if err != nil {
				t.Fatalf("new agent: %v", err)
			}
			content, meta := ag.applyToolResultBoundary(context.Background(), llm.TextContent("bounded derived result"), map[string]any{
				artifact.SourceManifestsMetadataKey: tt.value,
			}, toolName, toolCallID)
			for _, want := range []string{"stage=artifact_lineage", "action=restore_verified_source_manifests", "complete=false", "recoverable=false"} {
				if !strings.Contains(content.PlainText(), want) {
					t.Fatalf("lineage fallback missing %q: %q", want, content.PlainText())
				}
			}
			if meta["artifact_stage"] != artifactLineageFailureStage || meta["artifact_recoverable"] != false {
				t.Fatalf("lineage fallback metadata = %#v", meta)
			}
			if _, exists := meta[artifact.SourceManifestsMetadataKey]; exists {
				t.Fatalf("invalid source manifests remained exposed in fallback metadata: %#v", meta)
			}
			if len(sink.requests) != beforeRequests {
				t.Fatalf("invalid source metadata reached sink: requests=%d before=%d", len(sink.requests), beforeRequests)
			}
		})
	}
}

func TestAgentArtifactBoundaryRejectsSinkThatDropsDerivedLineage(t *testing.T) {
	const (
		toolName   = "shell"
		toolCallID = "call-dropped-lineage"
	)
	capability := artifactBoundaryCapability("Call artifact_read with object_ref and byte range.")
	sink := &artifactBoundarySink{}
	owner := artifactBoundaryOwner()
	owner.ToolName = toolName
	owner.ToolCallID = toolCallID
	owner.Stream = "stdout"
	owner.Part = "process_stdout"
	source, err := sink.Put(context.Background(), artifact.PutRequest{
		ObjectKind: artifact.ObjectKindRawStream,
		Owner:      owner,
		Content:    []byte("source bytes"),
		Retention: artifact.Retention{
			Class:     artifact.RetentionDurable,
			CreatedAt: time.Date(2026, 7, 20, 14, 0, 0, 0, time.UTC),
		},
		ContentType: "text/plain",
		Encoding:    artifact.EncodingUTF8,
		Recovery:    capability.Recovery,
	})
	if err != nil {
		t.Fatalf("persist source: %v", err)
	}
	sink.mutate = func(manifest *artifact.Manifest) {
		manifest.Lineage = nil
	}
	ag, err := New(Config{
		LLM:                        &stubModel{},
		MaxToolResultBytes:         4_096,
		MaxToolResultTokens:        2_048,
		ArtifactOwner:              artifactBoundaryOwner(),
		ArtifactSink:               sink,
		ArtifactResolver:           sink,
		ArtifactResolverCapability: capability,
		ArtifactEnvelopeCodec:      artifact.JSONEnvelopeCodec{},
		Warningf:                   func(string, ...any) {},
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	content, meta := ag.applyToolResultBoundary(context.Background(), llm.TextContent("bounded derived result"), map[string]any{
		artifact.SourceManifestsMetadataKey: []artifact.Manifest{source},
	}, toolName, toolCallID)
	if !strings.Contains(content.PlainText(), "stage=artifact_sink") || !strings.Contains(content.PlainText(), "recoverable=false") {
		t.Fatalf("sink lineage loss was not rejected visibly: %q", content.PlainText())
	}
	if meta["artifact_stage"] != artifactSinkFailureStage {
		t.Fatalf("sink lineage failure metadata = %#v", meta)
	}
	if _, exists := meta["artifact_manifest"]; exists {
		t.Fatalf("sink lineage loss projected canonical metadata: %#v", meta)
	}
}

func TestAgentArtifactOwnerProviderResolvesCurrentOwnerPerBoundary(t *testing.T) {
	large := strings.Repeat("dynamic-owner-result-", 8_000)
	sink := &artifactBoundarySink{}
	capability := artifactBoundaryCapability("Call artifact_read with object_ref and byte range.")
	var current atomic.Value
	firstOwner := artifactBoundaryOwner()
	firstOwner.SubjectID = "session-one"
	secondOwner := artifactBoundaryOwner()
	secondOwner.SubjectID = "session-two"
	current.Store(firstOwner)
	var providerCalls atomic.Int64
	ag, err := New(Config{
		LLM:                 &stubModel{},
		MaxToolResultBytes:  4_096,
		MaxToolResultTokens: 1_500,
		ArtifactOwner:       artifact.Owner{WorkspaceID: "stale-static", SubjectKind: artifact.SubjectKindSession, SubjectID: "stale-static"},
		ArtifactOwnerProvider: func(context.Context) (artifact.Owner, error) {
			providerCalls.Add(1)
			return current.Load().(artifact.Owner), nil
		},
		ArtifactSink:               sink,
		ArtifactResolverCapability: capability,
		ArtifactEnvelopeCodec:      artifact.JSONEnvelopeCodec{},
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	firstContent, _ := ag.applyToolResultBoundary(context.Background(), llm.TextContent(large), nil, "large_result", "call-one")
	firstEnvelope, found, err := (artifact.JSONEnvelopeCodec{}).Decode(firstContent.PlainText())
	if err != nil || !found {
		t.Fatalf("decode first envelope: found=%v err=%v", found, err)
	}
	current.Store(secondOwner)
	secondContent, _ := ag.applyToolResultBoundary(context.Background(), llm.TextContent(large+"second"), nil, "large_result", "call-two")
	secondEnvelope, found, err := (artifact.JSONEnvelopeCodec{}).Decode(secondContent.PlainText())
	if err != nil || !found {
		t.Fatalf("decode second envelope: found=%v err=%v", found, err)
	}
	validated, validatedMeta := ag.applyToolResultBoundary(context.Background(), secondContent, nil, "large_result", "call-two")
	validatedEnvelope, found, err := (artifact.JSONEnvelopeCodec{}).Decode(validated.PlainText())
	if err != nil || !found || validatedEnvelope.Manifest.Owner.SubjectID != secondOwner.SubjectID {
		t.Fatalf("validate current small envelope: found=%v err=%v envelope=%#v", found, err, validatedEnvelope)
	}
	if _, ok := validatedMeta["artifact_manifest"].(artifact.Manifest); !ok {
		t.Fatalf("validated small envelope metadata = %#v", validatedMeta)
	}
	oldSessionEnvelope, _, err := (artifact.JSONEnvelopeCodec{}).Encode(artifact.Envelope{
		Manifest: firstEnvelope.Manifest,
		Preview:  "old-session-preview",
		Continuation: &artifact.Continuation{
			ObjectRef: firstEnvelope.Manifest.ObjectRef,
			RangeUnit: artifact.RangeUnitBytes,
		},
	}, artifact.Budget{MaxBytes: 256 * 1024, MaxTokens: 256 * 1024})
	if err != nil {
		t.Fatalf("encode old-session envelope: %v", err)
	}
	if ag.toolResultExceedsBudget(oldSessionEnvelope) {
		t.Fatalf("old-session envelope bytes=%d unexpectedly exceeds boundary budget", len(oldSessionEnvelope))
	}
	rejected, rejectedMeta := ag.applyToolResultBoundary(context.Background(), llm.TextContent(oldSessionEnvelope), nil, "large_result", "call-one")
	if !strings.Contains(rejected.PlainText(), "stage=artifact_decode") || rejectedMeta["artifact_recoverable"] != false {
		t.Fatalf("old-session canonical envelope was accepted by new owner: result=%q metadata=%#v", rejected.PlainText(), rejectedMeta)
	}
	if _, exists := rejectedMeta["artifact_manifest"]; exists {
		t.Fatalf("owner-mismatched envelope projected canonical metadata: %#v", rejectedMeta)
	}

	if providerCalls.Load() != 4 {
		t.Fatalf("owner provider calls = %d, want 4", providerCalls.Load())
	}
	if len(sink.requests) != 2 {
		t.Fatalf("sink requests = %d, want 2", len(sink.requests))
	}
	if got := sink.requests[0].Owner; got.SubjectID != firstOwner.SubjectID || got.ToolCallID != "call-one" || got.ToolName != "large_result" {
		t.Fatalf("first request owner = %#v", got)
	}
	if got := sink.requests[1].Owner; got.SubjectID != secondOwner.SubjectID || got.ToolCallID != "call-two" || got.ToolName != "large_result" {
		t.Fatalf("second request owner = %#v", got)
	}
	if firstEnvelope.Manifest.Owner.SubjectID != firstOwner.SubjectID || secondEnvelope.Manifest.Owner.SubjectID != secondOwner.SubjectID {
		t.Fatalf("provider owners = (%#v, %#v)", firstEnvelope.Manifest.Owner, secondEnvelope.Manifest.Owner)
	}
	if firstEnvelope.Manifest.Owner.WorkspaceID == "stale-static" || secondEnvelope.Manifest.Owner.WorkspaceID == "stale-static" {
		t.Fatal("dynamic owner provider did not take precedence over static compatibility owner")
	}
	if _, err := sink.Resolve(context.Background(), artifact.ResolveRequest{ObjectRef: firstEnvelope.Manifest.ObjectRef, Owner: sink.requests[0].Owner}); err != nil {
		t.Fatalf("old session object lost after owner switch: %v", err)
	}
}

func TestAgentArtifactOwnerProviderFailureIsBoundedAndSkipsSink(t *testing.T) {
	large := strings.Repeat("owner-provider-failure-", 8_000)
	sink := &artifactBoundarySink{}
	ag, err := New(Config{
		LLM:                 &stubModel{},
		MaxToolResultBytes:  1_024,
		MaxToolResultTokens: 300,
		ArtifactOwnerProvider: func(context.Context) (artifact.Owner, error) {
			return artifact.Owner{}, errors.New("current session unavailable")
		},
		ArtifactSink:               sink,
		ArtifactResolverCapability: artifactBoundaryCapability("Call artifact_read with object_ref and byte range."),
		ArtifactEnvelopeCodec:      artifact.JSONEnvelopeCodec{},
		Warningf:                   func(string, ...any) {},
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	content, meta := ag.applyToolResultBoundary(context.Background(), llm.TextContent(large), nil, "large_result", "call-owner-failure")
	if len(content.PlainText()) > 1_024 {
		t.Fatalf("fallback bytes = %d, want <= 1024", len(content.PlainText()))
	}
	for _, want := range []string{"stage=artifact_owner", "action=resolve_current_artifact_owner", "complete=false", "recoverable=false"} {
		if !strings.Contains(content.PlainText(), want) {
			t.Fatalf("owner fallback missing %q: %q", want, content.PlainText())
		}
	}
	if len(sink.requests) != 0 {
		t.Fatalf("sink requests = %d, want 0", len(sink.requests))
	}
	if meta["artifact_stage"] != artifactOwnerFailureStage || meta["artifact_action"] != "resolve_current_artifact_owner" {
		t.Fatalf("owner failure metadata = %#v", meta)
	}
}

func TestAgentArtifactOwnerProviderConcurrentBoundariesUseWholeSnapshots(t *testing.T) {
	large := strings.Repeat("concurrent-owner-result-", 8_000)
	sink := &artifactBoundarySink{}
	ownerA := artifact.Owner{WorkspaceID: "workspace-a", SubjectKind: artifact.SubjectKindSession, SubjectID: "session-a"}
	ownerB := artifact.Owner{WorkspaceID: "workspace-b", SubjectKind: artifact.SubjectKindRun, SubjectID: "run-b"}
	var current atomic.Value
	current.Store(ownerA)
	ag, err := New(Config{
		LLM:                 &stubModel{},
		MaxToolResultBytes:  4_096,
		MaxToolResultTokens: 1_500,
		ArtifactOwnerProvider: func(context.Context) (artifact.Owner, error) {
			return current.Load().(artifact.Owner), nil
		},
		ArtifactSink:               sink,
		ArtifactResolverCapability: artifactBoundaryCapability("Call artifact_read with object_ref and byte range."),
		ArtifactEnvelopeCodec:      artifact.JSONEnvelopeCodec{},
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}

	const calls = 48
	var wg sync.WaitGroup
	wg.Add(calls)
	for i := 0; i < calls; i++ {
		if i%2 == 0 {
			current.Store(ownerA)
		} else {
			current.Store(ownerB)
		}
		go func(i int) {
			defer wg.Done()
			ag.applyToolResultBoundary(context.Background(), llm.TextContent(large), nil, "large_result", fmt.Sprintf("call-%02d", i))
		}(i)
	}
	wg.Wait()

	sink.mu.Lock()
	requests := append([]artifact.PutRequest(nil), sink.requests...)
	sink.mu.Unlock()
	if len(requests) != calls {
		t.Fatalf("sink requests = %d, want %d", len(requests), calls)
	}
	for _, request := range requests {
		owner := request.Owner
		isA := owner.WorkspaceID == ownerA.WorkspaceID && owner.SubjectKind == ownerA.SubjectKind && owner.SubjectID == ownerA.SubjectID
		isB := owner.WorkspaceID == ownerB.WorkspaceID && owner.SubjectKind == ownerB.SubjectKind && owner.SubjectID == ownerB.SubjectID
		if !isA && !isB {
			t.Fatalf("mixed owner snapshot = %#v", owner)
		}
		if owner.ToolName != "large_result" || !strings.HasPrefix(owner.ToolCallID, "call-") {
			t.Fatalf("active tool binding missing from owner = %#v", owner)
		}
	}
}

func TestAgentArtifactBoundarySinkFailureIsBoundedAndDoesNotInventRef(t *testing.T) {
	sink := &artifactBoundarySink{err: errors.New("disk unavailable")}
	large := strings.Repeat("plain-output-", 20_000)
	var warnings []string
	model := &stubModel{toolName: "large_result", toolArgs: `{}`, toolID: "call-sink-failure"}
	ag, err := New(Config{
		LLM:                        model,
		Tools:                      []tools.Tool{artifactBoundaryTool(large)},
		MaxToolResultBytes:         1_024,
		MaxToolResultTokens:        300,
		ArtifactOwner:              artifactBoundaryOwner(),
		ArtifactSink:               sink,
		ArtifactResolverCapability: artifactBoundaryCapability("Call artifact_read with object_ref and byte range."),
		ArtifactEnvelopeCodec:      artifact.JSONEnvelopeCodec{},
		Warningf: func(format string, args ...any) {
			warnings = append(warnings, fmt.Sprintf(format, args...))
		},
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("run")))
	result, ok := findToolResult(events, "large_result")
	if !ok {
		t.Fatal("missing tool result event")
	}
	if len(result.Result) > 1_024 || !utf8.ValidString(result.Result) {
		t.Fatalf("fallback result is not UTF-8 byte-bounded: bytes=%d", len(result.Result))
	}
	for _, required := range []string{"[WARN]", "stage=artifact_sink", "action=", "complete=false", "recoverable=false"} {
		if !strings.Contains(result.Result, required) {
			t.Fatalf("fallback missing %q: %q", required, result.Result)
		}
	}
	if strings.Contains(result.Result, artifact.EnvelopeStartMarker) || strings.Contains(result.Result, "obj:v1:") {
		t.Fatalf("sink failure invented or retained a canonical ref: %q", result.Result)
	}
	if _, exists := result.Metadata["artifact_manifest"]; exists {
		t.Fatalf("sink failure projected a canonical manifest: %#v", result.Metadata)
	}
	if result.Metadata["artifact_complete"] != false || result.Metadata["artifact_recoverable"] != false {
		t.Fatalf("sink failure metadata does not disclose integrity state: %#v", result.Metadata)
	}
	if !strings.Contains(strings.Join(warnings, "\n"), "stage=artifact_sink") {
		t.Fatalf("missing host warning with owning stage: %q", warnings)
	}
}

func TestAgentArtifactBoundaryMissingResolverRegistrationDoesNotCallSink(t *testing.T) {
	sink := &artifactBoundarySink{}
	large := strings.Repeat("plain-output-", 20_000)
	model := &stubModel{toolName: "large_result", toolArgs: `{}`, toolID: "call-capability-missing"}
	ag, err := New(Config{
		LLM:                        model,
		Tools:                      []tools.Tool{artifactBoundaryTool(large)},
		MaxToolResultBytes:         1_024,
		MaxToolResultTokens:        300,
		ArtifactOwner:              artifactBoundaryOwner(),
		ArtifactSink:               sink,
		ArtifactResolverCapability: artifact.ResolverCapability{},
		ArtifactEnvelopeCodec:      artifact.JSONEnvelopeCodec{},
		Warningf:                   func(string, ...any) {},
	})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("run")))
	result, ok := findToolResult(events, "large_result")
	if !ok {
		t.Fatal("missing tool result event")
	}
	if len(sink.requests) != 0 {
		t.Fatalf("sink called without a registered resolver capability: %d request(s)", len(sink.requests))
	}
	if !strings.Contains(result.Result, "stage=artifact_capability") || !strings.Contains(result.Result, "recoverable=false") {
		t.Fatalf("missing capability diagnostic: %q", result.Result)
	}
}

func TestAgentArtifactBoundaryRejectsSinkManifestThatDoesNotDescribeStoredBytes(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*artifact.Manifest)
	}{
		{
			name: "owner_mismatch",
			mutate: func(manifest *artifact.Manifest) {
				manifest.Owner.SubjectID = "different-session"
			},
		},
		{
			name: "hash_mismatch",
			mutate: func(manifest *artifact.Manifest) {
				manifest.ObjectMeasurement.SHA256 = strings.Repeat("a", 64)
			},
		},
		{
			name: "byte_count_mismatch",
			mutate: func(manifest *artifact.Manifest) {
				wrong := *manifest.ObjectMeasurement.Bytes + 1
				manifest.ObjectMeasurement.Bytes = &wrong
			},
		},
		{
			name: "recovery_contract_mismatch",
			mutate: func(manifest *artifact.Manifest) {
				manifest.Recovery.Tool = "unregistered_artifact_reader"
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sink := &artifactBoundarySink{mutate: tt.mutate}
			large := strings.Repeat("manifest-integrity-", 10_000)
			model := &stubModel{toolName: "large_result", toolArgs: `{}`, toolID: "call-invalid-manifest"}
			ag, err := New(Config{
				LLM:                        model,
				Tools:                      []tools.Tool{artifactBoundaryTool(large)},
				MaxToolResultBytes:         1_024,
				MaxToolResultTokens:        300,
				ArtifactOwner:              artifactBoundaryOwner(),
				ArtifactSink:               sink,
				ArtifactResolverCapability: artifactBoundaryCapability("Call artifact_read with object_ref and byte range."),
				ArtifactEnvelopeCodec:      artifact.JSONEnvelopeCodec{},
				Warningf:                   func(string, ...any) {},
			})
			if err != nil {
				t.Fatalf("new agent: %v", err)
			}
			events := collectEvents(ag.QueryStream(context.Background(), llm.TextContent("run")))
			result, ok := findToolResult(events, "large_result")
			if !ok {
				t.Fatal("missing tool result event")
			}
			if !strings.Contains(result.Result, "stage=artifact_sink") ||
				!strings.Contains(result.Result, "complete=false") ||
				!strings.Contains(result.Result, "recoverable=false") {
				t.Fatalf("invalid manifest was not rejected visibly: %q", result.Result)
			}
			if strings.Contains(result.Result, artifact.EnvelopeStartMarker) || strings.Contains(result.Result, "obj:v1:") {
				t.Fatalf("invalid manifest leaked a canonical claim: %q", result.Result)
			}
			if _, exists := result.Metadata["artifact_manifest"]; exists {
				t.Fatalf("invalid manifest projected into metadata: %#v", result.Metadata)
			}
		})
	}
}

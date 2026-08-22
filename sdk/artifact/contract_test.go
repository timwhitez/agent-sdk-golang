package artifact

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

func int64Pointer(v int64) *int64 { return &v }

func boolPointer(v bool) *bool { return &v }

func timePointer(v time.Time) *time.Time { return &v }

func validTestManifest(content []byte) Manifest {
	now := time.Date(2026, 7, 20, 12, 0, 0, 0, time.UTC)
	return Manifest{
		SchemaVersion: SchemaVersion,
		ObjectRef:     "obj:v1:logical-result-001",
		ObjectKind:    ObjectKindLogicalToolResult,
		Owner: Owner{
			WorkspaceID: "workspace-1",
			SubjectKind: SubjectKindSession,
			SubjectID:   "session-1",
			ToolCallID:  "call-1",
			ToolName:    "read",
		},
		Complete:    true,
		Recoverable: true,
		ObjectMeasurement: Measurement{
			Bytes:             int64Pointer(int64(len(content))),
			SHA256:            DigestSHA256(content),
			MeasurementSource: "host_sink",
		},
		Preview: Preview{
			Kind:      PreviewKindPrefix,
			Truncated: true,
			Ranges: []Range{{
				Unit:  RangeUnitBytes,
				Start: 0,
				End:   int64(len(content)),
			}},
		},
		Retention: Retention{
			Class:           RetentionDurable,
			CreatedAt:       now,
			GCEligibleAfter: timePointer(now.Add(24 * time.Hour)),
		},
		ContentType: "text/plain",
		Encoding:    EncodingUTF8,
		Recovery: Recovery{
			Capability:        "goode.artifact.resolve.v1",
			Tool:              "artifact_read",
			AllowedRangeUnits: []RangeUnit{RangeUnitBytes},
			Instruction:       "Call artifact_read with object_ref and a byte range.",
		},
	}
}

func TestManifestValidationAcceptsCompleteRecoverableObject(t *testing.T) {
	manifest := validTestManifest([]byte("complete object"))
	if err := manifest.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
}

func TestManifestValidationRejectsUnknownOrContradictoryMeasurements(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Manifest)
		want   string
	}{
		{
			name: "complete_without_object_bytes",
			mutate: func(m *Manifest) {
				m.ObjectMeasurement.Bytes = nil
			},
			want: "complete object requires measured object bytes",
		},
		{
			name: "complete_without_object_hash",
			mutate: func(m *Manifest) {
				m.ObjectMeasurement.SHA256 = ""
			},
			want: "complete object requires object sha256",
		},
		{
			name: "object_measurement_completeness_conflicts",
			mutate: func(m *Manifest) {
				m.ObjectMeasurement.Complete = boolPointer(false)
			},
			want: "object_measurement.complete conflicts with manifest complete",
		},
		{
			name: "partial_source_claims_full_hash",
			mutate: func(m *Manifest) {
				m.SourceMeasurement = &Measurement{
					Complete: boolPointer(false),
					Bytes:    int64Pointer(100),
					SHA256:   strings.Repeat("a", 64),
				}
			},
			want: "incomplete source cannot claim sha256",
		},
		{
			name: "complete_source_without_hash",
			mutate: func(m *Manifest) {
				m.SourceMeasurement = &Measurement{
					Complete: boolPointer(true),
					Bytes:    int64Pointer(100),
				}
			},
			want: "complete source requires source sha256",
		},
		{
			name: "unknown_is_not_zero",
			mutate: func(m *Manifest) {
				negative := int64(-1)
				m.SourceMeasurement = &Measurement{Bytes: &negative}
			},
			want: "source_measurement.bytes must be non-negative",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			manifest := validTestManifest([]byte("complete object"))
			tt.mutate(&manifest)
			err := manifest.Validate()
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Validate error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestManifestValidationRejectsInvalidOwnerLineageAndRetention(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Manifest)
		want   string
	}{
		{
			name: "path_like_ref",
			mutate: func(m *Manifest) {
				m.ObjectRef = "/tmp/tool-output.txt"
			},
			want: "object_ref must be opaque",
		},
		{
			name: "missing_subject",
			mutate: func(m *Manifest) {
				m.Owner.SubjectID = ""
			},
			want: "owner subject_id is required",
		},
		{
			name: "self_lineage",
			mutate: func(m *Manifest) {
				m.Lineage = &Lineage{SourceRef: m.ObjectRef, Transformation: "tool_serialize_v1"}
			},
			want: "lineage cannot reference object_ref itself",
		},
		{
			name: "derived_without_transformation",
			mutate: func(m *Manifest) {
				m.Lineage = &Lineage{SourceRef: "obj:v1:raw-001"}
			},
			want: "lineage transformation is required",
		},
		{
			name: "durable_expiry",
			mutate: func(m *Manifest) {
				m.Retention.ExpiresAt = timePointer(m.Retention.CreatedAt.Add(time.Hour))
			},
			want: "expires_at is allowed only for ephemeral objects",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			manifest := validTestManifest([]byte("complete object"))
			tt.mutate(&manifest)
			err := manifest.Validate()
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Validate error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestEnvelopeCodecPreservesRecoveryFieldsWithinByteAndTokenBudgets(t *testing.T) {
	content := []byte(strings.Repeat("raw-source-", 2_000))
	manifest := validTestManifest(content)
	preview := strings.Repeat("界🙂tool-output ", 2_000)
	envelope := Envelope{
		Manifest: manifest,
		Preview:  preview,
		Continuation: &Continuation{
			ObjectRef: manifest.ObjectRef,
			RangeUnit: RangeUnitBytes,
			Next:      0,
		},
	}
	estimate := func(text string) int {
		return len([]rune(text))
	}
	budget := Budget{MaxBytes: 4_096, MaxTokens: 2_000, EstimateTokens: estimate}

	encoded, normalized, err := (JSONEnvelopeCodec{}).Encode(envelope, budget)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if len(encoded) > budget.MaxBytes {
		t.Fatalf("encoded bytes = %d, want <= %d", len(encoded), budget.MaxBytes)
	}
	if tokens := estimate(encoded); tokens > budget.MaxTokens {
		t.Fatalf("encoded tokens = %d, want <= %d", tokens, budget.MaxTokens)
	}
	if !utf8.ValidString(encoded) || strings.ContainsRune(encoded, utf8.RuneError) {
		t.Fatalf("encoded envelope is not clean UTF-8: %q", encoded)
	}
	for _, want := range []string{
		EnvelopeStartMarker,
		EnvelopeEndMarker,
		manifest.ObjectRef,
		manifest.Recovery.Capability,
		manifest.Recovery.Tool,
		`"complete":true`,
		`"recoverable":true`,
	} {
		if !strings.Contains(encoded, want) {
			t.Fatalf("encoded envelope missing %q: %s", want, encoded)
		}
	}
	if normalized.Preview == preview {
		t.Fatal("oversized preview was not reduced")
	}
	if normalized.Manifest.VisibleMeasurement.Bytes == nil || *normalized.Manifest.VisibleMeasurement.Bytes != int64(len(normalized.Preview)) {
		t.Fatalf("visible byte measurement = %#v, preview bytes=%d", normalized.Manifest.VisibleMeasurement.Bytes, len(normalized.Preview))
	}
	if normalized.Manifest.VisibleMeasurement.EstimatorTokens == nil || *normalized.Manifest.VisibleMeasurement.EstimatorTokens != int64(estimate(normalized.Preview)) {
		t.Fatalf("visible token measurement = %#v", normalized.Manifest.VisibleMeasurement.EstimatorTokens)
	}
	if len(normalized.Manifest.Preview.Ranges) != 1 || normalized.Manifest.Preview.Ranges[0].End != int64(len(normalized.Preview)) {
		t.Fatalf("normalized visible ranges = %#v", normalized.Manifest.Preview.Ranges)
	}

	decoded, ok, err := (JSONEnvelopeCodec{}).Decode(encoded)
	if err != nil || !ok {
		t.Fatalf("Decode: ok=%v err=%v", ok, err)
	}
	if decoded.Manifest.ObjectRef != manifest.ObjectRef || decoded.Manifest.ObjectMeasurement.SHA256 != manifest.ObjectMeasurement.SHA256 {
		t.Fatalf("decoded identity/integrity changed: %#v", decoded.Manifest)
	}
	if decoded.Preview != normalized.Preview {
		t.Fatalf("decoded preview differs from encoded preview")
	}
}

func TestEnvelopeCodecMarksShortenedFullPreviewAsTruncatedPrefix(t *testing.T) {
	content := []byte(strings.Repeat("full-logical-source-", 1_000))
	manifest := validTestManifest(content)
	manifest.Preview = Preview{Kind: PreviewKindFull}
	envelope := Envelope{
		Manifest: manifest,
		Preview:  string(content),
		Continuation: &Continuation{
			ObjectRef: manifest.ObjectRef,
			RangeUnit: RangeUnitBytes,
		},
	}
	encoded, normalized, err := (JSONEnvelopeCodec{}).Encode(envelope, Budget{
		MaxBytes:  2_048,
		MaxTokens: 1_024,
	})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if len(encoded) > 2_048 || len(normalized.Preview) >= len(content) {
		t.Fatalf("preview was not bounded: encoded=%d preview=%d source=%d", len(encoded), len(normalized.Preview), len(content))
	}
	if normalized.Manifest.Preview.Kind != PreviewKindPrefix || !normalized.Manifest.Preview.Truncated {
		t.Fatalf("shortened full preview state = %#v", normalized.Manifest.Preview)
	}
	if len(normalized.Manifest.Preview.Ranges) != 1 || normalized.Manifest.Preview.Ranges[0].End != int64(len(normalized.Preview)) {
		t.Fatalf("shortened preview ranges = %#v", normalized.Manifest.Preview.Ranges)
	}
}

func TestEnvelopeCodecRejectsBudgetThatCannotFitFixedRecoveryFields(t *testing.T) {
	envelope := Envelope{
		Manifest: validTestManifest([]byte("complete object")),
		Preview:  strings.Repeat("x", 100),
	}
	_, _, err := (JSONEnvelopeCodec{}).Encode(envelope, Budget{
		MaxBytes:       64,
		MaxTokens:      64,
		EstimateTokens: func(text string) int { return len(text) },
	})
	if err == nil || !errors.Is(err, ErrEnvelopeBudgetTooSmall) || !strings.Contains(err.Error(), "fixed envelope fields") {
		t.Fatalf("Encode error = %v", err)
	}
}

func TestEnvelopeCodecFallsBackWhenEstimatorReturnsNonPositive(t *testing.T) {
	envelope := Envelope{
		Manifest: validTestManifest([]byte(strings.Repeat("source", 100))),
		Preview:  strings.Repeat("preview", 2_000),
	}
	encoded, _, err := (JSONEnvelopeCodec{}).Encode(envelope, Budget{
		MaxBytes:       2_048,
		MaxTokens:      512,
		EstimateTokens: func(string) int { return 0 },
	})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if len(encoded) > 2_048 || (len(encoded)+3)/4 > 512 {
		t.Fatalf("invalid estimator bypassed fallback budget: bytes=%d fallback_tokens=%d", len(encoded), (len(encoded)+3)/4)
	}
}

func TestEnvelopeCodecRejectsInvalidCanonicalClaims(t *testing.T) {
	envelope := Envelope{
		Manifest: validTestManifest([]byte("complete object")),
		Preview:  "preview",
	}
	envelope.Manifest.Recoverable = true
	envelope.Manifest.Recovery.Capability = ""
	_, _, err := (JSONEnvelopeCodec{}).Encode(envelope, Budget{MaxBytes: 4_096, MaxTokens: 4_096})
	if err == nil || !strings.Contains(err.Error(), "recoverable object requires recovery capability") {
		t.Fatalf("Encode error = %v", err)
	}
}

type interfaceTestSink struct{}

func (interfaceTestSink) Put(context.Context, PutRequest) (Manifest, error) {
	return Manifest{}, nil
}

type interfaceTestResolver struct{}

func (interfaceTestResolver) Resolve(context.Context, ResolveRequest) (ResolveResult, error) {
	return ResolveResult{}, nil
}

func TestHostCapabilityInterfacesRemainSeparateFromFilesystemPaths(t *testing.T) {
	var _ Sink = interfaceTestSink{}
	var _ Resolver = interfaceTestResolver{}

	request := ResolveRequest{
		ObjectRef: "obj:v1:logical-result-001",
		Owner: Owner{
			WorkspaceID: "workspace-1",
			SubjectKind: SubjectKindSession,
			SubjectID:   "session-1",
		},
		Range: &Range{Unit: RangeUnitBytes, Start: 100, End: 200},
	}
	if strings.ContainsAny(request.ObjectRef, `/\\`) {
		t.Fatalf("test object_ref is path-like: %q", request.ObjectRef)
	}
}

func TestResolverCapabilityRequiresExplicitRegistrationAndCompleteRecoveryContract(t *testing.T) {
	unregistered := ResolverCapability{
		Recovery: Recovery{
			Capability:  "goode.artifact.resolve.v1",
			Tool:        "artifact_read",
			Instruction: "Call artifact_read with object_ref.",
		},
	}
	if err := unregistered.Validate(); err != nil {
		t.Fatalf("unregistered capability should remain an available-but-disabled state: %v", err)
	}

	tests := []struct {
		name   string
		mutate func(*ResolverCapability)
		want   string
	}{
		{
			name: "missing_capability",
			mutate: func(capability *ResolverCapability) {
				capability.Recovery.Capability = ""
			},
			want: "capability is required",
		},
		{
			name: "missing_tool",
			mutate: func(capability *ResolverCapability) {
				capability.Recovery.Tool = ""
			},
			want: "tool is required",
		},
		{
			name: "missing_instruction",
			mutate: func(capability *ResolverCapability) {
				capability.Recovery.Instruction = ""
			},
			want: "instruction is required",
		},
		{
			name: "invalid_range_unit",
			mutate: func(capability *ResolverCapability) {
				capability.Recovery.AllowedRangeUnits = []RangeUnit{"records"}
			},
			want: "invalid range unit",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			capability := ResolverCapability{
				Registered: true,
				Recovery: Recovery{
					Capability:        "goode.artifact.resolve.v1",
					Tool:              "artifact_read",
					Instruction:       "Call artifact_read with object_ref.",
					AllowedRangeUnits: []RangeUnit{RangeUnitBytes},
				},
			}
			tt.mutate(&capability)
			err := capability.Validate()
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Validate error = %v, want %q", err, tt.want)
			}
		})
	}
}

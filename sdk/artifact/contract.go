package artifact

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"regexp"
	"strings"
	"time"
)

const SchemaVersion = 1

const (
	// SourceManifestsMetadataKey is the reserved tool-result metadata field for
	// ordered canonical objects that contributed to a derived logical result.
	SourceManifestsMetadataKey = "artifact_manifests"

	// TransformationToolSerializeV1 identifies the stable transformation from
	// one or more canonical producer objects to serialized logical tool-result
	// bytes.
	TransformationToolSerializeV1 = "tool_serialize_v1"
)

type ObjectKind string

const (
	ObjectKindRawStream           ObjectKind = "raw_stream"
	ObjectKindLogicalToolResult   ObjectKind = "logical_tool_result"
	ObjectKindProviderVisibleView ObjectKind = "provider_visible_view"
	ObjectKindCompactionMaterial  ObjectKind = "compaction_material"
)

type SubjectKind string

const (
	SubjectKindSession SubjectKind = "session"
	SubjectKindAgent   SubjectKind = "agent"
	SubjectKindRun     SubjectKind = "run"
)

type RangeUnit string

const (
	RangeUnitBytes   RangeUnit = "bytes"
	RangeUnitLines   RangeUnit = "lines"
	RangeUnitColumns RangeUnit = "columns"
)

type PreviewKind string

const (
	PreviewKindNone     PreviewKind = "none"
	PreviewKindFull     PreviewKind = "full"
	PreviewKindPrefix   PreviewKind = "prefix"
	PreviewKindHeadTail PreviewKind = "head_tail"
	PreviewKindRange    PreviewKind = "range"
	PreviewKindNested   PreviewKind = "nested"
)

type RetentionClass string

const (
	RetentionDurable   RetentionClass = "durable"
	RetentionEphemeral RetentionClass = "ephemeral"
)

const EncodingUTF8 = "utf-8"

var (
	opaqueRefPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9:._-]{0,255}$`)
	nameTokenPattern = regexp.MustCompile(`^[a-z][a-z0-9_.-]{0,63}$`)
	sha256Pattern    = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

type Owner struct {
	WorkspaceID string      `json:"workspace_id,omitempty"`
	SubjectKind SubjectKind `json:"subject_kind"`
	SubjectID   string      `json:"subject_id"`
	ToolCallID  string      `json:"tool_call_id,omitempty"`
	ToolName    string      `json:"tool_name,omitempty"`
	Stream      string      `json:"stream,omitempty"`
	Part        string      `json:"part,omitempty"`
}

type Lineage struct {
	SourceRef      string   `json:"source_ref,omitempty"`
	DerivedFrom    []string `json:"derived_from,omitempty"`
	ViewOf         string   `json:"view_of,omitempty"`
	Transformation string   `json:"transformation"`
}

type Measurement struct {
	Bytes             *int64 `json:"bytes,omitempty"`
	Lines             *int64 `json:"lines,omitempty"`
	EstimatorTokens   *int64 `json:"estimator_tokens,omitempty"`
	SHA256            string `json:"sha256,omitempty"`
	MeasurementSource string `json:"measurement_source,omitempty"`
	Complete          *bool  `json:"complete,omitempty"`
}

type Range struct {
	Unit  RangeUnit `json:"unit"`
	Start int64     `json:"start"`
	End   int64     `json:"end"`
}

type Preview struct {
	Kind      PreviewKind `json:"kind"`
	Ranges    []Range     `json:"ranges,omitempty"`
	Truncated bool        `json:"truncated"`
}

type Retention struct {
	Class           RetentionClass `json:"class"`
	CreatedAt       time.Time      `json:"created_at"`
	GCEligibleAfter *time.Time     `json:"gc_eligible_after,omitempty"`
	ExpiresAt       *time.Time     `json:"expires_at,omitempty"`
}

type Recovery struct {
	Capability        string      `json:"capability,omitempty"`
	Tool              string      `json:"tool,omitempty"`
	AllowedRangeUnits []RangeUnit `json:"allowed_range_units,omitempty"`
	Instruction       string      `json:"instruction,omitempty"`
}

// ResolverCapability is the host's explicit statement that the recovery
// contract exposed in provider-visible envelopes is currently registered.
// Recovery fields alone are insufficient: Registered must be true before a
// producer can claim that an object is recoverable.
type ResolverCapability struct {
	Registered bool
	Recovery   Recovery
}

func (c ResolverCapability) Validate() error {
	if !c.Registered {
		return nil
	}
	if strings.TrimSpace(c.Recovery.Capability) == "" {
		return fmt.Errorf("artifact resolver capability: capability is required when registered")
	}
	if strings.TrimSpace(c.Recovery.Tool) == "" {
		return fmt.Errorf("artifact resolver capability: tool is required when registered")
	}
	if strings.TrimSpace(c.Recovery.Instruction) == "" {
		return fmt.Errorf("artifact resolver capability: instruction is required when registered")
	}
	for _, unit := range c.Recovery.AllowedRangeUnits {
		if !validRangeUnit(unit) {
			return fmt.Errorf("artifact resolver capability: invalid range unit %q", unit)
		}
	}
	return nil
}

type Manifest struct {
	SchemaVersion      int          `json:"schema_version"`
	ObjectRef          string       `json:"object_ref"`
	ObjectKind         ObjectKind   `json:"object_kind"`
	Owner              Owner        `json:"owner"`
	Lineage            *Lineage     `json:"lineage,omitempty"`
	Complete           bool         `json:"complete"`
	Recoverable        bool         `json:"recoverable"`
	ObjectMeasurement  Measurement  `json:"object_measurement"`
	SourceMeasurement  *Measurement `json:"source_measurement,omitempty"`
	VisibleMeasurement Measurement  `json:"visible_measurement"`
	Preview            Preview      `json:"preview"`
	Retention          Retention    `json:"retention"`
	ContentType        string       `json:"content_type"`
	Encoding           string       `json:"encoding"`
	Recovery           Recovery     `json:"recovery"`
}

type Continuation struct {
	ObjectRef string    `json:"object_ref"`
	RangeUnit RangeUnit `json:"range_unit"`
	Next      int64     `json:"next"`
}

type Diagnostic struct {
	Severity string `json:"severity"`
	Stage    string `json:"stage"`
	Action   string `json:"action"`
	Message  string `json:"message"`
}

type Envelope struct {
	Manifest     Manifest      `json:"manifest"`
	Preview      string        `json:"preview"`
	Continuation *Continuation `json:"continuation,omitempty"`
	Diagnostics  []Diagnostic  `json:"diagnostics,omitempty"`
}

type PutRequest struct {
	ObjectKind  ObjectKind
	Owner       Owner
	Content     []byte
	Lineage     *Lineage
	Retention   Retention
	ContentType string
	Encoding    string
	Recovery    Recovery
}

// StreamPutRequest describes a byte object before a streaming producer knows
// its final measurements. StreamObjectWriter.Commit returns the complete
// manifest after the sink has atomically finalized the object.
type StreamPutRequest struct {
	ObjectKind  ObjectKind
	Owner       Owner
	Lineage     *Lineage
	Retention   Retention
	ContentType string
	Encoding    string
	Recovery    Recovery
}

type ResolveRequest struct {
	ObjectRef string
	Owner     Owner
	Range     *Range
}

type ResolveResult struct {
	Manifest Manifest
	Content  []byte
	Range    *Range
}

type Sink interface {
	Put(context.Context, PutRequest) (Manifest, error)
}

type StreamObjectWriter interface {
	Write([]byte) (int, error)
	Commit(context.Context) (Manifest, error)
	Abort(context.Context) error
}

type StreamSink interface {
	Begin(context.Context, StreamPutRequest) (StreamObjectWriter, error)
}

type Resolver interface {
	Resolve(context.Context, ResolveRequest) (ResolveResult, error)
}

type EnvelopeCodec interface {
	Encode(Envelope, Budget) (string, Envelope, error)
	Decode(string) (Envelope, bool, error)
}

func DigestSHA256(content []byte) string {
	sum := sha256.Sum256(content)
	return hex.EncodeToString(sum[:])
}

func (o Owner) Validate() error {
	return validateOwner(o)
}

func (m Manifest) Clone() Manifest {
	out := m
	out.Owner = m.Owner
	if m.Lineage != nil {
		lineage := *m.Lineage
		lineage.DerivedFrom = append([]string(nil), m.Lineage.DerivedFrom...)
		out.Lineage = &lineage
	}
	out.ObjectMeasurement = cloneMeasurement(m.ObjectMeasurement)
	if m.SourceMeasurement != nil {
		source := cloneMeasurement(*m.SourceMeasurement)
		out.SourceMeasurement = &source
	}
	out.VisibleMeasurement = cloneMeasurement(m.VisibleMeasurement)
	out.Preview.Ranges = append([]Range(nil), m.Preview.Ranges...)
	out.Retention = cloneRetention(m.Retention)
	out.Recovery.AllowedRangeUnits = append([]RangeUnit(nil), m.Recovery.AllowedRangeUnits...)
	return out
}

func (e Envelope) Clone() Envelope {
	out := e
	out.Manifest = e.Manifest.Clone()
	if e.Continuation != nil {
		continuation := *e.Continuation
		out.Continuation = &continuation
	}
	out.Diagnostics = append([]Diagnostic(nil), e.Diagnostics...)
	return out
}

func cloneMeasurement(m Measurement) Measurement {
	out := m
	if m.Bytes != nil {
		value := *m.Bytes
		out.Bytes = &value
	}
	if m.Lines != nil {
		value := *m.Lines
		out.Lines = &value
	}
	if m.EstimatorTokens != nil {
		value := *m.EstimatorTokens
		out.EstimatorTokens = &value
	}
	if m.Complete != nil {
		value := *m.Complete
		out.Complete = &value
	}
	return out
}

func cloneRetention(r Retention) Retention {
	out := r
	if r.GCEligibleAfter != nil {
		value := *r.GCEligibleAfter
		out.GCEligibleAfter = &value
	}
	if r.ExpiresAt != nil {
		value := *r.ExpiresAt
		out.ExpiresAt = &value
	}
	return out
}

func (m Manifest) Validate() error {
	if m.SchemaVersion != SchemaVersion {
		return fmt.Errorf("artifact manifest: unsupported schema_version %d", m.SchemaVersion)
	}
	if !opaqueRefPattern.MatchString(strings.TrimSpace(m.ObjectRef)) || strings.ContainsAny(m.ObjectRef, `/\\`) {
		return fmt.Errorf("artifact manifest: object_ref must be opaque and path-independent")
	}
	if !validObjectKind(m.ObjectKind) {
		return fmt.Errorf("artifact manifest: invalid object_kind %q", m.ObjectKind)
	}
	if err := validateOwner(m.Owner); err != nil {
		return err
	}
	if err := validateLineage(m.ObjectRef, m.Lineage); err != nil {
		return err
	}
	if err := validateMeasurement("object_measurement", m.ObjectMeasurement); err != nil {
		return err
	}
	if m.Complete {
		if m.ObjectMeasurement.Bytes == nil {
			return fmt.Errorf("artifact manifest: complete object requires measured object bytes")
		}
		if strings.TrimSpace(m.ObjectMeasurement.SHA256) == "" {
			return fmt.Errorf("artifact manifest: complete object requires object sha256")
		}
	}
	if m.ObjectMeasurement.Complete != nil && *m.ObjectMeasurement.Complete != m.Complete {
		return fmt.Errorf("artifact manifest: object_measurement.complete conflicts with manifest complete")
	}
	if m.Recoverable && !m.Complete {
		return fmt.Errorf("artifact manifest: recoverable object must be complete")
	}
	if m.SourceMeasurement != nil {
		if err := validateMeasurement("source_measurement", *m.SourceMeasurement); err != nil {
			return err
		}
		if m.SourceMeasurement.Complete != nil && !*m.SourceMeasurement.Complete && strings.TrimSpace(m.SourceMeasurement.SHA256) != "" {
			return fmt.Errorf("artifact manifest: incomplete source cannot claim sha256")
		}
		if m.SourceMeasurement.Complete != nil && *m.SourceMeasurement.Complete {
			if m.SourceMeasurement.Bytes == nil {
				return fmt.Errorf("artifact manifest: complete source requires measured source bytes")
			}
			if strings.TrimSpace(m.SourceMeasurement.SHA256) == "" {
				return fmt.Errorf("artifact manifest: complete source requires source sha256")
			}
		}
	}
	if err := validateMeasurement("visible_measurement", m.VisibleMeasurement); err != nil {
		return err
	}
	if err := validatePreview(m.Preview, m.VisibleMeasurement); err != nil {
		return err
	}
	if err := validateRetention(m.Retention); err != nil {
		return err
	}
	if strings.TrimSpace(m.ContentType) == "" {
		return fmt.Errorf("artifact manifest: content_type is required")
	}
	if strings.TrimSpace(m.Encoding) == "" {
		return fmt.Errorf("artifact manifest: encoding is required")
	}
	if m.Recoverable {
		if strings.TrimSpace(m.Recovery.Capability) == "" {
			return fmt.Errorf("artifact manifest: recoverable object requires recovery capability")
		}
		if strings.TrimSpace(m.Recovery.Tool) == "" {
			return fmt.Errorf("artifact manifest: recoverable object requires recovery tool")
		}
		if strings.TrimSpace(m.Recovery.Instruction) == "" {
			return fmt.Errorf("artifact manifest: recoverable object requires recovery instruction")
		}
	}
	for _, unit := range m.Recovery.AllowedRangeUnits {
		if !validRangeUnit(unit) {
			return fmt.Errorf("artifact manifest: invalid recovery range unit %q", unit)
		}
	}
	return nil
}

func (e Envelope) Validate() error {
	if err := e.Manifest.Validate(); err != nil {
		return err
	}
	if e.Continuation != nil {
		if e.Continuation.ObjectRef != e.Manifest.ObjectRef {
			return fmt.Errorf("artifact envelope: continuation object_ref does not match manifest")
		}
		if !validRangeUnit(e.Continuation.RangeUnit) {
			return fmt.Errorf("artifact envelope: invalid continuation range_unit %q", e.Continuation.RangeUnit)
		}
		if e.Continuation.Next < 0 {
			return fmt.Errorf("artifact envelope: continuation next must be non-negative")
		}
	}
	for i, diagnostic := range e.Diagnostics {
		if strings.TrimSpace(diagnostic.Stage) == "" || strings.TrimSpace(diagnostic.Action) == "" || strings.TrimSpace(diagnostic.Message) == "" {
			return fmt.Errorf("artifact envelope: diagnostic %d requires stage, action, and message", i)
		}
	}
	return nil
}

func validObjectKind(kind ObjectKind) bool {
	return nameTokenPattern.MatchString(string(kind))
}

func validRangeUnit(unit RangeUnit) bool {
	switch unit {
	case RangeUnitBytes, RangeUnitLines, RangeUnitColumns:
		return true
	default:
		return false
	}
}

func validateOwner(owner Owner) error {
	switch owner.SubjectKind {
	case SubjectKindSession, SubjectKindAgent, SubjectKindRun:
	default:
		return fmt.Errorf("artifact manifest: invalid owner subject_kind %q", owner.SubjectKind)
	}
	if strings.TrimSpace(owner.SubjectID) == "" {
		return fmt.Errorf("artifact manifest: owner subject_id is required")
	}
	if strings.ContainsAny(owner.SubjectID, "\x00\r\n") {
		return fmt.Errorf("artifact manifest: owner subject_id contains control characters")
	}
	return nil
}

func validateLineage(objectRef string, lineage *Lineage) error {
	if lineage == nil {
		return nil
	}
	refs := make([]string, 0, len(lineage.DerivedFrom)+2)
	refs = append(refs, strings.TrimSpace(lineage.SourceRef), strings.TrimSpace(lineage.ViewOf))
	refs = append(refs, lineage.DerivedFrom...)
	seen := map[string]struct{}{}
	hasRef := false
	for _, ref := range refs {
		ref = strings.TrimSpace(ref)
		if ref == "" {
			continue
		}
		hasRef = true
		if ref == objectRef {
			return fmt.Errorf("artifact manifest: lineage cannot reference object_ref itself")
		}
		if !opaqueRefPattern.MatchString(ref) || strings.ContainsAny(ref, `/\\`) {
			return fmt.Errorf("artifact manifest: lineage ref %q must be opaque", ref)
		}
		if _, exists := seen[ref]; exists {
			return fmt.Errorf("artifact manifest: duplicate lineage ref %q", ref)
		}
		seen[ref] = struct{}{}
	}
	if hasRef && strings.TrimSpace(lineage.Transformation) == "" {
		return fmt.Errorf("artifact manifest: lineage transformation is required")
	}
	if !hasRef && strings.TrimSpace(lineage.Transformation) != "" {
		return fmt.Errorf("artifact manifest: lineage transformation requires a source ref")
	}
	return nil
}

func validateMeasurement(name string, measurement Measurement) error {
	fields := []struct {
		name  string
		value *int64
	}{
		{name: "bytes", value: measurement.Bytes},
		{name: "lines", value: measurement.Lines},
		{name: "estimator_tokens", value: measurement.EstimatorTokens},
	}
	for _, field := range fields {
		if field.value != nil && *field.value < 0 {
			return fmt.Errorf("artifact manifest: %s.%s must be non-negative", name, field.name)
		}
	}
	if hash := strings.TrimSpace(measurement.SHA256); hash != "" && !sha256Pattern.MatchString(hash) {
		return fmt.Errorf("artifact manifest: %s.sha256 must be a lowercase SHA-256 digest", name)
	}
	return nil
}

func validatePreview(preview Preview, visible Measurement) error {
	switch preview.Kind {
	case PreviewKindNone, PreviewKindFull, PreviewKindPrefix, PreviewKindHeadTail, PreviewKindRange, PreviewKindNested:
	default:
		return fmt.Errorf("artifact manifest: invalid preview kind %q", preview.Kind)
	}
	if preview.Kind == PreviewKindFull && preview.Truncated {
		return fmt.Errorf("artifact manifest: full preview cannot be truncated")
	}
	if preview.Truncated && visible.Complete != nil && *visible.Complete {
		return fmt.Errorf("artifact manifest: visible_measurement.complete conflicts with truncated preview")
	}
	if preview.Kind == PreviewKindNone && len(preview.Ranges) > 0 {
		return fmt.Errorf("artifact manifest: none preview cannot declare ranges")
	}
	var lastEndByUnit = map[RangeUnit]int64{}
	for i, span := range preview.Ranges {
		if !validRangeUnit(span.Unit) {
			return fmt.Errorf("artifact manifest: preview range %d has invalid unit %q", i, span.Unit)
		}
		if span.Start < 0 || span.End <= span.Start {
			return fmt.Errorf("artifact manifest: preview range %d must be a non-empty half-open range", i)
		}
		if last, ok := lastEndByUnit[span.Unit]; ok && span.Start < last {
			return fmt.Errorf("artifact manifest: preview ranges overlap or are out of order")
		}
		lastEndByUnit[span.Unit] = span.End
	}
	if visible.Bytes != nil && *visible.Bytes > 0 && preview.Kind != PreviewKindNone && len(preview.Ranges) == 0 {
		return fmt.Errorf("artifact manifest: visible preview requires coverage ranges")
	}
	return nil
}

func validateRetention(retention Retention) error {
	switch retention.Class {
	case RetentionDurable, RetentionEphemeral:
	default:
		return fmt.Errorf("artifact manifest: invalid retention class %q", retention.Class)
	}
	if retention.CreatedAt.IsZero() {
		return fmt.Errorf("artifact manifest: retention created_at is required")
	}
	if retention.GCEligibleAfter != nil && retention.GCEligibleAfter.Before(retention.CreatedAt) {
		return fmt.Errorf("artifact manifest: gc_eligible_after precedes created_at")
	}
	if retention.ExpiresAt != nil {
		if retention.Class != RetentionEphemeral {
			return fmt.Errorf("artifact manifest: expires_at is allowed only for ephemeral objects")
		}
		if !retention.ExpiresAt.After(retention.CreatedAt) {
			return fmt.Errorf("artifact manifest: expires_at must follow created_at")
		}
	}
	return nil
}

package compaction

import (
	"bytes"
	"fmt"
	"reflect"
	"strings"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/artifact"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

const (
	compactionArtifactBindingStage  = "compaction_artifact_binding"
	compactionArtifactValidateStage = "compaction_artifact_validate"
	compactionArtifactWriteStage    = "compaction_artifact_write"
)

func (r *localReducer) canonicalArtifactsConfigured() bool {
	if r == nil || r.service == nil {
		return false
	}
	cfg := r.service.Config
	return cfg.ArtifactOwnerProvider != nil || cfg.ArtifactSink != nil || cfg.ArtifactResolver != nil ||
		cfg.ArtifactResolverCapability.Registered || strings.TrimSpace(cfg.ArtifactResolverCapability.Recovery.Capability) != ""
}

func (r *localReducer) validateCanonicalBinding() error {
	if r == nil || r.service == nil {
		return fmt.Errorf("canonical compaction service is unavailable")
	}
	cfg := r.service.Config
	if cfg.ArtifactOwnerProvider == nil {
		return fmt.Errorf("artifact owner provider is not configured")
	}
	if cfg.ArtifactSink == nil {
		return fmt.Errorf("artifact sink is not configured")
	}
	if cfg.ArtifactResolver == nil {
		return fmt.Errorf("artifact resolver is not configured")
	}
	if err := cfg.ArtifactResolverCapability.Validate(); err != nil {
		return err
	}
	if !cfg.ArtifactResolverCapability.Registered {
		return fmt.Errorf("artifact resolver capability is not registered")
	}
	return nil
}

func (r *localReducer) canonicalArtifactForContent(msg llm.Message, kind artifact.ObjectKind, content string) (artifact.Manifest, string, error) {
	if err := r.validateCanonicalBinding(); err != nil {
		return artifact.Manifest{}, compactionArtifactBindingStage, err
	}
	owner, err := r.canonicalArtifactOwner(msg)
	if err != nil {
		return artifact.Manifest{}, compactionArtifactBindingStage, err
	}
	codec := r.service.Config.ArtifactEnvelopeCodec
	if codec == nil {
		codec = artifact.JSONEnvelopeCodec{}
	}
	envelope, found, err := codec.Decode(content)
	if found {
		if err != nil {
			return artifact.Manifest{}, compactionArtifactValidateStage, err
		}
		if err := envelope.Validate(); err != nil {
			return artifact.Manifest{}, compactionArtifactValidateStage, err
		}
		if !reflect.DeepEqual(envelope.Manifest.Owner, owner) {
			return artifact.Manifest{}, compactionArtifactValidateStage, fmt.Errorf("canonical envelope owner does not match the active compaction subject")
		}
		resolved, err := r.resolveCanonicalArtifact(envelope.Manifest, owner, kind, nil)
		if err != nil {
			return artifact.Manifest{}, compactionArtifactValidateStage, err
		}
		return resolved, "", nil
	}

	request := artifact.PutRequest{
		ObjectKind: kind,
		Owner:      owner,
		Content:    []byte(content),
		Retention: artifact.Retention{
			Class:     artifact.RetentionDurable,
			CreatedAt: time.Now().UTC(),
		},
		ContentType: "text/plain",
		Encoding:    artifact.EncodingUTF8,
		Recovery:    cloneCanonicalRecovery(r.service.Config.ArtifactResolverCapability.Recovery),
	}
	manifest, err := r.service.Config.ArtifactSink.Put(r.ctx, request)
	if err != nil {
		return artifact.Manifest{}, compactionArtifactWriteStage, err
	}
	if err := validateCanonicalPutManifest(manifest, request); err != nil {
		return artifact.Manifest{}, compactionArtifactWriteStage, err
	}
	resolved, err := r.resolveCanonicalArtifact(manifest, owner, kind, request.Content)
	if err != nil {
		return artifact.Manifest{}, compactionArtifactValidateStage, err
	}
	return resolved, "", nil
}

func (r *localReducer) canonicalArtifactOwner(msg llm.Message) (artifact.Owner, error) {
	owner, err := r.service.Config.ArtifactOwnerProvider(r.ctx)
	if err != nil {
		return artifact.Owner{}, fmt.Errorf("resolve current compaction artifact owner: %w", err)
	}
	owner.ToolName = ""
	owner.ToolCallID = ""
	owner.Stream = ""
	owner.Part = ""
	switch msg.Role {
	case llm.RoleTool:
		owner.ToolName = strings.TrimSpace(msg.ToolName)
		owner.ToolCallID = strings.TrimSpace(msg.ToolCallID)
	case llm.RoleAssistant:
		owner.ToolName = "assistant"
	case llm.RoleUser:
		owner.ToolName = "user_code"
	default:
		return artifact.Owner{}, fmt.Errorf("unsupported compaction artifact role %q", msg.Role)
	}
	if strings.TrimSpace(owner.WorkspaceID) == "" {
		return artifact.Owner{}, fmt.Errorf("artifact owner workspace_id is required")
	}
	if err := owner.Validate(); err != nil {
		return artifact.Owner{}, err
	}
	return owner, nil
}

func (r *localReducer) validateCanonicalReplacement(msg llm.Message, replacement LedgerReplacement) (LedgerReplacement, string, bool) {
	if replacement.CanonicalArtifact == nil {
		warning := canonicalCompactionArtifactWarning(r.sessionID, msg.ToolName, msg.ToolCallID, compactionArtifactValidateStage, "legacy/unverified full_artifact has no canonical source descriptor")
		return LedgerReplacement{}, warning, false
	}
	if err := r.validateCanonicalBinding(); err != nil {
		warning := canonicalCompactionArtifactWarning(r.sessionID, msg.ToolName, msg.ToolCallID, compactionArtifactBindingStage, err.Error())
		return LedgerReplacement{}, warning, false
	}
	owner, err := r.canonicalArtifactOwner(msg)
	if err != nil {
		warning := canonicalCompactionArtifactWarning(r.sessionID, msg.ToolName, msg.ToolCallID, compactionArtifactBindingStage, err.Error())
		return LedgerReplacement{}, warning, false
	}
	wantKind := artifact.ObjectKindLogicalToolResult
	if msg.Role != llm.RoleTool {
		wantKind = artifact.ObjectKindCompactionMaterial
	}
	manifest, err := r.resolveCanonicalArtifact(*replacement.CanonicalArtifact, owner, wantKind, nil)
	if err != nil {
		warning := canonicalCompactionArtifactWarning(r.sessionID, msg.ToolName, msg.ToolCallID, compactionArtifactValidateStage, err.Error())
		return LedgerReplacement{}, warning, false
	}
	replacement.CanonicalArtifact = cloneCanonicalManifestPointer(manifest)
	replacement.FullArtifact = ""
	return replacement, "", true
}

func (r *localReducer) validateExistingPruneReplacement(msg llm.Message, parent, existing LedgerReplacement) (LedgerReplacement, string, bool) {
	if !r.canonicalArtifactsConfigured() {
		return existing, "", true
	}
	validated, warning, ok := r.validateCanonicalReplacement(msg, existing)
	if !ok {
		return LedgerReplacement{}, warning, false
	}
	if parent.CanonicalArtifact == nil || validated.CanonicalArtifact == nil ||
		parent.CanonicalArtifact.ObjectRef != validated.CanonicalArtifact.ObjectRef {
		warning := canonicalCompactionArtifactWarning(r.sessionID, msg.ToolName, msg.ToolCallID, compactionArtifactValidateStage, "prune replacement canonical object_ref does not match its snip parent")
		return LedgerReplacement{}, warning, false
	}
	return validated, "", true
}

func (r *localReducer) resolveCanonicalArtifact(descriptor artifact.Manifest, owner artifact.Owner, kind artifact.ObjectKind, exactContent []byte) (artifact.Manifest, error) {
	if err := validateVerifiedCanonicalManifest(descriptor, owner, kind, r.service.Config.ArtifactResolverCapability); err != nil {
		return artifact.Manifest{}, err
	}
	result, err := r.service.Config.ArtifactResolver.Resolve(r.ctx, artifact.ResolveRequest{
		ObjectRef: descriptor.ObjectRef,
		Owner:     owner,
	})
	if err != nil {
		return artifact.Manifest{}, fmt.Errorf("resolve canonical compaction artifact: %w", err)
	}
	if result.Range != nil {
		return artifact.Manifest{}, fmt.Errorf("resolver returned a partial range for full canonical validation")
	}
	if err := validateVerifiedCanonicalManifest(result.Manifest, owner, kind, r.service.Config.ArtifactResolverCapability); err != nil {
		return artifact.Manifest{}, fmt.Errorf("resolved canonical manifest invalid: %w", err)
	}
	if !canonicalManifestIdentityEqual(descriptor, result.Manifest) {
		return artifact.Manifest{}, fmt.Errorf("resolved canonical manifest identity does not match the ledger/envelope manifest")
	}
	if result.Manifest.ObjectMeasurement.Bytes == nil || int64(len(result.Content)) != *result.Manifest.ObjectMeasurement.Bytes {
		return artifact.Manifest{}, fmt.Errorf("resolved canonical byte count mismatch")
	}
	if got := artifact.DigestSHA256(result.Content); got != result.Manifest.ObjectMeasurement.SHA256 {
		return artifact.Manifest{}, fmt.Errorf("resolved canonical sha256 mismatch")
	}
	if exactContent != nil && !bytes.Equal(result.Content, exactContent) {
		return artifact.Manifest{}, fmt.Errorf("resolved canonical content does not match the bytes supplied to the sink")
	}
	return result.Manifest.Clone(), nil
}

func validateVerifiedCanonicalManifest(manifest artifact.Manifest, owner artifact.Owner, kind artifact.ObjectKind, capability artifact.ResolverCapability) error {
	if err := manifest.Validate(); err != nil {
		return err
	}
	if !manifest.Complete || !manifest.Recoverable {
		return fmt.Errorf("canonical manifest must be complete and recoverable")
	}
	if manifest.ObjectKind != kind {
		return fmt.Errorf("canonical manifest object_kind=%q cannot populate %q compaction source", manifest.ObjectKind, kind)
	}
	if manifest.ObjectKind == artifact.ObjectKindProviderVisibleView {
		return fmt.Errorf("provider-visible view cannot populate a verified compaction source")
	}
	if !reflect.DeepEqual(manifest.Owner, owner) {
		return fmt.Errorf("canonical manifest owner mismatch")
	}
	if manifest.Retention.Class != artifact.RetentionDurable || manifest.Retention.ExpiresAt != nil {
		return fmt.Errorf("canonical compaction artifact retention must be durable without expires_at")
	}
	if manifest.ObjectMeasurement.Bytes == nil || strings.TrimSpace(manifest.ObjectMeasurement.SHA256) == "" {
		return fmt.Errorf("canonical manifest must contain measured bytes and sha256")
	}
	if err := capability.Validate(); err != nil {
		return err
	}
	if !capability.Registered {
		return fmt.Errorf("canonical resolver capability is not registered")
	}
	if !canonicalRecoveryEqual(manifest.Recovery, capability.Recovery) {
		return fmt.Errorf("canonical recovery contract does not match the registered capability")
	}
	return nil
}

func validateCanonicalPutManifest(manifest artifact.Manifest, request artifact.PutRequest) error {
	capability := artifact.ResolverCapability{Registered: true, Recovery: cloneCanonicalRecovery(request.Recovery)}
	if err := validateVerifiedCanonicalManifest(manifest, request.Owner, request.ObjectKind, capability); err != nil {
		return fmt.Errorf("artifact sink returned invalid manifest: %w", err)
	}
	if manifest.ObjectMeasurement.Bytes == nil || *manifest.ObjectMeasurement.Bytes != int64(len(request.Content)) {
		return fmt.Errorf("artifact sink byte count mismatch")
	}
	if manifest.ObjectMeasurement.SHA256 != artifact.DigestSHA256(request.Content) {
		return fmt.Errorf("artifact sink sha256 mismatch")
	}
	if manifest.ContentType != request.ContentType || manifest.Encoding != request.Encoding {
		return fmt.Errorf("artifact sink content type or encoding mismatch")
	}
	if !reflect.DeepEqual(manifest.Lineage, request.Lineage) {
		return fmt.Errorf("artifact sink lineage mismatch")
	}
	return nil
}

func canonicalManifestIdentityEqual(left, right artifact.Manifest) bool {
	left = left.Clone()
	right = right.Clone()
	left.VisibleMeasurement = artifact.Measurement{}
	right.VisibleMeasurement = artifact.Measurement{}
	left.Preview = artifact.Preview{Kind: artifact.PreviewKindNone}
	right.Preview = artifact.Preview{Kind: artifact.PreviewKindNone}
	return reflect.DeepEqual(left, right)
}

func canonicalRecoveryEqual(left, right artifact.Recovery) bool {
	return left.Capability == right.Capability && left.Tool == right.Tool &&
		left.Instruction == right.Instruction && reflect.DeepEqual(left.AllowedRangeUnits, right.AllowedRangeUnits)
}

func cloneCanonicalRecovery(recovery artifact.Recovery) artifact.Recovery {
	out := recovery
	out.AllowedRangeUnits = append([]artifact.RangeUnit(nil), recovery.AllowedRangeUnits...)
	return out
}

func cloneCanonicalManifestPointer(manifest artifact.Manifest) *artifact.Manifest {
	clone := manifest.Clone()
	return &clone
}

func cloneCanonicalManifestPointerFrom(manifest *artifact.Manifest) *artifact.Manifest {
	if manifest == nil {
		return nil
	}
	return cloneCanonicalManifestPointer(*manifest)
}

func canonicalArtifactProjection(manifest artifact.Manifest) string {
	bytesCount := int64(0)
	if manifest.ObjectMeasurement.Bytes != nil {
		bytesCount = *manifest.ObjectMeasurement.Bytes
	}
	return fmt.Sprintf("object_ref=%s object_bytes=%d sha256=%s complete=%t recoverable=%t recovery_tool=%s recovery_instruction=%q",
		manifest.ObjectRef,
		bytesCount,
		manifest.ObjectMeasurement.SHA256,
		manifest.Complete,
		manifest.Recoverable,
		manifest.Recovery.Tool,
		manifest.Recovery.Instruction,
	)
}

func canonicalSnipReplacementText(msg llm.Message, original string, manifest artifact.Manifest) string {
	tool := strings.TrimSpace(msg.ToolName)
	if tool == "" {
		tool = "tool"
	}
	id := strings.TrimSpace(msg.ToolCallID)
	if id == "" {
		id = "-"
	}
	return fmt.Sprintf("[Tool result snipped: %s tool_call_id=%s lines=%d bytes=%d %s]", tool, id, countTextLines(original), len(original), canonicalArtifactProjection(manifest))
}

func canonicalToolPruneReplacementText(msg llm.Message, manifest artifact.Manifest) string {
	tool := strings.TrimSpace(msg.ToolName)
	if tool == "" {
		tool = "tool"
	}
	id := strings.TrimSpace(msg.ToolCallID)
	if id == "" {
		id = "-"
	}
	return fmt.Sprintf("[Tool result pruned: %s tool_call_id=%s %s]", tool, id, canonicalArtifactProjection(manifest))
}

func canonicalAssistantPruneReplacementText(original string, manifest artifact.Manifest, estimate tokenEstimator) string {
	return fmt.Sprintf("[Assistant text compacted: lines=%d bytes=%d %s]\n%s", countTextLines(original), len(original), canonicalArtifactProjection(manifest), assistantPreviewWithEstimator(original, estimate))
}

func canonicalUserCodeMicrocompactReplacementText(original string, manifest artifact.Manifest, estimate func(string) int) (string, bool) {
	block, ok := largestFencedCodeBlock(original)
	if !ok || estimate(strings.Join(block.lines, "\n")) < defaultUserCodeMicrocompactMinTokens {
		return "", false
	}
	allLines := strings.Split(strings.ReplaceAll(original, "\r\n", "\n"), "\n")
	out := make([]string, 0, len(allLines)-len(block.lines)+userCodePreviewHeadLines+userCodePreviewTailLines+4)
	out = append(out, allLines[:block.startLine]...)
	out = append(out, canonicalUserCodeBlockReplacementLines(block, manifest)...)
	if block.endLine+1 < len(allLines) {
		out = append(out, allLines[block.endLine+1:]...)
	}
	return strings.TrimSpace(strings.Join(out, "\n")), true
}

func canonicalUserCodeBlockReplacementLines(block fencedCodeBlock, manifest artifact.Manifest) []string {
	lang := strings.TrimSpace(block.language)
	if lang == "" {
		lang = "-"
	}
	hint := strings.TrimSpace(block.hint)
	if hint == "" {
		hint = "-"
	}
	header := fmt.Sprintf("[User code block compacted: language=%s hint=%s lines=%d bytes=%d %s]", lang, hint, len(block.lines), len(strings.Join(block.lines, "\n")), canonicalArtifactProjection(manifest))
	out := []string{header, "Preview:"}
	switch {
	case len(block.lines) <= userCodePreviewHeadLines+userCodePreviewTailLines:
		out = append(out, block.lines...)
	default:
		out = append(out, block.lines[:userCodePreviewHeadLines]...)
		out = append(out, fmt.Sprintf("[...%d middle lines omitted; full code object_ref=%s]", len(block.lines)-userCodePreviewHeadLines-userCodePreviewTailLines, manifest.ObjectRef))
		out = append(out, block.lines[len(block.lines)-userCodePreviewTailLines:]...)
	}
	return out
}

func canonicalCompactionArtifactWarning(sessionID, toolName, toolCallID, stage, detail string) string {
	return fmt.Sprintf("[WARN] Compaction canonical artifact rejected - session=%s stage=%s tool=%s tool_call_id=%s action=leaving original message in context: %s",
		strings.TrimSpace(sessionID), strings.TrimSpace(stage), strings.TrimSpace(toolName), strings.TrimSpace(toolCallID), strings.TrimSpace(detail))
}

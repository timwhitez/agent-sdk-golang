package agent

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/timwhitez/agent-sdk-golang/sdk/artifact"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

const (
	artifactSinkFailureStage       = "artifact_sink"
	artifactCapabilityFailureStage = "artifact_capability"
	artifactOwnerFailureStage      = "artifact_owner"
	artifactDecodeFailureStage     = "artifact_decode"
	artifactCodecFailureStage      = "artifact_codec"
	artifactLineageFailureStage    = "artifact_lineage"
)

func (a *Agent) applyToolResultBoundary(ctx context.Context, content llm.Content, meta map[string]any, toolName, toolCallID string) (llm.Content, map[string]any) {
	plain := content.PlainText()
	codec := a.artifactEnvelopeCodec
	if codec == nil {
		codec = artifact.JSONEnvelopeCodec{}
	}
	decoded, found, decodeErr := codec.Decode(plain)
	if found {
		if decodeErr != nil {
			return a.nonRecoverableToolResultFallback(plain, meta, artifactDecodeFailureStage, "return_valid_envelope_or_plain_result", decodeErr)
		}
		owner, err := a.boundArtifactOwner(ctx, toolName, toolCallID)
		if err != nil {
			return a.nonRecoverableToolResultFallback(plain, meta, artifactOwnerFailureStage, a.artifactOwnerFailureAction(), err)
		}
		if err := a.validateExistingArtifactEnvelope(decoded, owner); err != nil {
			return a.nonRecoverableToolResultFallback(plain, meta, artifactDecodeFailureStage, "restore_owner_and_resolver_binding", err)
		}
		encoded, normalized, err := codec.Encode(decoded, a.toolResultArtifactBudget())
		if err != nil {
			return a.nonRecoverableToolResultFallback(plain, meta, artifactCodecFailureStage, "increase_budget_or_shorten_fixed_recovery_fields", err)
		}
		return llm.TextContent(encoded), projectArtifactManifestMetadata(meta, normalized.Manifest, len(plain), len(encoded), a.maxToolResultBytes)
	}
	sourceManifests, hasSourceManifests, err := artifactSourceManifestsFromMetadata(meta)
	if err != nil {
		return a.nonRecoverableToolResultFallback(plain, withoutArtifactSourceManifests(meta), artifactLineageFailureStage, "restore_verified_source_manifests", err)
	}
	if !hasSourceManifests && !a.toolResultExceedsBudget(plain) {
		return content, meta
	}

	if a.artifactSink == nil {
		return a.nonRecoverableToolResultFallback(plain, meta, artifactSinkFailureStage, "configure_artifact_sink", fmt.Errorf("host artifact sink is not configured"))
	}
	if err := a.artifactResolverCapability.Validate(); err != nil {
		return a.nonRecoverableToolResultFallback(plain, meta, artifactCapabilityFailureStage, "register_valid_resolver_capability", err)
	}
	if !a.artifactResolverCapability.Registered {
		return a.nonRecoverableToolResultFallback(plain, meta, artifactCapabilityFailureStage, "register_resolver_capability", fmt.Errorf("host resolver capability is not registered"))
	}
	owner, err := a.boundArtifactOwner(ctx, toolName, toolCallID)
	if err != nil {
		return a.nonRecoverableToolResultFallback(plain, meta, artifactOwnerFailureStage, a.artifactOwnerFailureAction(), err)
	}
	var lineage *artifact.Lineage
	if hasSourceManifests {
		lineage, err = a.validateArtifactSourceManifests(sourceManifests, owner)
		if err != nil {
			return a.nonRecoverableToolResultFallback(plain, withoutArtifactSourceManifests(meta), artifactLineageFailureStage, "restore_verified_source_manifests", err)
		}
	}

	request := artifact.PutRequest{
		ObjectKind: artifact.ObjectKindLogicalToolResult,
		Owner:      owner,
		Content:    []byte(plain),
		Lineage:    lineage,
		Retention: artifact.Retention{
			Class:     artifact.RetentionDurable,
			CreatedAt: time.Now().UTC(),
		},
		ContentType: "text/plain",
		Encoding:    artifact.EncodingUTF8,
		Recovery:    cloneArtifactRecovery(a.artifactResolverCapability.Recovery),
	}
	manifest, err := a.artifactSink.Put(ctx, request)
	if err != nil {
		return a.nonRecoverableToolResultFallback(plain, meta, artifactSinkFailureStage, "retry_or_reconfigure_artifact_sink", err)
	}
	if err := validateArtifactSinkManifest(manifest, request); err != nil {
		return a.nonRecoverableToolResultFallback(plain, meta, artifactSinkFailureStage, "repair_sink_manifest_contract", err)
	}

	manifest = manifest.Clone()
	manifest.Preview = artifact.Preview{Kind: artifact.PreviewKindFull}
	manifest.VisibleMeasurement = artifact.Measurement{}
	envelope := artifact.Envelope{
		Manifest: manifest,
		Preview:  plain,
		Continuation: &artifact.Continuation{
			ObjectRef: manifest.ObjectRef,
			RangeUnit: artifact.RangeUnitBytes,
		},
	}
	encoded, normalized, err := codec.Encode(envelope, a.toolResultArtifactBudget())
	if err != nil {
		return a.nonRecoverableToolResultFallback(plain, meta, artifactCodecFailureStage, "increase_budget_or_shorten_fixed_recovery_fields", err)
	}
	return llm.TextContent(encoded), projectArtifactManifestMetadata(meta, normalized.Manifest, len(plain), len(encoded), a.maxToolResultBytes)
}

func withoutArtifactSourceManifests(meta map[string]any) map[string]any {
	out := cloneToolResultMetadata(meta)
	delete(out, artifact.SourceManifestsMetadataKey)
	return out
}

func artifactSourceManifestsFromMetadata(meta map[string]any) ([]artifact.Manifest, bool, error) {
	if meta == nil {
		return nil, false, nil
	}
	raw, exists := meta[artifact.SourceManifestsMetadataKey]
	if !exists {
		return nil, false, nil
	}
	manifests, ok := raw.([]artifact.Manifest)
	if !ok {
		return nil, true, fmt.Errorf("%s must contain []artifact.Manifest, got %T", artifact.SourceManifestsMetadataKey, raw)
	}
	if len(manifests) == 0 {
		return nil, true, fmt.Errorf("%s must contain at least one source manifest", artifact.SourceManifestsMetadataKey)
	}
	out := make([]artifact.Manifest, len(manifests))
	for i := range manifests {
		out[i] = manifests[i].Clone()
	}
	return out, true, nil
}

func (a *Agent) validateArtifactSourceManifests(manifests []artifact.Manifest, owner artifact.Owner) (*artifact.Lineage, error) {
	refs := make([]string, 0, len(manifests))
	seen := make(map[string]struct{}, len(manifests))
	for i, manifest := range manifests {
		if err := manifest.Validate(); err != nil {
			return nil, fmt.Errorf("source manifest %d is invalid: %w", i, err)
		}
		if !manifest.Complete || !manifest.Recoverable {
			return nil, fmt.Errorf("source manifest %d must be complete and recoverable", i)
		}
		if manifest.ObjectKind == artifact.ObjectKindProviderVisibleView {
			return nil, fmt.Errorf("source manifest %d cannot use provider_visible_view as a verified source", i)
		}
		expectedOwner := owner
		expectedOwner.Stream = manifest.Owner.Stream
		expectedOwner.Part = manifest.Owner.Part
		if !reflect.DeepEqual(manifest.Owner, expectedOwner) {
			return nil, fmt.Errorf("source manifest %d owner does not match the active tool call", i)
		}
		if manifest.Retention.Class != artifact.RetentionDurable || manifest.Retention.ExpiresAt != nil {
			return nil, fmt.Errorf("source manifest %d retention must be durable without expires_at", i)
		}
		if !artifactRecoveryEqual(manifest.Recovery, a.artifactResolverCapability.Recovery) {
			return nil, fmt.Errorf("source manifest %d recovery contract does not match the registered capability", i)
		}
		if _, duplicate := seen[manifest.ObjectRef]; duplicate {
			return nil, fmt.Errorf("source manifest %d repeats object_ref %q", i, manifest.ObjectRef)
		}
		seen[manifest.ObjectRef] = struct{}{}
		refs = append(refs, manifest.ObjectRef)
	}
	return &artifact.Lineage{
		DerivedFrom:    refs,
		Transformation: artifact.TransformationToolSerializeV1,
	}, nil
}

func (a *Agent) validateExistingArtifactEnvelope(envelope artifact.Envelope, owner artifact.Owner) error {
	if err := envelope.Validate(); err != nil {
		return err
	}
	if !reflect.DeepEqual(envelope.Manifest.Owner, owner) {
		return fmt.Errorf("canonical envelope owner does not match the active tool call")
	}
	if envelope.Manifest.Recoverable {
		if err := a.artifactResolverCapability.Validate(); err != nil {
			return err
		}
		if !a.artifactResolverCapability.Registered {
			return fmt.Errorf("canonical envelope recovery capability is not registered")
		}
		if !artifactRecoveryEqual(envelope.Manifest.Recovery, a.artifactResolverCapability.Recovery) {
			return fmt.Errorf("canonical envelope recovery contract does not match the registered capability")
		}
	}
	return nil
}

func (a *Agent) boundArtifactOwner(ctx context.Context, toolName, toolCallID string) (artifact.Owner, error) {
	owner := a.artifactOwner
	if a.artifactOwnerProvider != nil {
		current, err := a.artifactOwnerProvider(ctx)
		if err != nil {
			return artifact.Owner{}, fmt.Errorf("resolve current artifact owner: %w", err)
		}
		owner = current
	}
	owner.ToolName = strings.TrimSpace(toolName)
	owner.ToolCallID = strings.TrimSpace(toolCallID)
	probe := artifact.Manifest{
		SchemaVersion: artifact.SchemaVersion,
		ObjectRef:     "obj:v1:owner-validation",
		ObjectKind:    artifact.ObjectKindLogicalToolResult,
		Owner:         owner,
		Complete:      false,
		Recoverable:   false,
		Preview:       artifact.Preview{Kind: artifact.PreviewKindNone},
		Retention: artifact.Retention{
			Class:     artifact.RetentionDurable,
			CreatedAt: time.Unix(1, 0).UTC(),
		},
		ContentType: "text/plain",
		Encoding:    artifact.EncodingUTF8,
	}
	if err := probe.Validate(); err != nil {
		return artifact.Owner{}, fmt.Errorf("invalid artifact owner: %w", err)
	}
	return owner, nil
}

func (a *Agent) artifactOwnerFailureAction() string {
	if a.artifactOwnerProvider != nil {
		return "resolve_current_artifact_owner"
	}
	return "configure_artifact_owner"
}

func validateArtifactSinkManifest(manifest artifact.Manifest, request artifact.PutRequest) error {
	if err := manifest.Validate(); err != nil {
		return fmt.Errorf("artifact sink returned invalid manifest: %w", err)
	}
	if !manifest.Complete || !manifest.Recoverable {
		return fmt.Errorf("artifact sink manifest must be complete and recoverable")
	}
	if manifest.ObjectKind != request.ObjectKind {
		return fmt.Errorf("artifact sink object_kind mismatch: got %q want %q", manifest.ObjectKind, request.ObjectKind)
	}
	if !reflect.DeepEqual(manifest.Owner, request.Owner) {
		return fmt.Errorf("artifact sink owner mismatch")
	}
	if manifest.ObjectMeasurement.Bytes == nil || *manifest.ObjectMeasurement.Bytes != int64(len(request.Content)) {
		return fmt.Errorf("artifact sink byte count mismatch")
	}
	if manifest.ObjectMeasurement.SHA256 != artifact.DigestSHA256(request.Content) {
		return fmt.Errorf("artifact sink sha256 mismatch")
	}
	if manifest.Retention.Class != artifact.RetentionDurable || manifest.Retention.ExpiresAt != nil {
		return fmt.Errorf("artifact sink retention must be durable without expires_at")
	}
	if !artifactRecoveryEqual(manifest.Recovery, request.Recovery) {
		return fmt.Errorf("artifact sink recovery contract mismatch")
	}
	if !reflect.DeepEqual(manifest.Lineage, request.Lineage) {
		return fmt.Errorf("artifact sink lineage mismatch")
	}
	return nil
}

func artifactRecoveryEqual(left, right artifact.Recovery) bool {
	return left.Capability == right.Capability &&
		left.Tool == right.Tool &&
		left.Instruction == right.Instruction &&
		reflect.DeepEqual(left.AllowedRangeUnits, right.AllowedRangeUnits)
}

func cloneArtifactRecovery(recovery artifact.Recovery) artifact.Recovery {
	out := recovery
	out.AllowedRangeUnits = append([]artifact.RangeUnit(nil), recovery.AllowedRangeUnits...)
	return out
}

func projectArtifactManifestMetadata(meta map[string]any, manifest artifact.Manifest, originalBytes, visibleBytes, maxBytes int) map[string]any {
	meta = cloneToolResultMetadata(meta)
	meta["artifact_manifest"] = manifest.Clone()
	meta["result_truncated"] = manifest.Preview.Truncated
	meta["result_bytes"] = visibleBytes
	meta["result_original_bytes"] = originalBytes
	meta["result_max_bytes"] = maxBytes
	meta["truncated"] = manifest.Preview.Truncated
	meta["originalSize"] = originalBytes
	return meta
}

func cloneToolResultMetadata(meta map[string]any) map[string]any {
	if meta == nil {
		return map[string]any{}
	}
	out := make(map[string]any, len(meta)+8)
	for key, value := range meta {
		out[key] = value
	}
	return out
}

func (a *Agent) nonRecoverableToolResultFallback(plain string, meta map[string]any, stage, action string, cause error) (llm.Content, map[string]any) {
	diagnostic := fmt.Sprintf("[WARN] stage=%s action=%s complete=false recoverable=false", stage, action)
	if cause != nil {
		a.warnf("warning: %s detail=%v", diagnostic, cause)
	} else {
		a.warnf("warning: %s", diagnostic)
	}

	_, meta, dumpPath := truncateToolResultContent(llm.TextContent(plain), meta, a.maxToolResultBytes, a.warnf)
	if dumpPath != "" {
		now := toolResultDumpNow()
		lifecycle := a.registerToolResultDump(dumpPath, now)
		a.cleanupToolResultDumps(now, false)
		meta = cloneToolResultMetadata(meta)
		meta["result_output_ttl_ms"] = a.toolResultDumpTTL.Milliseconds()
		meta["result_output_created_at"] = lifecycle.CreatedAt.UTC().Format(time.RFC3339)
		meta["result_output_expires_at"] = lifecycle.ExpiresAt.UTC().Format(time.RFC3339)
		meta["result_output_expiry_policy"] = toolResultDumpExpiryPolicy
	}
	meta = cloneToolResultMetadata(meta)
	delete(meta, "artifact_manifest")
	meta["artifact_complete"] = false
	meta["artifact_recoverable"] = false
	meta["artifact_stage"] = stage
	meta["artifact_action"] = action

	bounded := boundedToolResultFallback(plain, diagnostic, a.toolResultArtifactBudget())
	meta["result_truncated"] = true
	meta["result_original_bytes"] = len(plain)
	meta["result_bytes"] = len(bounded)
	meta["result_max_bytes"] = a.maxToolResultBytes
	meta["truncated"] = true
	meta["originalSize"] = len(plain)
	return llm.TextContent(bounded), meta
}

func (a *Agent) toolResultArtifactBudget() artifact.Budget {
	return artifact.Budget{
		MaxBytes:       a.maxToolResultBytes,
		MaxTokens:      a.maxToolResultTokens,
		EstimateTokens: a.toolResultTokenEstimator,
	}
}

func (a *Agent) toolResultExceedsBudget(text string) bool {
	if a.maxToolResultBytes > 0 && len(text) > a.maxToolResultBytes {
		return true
	}
	return a.maxToolResultTokens > 0 && estimateBoundaryTokens(a.toolResultTokenEstimator, text) > a.maxToolResultTokens
}

func boundedToolResultFallback(plain, diagnostic string, budget artifact.Budget) string {
	estimate := func(text string) int { return estimateBoundaryTokens(budget.EstimateTokens, text) }
	fits := func(text string) bool {
		return len(text) <= budget.MaxBytes && estimate(text) <= budget.MaxTokens
	}
	if !fits(diagnostic) {
		return utf8BudgetPrefix(diagnostic, budget, estimate)
	}
	suffix := "\n" + diagnostic
	if plain == "" {
		return diagnostic
	}
	low, high := 0, len(plain)
	best := diagnostic
	for low <= high {
		mid := low + (high-low)/2
		prefix := utf8PrefixBytes(plain, mid)
		candidate := prefix + suffix
		if fits(candidate) {
			best = candidate
			low = mid + 1
			continue
		}
		high = mid - 1
	}
	return best
}

func utf8BudgetPrefix(text string, budget artifact.Budget, estimate func(string) int) string {
	low, high := 0, len(text)
	best := ""
	for low <= high {
		mid := low + (high-low)/2
		candidate := utf8PrefixBytes(text, mid)
		if len(candidate) <= budget.MaxBytes && estimate(candidate) <= budget.MaxTokens {
			best = candidate
			low = mid + 1
			continue
		}
		high = mid - 1
	}
	return best
}

func utf8PrefixBytes(text string, maxBytes int) string {
	if maxBytes <= 0 {
		return ""
	}
	if len(text) <= maxBytes {
		return text
	}
	cut := maxBytes
	for cut > 0 && !utf8.ValidString(text[:cut]) {
		cut--
	}
	return text[:cut]
}

func estimateBoundaryTokens(estimator func(string) int, text string) int {
	if text == "" {
		return 0
	}
	if estimator != nil {
		if tokens := estimator(text); tokens > 0 {
			return tokens
		}
	}
	return (len(text) + 3) / 4
}

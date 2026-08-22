package accounting

import (
	"encoding/json"
	"reflect"
	"sort"
	"strings"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	sdkartifact "github.com/timwhitez/agent-sdk-golang/sdk/artifact"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type ToolResultInput struct {
	Tool     string
	Original string
	Visible  string
	IsError  bool
	Metadata map[string]any
}

func ProjectToolResult(input ToolResultInput, estimator Estimator) Payload {
	status := StatusSuccess
	if input.IsError {
		status = StatusError
	}
	originalBytes := int64(len(input.Original))
	visibleBytes := int64(len(input.Visible))
	payload := Payload{
		SchemaVersion: SchemaVersion,
		EventKind:     EventKindToolResult,
		Status:        status,
		ToolKind:      canonicalToolFamily(input.Tool),
		Measurements: Measurements{
			OriginalBytes:     &originalBytes,
			VisibleBytes:      &visibleBytes,
			MeasurementSource: "runtime",
		},
	}
	if identity, ok := estimator.identity(); ok {
		if original, ok := safeEstimate(estimator.EstimateTokens, input.Original); ok {
			value := int64(original)
			payload.Measurements.OriginalTokens = &value
		}
		if visible, ok := safeEstimate(estimator.EstimateTokens, input.Visible); ok {
			value := int64(visible)
			payload.Measurements.VisibleTokens = &value
		}
		if payload.Measurements.OriginalTokens != nil && payload.Measurements.VisibleTokens != nil {
			payload.Estimator = identity
		}
	}
	payload.Layers = projectLayers(input.Metadata, originalBytes, visibleBytes)
	payload.Artifact = projectArtifact(input.Metadata)
	payload.Scan = projectScan(input.Metadata)
	payload.RepeatedEvidence = projectRepeatedEvidence(input.Metadata)
	if payload.Scan != nil && payload.Scan.ScanComplete != nil && !*payload.Scan.ScanComplete {
		payload.Status = StatusPartial
	}
	return payload
}

func ProjectUsage(usage llm.Usage, estimators ...Estimator) Payload {
	payload := Payload{
		SchemaVersion: SchemaVersion,
		EventKind:     EventKindProviderUsage,
		Status:        StatusUnknown,
		Usage: &UsageDisposition{
			PromptTokensValid:     usage.PromptTokensValid,
			PromptTokensSource:    boundedIdentifier(usage.PromptTokensSource),
			PromptTokensSemantics: boundedIdentifier(usage.PromptTokensSemantics),
			SummationPolicy:       "usage_sum_v1",
		},
	}
	if usage.PromptTokens > 0 || strings.TrimSpace(usage.PromptTokensSource) != "" {
		payload.Usage.EffectivePromptTokens = intPointer(usage.PromptTokens)
	}
	if usage.CompletionTokens > 0 {
		payload.Usage.CompletionTokens = intPointer(usage.CompletionTokens)
	}
	if usage.TotalTokens > 0 {
		payload.Usage.TotalTokens = intPointer(usage.TotalTokens)
	}
	payload.Usage.ProviderPromptTokens = cloneIntPointer(usage.ProviderPromptTokens)
	payload.Usage.ProviderTotalTokens = cloneIntPointer(usage.ProviderTotalTokens)
	payload.Usage.PromptCachedTokens = cloneIntPointer(usage.PromptCachedTokens)
	payload.Usage.CacheCreationTokens = cloneIntPointer(usage.PromptCacheCreationTokens)
	payload.Usage.PromptImageTokens = cloneIntPointer(usage.PromptImageTokens)
	payload.Usage.PromptUncachedTokens = cloneIntPointer(usage.PromptUncachedTokens)
	if payload.Usage.EffectivePromptTokens != nil || payload.Usage.CompletionTokens != nil || payload.Usage.TotalTokens != nil || payload.Usage.ProviderPromptTokens != nil || payload.Usage.ProviderTotalTokens != nil {
		payload.Status = StatusSuccess
	}
	if payload.Usage.EffectivePromptTokens != nil && payload.Usage.PromptTokensSource == llm.PromptTokensSourceEstimate && len(estimators) > 0 {
		if identity, ok := estimators[0].identity(); ok {
			payload.Estimator = identity
		}
	}
	return payload
}

func ProjectCompaction(result compaction.Result, estimator Estimator) Payload {
	payload := Payload{
		SchemaVersion: SchemaVersion,
		EventKind:     EventKindCompaction,
		Status:        StatusSuccess,
		Compaction: &CompactionDisposition{
			Compacted:        result.Compacted,
			Trigger:          boundedIdentifier(result.Trigger),
			Watermark:        boundedIdentifier(result.Watermark),
			TokenCountSource: boundedIdentifier(result.TokenCountSource),
			TiersApplied:     boundedUniqueStrings(result.TiersApplied, MaxLayers),
			CheckpointID:     boundedIdentifier(result.CheckpointID),
			GenerationDelta:  boolInt(result.Compacted),
			WarningCount:     len(result.Warnings),
			SummaryPresent:   strings.TrimSpace(result.Summary) != "",
		},
	}
	if result.OriginalTokens > 0 {
		payload.Compaction.OriginalTokens = intPointer(result.OriginalTokens)
	}
	if result.NewTokens > 0 {
		payload.Compaction.NewTokens = intPointer(result.NewTokens)
	}
	if result.CheckpointMessages > 0 {
		payload.Compaction.CheckpointMessages = intPointer(result.CheckpointMessages)
	}
	if result.TokenCountSource == compaction.TokenCountSourceEstimate && payload.Compaction.OriginalTokens != nil && payload.Compaction.NewTokens != nil {
		if identity, ok := estimator.identity(); ok {
			payload.Estimator = identity
		}
	}
	payload.Layers = []LayerDisposition{{
		Layer:     LayerCompaction,
		Truncated: boolPointer(result.Compacted),
		Complete:  boolPointer(result.Compacted),
		Reason:    boundedIdentifier(result.Trigger),
	}}
	if !result.Compacted {
		payload.Status = StatusUnknown
	}
	return payload
}

func projectLayers(meta map[string]any, boundaryOriginal, boundaryVisible int64) []LayerDisposition {
	layers := make([]LayerDisposition, 0, 3)
	if outputBytes, ok := metaInt64(meta, "output_bytes"); ok {
		producer := LayerDisposition{Layer: LayerProducer, OriginalBytes: &outputBytes}
		if limit, ok := metaInt64(meta, "output_bytes_limit", "output_max_bytes"); ok {
			producer.LimitBytes = &limit
			visible := outputBytes
			if visible > limit {
				visible = limit
			}
			producer.VisibleBytes = &visible
		}
		if truncated, ok := metaBool(meta, "output_truncated"); ok {
			producer.Truncated = &truncated
			complete := !truncated
			producer.Complete = &complete
		}
		layers = append(layers, producer)
	}
	if bytes, ok := metaInt64(meta, "bytes", "original_bytes"); ok {
		wrapper := LayerDisposition{Layer: LayerWrapper, OriginalBytes: &bytes}
		visible := boundaryOriginal
		wrapper.VisibleBytes = &visible
		if truncated, ok := metaBool(meta, "truncated", "wrapper_truncated"); ok {
			wrapper.Truncated = &truncated
			complete := !truncated
			wrapper.Complete = &complete
		}
		layers = append(layers, wrapper)
	}
	boundary := LayerDisposition{
		Layer:         LayerAgentBoundary,
		OriginalBytes: &boundaryOriginal,
		VisibleBytes:  &boundaryVisible,
	}
	truncated := boundaryVisible < boundaryOriginal
	if value, ok := metaBool(meta, "result_truncated"); ok {
		truncated = value
	}
	boundary.Truncated = &truncated
	complete := !truncated
	if value, ok := metaBool(meta, "artifact_complete"); ok && !value {
		complete = false
	}
	boundary.Complete = &complete
	if limit, ok := metaInt64(meta, "result_max_bytes"); ok {
		boundary.LimitBytes = &limit
	}
	layers = append(layers, boundary)
	if len(layers) > MaxLayers {
		layers = layers[:MaxLayers]
	}
	return layers
}

func projectArtifact(meta map[string]any) *ArtifactDisposition {
	if len(meta) == 0 {
		return nil
	}
	recoveryDisposition := ""
	if value, ok := metaString(meta, "artifact_recovery_disposition"); ok && validArtifactRecoveryDisposition(value) {
		recoveryDisposition = value
	}
	if manifest, ok := artifactManifest(meta["artifact_manifest"]); ok {
		disposition := &ArtifactDisposition{
			ObjectRef:           boundedObjectRef(manifest.ObjectRef),
			ObjectKind:          boundedIdentifier(string(manifest.ObjectKind)),
			Complete:            boolPointer(manifest.Complete),
			Recoverable:         boolPointer(manifest.Recoverable),
			RetentionClass:      boundedIdentifier(string(manifest.Retention.Class)),
			HashDisposition:     "invalid",
			RecoveryDisposition: recoveryDisposition,
		}
		if manifest.Validate() == nil {
			disposition.HashDisposition = "validated"
		}
		if manifest.Retention.ExpiresAt != nil {
			disposition.ExpiresAt = manifest.Retention.ExpiresAt.UTC().Format(time.RFC3339)
		}
		return disposition
	}
	_, legacyPath := metaString(meta, "result_output_path")
	complete, hasComplete := metaBool(meta, "artifact_complete")
	recoverable, hasRecoverable := metaBool(meta, "artifact_recoverable")
	if !legacyPath && !hasComplete && !hasRecoverable && recoveryDisposition == "" {
		return nil
	}
	disposition := &ArtifactDisposition{LegacyPathPresent: legacyPath, RecoveryDisposition: recoveryDisposition}
	if hasComplete {
		disposition.Complete = &complete
	}
	if hasRecoverable {
		disposition.Recoverable = &recoverable
	}
	return disposition
}

func projectScan(meta map[string]any) *ScanDisposition {
	scanComplete, hasScan := metaBool(meta, "scan_complete")
	if !hasScan {
		return nil
	}
	scan := &ScanDisposition{ScanComplete: &scanComplete}
	if value, ok := metaBool(meta, "return_truncated"); ok {
		scan.ReturnTruncated = &value
	}
	if value, ok := metaBool(meta, "matches_total_known"); ok {
		scan.MatchesTotalKnown = &value
	}
	scan.WalkEntriesSeen = metaIntPointer(meta, "walk_entries_seen")
	scan.EligibleCandidates = metaIntPointer(meta, "eligible_candidates")
	scan.FilesOpened = metaIntPointer(meta, "files_opened")
	scan.FilesMatched = metaIntPointer(meta, "files_matched")
	scan.MatchesReturned = metaIntPointer(meta, "matches_returned")
	scan.PageIndex = metaIntPointer(meta, "page_index", "scan_page")
	if reason, ok := metaString(meta, "budget_exhausted_reason"); ok {
		scan.BudgetReason = boundedIdentifier(reason)
	}
	if phase, ok := metaString(meta, "scan_phase", "snapshot_phase"); ok {
		scan.Phase = boundedIdentifier(phase)
	}
	scan.ContinuationPresent = metaNonEmpty(meta, "continuation")
	if value, ok := metaBool(meta, "continuation_consumed"); ok {
		scan.ContinuationConsumed = value
	}
	if value, ok := metaBool(meta, "snapshot_invalidated"); ok {
		scan.SnapshotInvalidated = value
	}
	scan.InFileRangePresent = metaHasAny(meta, "line_start_byte", "line_end_byte", "match_start_byte", "match_end_byte", "long_line_recovery_ranges")
	return scan
}

func projectRepeatedEvidence(meta map[string]any) *RepeatedEvidenceDisposition {
	if len(meta) == 0 {
		return nil
	}
	disposition, ok := metaString(meta, "evidence_disposition")
	if !ok {
		if suppressed, _ := metaBool(meta, "evidence_suppressed", "loop_guard_suppressed"); suppressed {
			disposition = "recovery_failed"
			ok = true
		}
	}
	if !ok {
		return nil
	}
	out := &RepeatedEvidenceDisposition{Disposition: boundedIdentifier(disposition)}
	if fingerprint, ok := metaString(meta, "evidence_fingerprint"); ok && strings.HasPrefix(fingerprint, "sha256:") {
		out.Fingerprint = boundedIdentifier(fingerprint)
	}
	out.PriorGeneration = metaIntPointer(meta, "evidence_prior_generation")
	out.RepeatCount = metaIntPointer(meta, "evidence_repeat_count", "evidence_executed")
	return out
}

func safeEstimate(estimate func(string) int, text string) (value int, ok bool) {
	defer func() {
		if recover() != nil {
			value, ok = 0, false
		}
	}()
	value = estimate(text)
	return value, value >= 0
}

func canonicalToolFamily(tool string) string {
	switch strings.ToLower(strings.TrimSpace(tool)) {
	case "grep_files":
		return "grep"
	case "read_file":
		return "read"
	case "ls", "list_dir":
		return "list"
	case "bashbash", "shell", "shell_command", "exec_command":
		return "bash"
	default:
		return boundedIdentifier(strings.ToLower(strings.TrimSpace(tool)))
	}
}

func artifactManifest(value any) (sdkartifact.Manifest, bool) {
	switch typed := value.(type) {
	case sdkartifact.Manifest:
		return typed.Clone(), true
	case *sdkartifact.Manifest:
		if typed != nil {
			return typed.Clone(), true
		}
	}
	encoded, err := json.Marshal(value)
	if err != nil || len(encoded) == 0 || string(encoded) == "null" {
		return sdkartifact.Manifest{}, false
	}
	var manifest sdkartifact.Manifest
	if json.Unmarshal(encoded, &manifest) != nil || strings.TrimSpace(manifest.ObjectRef) == "" {
		return sdkartifact.Manifest{}, false
	}
	return manifest, true
}

func metaIntPointer(meta map[string]any, keys ...string) *int64 {
	value, ok := metaInt64(meta, keys...)
	if !ok {
		return nil
	}
	return &value
}

func metaInt64(meta map[string]any, keys ...string) (int64, bool) {
	for _, key := range keys {
		value, ok := meta[key]
		if !ok {
			continue
		}
		switch typed := value.(type) {
		case int:
			return int64(typed), true
		case int8:
			return int64(typed), true
		case int16:
			return int64(typed), true
		case int32:
			return int64(typed), true
		case int64:
			return typed, true
		case uint:
			return int64(typed), uint64(typed) <= uint64(^uint64(0)>>1)
		case uint64:
			if typed <= uint64(^uint64(0)>>1) {
				return int64(typed), true
			}
		case float64:
			return int64(typed), typed >= 0 && typed == float64(int64(typed))
		case json.Number:
			parsed, err := typed.Int64()
			return parsed, err == nil
		}
	}
	return 0, false
}

func metaBool(meta map[string]any, keys ...string) (bool, bool) {
	for _, key := range keys {
		if value, ok := meta[key].(bool); ok {
			return value, true
		}
	}
	return false, false
}

func metaString(meta map[string]any, keys ...string) (string, bool) {
	for _, key := range keys {
		if value, ok := meta[key].(string); ok && strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value), true
		}
	}
	return "", false
}

func metaNonEmpty(meta map[string]any, key string) bool {
	value, ok := meta[key]
	if !ok || value == nil {
		return false
	}
	if text, ok := value.(string); ok {
		return strings.TrimSpace(text) != ""
	}
	return !reflect.ValueOf(value).IsZero()
}

func metaHasAny(meta map[string]any, keys ...string) bool {
	for _, key := range keys {
		if _, ok := meta[key]; ok {
			return true
		}
	}
	return false
}

func boundedUniqueStrings(values []string, max int) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, minInt(len(values), max))
	for _, value := range values {
		value = boundedIdentifier(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
		if len(out) >= max {
			break
		}
	}
	sort.Strings(out)
	return out
}

func boundedObjectRef(value string) string {
	value = strings.TrimSpace(value)
	if len(value) > MaxObjectRefBytes {
		value = value[:MaxObjectRefBytes]
	}
	return strings.ToValidUTF8(value, "?")
}

func intPointer(value int) *int64 {
	converted := int64(value)
	return &converted
}

func cloneIntPointer(value *int) *int64 {
	if value == nil {
		return nil
	}
	converted := int64(*value)
	return &converted
}

func boolPointer(value bool) *bool { return &value }

func boolInt(value bool) int {
	if value {
		return 1
	}
	return 0
}

func minInt(left, right int) int {
	if left < right {
		return left
	}
	return right
}

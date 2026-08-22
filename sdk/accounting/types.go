package accounting

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

const (
	SchemaVersion          = 1
	MaxPayloadBytes        = 12 * 1024
	MaxIdentifierBytes     = 128
	MaxObjectRefBytes      = 512
	MaxLayers              = 8
	EventKindToolResult    = "tool_result"
	EventKindProviderUsage = "provider_usage"
	EventKindCompaction    = "compaction"
	StatusSuccess          = "success"
	StatusError            = "error"
	StatusPartial          = "partial"
	StatusUnknown          = "unknown"
	LayerProducer          = "producer"
	LayerWrapper           = "wrapper"
	LayerAgentBoundary     = "agent_boundary"
	LayerCompaction        = "compaction"
)

var policyHashPattern = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)

// Estimator declares the identity of a local token estimator. EstimateTokens
// is deliberately excluded from JSON and is trusted only when the complete
// identity is valid.
type Estimator struct {
	Name           string           `json:"name"`
	Version        string           `json:"version"`
	PolicyHash     string           `json:"policy_hash"`
	EstimateTokens func(string) int `json:"-"`
}

type EstimatorIdentity struct {
	Name       string `json:"name"`
	Version    string `json:"version"`
	PolicyHash string `json:"policy_hash"`
}

type Measurements struct {
	OriginalBytes     *int64 `json:"original_bytes,omitempty"`
	VisibleBytes      *int64 `json:"visible_bytes,omitempty"`
	OriginalTokens    *int64 `json:"original_tokens,omitempty"`
	VisibleTokens     *int64 `json:"visible_tokens,omitempty"`
	MeasurementSource string `json:"measurement_source,omitempty"`
}

type LayerDisposition struct {
	Layer         string `json:"layer"`
	OriginalBytes *int64 `json:"original_bytes,omitempty"`
	VisibleBytes  *int64 `json:"visible_bytes,omitempty"`
	LimitBytes    *int64 `json:"limit_bytes,omitempty"`
	Truncated     *bool  `json:"truncated,omitempty"`
	Complete      *bool  `json:"complete,omitempty"`
	Reason        string `json:"reason,omitempty"`
}

type ArtifactDisposition struct {
	ObjectRef           string `json:"object_ref,omitempty"`
	ObjectKind          string `json:"object_kind,omitempty"`
	Complete            *bool  `json:"complete,omitempty"`
	Recoverable         *bool  `json:"recoverable,omitempty"`
	RetentionClass      string `json:"retention_class,omitempty"`
	ExpiresAt           string `json:"expires_at,omitempty"`
	HashDisposition     string `json:"hash_disposition,omitempty"`
	RecoveryDisposition string `json:"recovery_disposition,omitempty"`
	LegacyPathPresent   bool   `json:"legacy_path_present,omitempty"`
}

type ScanDisposition struct {
	Phase                string `json:"phase,omitempty"`
	ScanComplete         *bool  `json:"scan_complete,omitempty"`
	ReturnTruncated      *bool  `json:"return_truncated,omitempty"`
	MatchesTotalKnown    *bool  `json:"matches_total_known,omitempty"`
	WalkEntriesSeen      *int64 `json:"walk_entries_seen,omitempty"`
	EligibleCandidates   *int64 `json:"eligible_candidates,omitempty"`
	FilesOpened          *int64 `json:"files_opened,omitempty"`
	FilesMatched         *int64 `json:"files_matched,omitempty"`
	MatchesReturned      *int64 `json:"matches_returned,omitempty"`
	PageIndex            *int64 `json:"page_index,omitempty"`
	BudgetReason         string `json:"budget_reason,omitempty"`
	ContinuationPresent  bool   `json:"continuation_present,omitempty"`
	ContinuationConsumed bool   `json:"continuation_consumed,omitempty"`
	SnapshotInvalidated  bool   `json:"snapshot_invalidated,omitempty"`
	InFileRangePresent   bool   `json:"in_file_range_present,omitempty"`
}

type UsageDisposition struct {
	EffectivePromptTokens *int64 `json:"effective_prompt_tokens,omitempty"`
	CompletionTokens      *int64 `json:"completion_tokens,omitempty"`
	TotalTokens           *int64 `json:"total_tokens,omitempty"`
	ProviderPromptTokens  *int64 `json:"provider_prompt_tokens,omitempty"`
	ProviderTotalTokens   *int64 `json:"provider_total_tokens,omitempty"`
	PromptCachedTokens    *int64 `json:"prompt_cached_tokens,omitempty"`
	CacheCreationTokens   *int64 `json:"prompt_cache_creation_tokens,omitempty"`
	PromptImageTokens     *int64 `json:"prompt_image_tokens,omitempty"`
	PromptUncachedTokens  *int64 `json:"prompt_uncached_tokens,omitempty"`
	PromptTokensValid     bool   `json:"prompt_tokens_valid"`
	PromptTokensSource    string `json:"prompt_tokens_source,omitempty"`
	PromptTokensSemantics string `json:"prompt_tokens_semantics,omitempty"`
	SummationPolicy       string `json:"summation_policy"`
}

type CompactionDisposition struct {
	Compacted          bool     `json:"compacted"`
	Trigger            string   `json:"trigger,omitempty"`
	Watermark          string   `json:"watermark,omitempty"`
	OriginalTokens     *int64   `json:"original_tokens,omitempty"`
	NewTokens          *int64   `json:"new_tokens,omitempty"`
	TokenCountSource   string   `json:"token_count_source,omitempty"`
	TiersApplied       []string `json:"tiers_applied,omitempty"`
	CheckpointID       string   `json:"checkpoint_id,omitempty"`
	CheckpointMessages *int64   `json:"checkpoint_messages,omitempty"`
	GenerationDelta    int      `json:"generation_delta"`
	WarningCount       int      `json:"warning_count,omitempty"`
	SummaryPresent     bool     `json:"summary_present,omitempty"`
}

type RepeatedEvidenceDisposition struct {
	Disposition     string `json:"disposition,omitempty"`
	Fingerprint     string `json:"fingerprint,omitempty"`
	PriorGeneration *int64 `json:"prior_generation,omitempty"`
	RepeatCount     *int64 `json:"repeat_count,omitempty"`
}

type Payload struct {
	SchemaVersion    int                          `json:"schema_version"`
	EventKind        string                       `json:"event_kind"`
	Status           string                       `json:"status"`
	ToolKind         string                       `json:"tool_kind,omitempty"`
	Measurements     Measurements                 `json:"measurements,omitempty"`
	Estimator        *EstimatorIdentity           `json:"estimator,omitempty"`
	Layers           []LayerDisposition           `json:"layers,omitempty"`
	Artifact         *ArtifactDisposition         `json:"artifact,omitempty"`
	Scan             *ScanDisposition             `json:"scan,omitempty"`
	Usage            *UsageDisposition            `json:"usage,omitempty"`
	Compaction       *CompactionDisposition       `json:"compaction,omitempty"`
	RepeatedEvidence *RepeatedEvidenceDisposition `json:"repeated_evidence,omitempty"`
}

func (p Payload) Validate() error {
	if p.SchemaVersion != SchemaVersion {
		return fmt.Errorf("accounting: unsupported schema_version %d", p.SchemaVersion)
	}
	switch p.EventKind {
	case EventKindToolResult, EventKindProviderUsage, EventKindCompaction:
	default:
		return fmt.Errorf("accounting: unsupported event_kind %q", p.EventKind)
	}
	if !validIdentifier(p.EventKind, MaxIdentifierBytes) || !validIdentifier(p.Status, MaxIdentifierBytes) {
		return fmt.Errorf("accounting: event kind or status exceeds identifier bound")
	}
	if p.ToolKind != "" && !validIdentifier(p.ToolKind, MaxIdentifierBytes) {
		return fmt.Errorf("accounting: tool_kind exceeds identifier bound")
	}
	if len(p.Layers) > MaxLayers {
		return fmt.Errorf("accounting: layer count %d exceeds %d", len(p.Layers), MaxLayers)
	}
	for _, layer := range p.Layers {
		if !validIdentifier(layer.Layer, MaxIdentifierBytes) || (layer.Reason != "" && !validIdentifier(layer.Reason, MaxIdentifierBytes)) {
			return fmt.Errorf("accounting: invalid layer identifier")
		}
	}
	if p.Estimator != nil {
		if !validIdentifier(p.Estimator.Name, MaxIdentifierBytes) || !validIdentifier(p.Estimator.Version, MaxIdentifierBytes) || !policyHashPattern.MatchString(p.Estimator.PolicyHash) {
			return fmt.Errorf("accounting: invalid estimator identity")
		}
	}
	if p.Artifact != nil && len(p.Artifact.ObjectRef) > MaxObjectRefBytes {
		return fmt.Errorf("accounting: object_ref exceeds %d bytes", MaxObjectRefBytes)
	}
	if p.Artifact != nil && p.Artifact.RecoveryDisposition != "" && !validArtifactRecoveryDisposition(p.Artifact.RecoveryDisposition) {
		return fmt.Errorf("accounting: invalid artifact recovery disposition %q", p.Artifact.RecoveryDisposition)
	}
	return nil
}

func validArtifactRecoveryDisposition(value string) bool {
	switch value {
	case "hit", "miss", "expired", "corrupt", "unknown":
		return true
	default:
		return false
	}
}

func MarshalBounded(payload Payload) ([]byte, error) {
	if err := payload.Validate(); err != nil {
		return nil, err
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("accounting: encode payload: %w", err)
	}
	if len(encoded) > MaxPayloadBytes {
		return nil, fmt.Errorf("accounting: payload exceeds %d bytes (got %d)", MaxPayloadBytes, len(encoded))
	}
	return encoded, nil
}

func (e Estimator) identity() (*EstimatorIdentity, bool) {
	name := boundedIdentifier(e.Name)
	version := boundedIdentifier(e.Version)
	hash := strings.TrimSpace(strings.ToLower(e.PolicyHash))
	if name == "" || version == "" || !policyHashPattern.MatchString(hash) || e.EstimateTokens == nil {
		return nil, false
	}
	return &EstimatorIdentity{Name: name, Version: version, PolicyHash: hash}, true
}

func validIdentifier(value string, max int) bool {
	value = strings.TrimSpace(value)
	return value != "" && len(value) <= max
}

func boundedIdentifier(value string) string {
	value = strings.TrimSpace(value)
	if len(value) > MaxIdentifierBytes {
		value = value[:MaxIdentifierBytes]
	}
	return strings.ToValidUTF8(value, "?")
}

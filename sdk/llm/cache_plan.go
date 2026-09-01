package llm

// CachePlanSchemaVersion identifies the in-memory request-local plan model.
// It is not a persistence or Provider wire schema.
const CachePlanSchemaVersion = 1

type CacheTargetKind string

const (
	CacheAfterMessageBlock   CacheTargetKind = "after_message_block"
	CacheAfterMessage        CacheTargetKind = "after_message"
	CacheAfterToolDefinition CacheTargetKind = "after_tool_definition"
)

type CacheDirectivePolicy string

const (
	CacheBestEffort CacheDirectivePolicy = "best_effort"
	CacheRequired   CacheDirectivePolicy = "required"
)

type CacheTTL string

const (
	CacheTTLProviderDefault CacheTTL = ""
	CacheTTL5Minutes        CacheTTL = "5m"
	CacheTTL1Hour           CacheTTL = "1h"
)

// CacheTarget identifies an object in a normalized, already-materialized
// request. Only fields selected by Kind are meaningful. Fingerprints bind the
// intent without retaining Prompt, Tool Result, or Tool Definition text.
type CacheTarget struct {
	Kind CacheTargetKind

	MessageIndex int
	BlockOrdinal int
	ToolIndex    int

	ExpectedObjectFingerprint string
}

type CacheDirective struct {
	Target CacheTarget
	Policy CacheDirectivePolicy
	TTL    CacheTTL
}

// CachePlan is request-local optimization intent. It must not be written into
// conversation history or treated as proof that a Provider cache was hit.
// Provider adapters do not consume this type until explicit validation and
// capability-gated mapping are added.
type CachePlan struct {
	SchemaVersion      int
	RequestFingerprint string
	Directives         []CacheDirective
}

// CloneCachePlan returns an owned copy while preserving nil versus non-nil
// empty directive slices.
func CloneCachePlan(plan *CachePlan) *CachePlan {
	if plan == nil {
		return nil
	}
	cloned := *plan
	if plan.Directives != nil {
		cloned.Directives = make([]CacheDirective, len(plan.Directives))
		copy(cloned.Directives, plan.Directives)
	}
	return &cloned
}

// PromptCacheCapabilities describes explicit request controls implemented by
// one concrete Provider client. Callers must not infer it from a provider name.
type PromptCacheCapabilities struct {
	ExplicitMessageBoundary bool
	ExplicitContentBlock    bool
	ExplicitToolDefinition  bool
	SupportedTTLs           []CacheTTL
	MaxBreakpoints          int
	UsageTelemetry          bool
}

// Clone returns an owned capability snapshot.
func (capabilities PromptCacheCapabilities) Clone() PromptCacheCapabilities {
	cloned := capabilities
	if capabilities.SupportedTTLs != nil {
		cloned.SupportedTTLs = make([]CacheTTL, len(capabilities.SupportedTTLs))
		copy(cloned.SupportedTTLs, capabilities.SupportedTTLs)
	}
	return cloned
}

// PromptCacheCapabilityProvider is an optional interface implemented by a
// concrete client only when it can report explicit cache controls accurately.
type PromptCacheCapabilityProvider interface {
	PromptCacheCapabilities() PromptCacheCapabilities
}

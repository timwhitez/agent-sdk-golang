package llm

import (
	"fmt"
	"reflect"
	"strings"
)

// CacheTargetDescriptor locates a logical boundary, not a Provider wire index.
// Source is text, content_block, tool_call, tool_result, or tool_definition.
// SourceIndex is the
// original Blocks/ToolCalls index, or -1 for Text and the outer tool result.
// It contains no content, names, IDs, or content fingerprints.
type CacheTargetDescriptor struct {
	Target      CacheTarget
	Source      string
	SourceIndex int
}

// CacheTargetView owns one materialized logical request. Retain this view from
// plan construction through validation; rebuilding it from a changed request
// cannot establish that an older plan still names the intended objects.
// It is not a model snapshot, persistence identity, or Provider capability.
// Read-only methods may run concurrently; callers must own their input graphs.
type CacheTargetView struct {
	request *InvokeRequest
	targets []CacheTargetDescriptor
}

// CachePlanValidationError exposes only a fixed reason and directive ordinal.
// DirectiveIndex is -1 for a request/plan-wide failure. There is deliberately
// no wrapped clone error, request content, or fingerprint in diagnostics.
type CachePlanValidationError struct {
	Reason         string
	DirectiveIndex int
}

func (e *CachePlanValidationError) Error() string {
	return fmt.Sprintf("cache plan: %s (directive %d)", e.Reason, e.DirectiveIndex)
}

func cachePlanError(reason string, index int) error {
	return &CachePlanValidationError{Reason: reason, DirectiveIndex: index}
}

// NewCacheTargetView clones using the ordinary request ownership path. It does
// not call a Provider, alter request/history, or generate a content hash.
// Plan metadata is excluded from request identity. Exact Go-value equality is
// intentionally conservative (including nil/empty and non-wire options).
func NewCacheTargetView(request InvokeRequest) (*CacheTargetView, error) {
	request.CachePlan = nil
	owned, err := CloneInvokeRequest(request)
	if err != nil {
		return nil, cachePlanError("uncloneable_request", -1)
	}
	return &CacheTargetView{request: &owned, targets: CacheTargets(owned)}, nil
}

// CacheTargets returns owned, content-free logical target metadata for a request.
// It is the same projection used by CacheTargetView, not a new binding or proof
// that a stale plan is valid. Callers must own inputs while this reads them.
func CacheTargets(request InvokeRequest) []CacheTargetDescriptor {
	var targets []CacheTargetDescriptor
	for i, message := range request.Messages {
		if message.Role != RoleSystem && message.Role != RoleUser && message.Role != RoleAssistant && message.Role != RoleTool {
			continue
		}
		ordinal := 0
		start := len(targets)
		appendBlock := func(source string, index int) {
			targets = append(targets, CacheTargetDescriptor{
				Target: CacheTarget{Kind: CacheAfterMessageBlock, MessageIndex: i, BlockOrdinal: ordinal},
				Source: source, SourceIndex: index,
			})
			ordinal++
		}
		if message.Role == RoleTool {
			// A result is one logical outer block, even if its Provider content
			// is flattened, nested, empty, or merged with adjacent results.
			appendBlock("tool_result", -1)
		} else {
			if strings.TrimSpace(message.Content.Text) != "" {
				appendBlock("text", -1)
			}
			for j, block := range message.Content.Blocks {
				if IsProviderStateBlock(block) {
					continue
				}
				// Only structurally visible supported shapes get targets. Empty,
				// unknown, thinking and redacted-thinking blocks are not targets.
				switch block.Type {
				case "text":
					if strings.TrimSpace(block.Text) == "" {
						continue
					}
				case "image_url":
					if block.ImageURL == nil || strings.TrimSpace(block.ImageURL.URL) == "" {
						continue
					}
				case "document":
					if block.Source == nil || block.Source.Data == "" || block.Source.MediaType == "" {
						continue
					}
				default:
					continue
				}
				appendBlock("content_block", j)
			}
			if message.Role == RoleAssistant {
				for j := range message.ToolCalls {
					appendBlock("tool_call", j)
				}
			}
		}
		if len(targets) > start {
			last := targets[len(targets)-1]
			// This is a logical message boundary. Provider mapping must still
			// reject unsupported/hidden trailing wire blocks instead of guessing.
			last.Target = CacheTarget{Kind: CacheAfterMessage, MessageIndex: i}
			targets = append(targets, last)
		}
	}
	for i := range request.Tools {
		targets = append(targets, CacheTargetDescriptor{
			Target: CacheTarget{Kind: CacheAfterToolDefinition, ToolIndex: i},
			Source: "tool_definition", SourceIndex: i,
		})
	}
	return targets
}

// Targets returns an owned list. BlockOrdinal indexes visible logical blocks:
// nonempty Text, addressable explicit Blocks, then assistant ToolCalls. Opaque
// and hidden blocks do not consume ordinals. Tool messages expose only their
// outer result, never nested content. Message boundaries alias the final logical
// target for duplicate detection; this is not permission to drop wire blocks.
func (view *CacheTargetView) Targets() []CacheTargetDescriptor {
	if view == nil {
		return nil
	}
	return append([]CacheTargetDescriptor(nil), view.targets...)
}

// Validate checks a plan against this retained view and the current request.
// This is a pure check; built-in admission reuses it through AdmitCachePlan.
// Nil plans require no check. Fingerprint fields from the experimental API are
// unsupported here; the retained owned snapshot supplies the binding instead.
// Both policies require valid structure; capability/TTL support and best-effort
// downgrade are handled by Decide and the Provider admission layer.
func (view *CacheTargetView) Validate(request InvokeRequest, plan *CachePlan) error {
	if plan == nil {
		return nil
	}
	if view == nil || view.request == nil {
		return cachePlanError("missing_view", -1)
	}
	request.CachePlan = nil
	if !reflect.DeepEqual(*view.request, request) {
		return cachePlanError("stale_request", -1)
	}
	if plan.SchemaVersion != CachePlanSchemaVersion {
		return cachePlanError("unsupported_schema", -1)
	}
	if plan.RequestFingerprint != "" {
		return cachePlanError("unsupported_fingerprint", -1)
	}
	// ponytail: linear target lookup; index only if measured large plans need it.
	seen := make(map[CacheTargetDescriptor]bool, len(plan.Directives))
	for i, directive := range plan.Directives {
		if directive.Policy != CacheBestEffort && directive.Policy != CacheRequired {
			return cachePlanError("invalid_policy", i)
		}
		if directive.TTL != CacheTTLProviderDefault && directive.TTL != CacheTTL5Minutes && directive.TTL != CacheTTL1Hour {
			return cachePlanError("invalid_ttl", i)
		}
		if directive.Target.ExpectedObjectFingerprint != "" {
			return cachePlanError("unsupported_fingerprint", i)
		}
		target := directive.Target
		switch target.Kind {
		case CacheAfterMessageBlock:
			target.ToolIndex = 0
		case CacheAfterMessage:
			target.ToolIndex, target.BlockOrdinal = 0, 0
		case CacheAfterToolDefinition:
			target.MessageIndex, target.BlockOrdinal = 0, 0
		default:
			return cachePlanError("invalid_target", i)
		}
		found := false
		for _, candidate := range view.targets {
			if candidate.Target != target {
				continue
			}
			// Canonicalize message/final-block aliases by their original source.
			candidate.Target.Kind = ""
			candidate.Target.BlockOrdinal = 0
			if seen[candidate] {
				return cachePlanError("duplicate_boundary", i)
			}
			seen[candidate] = true
			found = true
			break
		}
		if !found {
			return cachePlanError("invalid_target", i)
		}
	}
	return nil
}

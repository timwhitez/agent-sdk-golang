package llm

import "reflect"

// CacheDirectiveDecision is bounded, content-free metadata for one original
// directive. Accepted means eligible under reported capabilities, not mapped,
// sent, or cached. Reason is accepted, unsupported_target, unsupported_ttl,
// unmappable_target, breakpoint_limit, ttl_order_conflict, or duplicate_boundary.
// No content, names, IDs, hashes or
// Provider strings are kept.
type CacheDirectiveDecision struct {
	DirectiveIndex int
	Accepted       bool
	Reason         string
}

// CachePlanDecision owns a filtered plan and decisions in original order.
// It is not send authorization: retain the original view and revalidate at the
// actual owned-request/serializer boundary. A mutable model handle or capability
// report is not a concrete-model snapshot or proof of exact wire mapping.
type CachePlanDecision struct {
	Plan         *CachePlan
	Directives   []CacheDirectiveDecision
	Capabilities PromptCacheCapabilities
}

// Decide validates against the retained view, queries the actual model's
// optional capability interface once, and allocates logical breakpoints.
// It never invokes the model, changes request/history, or guesses from names.
// Built-in admission reuses this helper; it can also be called without sending.
//
// Invalid/stale plans fail for both policies. Required directives reserve
// capacity first; remaining best-effort directives are kept in input order when
// compatible with required/previously selected wire TTL boundaries.
// Unsupported required directives or required overflow reject the entire plan
// with a fixed CachePlanValidationError, never a partially accepted result.
func (view *CacheTargetView) Decide(request InvokeRequest, plan *CachePlan, model ChatModel) (*CachePlanDecision, error) {
	// Capability callbacks must not be able to rewrite the validated plan via
	// aliases to the caller's input graph.
	plan = CloneCachePlan(plan)
	if err := view.Validate(request, plan); err != nil {
		return nil, err
	}
	if plan == nil {
		return &CachePlanDecision{}, nil
	}
	result := &CachePlanDecision{Plan: plan}
	if len(plan.Directives) == 0 {
		return result, nil
	}
	snapshot := view.request
	value := reflect.ValueOf(model)
	if !value.IsValid() || (value.Kind() == reflect.Pointer && value.IsNil()) {
		return nil, cachePlanError("missing_model", -1)
	}
	if provider, ok := model.(PromptCacheCapabilityProvider); ok {
		result.Capabilities = provider.PromptCacheCapabilities().Clone()
	}
	caps := result.Capabilities
	if caps.MaxBreakpoints < 0 {
		return nil, cachePlanError("invalid_capabilities", -1)
	}
	for _, ttl := range caps.SupportedTTLs {
		if ttl != CacheTTLProviderDefault && ttl != CacheTTL5Minutes && ttl != CacheTTL1Hour {
			return nil, cachePlanError("invalid_capabilities", -1)
		}
	}
	var eligible []bool
	if mapper, ok := model.(PromptCacheTargetEligibilityProvider); ok {
		// Do not expose the retained snapshot or caller graph to a provider hook.
		mappingRequest, err := CloneInvokeRequest(*snapshot)
		if err != nil {
			return nil, cachePlanError("uncloneable_request", -1)
		}
		targets := make([]CacheTarget, len(plan.Directives))
		for i, directive := range plan.Directives {
			targets[i] = directive.Target
		}
		mapping := mapper.PromptCacheTargetEligibility(mappingRequest, targets)
		if len(mapping) != len(plan.Directives) {
			return nil, cachePlanError("invalid_capabilities", -1)
		}
		eligible = append([]bool(nil), mapping...)
	}
	var order []int
	for _, directive := range plan.Directives {
		if directive.TTL != CacheTTL1Hour {
			continue
		}
		if provider, ok := model.(PromptCacheTTLOrderProvider); ok {
			copy, err := CloneInvokeRequest(*snapshot)
			if err != nil {
				return nil, cachePlanError("uncloneable_request", -1)
			}
			targets := make([]CacheTarget, len(plan.Directives))
			for i, d := range plan.Directives {
				targets[i] = d.Target
			}
			positions := provider.PromptCacheTTLOrder(copy, targets)
			if len(positions) != len(targets) {
				return nil, cachePlanError("invalid_capabilities", -1)
			}
			order = append([]int(nil), positions...)
		}
		break
	}
	result.Directives = make([]CacheDirectiveDecision, len(plan.Directives))
	required := 0
	for i, directive := range plan.Directives {
		reason := "accepted"
		supported := false
		switch directive.Target.Kind {
		case CacheAfterMessage:
			supported = caps.ExplicitMessageBoundary
		case CacheAfterMessageBlock:
			supported = caps.ExplicitContentBlock
		case CacheAfterToolDefinition:
			supported = caps.ExplicitToolDefinition
		}
		if !supported {
			reason = "unsupported_target"
		} else if directive.TTL != CacheTTLProviderDefault {
			ttlSupported := false
			for _, ttl := range caps.SupportedTTLs {
				if ttl == directive.TTL {
					ttlSupported = true
					break
				}
			}
			if !ttlSupported {
				reason = "unsupported_ttl"
			}
		}
		if reason == "accepted" && eligible != nil && !eligible[i] {
			reason = "unmappable_target"
		}
		if reason == "accepted" && order != nil && order[i] < 0 {
			reason = "unmappable_target"
		}
		if directive.Policy == CacheRequired {
			if reason != "accepted" {
				return nil, cachePlanError(reason, i)
			}
			required++
			if required > caps.MaxBreakpoints {
				return nil, cachePlanError("breakpoint_limit", i)
			}
		}
		result.Directives[i] = CacheDirectiveDecision{DirectiveIndex: i, Reason: reason}
	}
	remaining := caps.MaxBreakpoints - required
	selected := make([]int, 0, required)
	conflict := func(index int) string {
		if order == nil {
			return ""
		}
		for _, previous := range selected {
			if order[index] == order[previous] {
				return "duplicate_boundary"
			}
			long, previousLong := plan.Directives[index].TTL == CacheTTL1Hour, plan.Directives[previous].TTL == CacheTTL1Hour
			if long != previousLong && ((long && order[index] > order[previous]) || (!long && order[index] < order[previous])) {
				return "ttl_order_conflict"
			}
		}
		return ""
	}
	for i, d := range plan.Directives {
		if d.Policy != CacheRequired {
			continue
		}
		if reason := conflict(i); reason != "" {
			return nil, cachePlanError(reason, i)
		}
		selected = append(selected, i)
	}
	// Decide optional entries before compacting the slice: conflict lookup must
	// keep the original directive indexes and TTLs throughout selection.
	for i, d := range plan.Directives {
		decision := &result.Directives[i]
		if decision.Reason != "accepted" {
			continue
		}
		if d.Policy == CacheBestEffort {
			if reason := conflict(i); reason != "" {
				decision.Reason = reason
				continue
			}
			if remaining == 0 {
				decision.Reason = "breakpoint_limit"
				continue
			}
			remaining--
			selected = append(selected, i)
		}
		decision.Accepted = true
	}
	// Filter the owned slice, preserving nil/empty ownership and original order.
	directives := result.Plan.Directives
	result.Plan.Directives = result.Plan.Directives[:0]
	for i, directive := range directives {
		if result.Directives[i].Accepted {
			result.Plan.Directives = append(result.Plan.Directives, directive)
		}
	}
	return result, nil
}

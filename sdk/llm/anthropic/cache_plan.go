package anthropic

import "github.com/timwhitez/agent-sdk-golang/sdk/llm"

// applyToolCachePlan only touches newly serialized payload objects. Preserve
// block shape/order/text while removing legacy breakpoints; do not rewrite
// request/history Cache flags or reserialize system blocks as a plain string.
func applyToolCachePlan(plan *llm.CachePlan, system any, messages []messageParam, tools []toolParam) error {
	if len(plan.Directives) > 4 {
		return &llm.CachePlanValidationError{Reason: "breakpoint_limit", DirectiveIndex: -1}
	}
	for i, directive := range plan.Directives {
		if directive.Target.Kind != llm.CacheAfterToolDefinition || directive.Target.ToolIndex < 0 || directive.Target.ToolIndex >= len(tools) {
			return &llm.CachePlanValidationError{Reason: "unmappable_target", DirectiveIndex: i}
		}
		if directive.TTL != llm.CacheTTLProviderDefault && directive.TTL != llm.CacheTTL5Minutes {
			return &llm.CachePlanValidationError{Reason: "unsupported_ttl", DirectiveIndex: i}
		}
	}
	if blocks, ok := system.([]contentBlockParam); ok {
		for i := range blocks {
			blocks[i].CacheCtrl = nil
		}
	}
	for i := range messages {
		for j := range messages[i].Content {
			messages[i].Content[j].CacheCtrl = nil
		}
	}
	for i := range tools {
		tools[i].CacheCtrl = nil
	}
	for _, directive := range plan.Directives {
		tools[directive.Target.ToolIndex].CacheCtrl = &cacheControl{Type: "ephemeral", TTL: directive.TTL}
	}
	return nil
}

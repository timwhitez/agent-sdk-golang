package anthropic

import (
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type messageCacheLocation struct {
	system, plain, eligible bool
	message, block          int
}

func cacheableContentBoundary(source llm.ContentBlock, wire contentBlockParam) bool {
	switch source.Type {
	case "text":
		return wire.Type == "text" && strings.TrimSpace(wire.Text) != ""
	case "image_url":
		return wire.Type == "image" && wire.Source != nil
	default:
		// Never treat a placeholder for unsupported content as its source.
		return false
	}
}

// PromptCacheTargetEligibility uses the same traversal as the actual builder,
// with warnings disabled. Tool-only plans do not need message serialization.
func (c *Client) PromptCacheTargetEligibility(request llm.InvokeRequest, targets []llm.CacheTarget) []bool {
	eligible := make([]bool, len(targets))
	var locations []messageCacheLocation
	for _, target := range targets {
		if target.Kind == llm.CacheAfterMessage {
			locations = make([]messageCacheLocation, len(request.Messages))
			break
		}
	}
	if locations != nil {
		if _, _, err := serializeMessagesWithLocations(request.Messages, nil, locations); err != nil {
			locations = nil
		}
	}
	for i, target := range targets {
		switch target.Kind {
		case llm.CacheAfterToolDefinition:
			eligible[i] = target.ToolIndex >= 0 && target.ToolIndex < len(request.Tools)
		case llm.CacheAfterMessage:
			eligible[i] = target.MessageIndex >= 0 && target.MessageIndex < len(locations) && locations[target.MessageIndex].eligible
		}
	}
	return eligible
}

// applyCachePlan only touches newly serialized objects, after all destinations
// pass validation. A collapsed system string can be wrapped whole, not split
// to invent earlier source boundaries. Existing structured blocks keep shape.
func applyCachePlan(plan *llm.CachePlan, system *any, messages []messageParam, tools []toolParam, locations []messageCacheLocation) error {
	if len(plan.Directives) > 4 {
		return &llm.CachePlanValidationError{Reason: "breakpoint_limit", DirectiveIndex: -1}
	}
	sysBlocks, _ := (*system).([]contentBlockParam)
	var wrapped []contentBlockParam
	destinations := make([]**cacheControl, 0, len(plan.Directives))
	for i, directive := range plan.Directives {
		var destination **cacheControl
		target := directive.Target
		switch target.Kind {
		case llm.CacheAfterToolDefinition:
			if target.ToolIndex >= 0 && target.ToolIndex < len(tools) {
				destination = &tools[target.ToolIndex].CacheCtrl
			}
		case llm.CacheAfterMessage:
			if target.MessageIndex >= 0 && target.MessageIndex < len(locations) {
				location := locations[target.MessageIndex]
				if location.eligible {
					if location.system {
						if location.plain {
							if text, ok := (*system).(string); ok {
								if wrapped == nil {
									wrapped = []contentBlockParam{{Type: "text", Text: text}}
								}
								destination = &wrapped[0].CacheCtrl
							}
						} else if location.block >= 0 && location.block < len(sysBlocks) {
							destination = &sysBlocks[location.block].CacheCtrl
						}
					} else if location.message >= 0 && location.message < len(messages) && location.block >= 0 && location.block < len(messages[location.message].Content) {
						destination = &messages[location.message].Content[location.block].CacheCtrl
					}
				}
			}
		}
		if destination == nil {
			return &llm.CachePlanValidationError{Reason: "unmappable_target", DirectiveIndex: i}
		}
		if directive.TTL != llm.CacheTTLProviderDefault && directive.TTL != llm.CacheTTL5Minutes {
			return &llm.CachePlanValidationError{Reason: "unsupported_ttl", DirectiveIndex: i}
		}
		destinations = append(destinations, destination)
	}
	for i := range sysBlocks {
		sysBlocks[i].CacheCtrl = nil
	}
	for i := range messages {
		for j := range messages[i].Content {
			messages[i].Content[j].CacheCtrl = nil
		}
	}
	for i := range tools {
		tools[i].CacheCtrl = nil
	}
	for i, directive := range plan.Directives {
		*destinations[i] = &cacheControl{Type: "ephemeral", TTL: directive.TTL}
	}
	if wrapped != nil {
		*system = wrapped
	}
	return nil
}

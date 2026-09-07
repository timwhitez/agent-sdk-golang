package agent

import (
	"reflect"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// executionFrame owns one logical request, not the final provider payload.
// For continued tool calls it is the finalizing request's dispatch snapshot;
// the merged arguments may originate from multiple earlier requests.
// Model/Handler values retain runtime handles, not immutable closure state or
// a dynamic model wrapper's concrete target. Nothing here is persisted/emitted.
type executionFrame struct {
	model      llm.ChatModel
	request    llm.InvokeRequest
	exact      map[string]tools.Tool
	normalized map[string]tools.Tool
}

func newExecutionFrame(model llm.ChatModel, request llm.InvokeRequest, exact, normalized map[string]tools.Tool) (*executionFrame, error) {
	owned, err := llm.CloneInvokeRequest(request)
	if err != nil {
		return nil, err
	}
	cloneRegistry := func(source map[string]tools.Tool) (map[string]tools.Tool, error) {
		if source == nil {
			return nil, nil
		}
		result := make(map[string]tools.Tool, len(source))
		for name, tool := range source {
			cloned, err := llm.CloneInvokeRequest(llm.InvokeRequest{Tools: []llm.ToolDefinition{tool.Definition()}})
			if err != nil {
				return nil, err
			}
			tool.Schema = cloned.Tools[0].Parameters
			result[name] = tool
		}
		return result, nil
	}
	ownedExact, err := cloneRegistry(exact)
	if err != nil {
		return nil, err
	}
	ownedNormalized, err := cloneRegistry(normalized)
	if err != nil {
		return nil, err
	}
	return &executionFrame{model: model, request: owned, exact: ownedExact, normalized: ownedNormalized}, nil
}

// validBindings checks structural agreement, not mutable closure identity.
// Hidden tools and the dispatch-time internal invalid fallback remain legal.
func (frame *executionFrame) validBindings() bool {
	for _, definition := range frame.request.Tools {
		tool, ok := frame.exact[definition.Name]
		if !ok || tool.Hidden || !reflect.DeepEqual(definition, tool.Definition()) {
			return false
		}
	}
	for _, normalized := range frame.normalized {
		exact, ok := frame.exact[normalized.Name]
		if !ok || normalized.Hidden != exact.Hidden || normalized.EphemeralKeep != exact.EphemeralKeep ||
			(normalized.Handler == nil) != (exact.Handler == nil) || !reflect.DeepEqual(normalized.Definition(), exact.Definition()) {
			return false
		}
	}
	return true
}

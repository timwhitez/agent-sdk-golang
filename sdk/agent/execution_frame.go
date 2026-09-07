package agent

import (
	"reflect"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// executionFrame shadows one logical request, not the final provider payload.
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

func (a *Agent) observeFrameAdvertisement(frame *executionFrame) {
	if frame == nil {
		return
	}
	for index, definition := range frame.request.Tools {
		tool, ok := frame.exact[definition.Name]
		if !ok || tool.Hidden || !reflect.DeepEqual(definition, tool.Definition()) {
			a.warnf("warning: execution frame shadow mismatch: advertised_tool[%d]", index)
		}
	}
}

func (a *Agent) observeFrameResolution(frame *executionFrame, index int, name string, actual tools.Tool, resolved string, found, alias bool) {
	if frame == nil {
		return
	}
	expected, expectedName, expectedFound, expectedAlias := resolveToolByName(name, frame.exact, frame.normalized)
	if !expectedFound {
		expectedName = "invalid"
		var ok bool
		expected, ok = frame.exact["invalid"]
		if !ok {
			expected = autoInvalidTool()
		}
	}
	// Functions cannot prove closure identity through comparison. Check only
	// resolution and metadata; retain the captured Handler for later authority.
	if expectedName != resolved || expectedFound != found || expectedAlias != alias ||
		expected.Hidden != actual.Hidden || expected.EphemeralKeep != actual.EphemeralKeep ||
		(expected.Handler == nil) != (actual.Handler == nil) || !reflect.DeepEqual(expected.Definition(), actual.Definition()) {
		a.warnf("warning: execution frame shadow mismatch: resolved_tool[%d]", index)
	}
}

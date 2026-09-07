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
// a dynamic model wrapper's concrete target. Only the opaque id is correlated
// in event metadata; request/model/resolver content is not emitted here.
type executionFrame struct {
	id         string
	model      llm.ChatModel
	request    llm.InvokeRequest
	exact      map[string]tools.Tool
	normalized map[string]tools.Tool
}

func newExecutionFrame(id string, model llm.ChatModel, request llm.InvokeRequest, exact, normalized map[string]tools.Tool) (*executionFrame, error) {
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
	return &executionFrame{id: id, model: model, request: owned, exact: ownedExact, normalized: ownedNormalized}, nil
}

// eventCorrelation is copied at the producer, never read from ambient output
// state. The original eventOutput remains the sequence/backpressure owner.
type eventCorrelation struct {
	frameID string
	attempt uint64
}

// frameInvocation is query-driver-local bookkeeping, separate from immutable
// executionFrame state. Only actual ChatModel entries advance the counter.
type frameInvocation struct {
	frameID string
	entries uint64
	current uint64
}

func (s *frameInvocation) resetAttempt() {
	if s != nil {
		s.current = 0
	}
}
func (s *frameInvocation) enter() eventCorrelation {
	if s == nil {
		return eventCorrelation{}
	}
	s.entries++
	s.current = s.entries
	return s.correlation()
}
func (s *frameInvocation) correlation() eventCorrelation {
	if s == nil || s.frameID == "" {
		return eventCorrelation{}
	}
	return eventCorrelation{frameID: s.frameID, attempt: s.current}
}

// A completion retained across retry backoff belongs to the last actual model
// entry even when no invocation is currently active. Use only with that returned
// completion, not to attribute a backoff/cancellation error to the old attempt.
func (s *frameInvocation) completionCorrelation() eventCorrelation {
	if s == nil || s.frameID == "" {
		return eventCorrelation{}
	}
	return eventCorrelation{frameID: s.frameID, attempt: s.entries}
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

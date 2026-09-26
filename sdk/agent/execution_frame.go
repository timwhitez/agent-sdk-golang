package agent

import (
	"fmt"
	"reflect"
	"slices"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
	"github.com/timwhitez/agent-sdk-golang/sdk/tools"
)

// executionFrame owns one logical request, not the final provider payload.
// For continued tool calls it is the finalizing request's dispatch snapshot;
// the merged arguments may originate from multiple earlier requests.
// Handlers retain runtime handles, not immutable closure state. The driver
// captures a model configuration only through explicit FrameModelBinder support;
// unbound wrappers retain legacy behavior. Only the opaque id is correlated
// in event metadata; request/model/resolver content is not emitted here.
type executionFrame struct {
	controlSource string
	// historySource is the Frame whose usage triggered an automatic
	// compaction published into this request's history; empty is unknown.
	historySource string
	// recoverySource is the Frame whose stalled stream made the driver append
	// a recovery reminder into this request's history; empty is unknown.
	recoverySource string
	// steeringSource is the Frame whose execution a real user steering
	// message interrupted or extended before it entered this request's
	// history; empty is unknown or no accepted steering.
	steeringSource string
	// continuationSources are the Frames whose responses produced the tool
	// calls this request answers with not-yet-answered tool results, in
	// build order; empty is unknown or none.
	continuationSources []string
	// hostPublication is the HistoryPublication.Revision whose system
	// messages this request carries; zero is unknown.
	hostPublication uint64
	id              string
	model           llm.ChatModel
	request         llm.InvokeRequest
	exact           map[string]tools.Tool
	normalized      map[string]tools.Tool
}

// executionFrameID names the ordinal-th Frame (one-based) of a Query.
func executionFrameID(queryID string, ordinal int) string {
	return fmt.Sprintf("%s/frame/%d", queryID, ordinal)
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
	// Intervention labels; empty is unreported. An empty stage with a kind
	// means applied; only an observe-only detection sets another stage.
	intervention       string
	interventionStage  string
	interventionResult string
	interventionStrike uint64
	historySource      string
	recoverySource     string
	steeringSource     string
	// continuationSources is shared, read-only; envelopes receive copies.
	continuationSources []string
	controlSource       string
	hostPublication     uint64
	toolBlockID         string
	toolCallOrdinal     uint64
	blockCallCount      uint64
	frameID             string
	attempt             uint64
}

// frameInvocation is query-driver-local bookkeeping, separate from immutable
// executionFrame state. Only actual ChatModel entries advance the counter.
type frameInvocation struct {
	historySource       string
	recoverySource      string
	steeringSource      string
	continuationSources []string
	controlSource       string
	hostPublication     uint64
	frameID             string
	entries             uint64
	current             uint64
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
	return eventCorrelation{frameID: s.frameID, attempt: s.current, controlSource: s.controlSource, historySource: s.historySource, recoverySource: s.recoverySource, steeringSource: s.steeringSource, continuationSources: s.continuationSources, hostPublication: s.hostPublication}
}

// A completion retained across retry backoff belongs to the last actual model
// entry even when no invocation is currently active. Use only with that returned
// completion, not to attribute a backoff/cancellation error to the old attempt.
func (s *frameInvocation) completionCorrelation() eventCorrelation {
	if s == nil || s.frameID == "" {
		return eventCorrelation{}
	}
	return eventCorrelation{frameID: s.frameID, attempt: s.entries, controlSource: s.controlSource, historySource: s.historySource, recoverySource: s.recoverySource, steeringSource: s.steeringSource, continuationSources: s.continuationSources, hostPublication: s.hostPublication}
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

// addFrameSource returns sources (ascending, unique) with ordinal added.
func addFrameSource(sources []int, ordinal int) []int {
	index, found := slices.BinarySearch(sources, ordinal)
	if found {
		return sources
	}
	return slices.Insert(slices.Clone(sources), index, ordinal)
}

// unionFrameSources returns the ascending union of per-call source sets.
func unionFrameSources(perCall [][]int) []int {
	var union []int
	for _, sources := range perCall {
		for _, ordinal := range sources {
			union = addFrameSource(union, ordinal)
		}
	}
	return union
}

// continuationFrameIDs names the source Frames of a continuation, or nil
// (unreported) when they exceed MaxRequestContinuationSources.
func continuationFrameIDs(queryID string, sources []int) []string {
	if len(sources) > MaxRequestContinuationSources {
		return nil
	}
	ids := make([]string, 0, len(sources))
	for _, ordinal := range sources {
		ids = append(ids, executionFrameID(queryID, ordinal))
	}
	return ids
}

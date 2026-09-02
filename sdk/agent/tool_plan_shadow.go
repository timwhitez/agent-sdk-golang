package agent

type toolPlanClass uint8
type toolResolutionState uint8
type toolArgsState uint8

const (
	toolPlanUnknown toolPlanClass = iota
	toolPlanExclusive
)

const (
	toolResolutionExact toolResolutionState = iota
	toolResolutionNormalizedAlias
	toolResolutionUnknownFallback
)

const (
	toolArgsNormalized toolArgsState = iota
	toolArgsInvalid
)

// toolPlanningObservation intentionally excludes names, arguments, schemas,
// handlers, and dependency state.
type toolPlanningObservation struct {
	ordinal    int
	resolution toolResolutionState
	args       toolArgsState
}

type toolCallPlan struct {
	ordinal int
	class   toolPlanClass
}

// shadowToolCallPlan is observe-only. Without an explicit effect contract,
// every reached call remains exclusive and the legacy loop stays authoritative.
func shadowToolCallPlan(observation toolPlanningObservation) toolCallPlan {
	return toolCallPlan{ordinal: observation.ordinal, class: toolPlanExclusive}
}

func (a *Agent) observeToolCallPlan(legacy, shadow toolCallPlan) {
	if legacy == shadow {
		return
	}
	a.warnf(
		"agent: tool planner shadow mismatch: legacy_ordinal=%d shadow_ordinal=%d legacy_class=%s shadow_class=%s",
		legacy.ordinal,
		shadow.ordinal,
		safeToolPlanClass(legacy.class),
		safeToolPlanClass(shadow.class),
	)
}

func safeToolPlanClass(class toolPlanClass) string {
	if class == toolPlanExclusive {
		return "exclusive"
	}
	return "unknown"
}

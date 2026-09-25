package agent

type interventionAction uint8

const (
	interventionActionProceed interventionAction = iota
	interventionActionSuppressTool
)

type interventionDecision struct {
	action         interventionAction
	queueReminder  bool
	downgradeGuard bool
}

type repeatedSignatureObservation struct {
	count              int
	threshold          int
	exhausted          bool
	lastResultRecycled bool
	reminderConfigured bool
	nextStrike         int
	strikeLimit        int
}

func decideRepeatedSignatureIntervention(observation repeatedSignatureObservation) interventionDecision {
	detected := observation.threshold > 0 && observation.count >= observation.threshold
	suppress := detected && (!observation.exhausted || observation.lastResultRecycled)
	action := interventionActionProceed
	if suppress {
		action = interventionActionSuppressTool
	}
	return interventionDecision{
		action:         action,
		queueReminder:  suppress && (observation.exhausted || observation.reminderConfigured),
		downgradeGuard: suppress && !observation.exhausted && observation.strikeLimit > 0 && observation.nextStrike >= observation.strikeLimit,
	}
}

// Intervention lifecycle labels reported on EventEnvelope. Only an accepted
// application is reported: a proposed decision whose history commit never
// happened produces no intervention fields.
const (
	InterventionRepeatedToolSignature = "repeated_tool_signature"
	// InterventionRequireDone is the RequireDone reminder guard; it is
	// reported when its bounded safety fallback accepted a partial answer.
	InterventionRequireDone = "require_done_reminder"
	// InterventionEvidenceProgress suppresses a repeated evidence read that
	// adds no new coverage.
	InterventionEvidenceProgress      = "evidence_progress"
	InterventionStageApplied          = "applied"
	InterventionResultToolSuppressed  = "tool_suppressed"
	InterventionResultReminderQueued  = "reminder_queued"
	InterventionResultGuardDowngraded = "guard_downgraded"
	// InterventionResultSafetyFallback: the reminder budget was spent and the
	// latest answer was accepted as a partial final response.
	InterventionResultSafetyFallback = "safety_fallback_accepted"
	// InterventionStreamIdleRecovery: a provider stream stalled and the
	// driver continued with a recovery reminder instead of ending the turn.
	InterventionStreamIdleRecovery = "stream_idle_recovery"
	// InterventionResultRecoveryReminderAppended: the recovery reminder was
	// appended to history before the event.
	InterventionResultRecoveryReminderAppended = "recovery_reminder_appended"
	// InterventionContextOverflowRecovery: a typed provider context overflow
	// was recovered by compacting history for a new request.
	InterventionContextOverflowRecovery = "context_overflow_recovery"
	// InterventionResultHistoryCompacted: compaction changed history before
	// the event; the next request is a new logical request.
	InterventionResultHistoryCompacted = "history_compacted"
	// InterventionThinkingOnly is the opt-in, observe-only thinking-only
	// detector (Config.ObserveThinkingOnlyResponses). It is only ever reported
	// with InterventionStageDetected and InterventionResultObservedOnly: a
	// detection, never an application. No recovery exists for it.
	InterventionThinkingOnly = "thinking_only"
	// InterventionStageDetected marks a detection that changed nothing; it is
	// not "applied" and does not imply that any decision or action followed.
	InterventionStageDetected = "detected"
	// InterventionResultObservedOnly: the condition was observed and the
	// request, tool choice, thinking controls, model call count, history and
	// opaque provider state were left unchanged.
	InterventionResultObservedOnly = "observed_only"
)

type interventionStage uint8

const (
	interventionProposed interventionStage = iota + 1
	interventionApplied
)

// interventionRecord is the package-local lifecycle of one intervention
// decision. It carries only fixed labels and counters, never signatures,
// arguments or prompt text.
type interventionRecord struct {
	kind            string
	stage           interventionStage
	strike          int
	strikeLimit     int
	reminderQueued  bool
	guardDowngraded bool
}

// withIntervention returns c with the applied intervention labels.
func (c eventCorrelation) withIntervention(result string, strike int) eventCorrelation {
	return c.withInterventionKind(InterventionRepeatedToolSignature, result, strike)
}

// withInterventionDetection marks an event reporting an observe-only
// detection of kind: stage "detected", never "applied", and no strike.
func (c eventCorrelation) withInterventionDetection(kind, result string) eventCorrelation {
	c.intervention = kind
	c.interventionStage = InterventionStageDetected
	c.interventionResult = result
	c.interventionStrike = 0
	return c
}

// withInterventionKind marks an event produced by an applied intervention of
// kind; strike is its one-based applied count in this Query (0 omits it).
func (c eventCorrelation) withInterventionKind(kind, result string, strike int) eventCorrelation {
	c.intervention = kind
	c.interventionResult = result
	if strike > 0 {
		c.interventionStrike = uint64(strike)
	}
	return c
}

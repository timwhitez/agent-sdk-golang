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
	InterventionStageApplied          = "applied"
	InterventionResultToolSuppressed  = "tool_suppressed"
	InterventionResultReminderQueued  = "reminder_queued"
	InterventionResultGuardDowngraded = "guard_downgraded"
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
	c.intervention = InterventionRepeatedToolSignature
	c.interventionResult = result
	if strike > 0 {
		c.interventionStrike = uint64(strike)
	}
	return c
}

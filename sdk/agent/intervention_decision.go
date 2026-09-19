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

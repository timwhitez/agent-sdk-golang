package agent

type interventionKind uint8
type interventionAction uint8

const (
	interventionKindUnknown interventionKind = iota
	interventionKindRepeatedSignature

	interventionActionProceed interventionAction = iota
	interventionActionSuppressTool
)

type interventionDetection struct {
	kind   interventionKind
	active bool
}

type interventionDecision struct {
	detection      interventionDetection
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

func shadowRepeatedSignatureIntervention(observation repeatedSignatureObservation) interventionDecision {
	detected := observation.threshold > 0 && observation.count >= observation.threshold
	suppress := detected && (!observation.exhausted || observation.lastResultRecycled)
	action := interventionActionProceed
	if suppress {
		action = interventionActionSuppressTool
	}
	return interventionDecision{
		detection:      interventionDetection{kind: interventionKindRepeatedSignature, active: detected},
		action:         action,
		queueReminder:  suppress && (observation.exhausted || observation.reminderConfigured),
		downgradeGuard: suppress && !observation.exhausted && observation.strikeLimit > 0 && observation.nextStrike >= observation.strikeLimit,
	}
}

func (a *Agent) observeRepeatedSignatureIntervention(legacy, shadow interventionDecision) {
	if a.repeatInterventionShadowObserved != nil {
		a.repeatInterventionShadowObserved(legacy, shadow)
	}
	if legacy == shadow {
		return
	}
	a.warnf(
		"agent: repeated-signature intervention shadow mismatch: legacy_kind=%s shadow_kind=%s legacy_detected=%t shadow_detected=%t legacy_action=%s shadow_action=%s legacy_reminder=%t shadow_reminder=%t legacy_downgrade=%t shadow_downgrade=%t",
		safeInterventionKind(legacy.detection.kind),
		safeInterventionKind(shadow.detection.kind),
		legacy.detection.active,
		shadow.detection.active,
		safeInterventionAction(legacy.action),
		safeInterventionAction(shadow.action),
		legacy.queueReminder,
		shadow.queueReminder,
		legacy.downgradeGuard,
		shadow.downgradeGuard,
	)
}

func safeInterventionKind(kind interventionKind) string {
	if kind == interventionKindRepeatedSignature {
		return "repeated_signature"
	}
	return "unknown"
}

func safeInterventionAction(action interventionAction) string {
	if action == interventionActionSuppressTool {
		return "suppress_tool"
	}
	if action == interventionActionProceed {
		return "proceed"
	}
	return "unknown"
}

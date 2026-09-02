package agent

type compactionDecision struct {
	run             bool
	trigger         string
	targetWatermark string
}

type automaticCompactionObservation struct {
	overflow          bool
	ordinaryAdmission bool
	trigger           string
	targetWatermark   string
}

func shadowAutomaticCompactionDecision(observation automaticCompactionObservation) compactionDecision {
	return compactionDecision{
		run:             observation.overflow || observation.ordinaryAdmission,
		trigger:         observation.trigger,
		targetWatermark: observation.targetWatermark,
	}
}

func (a *Agent) observeAutomaticCompactionDecision(legacy, shadow compactionDecision) {
	if a.compactionShadowObserved != nil {
		a.compactionShadowObserved()
	}
	if legacy != shadow {
		a.warnf(
			"agent: automatic compaction decision shadow mismatch: legacy_run=%t shadow_run=%t legacy_trigger=%s shadow_trigger=%s legacy_watermark=%s shadow_watermark=%s",
			legacy.run,
			shadow.run,
			safeCompactionTrigger(legacy.trigger),
			safeCompactionTrigger(shadow.trigger),
			safeCompactionWatermark(legacy.targetWatermark),
			safeCompactionWatermark(shadow.targetWatermark),
		)
	}
}

func safeCompactionTrigger(trigger string) string {
	switch trigger {
	case "usage", "overflow", "todo_checkpoint", "retry_checkpoint", "placeholder_pressure":
		return trigger
	default:
		return "unknown"
	}
}

func safeCompactionWatermark(watermark string) string {
	switch watermark {
	case "":
		return "none"
	case "snip", "prune", "summarize", "overflow", "placeholder_cleanup":
		return watermark
	default:
		return "unknown"
	}
}

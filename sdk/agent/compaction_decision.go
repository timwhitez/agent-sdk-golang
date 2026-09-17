package agent

import (
	"context"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type compactionDecision struct {
	run             bool
	trigger         string
	targetWatermark string
}

func (a *Agent) automaticCompactionDecision(ctx context.Context, usage *llm.Usage) compactionDecision {
	trigger, watermark := a.compactionTriggerAndWatermarkForUsage(usage)
	return compactionDecision{
		run:             watermark == "overflow" || a.shouldAttemptCompactionUsage(ctx, usage),
		trigger:         trigger,
		targetWatermark: watermark,
	}
}

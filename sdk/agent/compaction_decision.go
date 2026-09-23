package agent

import (
	"context"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/compaction"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// compactionEntry names which public or driver entry produced a decision.
// Entries keep their own thresholds, force and wait semantics; the decision
// only makes each entry's policy explicit and sampled once.
type compactionEntry uint8

const (
	compactionEntryAutomatic compactionEntry = iota
	// compactionEntryManual covers CompactNow and CompactPipelineNow, which
	// hosts use for manual compaction and prompt preflight.
	compactionEntryManual
)

// compactionDecision is the single policy value handed to the compaction
// executor. Every admission input is read when the decision is made; the
// synchronous or asynchronous executor never samples policy state again.
type compactionDecision struct {
	run             bool
	trigger         string
	targetWatermark string
	// allowSummary is the summary-tier admission sampled with the decision.
	allowSummary bool
	entry        compactionEntry
	// request is the explicit host request of a manual entry.
	request compaction.PipelineRequest
}

func (a *Agent) automaticCompactionDecision(ctx context.Context, usage *llm.Usage) compactionDecision {
	trigger, watermark := a.compactionTriggerAndWatermarkForUsage(usage)
	run := watermark == "overflow" || a.shouldAttemptCompactionUsage(ctx, usage)
	return compactionDecision{
		run:             run,
		trigger:         trigger,
		targetWatermark: watermark,
		allowSummary:    run && a.compactionSummaryAllowed(),
		entry:           compactionEntryAutomatic,
	}
}

// manualCompactionDecision expresses the manual/preflight policy: the host's
// explicit request runs as given (automatic cooldown and summary suppression
// are not inherited), an overlapping run is rejected rather than joined, and a
// successful run satisfies pending todo/retry work without touching the
// automatic failure streak.
func manualCompactionDecision(req compaction.PipelineRequest) compactionDecision {
	return compactionDecision{
		run:             true,
		trigger:         req.Trigger,
		targetWatermark: req.TargetWatermark,
		allowSummary:    req.AllowSummary,
		entry:           compactionEntryManual,
		request:         req,
	}
}

// pipelineRequest returns the executor request for this decision.
func (d compactionDecision) pipelineRequest(usage *llm.Usage) compaction.PipelineRequest {
	if d.entry == compactionEntryManual {
		return d.request
	}
	return compaction.PipelineRequest{
		Trigger:         d.trigger,
		Usage:           usage,
		TargetWatermark: d.targetWatermark,
		AllowSummary:    d.allowSummary,
	}
}

// rejectsOverlap reports whether an in-flight run is an ErrAgentBusy error
// (manual) instead of a silent skip (automatic).
func (d compactionDecision) rejectsOverlap() bool {
	return d.entry == compactionEntryManual
}

// clearsPending reports which pending work a successful run satisfies.
func (d compactionDecision) clearsPending() (todo, retry bool) {
	if d.entry == compactionEntryManual {
		return true, true
	}
	return d.trigger == "todo_checkpoint", d.trigger == "retry_checkpoint"
}

// recordsOutcome reports whether the run feeds the automatic failure streak
// and cooldown.
func (d compactionDecision) recordsOutcome() bool {
	return d.entry == compactionEntryAutomatic
}

package agent

import (
	"context"
	"fmt"
	"sync"

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
	// inputEpoch is the user-input epoch the decision was made in.
	inputEpoch uint64
	// request is the explicit host request of a manual entry.
	request compaction.PipelineRequest
}

func (a *Agent) automaticCompactionDecision(ctx context.Context, usage *llm.Usage) compactionDecision {
	trigger, watermark := a.compactionTriggerAndWatermarkForUsage(usage)
	run := watermark == "overflow" || a.shouldAttemptCompactionUsage(ctx, usage)
	// Sampled on every decision, run or not, so a decision below the summary
	// threshold clears the suppression.
	epoch := a.userInputEpoch.Load()
	suppressed := a.automaticSummarySuppressed(epoch, usage)
	return compactionDecision{
		run:             run,
		trigger:         trigger,
		targetWatermark: watermark,
		allowSummary:    run && a.compactionSummaryAllowed() && !suppressed,
		entry:           compactionEntryAutomatic,
		inputEpoch:      epoch,
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

// overflowCompactionMark records the latest overflow compaction: the user
// input it ran in (epoch+1, 0 when unset), the Frame whose usage drove it and
// the estimated history it left.
type overflowCompactionMark struct {
	mu            sync.Mutex
	epoch         uint64
	frameID       string
	historyTokens int
}

// noteOverflowCompaction arms fixedPromptExceedsWindow after an overflow
// compaction run, including one whose summary was rejected as not smaller.
func (a *Agent) noteOverflowCompaction(frameID string, historyTokens int) {
	m := &a.overflowCompaction
	m.mu.Lock()
	defer m.mu.Unlock()
	m.epoch, m.frameID, m.historyTokens = a.userInputEpoch.Load()+1, frameID, historyTokens
}

// fixedPromptExceedsWindow is the overflow side of the ineffective-summary
// guard. Compaction shrinks only history, not the system prompt and tool
// definitions the provider also counts. The first decision driven by a later
// Frame of the same user input therefore checks the provider-reported prompt
// of that Frame's request: still at or above the overflow limit means the
// compacted request did not fit, and compacting or trimming again cannot help,
// so the turn ends with this error. Only provider usage stops the turn; a
// local estimate does not count images and would misjudge the fixed share.
// Decisions on the compacting Frame itself still carry its pre-compaction
// usage and are not judged.
func (a *Agent) fixedPromptExceedsWindow(frameID string, usage *llm.Usage) error {
	m := &a.overflowCompaction
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.epoch == 0 || frameID == "" || frameID == m.frameID {
		return nil
	}
	limit := a.compactor.OverflowLimit()
	if m.epoch != a.userInputEpoch.Load()+1 || usage == nil || usage.PromptTokensSource == llm.PromptTokensSourceEstimate ||
		!llm.PromptUsageIsProviderValid(usage) || usage.PromptTokens < limit {
		m.epoch = 0
		return nil
	}
	// Stays armed, so every later decision on this Frame reports the same stop.
	return &fixedPromptExceedsWindowError{promptTokens: usage.PromptTokens, historyTokens: m.historyTokens, limit: limit}
}

type fixedPromptExceedsWindowError struct {
	promptTokens, historyTokens, limit int
}

func (e *fixedPromptExceedsWindowError) Error() string {
	hint := "increase the context window or send fewer tool definitions / a shorter system prompt"
	if e.historyTokens >= e.limit {
		hint = "the compacted history alone reaches the limit; increase the context window"
	}
	return fmt.Sprintf("context window cannot hold the request after overflow compaction: provider reported %d prompt tokens, overflow limit %d; compacted history is about %d estimated tokens, so the fixed prompt (system prompt and tool definitions) is about %d tokens; %s",
		e.promptTokens, e.limit, e.historyTokens, e.promptTokens-e.historyTokens, hint)
}

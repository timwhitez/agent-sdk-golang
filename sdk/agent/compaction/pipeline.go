package compaction

import (
	"context"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

// CompactPipeline is the canonical compaction state machine used by automatic,
// preflight, manual, Todo/retry checkpoint, and overflow entry points.
func (s *Service) CompactPipeline(ctx context.Context, model llm.ChatModel, messages []llm.Message, req PipelineRequest) ([]llm.Message, Result, error) {
	if s == nil || !s.Config.Enabled {
		return messages, Result{Compacted: false}, nil
	}
	if ctx != nil && ctx.Err() != nil {
		return messages, Result{Compacted: false}, ctx.Err()
	}
	if s.Config.LedgerStore != nil && s.Config.CheckpointWriter != nil {
		return s.compactPipelineWithDeferredLedger(ctx, model, messages, req)
	}
	return s.compactPipeline(ctx, model, messages, req)
}

func (s *Service) compactPipelineWithDeferredLedger(ctx context.Context, model llm.ChatModel, messages []llm.Message, req PipelineRequest) ([]llm.Message, Result, error) {
	sessionID := strings.TrimSpace(s.Config.SessionID)
	previous, loadWarnings, err := s.loadLedger(ctx, sessionID)
	if err != nil {
		return messages, Result{Compacted: false, Warnings: append([]string(nil), loadWarnings...)}, err
	}
	txStore := &deferredLedgerStore{ledger: previous.Clone()}
	tx := *s
	tx.Config = s.Config
	tx.Config.LedgerStore = txStore
	// The outer service owns the real checkpoint transaction. Clearing the
	// writer here lets every tier save into the in-memory ledger copy without
	// recursively starting another deferred transaction.
	tx.Config.CheckpointWriter = nil

	out, res, err := tx.compactPipeline(ctx, model, messages, req)
	if len(loadWarnings) > 0 {
		res.Warnings = append(append([]string(nil), loadWarnings...), res.Warnings...)
	}
	if !res.Compacted || !txStore.saved {
		return out, res, err
	}
	res.previousLedger = previous.Clone()
	res.pendingLedger = txStore.ledger.Clone()
	return out, res, err
}

func (s *Service) compactPipeline(ctx context.Context, model llm.ChatModel, messages []llm.Message, req PipelineRequest) ([]llm.Message, Result, error) {

	trigger := strings.TrimSpace(req.Trigger)
	if trigger == "" {
		trigger = "usage"
	}
	usage := pipelineUsage(req)
	target := strings.TrimSpace(req.TargetWatermark)
	if target == "" {
		target = s.WatermarkForUsage(usage)
	}
	if req.ForceSummary && target == "" {
		target = "summarize"
	}
	if target == tierPlaceholderCleanup {
		out, res, err := s.CompactDestroyedPlaceholders(ctx, messages, usage)
		res.Trigger = trigger
		return out, res, err
	}

	additionalTokens := maxPipelineInt(req.AdditionalTokens, 0)
	originalEstimate := s.approximateMessageTokens(messages) + additionalTokens
	decisionTokens := s.TotalTokens(usage)
	if decisionTokens <= 0 {
		decisionTokens = originalEstimate
	}
	currentDecisionTokens := decisionTokens
	currentEstimate := originalEstimate
	current := messages
	results := make([]Result, 0, 3)
	targetRank := pipelineWatermarkRank(target)

	if targetRank >= pipelineWatermarkRank(tierSnip) && currentDecisionTokens >= s.snipThreshold() {
		out, res, err := s.compactLocalWithWatermark(ctx, current, pipelineLocalUsage(usage, currentDecisionTokens), tierSnip)
		results = append(results, res)
		if err != nil {
			merged := mergePipelineResults(results...)
			finalizePipelineResult(&merged, trigger, target, originalEstimate, currentEstimate, usage)
			return messages, merged, err
		}
		if res.Compacted {
			current = out
			currentEstimate = s.approximateMessageTokens(current) + additionalTokens
			currentDecisionTokens = currentEstimate
		}
	}

	if targetRank >= pipelineWatermarkRank(tierPrune) && currentDecisionTokens >= s.pruneThreshold() {
		out, res, err := s.compactLocalWithWatermark(ctx, current, pipelineLocalUsage(usage, currentDecisionTokens), tierPrune)
		results = append(results, res)
		if err != nil {
			merged := mergePipelineResults(results...)
			finalizePipelineResult(&merged, trigger, target, originalEstimate, currentEstimate, usage)
			return current, merged, err
		}
		if res.Compacted {
			current = out
			currentEstimate = s.approximateMessageTokens(current) + additionalTokens
			currentDecisionTokens = currentEstimate
		}
	}

	shouldSummarize := req.ForceSummary || (req.AllowSummary && targetRank >= pipelineWatermarkRank("summarize") && currentDecisionTokens >= s.threshold())
	if shouldSummarize {
		out, res, err := s.compactSummary(ctx, model, current)
		if res.NewTokens > 0 && res.Compacted {
			res.NewTokens += additionalTokens
		}
		results = append(results, res)
		merged := mergePipelineResults(results...)
		finalizePipelineResult(&merged, trigger, target, originalEstimate, merged.NewTokens, usage)
		if err != nil {
			return current, merged, err
		}
		return out, merged, nil
	}

	merged := mergePipelineResults(results...)
	finalizePipelineResult(&merged, trigger, target, originalEstimate, currentEstimate, usage)
	return current, merged, nil
}

func pipelineUsage(req PipelineRequest) *llm.Usage {
	if req.EstimatedTokens > 0 {
		return llm.WithPromptEstimate(req.Usage, req.EstimatedTokens)
	}
	return llm.NormalizeUsage(req.Usage)
}

func pipelineLocalUsage(base *llm.Usage, total int) *llm.Usage {
	if total < 0 {
		total = 0
	}
	if base == nil {
		return llm.WithPromptEstimate(nil, total)
	}
	out := llm.CloneUsage(base)
	out.PromptTokens = total
	out.TotalTokens = total
	out.CompletionTokens = 0
	if strings.TrimSpace(out.PromptTokensSource) == "" {
		out.PromptTokensSource = llm.PromptTokensSourceEstimate
	}
	return out
}

func pipelineWatermarkRank(watermark string) int {
	switch strings.TrimSpace(watermark) {
	case tierPlaceholderCleanup:
		return 1
	case tierSnip:
		return 2
	case tierPrune:
		return 3
	case "summarize", "overflow":
		return 4
	default:
		return 0
	}
}

func mergePipelineResults(results ...Result) Result {
	merged := Result{}
	seenTiers := map[string]struct{}{}
	// Watermark and NewTokens report what compaction actually achieved, so they
	// may only be adopted from a tier that reported Compacted. A tier that ran
	// but changed nothing still returns its own watermark and NewTokens equal to
	// the original token count (a summary rejected by the quality gate is the
	// common case): letting it win would erase the reduction the preceding
	// snip/prune tiers really made, report "compacted size == original size",
	// and make the host believe the context never came back under threshold.
	fallbackWatermark := ""
	for _, res := range results {
		merged.Compacted = merged.Compacted || res.Compacted
		if merged.Trigger == "" && strings.TrimSpace(res.Trigger) != "" {
			merged.Trigger = strings.TrimSpace(res.Trigger)
		}
		if watermark := strings.TrimSpace(res.Watermark); watermark != "" {
			if res.Compacted {
				merged.Watermark = watermark
			} else {
				fallbackWatermark = watermark
			}
		}
		if merged.Usage == nil && res.Usage != nil {
			merged.Usage = llm.CloneUsage(res.Usage)
		}
		if merged.TokenCountSource == "" && strings.TrimSpace(res.TokenCountSource) != "" {
			merged.TokenCountSource = strings.TrimSpace(res.TokenCountSource)
		}
		if merged.OriginalTokens <= 0 && res.OriginalTokens > 0 {
			merged.OriginalTokens = res.OriginalTokens
		}
		if res.NewTokens > 0 && res.Compacted {
			merged.NewTokens = res.NewTokens
		}
		for _, tier := range res.TiersApplied {
			tier = strings.TrimSpace(tier)
			if tier == "" {
				continue
			}
			if _, exists := seenTiers[tier]; exists {
				continue
			}
			seenTiers[tier] = struct{}{}
			merged.TiersApplied = append(merged.TiersApplied, tier)
		}
		if strings.TrimSpace(res.SnapshotPath) != "" {
			merged.SnapshotPath = strings.TrimSpace(res.SnapshotPath)
		}
		if strings.TrimSpace(res.LedgerPath) != "" {
			merged.LedgerPath = strings.TrimSpace(res.LedgerPath)
		}
		merged.Warnings = append(merged.Warnings, res.Warnings...)
		if strings.TrimSpace(res.Summary) != "" {
			merged.Summary = strings.TrimSpace(res.Summary)
		}
		if res.pendingLedger != nil {
			merged.pendingLedger = res.pendingLedger.Clone()
			merged.previousLedger = res.previousLedger.Clone()
		}
	}
	if merged.Watermark == "" {
		// No tier reported a successful compaction; the last attempted tier's
		// watermark is still the most accurate description of what was tried.
		merged.Watermark = fallbackWatermark
	}
	return merged
}

func finalizePipelineResult(res *Result, trigger, target string, originalTokens, newTokens int, usage *llm.Usage) {
	if res == nil {
		return
	}
	res.Trigger = strings.TrimSpace(trigger)
	if res.Trigger == "" {
		res.Trigger = "usage"
	}
	if strings.TrimSpace(res.Watermark) == "" {
		res.Watermark = strings.TrimSpace(target)
	}
	if usage != nil {
		res.Usage = llm.CloneUsage(usage)
	}
	if originalTokens > 0 {
		res.OriginalTokens = originalTokens
	}
	if newTokens > 0 {
		res.NewTokens = newTokens
	} else if res.NewTokens <= 0 {
		res.NewTokens = res.OriginalTokens
	}
	if res.OriginalTokens > 0 || res.NewTokens > 0 {
		res.TokenCountSource = TokenCountSourceEstimate
	}
}

func maxPipelineInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

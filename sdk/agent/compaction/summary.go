package compaction

import (
	"fmt"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/agent/messageorigin"
	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

const (
	summaryCheckpointMarkerPrefix = "<!-- goode-compaction-checkpoint-id="
	summaryCheckpointMarkerSuffix = " -->"
)

func incrementalSummaryContext(messages []llm.Message, ledger *Ledger, keepRecentUsers int, estimate tokenEstimator) (string, string) {
	if ledger == nil || ledger.Summary == nil || ledger.Summary.Version <= 0 {
		return "", ""
	}
	binding, err := validateLedgerSummaryBinding(messages, ledger.Summary)
	if err != nil {
		return "", incrementalSummaryRebuildWarning(err.Error())
	}
	delta := summaryDeltaMessages(messages, binding.summaryIndex+1, keepRecentUsers, estimate)
	if strings.TrimSpace(delta) == "" {
		delta = fallbackSummaryContext
	}
	return fmt.Sprintf("## Previous Summary\n%s\n\n## Delta Messages\n%s", binding.summary, delta), ""
}

type ledgerSummaryBinding struct {
	summaryIndex int
	summary      string
}

// ValidateLedgerSummaryBinding verifies that one ledger summary generation is
// the exact current compaction_summary in effective provider history. Hosts use
// this shared validator when a durable lifecycle decision must agree with the
// SDK's incremental-summary authority semantics.
func ValidateLedgerSummaryBinding(messages []llm.Message, meta *LedgerSummary) error {
	_, err := validateLedgerSummaryBinding(messages, meta)
	return err
}

func validateLedgerSummaryBinding(messages []llm.Message, meta *LedgerSummary) (ledgerSummaryBinding, error) {
	if meta == nil || meta.Version <= 0 {
		return ledgerSummaryBinding{}, fmt.Errorf("summary generation metadata missing")
	}
	if strings.TrimSpace(meta.MessageName) != CompactionSummaryMessageName {
		return ledgerSummaryBinding{}, fmt.Errorf("message name mismatch")
	}
	if strings.TrimSpace(meta.SummaryHash) == "" {
		return ledgerSummaryBinding{}, fmt.Errorf("summary hash missing")
	}
	if strings.TrimSpace(meta.CoveredStartKey) == "" || strings.TrimSpace(meta.CoveredEndKey) == "" {
		return ledgerSummaryBinding{}, fmt.Errorf("coverage metadata missing")
	}
	if strings.TrimSpace(meta.CheckpointID) == "" {
		return ledgerSummaryBinding{}, fmt.Errorf("coverage checkpoint identity missing")
	}

	summaryIndex := -1
	previousSummary := ""
	checkpointID := ""
	matches := 0
	summaryMessages := 0
	for i, msg := range messages {
		if !isCompactionSummaryMessage(msg) {
			continue
		}
		summaryMessages++
		candidateSummary, candidateCheckpointID, ok := splitSummaryCheckpoint(msg.Content.PlainText())
		if !ok || strings.TrimSpace(candidateCheckpointID) != strings.TrimSpace(meta.CheckpointID) {
			continue
		}
		summaryIndex = i
		previousSummary = candidateSummary
		checkpointID = candidateCheckpointID
		matches++
	}
	if summaryMessages != 1 || matches != 1 || summaryIndex < 0 || strings.TrimSpace(previousSummary) == "" {
		return ledgerSummaryBinding{}, fmt.Errorf("current history topology mismatch: summary_messages=%d matching_checkpoint_summaries=%d", summaryMessages, matches)
	}
	if strings.TrimSpace(checkpointID) != strings.TrimSpace(meta.CheckpointID) {
		return ledgerSummaryBinding{}, fmt.Errorf("coverage checkpoint identity mismatch")
	}
	if ContentHash(previousSummary) != strings.TrimSpace(meta.SummaryHash) {
		return ledgerSummaryBinding{}, fmt.Errorf("summary hash mismatch")
	}
	if got := ledgerSummaryCheckpointID(meta); got != strings.TrimSpace(meta.CheckpointID) {
		return ledgerSummaryBinding{}, fmt.Errorf("coverage mismatch: checkpoint identity does not match ledger coverage metadata")
	}
	for i := 0; i < summaryIndex; i++ {
		msg := messages[i]
		if msg.Destroyed || msg.Role == llm.RoleSystem || messageorigin.IsInternalMessage(msg) {
			continue
		}
		return ledgerSummaryBinding{}, fmt.Errorf("current history topology mismatch: non-system message precedes checkpoint summary")
	}
	return ledgerSummaryBinding{summaryIndex: summaryIndex, summary: previousSummary}, nil
}

func incrementalSummaryRebuildWarning(reason string) string {
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "ledger integrity could not be proven"
	}
	return fmt.Sprintf("[WARN] Incremental compaction ledger integrity mismatch - using full rebuild from current history. (reason=%s action=continuing safely; a successful rebuild will refresh stale summary-ledger metadata)", reason)
}

func summaryDeltaMessages(messages []llm.Message, start int, keepRecentUsers int, estimate tokenEstimator) string {
	if len(messages) == 0 {
		return ""
	}
	if start < 0 {
		start = 0
	}
	if start >= len(messages) {
		return ""
	}
	protectedUsers := recentUserIndexes(messages, start, keepRecentUsers)
	var b strings.Builder
	for i := start; i < len(messages); i++ {
		msg := messages[i]
		if msg.Destroyed || messageorigin.IsInternalMessage(msg) {
			continue
		}
		if _, ok := protectedUsers[i]; ok {
			continue
		}
		text := strings.TrimSpace(msg.Content.PlainText())
		if text == "" && len(msg.ToolCalls) == 0 {
			continue
		}
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString("- ")
		b.WriteString(string(msg.Role))
		if strings.TrimSpace(msg.Name) != "" {
			b.WriteString(" name=")
			b.WriteString(strings.TrimSpace(msg.Name))
		}
		if strings.TrimSpace(msg.ToolName) != "" {
			b.WriteString(" tool=")
			b.WriteString(strings.TrimSpace(msg.ToolName))
		}
		if strings.TrimSpace(msg.ToolCallID) != "" {
			b.WriteString(" tool_call_id=")
			b.WriteString(strings.TrimSpace(msg.ToolCallID))
		}
		if len(msg.ToolCalls) > 0 {
			b.WriteString(" tool_calls=")
			for i, call := range msg.ToolCalls {
				if i > 0 {
					b.WriteByte(',')
				}
				b.WriteString(strings.TrimSpace(call.Function.Name))
				if strings.TrimSpace(call.ID) != "" {
					b.WriteString("/")
					b.WriteString(strings.TrimSpace(call.ID))
				}
			}
		}
		if text != "" {
			b.WriteString(": ")
			b.WriteString(truncateSummaryDeltaTextWithEstimator(text, estimate))
		}
	}
	return b.String()
}

func recentUserIndexes(messages []llm.Message, start int, keepRecentUsers int) map[int]struct{} {
	protected := map[int]struct{}{}
	if keepRecentUsers <= 0 || start >= len(messages) {
		return protected
	}
	if start < 0 {
		start = 0
	}
	for i := len(messages) - 1; i >= start && len(protected) < keepRecentUsers; i-- {
		msg := messages[i]
		if !messageorigin.IsRealUserMessage(msg) {
			continue
		}
		protected[i] = struct{}{}
	}
	return protected
}

func truncateSummaryDeltaText(text string) string {
	return truncateSummaryDeltaTextWithEstimator(text, approximateTextTokens)
}

func truncateSummaryDeltaTextWithEstimator(text string, estimate tokenEstimator) string {
	return truncateTextToTokenBudget(text, summaryDeltaTokenBudget, estimate)
}

func stripSummaryPrefix(text string) string {
	summary, _, _ := splitSummaryCheckpoint(text)
	return summary
}

func nextLedgerSummary(prev *LedgerSummary, messages []llm.Message, summary, sourceSnapshot string) *LedgerSummary {
	version := 1
	if prev != nil && prev.Version > 0 {
		version = prev.Version + 1
	}
	start, end := summaryCoverageKeys(messages)
	meta := &LedgerSummary{
		Version:         version,
		MessageName:     CompactionSummaryMessageName,
		SummaryHash:     ContentHash(summary),
		CoveredStartKey: start,
		CoveredEndKey:   end,
		SourceSnapshot:  strings.TrimSpace(sourceSnapshot),
	}
	meta.CheckpointID = ledgerSummaryCheckpointID(meta)
	return meta
}

func ledgerSummaryCheckpointID(meta *LedgerSummary) string {
	if meta == nil {
		return ""
	}
	payload := strings.Join([]string{
		fmt.Sprintf("version=%d", meta.Version),
		"message_name=" + strings.TrimSpace(meta.MessageName),
		"summary_hash=" + strings.TrimSpace(meta.SummaryHash),
		"covered_start_key=" + strings.TrimSpace(meta.CoveredStartKey),
		"covered_end_key=" + strings.TrimSpace(meta.CoveredEndKey),
		"source_snapshot=" + strings.TrimSpace(meta.SourceSnapshot),
	}, "\n")
	return ContentHash(payload)
}

func withSummaryCheckpoint(summary string, meta *LedgerSummary) string {
	checkpointID := ""
	if meta != nil {
		checkpointID = strings.TrimSpace(meta.CheckpointID)
	}
	if checkpointID == "" {
		return WithSummaryPrefix(summary)
	}
	marker := summaryCheckpointMarkerPrefix + checkpointID + summaryCheckpointMarkerSuffix
	return WithSummaryPrefix(marker + "\n\n" + strings.TrimSpace(summary))
}

func splitSummaryCheckpoint(text string) (summary string, checkpointID string, ok bool) {
	text = strings.TrimSpace(text)
	text = strings.TrimSpace(strings.TrimPrefix(text, DefaultSummaryPrefix))
	if !strings.HasPrefix(text, summaryCheckpointMarkerPrefix) {
		return text, "", false
	}
	end := strings.Index(text, summaryCheckpointMarkerSuffix)
	if end < len(summaryCheckpointMarkerPrefix) {
		return text, "", false
	}
	checkpointID = strings.TrimSpace(text[len(summaryCheckpointMarkerPrefix):end])
	summary = strings.TrimSpace(text[end+len(summaryCheckpointMarkerSuffix):])
	if checkpointID == "" || summary == "" {
		return summary, checkpointID, false
	}
	return summary, checkpointID, true
}

func summaryCoverageKeys(messages []llm.Message) (string, string) {
	if len(messages) == 0 {
		return "", ""
	}
	first := ""
	last := ""
	for i, msg := range messages {
		if msg.Destroyed {
			continue
		}
		key := StableMessageKey(MessageKeyInput{
			Role:           string(msg.Role),
			ToolCallID:     msg.ToolCallID,
			ToolName:       msg.ToolName,
			OriginalText:   msg.Content.PlainText(),
			FirstSeenIndex: i,
		})
		if first == "" {
			first = key
		}
		last = key
	}
	return first, last
}

package compaction

import (
	"fmt"
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func incrementalSummaryContext(messages []llm.Message, ledger *Ledger, keepRecentUsers int) string {
	if ledger == nil || ledger.Summary == nil || ledger.Summary.Version <= 0 {
		return ""
	}
	summaryIndex := -1
	previousSummary := ""
	for i, msg := range messages {
		if !isCompactionSummaryMessage(msg) {
			continue
		}
		summaryIndex = i
		previousSummary = stripSummaryPrefix(msg.Content.PlainText())
	}
	if summaryIndex < 0 || strings.TrimSpace(previousSummary) == "" {
		return ""
	}
	delta := summaryDeltaMessages(messages, summaryIndex+1, keepRecentUsers)
	if strings.TrimSpace(delta) == "" {
		delta = fallbackSummaryContext
	}
	return fmt.Sprintf("Merge the previous compaction summary with the new delta messages.\n\n## Previous summary\n%s\n\n## Delta messages\n%s\n\nReturn one updated summary in <summary></summary> tags.", previousSummary, delta)
}

func summaryDeltaMessages(messages []llm.Message, start int, keepRecentUsers int) string {
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
		if msg.Destroyed || isCompactionSummaryMessage(msg) {
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
			b.WriteString(truncateSummaryDeltaText(text))
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
		if msg.Destroyed || msg.Role != llm.RoleUser || isCompactionSummaryMessage(msg) {
			continue
		}
		protected[i] = struct{}{}
	}
	return protected
}

func truncateSummaryDeltaText(text string) string {
	const max = 1200
	text = strings.TrimSpace(text)
	if len(text) <= max {
		return text
	}
	return strings.TrimSpace(text[:max]) + "..."
}

func stripSummaryPrefix(text string) string {
	text = strings.TrimSpace(text)
	text = strings.TrimPrefix(text, DefaultSummaryPrefix)
	return strings.TrimSpace(text)
}

func nextLedgerSummary(prev *LedgerSummary, messages []llm.Message, summary string) *LedgerSummary {
	version := 1
	if prev != nil && prev.Version > 0 {
		version = prev.Version + 1
	}
	start, end := summaryCoverageKeys(messages)
	return &LedgerSummary{
		Version:         version,
		MessageName:     CompactionSummaryMessageName,
		SummaryHash:     ContentHash(summary),
		CoveredStartKey: start,
		CoveredEndKey:   end,
	}
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

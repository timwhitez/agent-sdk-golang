package compaction

import (
	"strings"
	"testing"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

func TestValidateLedgerSummaryBindingAcceptsCurrentEffectiveSummary(t *testing.T) {
	source := []llm.Message{
		llm.NewSystemMessage("system"),
		llm.NewUserMessage("first request"),
		llm.NewAssistantMessage("first answer", nil),
	}
	meta := nextLedgerSummary(nil, source, "bounded current summary", "")
	effective := []llm.Message{
		llm.NewSystemMessage("system"),
		{
			Role:    llm.RoleUser,
			Name:    CompactionSummaryMessageName,
			Content: llm.TextContent(withSummaryCheckpoint("bounded current summary", meta)),
		},
		llm.NewUserMessage("delta request"),
	}

	if err := ValidateLedgerSummaryBinding(effective, meta); err != nil {
		t.Fatalf("ValidateLedgerSummaryBinding() error = %v", err)
	}
}

func TestValidateLedgerSummaryBindingRejectsMissingDuplicateAndDrift(t *testing.T) {
	source := []llm.Message{
		llm.NewUserMessage("first request"),
		llm.NewAssistantMessage("first answer", nil),
	}
	meta := nextLedgerSummary(nil, source, "bounded current summary", "")
	current := llm.Message{
		Role:    llm.RoleUser,
		Name:    CompactionSummaryMessageName,
		Content: llm.TextContent(withSummaryCheckpoint("bounded current summary", meta)),
	}

	tests := []struct {
		name     string
		messages []llm.Message
		meta     *LedgerSummary
		want     string
	}{
		{name: "missing", messages: []llm.Message{llm.NewSystemMessage("system")}, meta: meta, want: "topology mismatch"},
		{name: "duplicate", messages: []llm.Message{current, current}, meta: meta, want: "topology mismatch"},
		{name: "summary hash drift", messages: []llm.Message{{Role: llm.RoleUser, Name: CompactionSummaryMessageName, Content: llm.TextContent(withSummaryCheckpoint("different summary", meta))}}, meta: meta, want: "summary hash mismatch"},
		{name: "coverage drift", messages: []llm.Message{current}, meta: func() *LedgerSummary {
			drift := *meta
			drift.CoveredEndKey = "drifted"
			return &drift
		}(), want: "checkpoint identity does not match"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateLedgerSummaryBinding(tt.messages, tt.meta)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("ValidateLedgerSummaryBinding() error = %v, want containing %q", err, tt.want)
			}
		})
	}
}

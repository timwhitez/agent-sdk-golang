package compaction

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

const (
	tierSnip             = "snip"
	defaultSnipMinTokens = 64
)

var truncationArtifactRe = regexp.MustCompile(`(?i)(?:saved to|full output:|full_output=)\s+([^\]\s]+(?:/|\\)truncated(?:/|\\)[^\]\s]+)`)

func (s *Service) CompactLocal(ctx context.Context, messages []llm.Message, usage *llm.Usage) ([]llm.Message, Result, error) {
	if s == nil || !s.Config.Enabled {
		return messages, Result{Compacted: false}, nil
	}
	if ctx != nil && ctx.Err() != nil {
		return messages, Result{Compacted: false}, ctx.Err()
	}
	if len(messages) == 0 {
		return messages, Result{Compacted: false}, nil
	}
	sessionID := strings.TrimSpace(s.Config.SessionID)
	ledger, loadWarnings, err := s.loadLedger(ctx, sessionID)
	if err != nil {
		return messages, Result{Compacted: false}, err
	}
	originalTokens := s.TotalTokens(usage)
	if originalTokens <= 0 {
		originalTokens = approximateMessageTokens(messages)
	}
	reducer := localReducer{
		service:       s,
		ctx:           ctx,
		sessionID:     sessionID,
		ledger:        ledger,
		replacements:  ledgerReplacementIndex(ledger),
		protectedFrom: protectedMessageStart(len(messages), s.protectedRecentMessages()),
	}
	out, warnings, createdOrReused := reducer.snip(messages)
	warnings = append(loadWarnings, warnings...)
	if createdOrReused == 0 {
		return messages, Result{
			Compacted:      false,
			Trigger:        "usage",
			Watermark:      tierSnip,
			Usage:          cloneUsage(usage),
			OriginalTokens: originalTokens,
			NewTokens:      originalTokens,
			Warnings:       warnings,
		}, nil
	}
	ledger.UpdatedAt = time.Now().UTC()
	ledger.ContextWindow = s.contextWindow()
	if err := ledger.Validate(sessionID); err != nil {
		return messages, Result{Compacted: false, Warnings: warnings}, err
	}
	if err := s.saveLedger(ctx, sessionID, ledger); err != nil {
		return messages, Result{Compacted: false, Warnings: warnings}, err
	}
	newTokens := approximateMessageTokens(out)
	res := Result{
		Compacted:      true,
		Trigger:        "usage",
		Watermark:      tierSnip,
		Usage:          cloneUsage(usage),
		OriginalTokens: originalTokens,
		NewTokens:      newTokens,
		TiersApplied:   []string{tierSnip},
		Warnings:       warnings,
	}
	res.LedgerPath = strings.TrimSpace(s.Config.LedgerPath)
	return out, res, nil
}

type localReducer struct {
	service       *Service
	ctx           context.Context
	sessionID     string
	ledger        *Ledger
	replacements  map[string]LedgerReplacement
	protectedFrom int
}

func (r *localReducer) snip(messages []llm.Message) ([]llm.Message, []string, int) {
	out := make([]llm.Message, len(messages))
	copy(out, messages)
	warnings := []string{}
	changed := 0
	for i, msg := range messages {
		if !r.eligibleToolMessage(i, msg) {
			continue
		}
		original := msg.Content.PlainText()
		key := StableMessageKey(MessageKeyInput{
			Role:           string(msg.Role),
			ToolCallID:     msg.ToolCallID,
			ToolName:       msg.ToolName,
			OriginalText:   original,
			FirstSeenIndex: i,
		})
		partKey := "content-0"
		lookupKey := replacementLookupKey(key, partKey)
		repl, ok := r.replacements[lookupKey]
		if !ok {
			created, warning, ok := r.createSnipReplacement(msg, key, partKey, original)
			if warning != "" {
				warnings = append(warnings, warning)
			}
			if !ok {
				continue
			}
			repl = created
			r.ledger.Replacements = append(r.ledger.Replacements, repl)
			r.replacements[lookupKey] = repl
		}
		msg.Content = llm.TextContent(repl.ReplacementText)
		out[i] = msg
		changed++
	}
	return out, warnings, changed
}

func (r *localReducer) eligibleToolMessage(index int, msg llm.Message) bool {
	if index >= r.protectedFrom {
		return false
	}
	if msg.Role != llm.RoleTool || msg.Destroyed {
		return false
	}
	if isProtectedTool(msg.ToolName, r.service.protectedTools) {
		return false
	}
	text := msg.Content.PlainText()
	if strings.TrimSpace(text) == "" {
		return false
	}
	return approximateTextTokens(text) >= defaultSnipMinTokens
}

func (r *localReducer) createSnipReplacement(msg llm.Message, messageKey, partKey, original string) (LedgerReplacement, string, bool) {
	artifactPath := extractTruncationArtifactPath(original)
	if artifactPath == "" {
		if r.service.Config.ToolArtifactWriter == nil {
			return LedgerReplacement{}, compactionArtifactWarning(r.sessionID, msg.ToolName, msg.ToolCallID, "write", "no artifact writer configured"), false
		}
		artifact, err := r.service.Config.ToolArtifactWriter.SaveCompactionArtifact(r.ctx, ArtifactRequest{
			SessionID:  r.sessionID,
			MessageKey: messageKey,
			PartKey:    partKey,
			ToolName:   msg.ToolName,
			ToolCallID: msg.ToolCallID,
			Content:    original,
		})
		if err != nil {
			return LedgerReplacement{}, compactionArtifactWarning(r.sessionID, msg.ToolName, msg.ToolCallID, "write", err.Error()), false
		}
		artifactPath = strings.TrimSpace(artifact.Path)
		if artifactPath == "" {
			return LedgerReplacement{}, compactionArtifactWarning(r.sessionID, msg.ToolName, msg.ToolCallID, "write", "artifact writer returned empty path"), false
		}
	}
	replacementText := snipReplacementText(msg, original, artifactPath)
	now := time.Now().UTC()
	return LedgerReplacement{
		MessageKey:      messageKey,
		PartKey:         partKey,
		Role:            string(msg.Role),
		ToolName:        strings.TrimSpace(msg.ToolName),
		Tier:            tierSnip,
		OriginalHash:    ContentHash(original),
		ReplacementHash: ContentHash(replacementText),
		ReplacementText: replacementText,
		FullArtifact:    artifactPath,
		CreatedAt:       now,
		OriginalText:    original,
	}, "", true
}

func snipReplacementText(msg llm.Message, original, artifactPath string) string {
	tool := strings.TrimSpace(msg.ToolName)
	if tool == "" {
		tool = "tool"
	}
	id := strings.TrimSpace(msg.ToolCallID)
	if id == "" {
		id = "-"
	}
	lines := countTextLines(original)
	return fmt.Sprintf("[Tool result snipped: %s tool_call_id=%s lines=%d bytes=%d full_output=%s]", tool, id, lines, len(original), strings.TrimSpace(artifactPath))
}

func countTextLines(text string) int {
	if text == "" {
		return 0
	}
	return strings.Count(text, "\n") + 1
}

func extractTruncationArtifactPath(text string) string {
	m := truncationArtifactRe.FindStringSubmatch(text)
	if len(m) < 2 {
		return ""
	}
	return strings.TrimSpace(strings.TrimRight(m[1], ".,;"))
}

func compactionArtifactWarning(sessionID, toolName, toolCallID, stage, detail string) string {
	return fmt.Sprintf("[WARN] Compaction artifact not saved - session=%s stage=%s tool=%s tool_call_id=%s action=leaving original tool result in context: %s", strings.TrimSpace(sessionID), strings.TrimSpace(stage), strings.TrimSpace(toolName), strings.TrimSpace(toolCallID), strings.TrimSpace(detail))
}

func protectedMessageStart(length int, protected int) int {
	if protected <= 0 {
		return length
	}
	start := length - protected
	if start < 0 {
		return 0
	}
	return start
}

func (s *Service) protectedRecentMessages() int {
	if s == nil {
		return DefaultKeepRecentUserMessages
	}
	n := s.Config.ProtectedRecentMessages
	if n <= 0 {
		n = DefaultKeepRecentUserMessages
	}
	return n
}

func ledgerReplacementIndex(ledger *Ledger) map[string]LedgerReplacement {
	out := map[string]LedgerReplacement{}
	if ledger == nil {
		return out
	}
	for _, repl := range ledger.Replacements {
		out[replacementLookupKey(repl.MessageKey, repl.PartKey)] = repl
	}
	return out
}

func replacementLookupKey(messageKey, partKey string) string {
	return strings.TrimSpace(messageKey) + "\x00" + strings.TrimSpace(partKey)
}

type warningLedgerStore interface {
	LoadCompactionLedger(context.Context, string) (*Ledger, string, error)
}

func (s *Service) loadLedger(ctx context.Context, sessionID string) (*Ledger, []string, error) {
	if s == nil || s.Config.LedgerStore == nil {
		return NewLedger(sessionID), nil, nil
	}
	if store, ok := s.Config.LedgerStore.(warningLedgerStore); ok {
		ledger, warning, err := store.LoadCompactionLedger(ctx, sessionID)
		warnings := []string{}
		if strings.TrimSpace(warning) != "" {
			warnings = append(warnings, strings.TrimSpace(warning))
		}
		if err != nil {
			return nil, warnings, err
		}
		if ledger == nil {
			return NewLedger(sessionID), warnings, nil
		}
		if err := ledger.Validate(sessionID); err != nil {
			return nil, warnings, err
		}
		return ledger, warnings, nil
	}
	ledger, err := s.Config.LedgerStore.Load(ctx, sessionID)
	if err != nil {
		return nil, nil, err
	}
	if ledger == nil {
		return NewLedger(sessionID), nil, nil
	}
	if err := ledger.Validate(sessionID); err != nil {
		return nil, nil, err
	}
	return ledger, nil, nil
}

func (s *Service) saveLedger(ctx context.Context, sessionID string, ledger *Ledger) error {
	if s == nil || s.Config.LedgerStore == nil {
		return nil
	}
	return s.Config.LedgerStore.Save(ctx, sessionID, ledger)
}

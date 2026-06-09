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
	tierSnip                             = "snip"
	tierPrune                            = "prune"
	tierMicrocompact                     = "microcompact"
	defaultSnipMinTokens                 = 64
	defaultUserCodeMicrocompactMinTokens = 96
	userCodePreviewHeadLines             = 12
	userCodePreviewTailLines             = 8
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
	watermark := s.WatermarkForUsage(usage)
	if watermark == "" || watermark == "overflow" || watermark == "summarize" {
		watermark = tierSnip
	}
	reducer := localReducer{
		service:       s,
		ctx:           ctx,
		sessionID:     sessionID,
		ledger:        ledger,
		replacements:  ledgerReplacementIndex(ledger),
		protectedFrom: protectedMessageStart(len(messages), s.protectedRecentMessages()),
		latestUser:    latestRealUserIndex(messages),
		watermark:     watermark,
	}
	out, warnings, createdOrReused, tiers := reducer.reduce(messages)
	warnings = append(loadWarnings, warnings...)
	if createdOrReused == 0 {
		return messages, Result{
			Compacted:      false,
			Trigger:        "usage",
			Watermark:      watermark,
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
		Watermark:      watermark,
		Usage:          cloneUsage(usage),
		OriginalTokens: originalTokens,
		NewTokens:      newTokens,
		TiersApplied:   tiers,
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
	latestUser    int
	watermark     string
}

func (r *localReducer) reduce(messages []llm.Message) ([]llm.Message, []string, int, []string) {
	out := make([]llm.Message, len(messages))
	copy(out, messages)
	warnings := []string{}
	changed := 0
	applied := map[string]struct{}{}
	for i, msg := range messages {
		var repl LedgerReplacement
		var warning string
		var ok bool
		switch {
		case msg.Role == llm.RoleTool:
			repl, warning, ok = r.reduceToolMessage(i, msg)
		case r.isPrune() && msg.Role == llm.RoleAssistant:
			repl, warning, ok = r.reduceAssistantMessage(i, msg)
		case r.isPrune() && msg.Role == llm.RoleUser:
			repl, warning, ok = r.reduceUserCodeMessage(i, msg)
		default:
			continue
		}
		if warning != "" {
			warnings = append(warnings, warning)
		}
		if !ok {
			continue
		}
		msg.Content = llm.TextContent(repl.ReplacementText)
		out[i] = msg
		changed++
		applied[repl.Tier] = struct{}{}
	}
	return out, warnings, changed, orderedLocalTiers(applied)
}

func (r *localReducer) isPrune() bool {
	return r.watermark == tierPrune
}

func (r *localReducer) reduceToolMessage(index int, msg llm.Message) (LedgerReplacement, string, bool) {
	if r.isPrune() {
		text := msg.Content.PlainText()
		if parent, ok := r.findReplacementByText(text, msg); ok {
			if parent.Tier == tierPrune {
				return parent, "", true
			}
			pruned := r.createToolPruneReplacement(msg, parent)
			pruneKey := replacementLookupKey(pruned.MessageKey, pruned.PartKey)
			if existing, exists := r.replacements[pruneKey]; exists && existing.Tier == tierPrune {
				return existing, "", true
			}
			r.ledger.Replacements = append(r.ledger.Replacements, pruned)
			r.replacements[pruneKey] = pruned
			return pruned, "", true
		}
	}
	if !r.eligibleToolMessage(index, msg) {
		return LedgerReplacement{}, "", false
	}
	original := msg.Content.PlainText()
	key := StableMessageKey(MessageKeyInput{
		Role:           string(msg.Role),
		ToolCallID:     msg.ToolCallID,
		ToolName:       msg.ToolName,
		OriginalText:   original,
		FirstSeenIndex: index,
	})
	partKey := "content-0"
	lookupKey := replacementLookupKey(key, partKey)
	repl, ok := r.replacements[lookupKey]
	if !ok {
		created, warning, ok := r.createSnipReplacement(msg, key, partKey, original)
		if !ok {
			return LedgerReplacement{}, warning, false
		}
		repl = created
		r.ledger.Replacements = append(r.ledger.Replacements, repl)
		r.replacements[lookupKey] = repl
	}
	if r.isPrune() && repl.Tier == tierSnip {
		pruned := r.createToolPruneReplacement(msg, repl)
		pruneKey := replacementLookupKey(pruned.MessageKey, pruned.PartKey)
		if existing, exists := r.replacements[pruneKey]; exists && existing.Tier == tierPrune {
			return existing, "", true
		}
		r.ledger.Replacements = append(r.ledger.Replacements, pruned)
		r.replacements[pruneKey] = pruned
		return pruned, "", true
	}
	return repl, "", true
}

func (r *localReducer) findReplacementByText(text string, msg llm.Message) (LedgerReplacement, bool) {
	hash := ContentHash(text)
	for _, repl := range r.ledger.Replacements {
		if repl.ReplacementHash != hash {
			continue
		}
		if repl.Role != string(msg.Role) {
			continue
		}
		if strings.TrimSpace(repl.ToolName) != "" && strings.TrimSpace(repl.ToolName) != strings.TrimSpace(msg.ToolName) {
			continue
		}
		return repl, true
	}
	return LedgerReplacement{}, false
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

func (r *localReducer) reduceAssistantMessage(index int, msg llm.Message) (LedgerReplacement, string, bool) {
	if !r.eligibleAssistantMessage(index, msg) {
		return LedgerReplacement{}, "", false
	}
	original := msg.Content.PlainText()
	key := StableMessageKey(MessageKeyInput{
		Role:           string(msg.Role),
		OriginalText:   original,
		FirstSeenIndex: index,
	})
	partKey := "content-0"
	lookupKey := replacementLookupKey(key, partKey)
	if repl, ok := r.replacements[lookupKey]; ok {
		return repl, "", true
	}
	if r.service.Config.ToolArtifactWriter == nil {
		return LedgerReplacement{}, compactionArtifactWarning(r.sessionID, "assistant", "", "write", "no artifact writer configured"), false
	}
	artifact, err := r.service.Config.ToolArtifactWriter.SaveCompactionArtifact(r.ctx, ArtifactRequest{
		SessionID:  r.sessionID,
		MessageKey: key,
		PartKey:    partKey,
		ToolName:   "assistant",
		Content:    original,
	})
	if err != nil {
		return LedgerReplacement{}, compactionArtifactWarning(r.sessionID, "assistant", "", "write", err.Error()), false
	}
	artifactPath := strings.TrimSpace(artifact.Path)
	if artifactPath == "" {
		return LedgerReplacement{}, compactionArtifactWarning(r.sessionID, "assistant", "", "write", "artifact writer returned empty path"), false
	}
	text := assistantPruneReplacementText(original, artifactPath)
	repl := LedgerReplacement{
		MessageKey:      key,
		PartKey:         partKey,
		Role:            string(msg.Role),
		Tier:            tierPrune,
		OriginalHash:    ContentHash(original),
		ReplacementHash: ContentHash(text),
		ReplacementText: text,
		FullArtifact:    artifactPath,
		CreatedAt:       time.Now().UTC(),
		OriginalText:    original,
	}
	r.ledger.Replacements = append(r.ledger.Replacements, repl)
	r.replacements[lookupKey] = repl
	return repl, "", true
}

func (r *localReducer) eligibleAssistantMessage(index int, msg llm.Message) bool {
	if index >= r.protectedFrom {
		return false
	}
	if msg.Destroyed || len(msg.ToolCalls) > 0 {
		return false
	}
	text := msg.Content.PlainText()
	if strings.TrimSpace(text) == "" {
		return false
	}
	return approximateTextTokens(text) >= defaultSnipMinTokens
}

func (r *localReducer) reduceUserCodeMessage(index int, msg llm.Message) (LedgerReplacement, string, bool) {
	if !r.eligibleUserCodeMessage(index, msg) {
		return LedgerReplacement{}, "", false
	}
	original := msg.Content.PlainText()
	key := StableMessageKey(MessageKeyInput{
		Role:           string(msg.Role),
		OriginalText:   original,
		FirstSeenIndex: index,
	})
	partKey := "content-0"
	lookupKey := replacementLookupKey(key, partKey)
	if repl, ok := r.replacements[lookupKey]; ok {
		return repl, "", true
	}
	if r.service.Config.ToolArtifactWriter == nil {
		return LedgerReplacement{}, userCodeArtifactWarning(r.sessionID, "write", "no artifact writer configured"), false
	}
	artifact, err := r.service.Config.ToolArtifactWriter.SaveCompactionArtifact(r.ctx, ArtifactRequest{
		SessionID:  r.sessionID,
		MessageKey: key,
		PartKey:    partKey,
		ToolName:   "user_code",
		Content:    original,
	})
	if err != nil {
		return LedgerReplacement{}, userCodeArtifactWarning(r.sessionID, "write", err.Error()), false
	}
	artifactPath := strings.TrimSpace(artifact.Path)
	if artifactPath == "" {
		return LedgerReplacement{}, userCodeArtifactWarning(r.sessionID, "write", "artifact writer returned empty path"), false
	}
	text, ok := userCodeMicrocompactReplacementText(original, artifactPath)
	if !ok {
		return LedgerReplacement{}, "", false
	}
	repl := LedgerReplacement{
		MessageKey:      key,
		PartKey:         partKey,
		Role:            string(msg.Role),
		Tier:            tierMicrocompact,
		OriginalHash:    ContentHash(original),
		ReplacementHash: ContentHash(text),
		ReplacementText: text,
		FullArtifact:    artifactPath,
		CreatedAt:       time.Now().UTC(),
		OriginalText:    original,
	}
	r.ledger.Replacements = append(r.ledger.Replacements, repl)
	r.replacements[lookupKey] = repl
	return repl, "", true
}

func (r *localReducer) eligibleUserCodeMessage(index int, msg llm.Message) bool {
	if !r.service.Config.EnableUserCodeMicrocompact {
		return false
	}
	if index >= r.protectedFrom || index == r.latestUser || msg.Destroyed || msg.Role != llm.RoleUser || isCompactionSummaryMessage(msg) {
		return false
	}
	text := msg.Content.PlainText()
	if strings.TrimSpace(text) == "" {
		return false
	}
	return userCodeMicrocompactCandidate(text)
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

func (r *localReducer) createToolPruneReplacement(msg llm.Message, parent LedgerReplacement) LedgerReplacement {
	text := toolPruneReplacementText(msg, parent.FullArtifact)
	return LedgerReplacement{
		MessageKey:            parent.MessageKey + "/tier:prune",
		PartKey:               parent.PartKey,
		Role:                  string(msg.Role),
		ToolName:              strings.TrimSpace(msg.ToolName),
		Tier:                  tierPrune,
		OriginalHash:          parent.ReplacementHash,
		ReplacementHash:       ContentHash(text),
		ReplacementText:       text,
		FullArtifact:          strings.TrimSpace(parent.FullArtifact),
		ParentReplacementHash: parent.ReplacementHash,
		CreatedAt:             time.Now().UTC(),
	}
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

func toolPruneReplacementText(msg llm.Message, artifactPath string) string {
	tool := strings.TrimSpace(msg.ToolName)
	if tool == "" {
		tool = "tool"
	}
	id := strings.TrimSpace(msg.ToolCallID)
	if id == "" {
		id = "-"
	}
	return fmt.Sprintf("[Tool result pruned: %s tool_call_id=%s full_output=%s]", tool, id, strings.TrimSpace(artifactPath))
}

func assistantPruneReplacementText(original, artifactPath string) string {
	return fmt.Sprintf("[Assistant text compacted: lines=%d bytes=%d full_output=%s]\n%s", countTextLines(original), len(original), strings.TrimSpace(artifactPath), assistantPreview(original))
}

type fencedCodeBlock struct {
	startLine int
	endLine   int
	info      string
	language  string
	hint      string
	lines     []string
}

func userCodeMicrocompactCandidate(text string) bool {
	block, ok := largestFencedCodeBlock(text)
	if !ok {
		return false
	}
	return approximateTextTokens(strings.Join(block.lines, "\n")) >= defaultUserCodeMicrocompactMinTokens
}

func userCodeMicrocompactReplacementText(original, artifactPath string) (string, bool) {
	block, ok := largestFencedCodeBlock(original)
	if !ok {
		return "", false
	}
	if approximateTextTokens(strings.Join(block.lines, "\n")) < defaultUserCodeMicrocompactMinTokens {
		return "", false
	}
	allLines := strings.Split(strings.ReplaceAll(original, "\r\n", "\n"), "\n")
	out := make([]string, 0, len(allLines)-len(block.lines)+userCodePreviewHeadLines+userCodePreviewTailLines+4)
	out = append(out, allLines[:block.startLine]...)
	out = append(out, userCodeBlockReplacementLines(block, artifactPath)...)
	if block.endLine+1 < len(allLines) {
		out = append(out, allLines[block.endLine+1:]...)
	}
	return strings.TrimSpace(strings.Join(out, "\n")), true
}

func largestFencedCodeBlock(text string) (fencedCodeBlock, bool) {
	lines := strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
	best := fencedCodeBlock{}
	haveBest := false
	open := -1
	info := ""
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if !strings.HasPrefix(trimmed, "```") {
			continue
		}
		if open < 0 {
			open = i
			info = strings.TrimSpace(strings.TrimPrefix(trimmed, "```"))
			continue
		}
		blockLines := append([]string(nil), lines[open+1:i]...)
		candidate := fencedCodeBlock{
			startLine: open,
			endLine:   i,
			info:      info,
			lines:     blockLines,
		}
		candidate.language, candidate.hint = parseFenceInfo(info)
		if !haveBest || len(candidate.lines) > len(best.lines) {
			best = candidate
			haveBest = true
		}
		open = -1
		info = ""
	}
	return best, haveBest
}

func parseFenceInfo(info string) (string, string) {
	fields := strings.Fields(strings.TrimSpace(info))
	if len(fields) == 0 {
		return "", ""
	}
	lang := fields[0]
	hint := ""
	if len(fields) > 1 {
		hint = strings.Join(fields[1:], " ")
	}
	return lang, hint
}

func userCodeBlockReplacementLines(block fencedCodeBlock, artifactPath string) []string {
	lang := strings.TrimSpace(block.language)
	if lang == "" {
		lang = "-"
	}
	hint := strings.TrimSpace(block.hint)
	if hint == "" {
		hint = "-"
	}
	header := fmt.Sprintf("[User code block compacted: language=%s hint=%s lines=%d bytes=%d full_output=%s]", lang, hint, len(block.lines), len(strings.Join(block.lines, "\n")), strings.TrimSpace(artifactPath))
	out := []string{header}
	out = append(out, "Preview:")
	switch {
	case len(block.lines) <= userCodePreviewHeadLines+userCodePreviewTailLines:
		out = append(out, block.lines...)
	default:
		out = append(out, block.lines[:userCodePreviewHeadLines]...)
		out = append(out, fmt.Sprintf("[...%d middle lines omitted; full code: %s]", len(block.lines)-userCodePreviewHeadLines-userCodePreviewTailLines, strings.TrimSpace(artifactPath)))
		out = append(out, block.lines[len(block.lines)-userCodePreviewTailLines:]...)
	}
	return out
}

func assistantPreview(text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}
	const maxPreview = 220
	preview := text
	if len(preview) > maxPreview {
		preview = strings.TrimSpace(preview[:maxPreview]) + "..."
	}
	tokens := extractKeyTokens(text, 6)
	if len(tokens) == 0 {
		return preview
	}
	return preview + "\nKey tokens: " + strings.Join(tokens, " ")
}

func extractKeyTokens(text string, maxTokens int) []string {
	raw := strings.FieldsFunc(text, func(r rune) bool {
		return r == ' ' || r == '\n' || r == '\t' || r == ',' || r == ';' || r == ':' || r == ')' || r == '(' || r == '[' || r == ']'
	})
	out := []string{}
	seen := map[string]struct{}{}
	for _, tok := range raw {
		tok = strings.Trim(tok, "\"'`")
		if tok == "" {
			continue
		}
		isKey := strings.Contains(tok, "/") || strings.Contains(tok, "\\") || strings.Contains(tok, ".") || strings.Contains(tok, "=") || strings.Contains(strings.ToLower(tok), "error")
		if !isKey {
			continue
		}
		if _, ok := seen[tok]; ok {
			continue
		}
		seen[tok] = struct{}{}
		out = append(out, tok)
		if len(out) >= maxTokens {
			break
		}
	}
	return out
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

func userCodeArtifactWarning(sessionID, stage, detail string) string {
	return fmt.Sprintf("[WARN] Compaction artifact not saved - session=%s stage=%s role=user action=leaving original user code in context: %s", strings.TrimSpace(sessionID), strings.TrimSpace(stage), strings.TrimSpace(detail))
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

func latestRealUserIndex(messages []llm.Message) int {
	for i := len(messages) - 1; i >= 0; i-- {
		msg := messages[i]
		if msg.Role == llm.RoleUser && !msg.Destroyed && !isCompactionSummaryMessage(msg) {
			return i
		}
	}
	return -1
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

func orderedLocalTiers(applied map[string]struct{}) []string {
	out := []string{}
	if _, ok := applied[tierSnip]; ok {
		out = append(out, tierSnip)
	}
	if _, ok := applied[tierPrune]; ok {
		out = append(out, tierPrune)
	}
	if _, ok := applied[tierMicrocompact]; ok {
		out = append(out, tierMicrocompact)
	}
	return out
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

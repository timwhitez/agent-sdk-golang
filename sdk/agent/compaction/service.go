package compaction

import (
	"context"
	"fmt"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type Service struct {
	Config Config

	// ContextWindow optionally overrides default window.
	ContextWindow int

	summaryPromptFn SummaryPromptFunc
	protectedTools  map[string]struct{}
}

const fallbackSummaryContext = "[compaction] no eligible prior messages were available; summarize from available tool state only."

func NewService(cfg *Config) *Service {
	c := DefaultConfig()
	ctxWindow := DefaultContextWindow
	if cfg != nil {
		c = *cfg
		if cfg.ContextWindow > 0 {
			ctxWindow = cfg.ContextWindow
		}
	}
	if ctxWindow <= 0 {
		ctxWindow = DefaultContextWindow
	}
	if c.ThresholdRatio <= 0 {
		c.ThresholdRatio = DefaultThresholdRatio
	}
	if c.CompactionTimeout <= 0 {
		c.CompactionTimeout = DefaultCompactionTimeout
	}
	if c.CompactionRetries <= 0 {
		c.CompactionRetries = DefaultCompactionRetries
	}
	if c.CompactionRetryBackoff <= 0 {
		c.CompactionRetryBackoff = DefaultCompactionRetryBackoff
	}
	if c.MinSummaryCharsForToolContext <= 0 {
		c.MinSummaryCharsForToolContext = DefaultMinSummaryCharsForToolContext
	}
	if c.ToolSnapshotMaxEntries <= 0 {
		c.ToolSnapshotMaxEntries = DefaultToolSnapshotMaxEntries
	}
	if c.ToolSnapshotMaxChars <= 0 {
		c.ToolSnapshotMaxChars = DefaultToolSnapshotMaxChars
	}
	if c.SnipThresholdRatio <= 0 {
		c.SnipThresholdRatio = DefaultSnipThresholdRatio
	}
	if c.SnipThresholdRatio >= c.ThresholdRatio {
		c.SnipThresholdRatio = c.ThresholdRatio
	}
	if c.PruneThresholdRatio <= 0 {
		c.PruneThresholdRatio = DefaultPruneThresholdRatio
	}
	if c.PruneThresholdRatio >= c.ThresholdRatio {
		c.PruneThresholdRatio = c.ThresholdRatio
	}
	if c.PruneThresholdRatio < c.SnipThresholdRatio {
		c.PruneThresholdRatio = c.SnipThresholdRatio
	}
	if c.ProtectedRecentMessages <= 0 {
		c.ProtectedRecentMessages = DefaultKeepRecentUserMessages
	}
	return &Service{
		Config:          c,
		ContextWindow:   ctxWindow,
		summaryPromptFn: resolveSummaryPrompt(c.SummaryPrompt),
		protectedTools:  normalizeToolSet(c.ProtectedTools),
	}
}

func (s *Service) threshold() int {
	window := s.contextWindow()
	return int(float64(window) * s.Config.ThresholdRatio)
}

func (s *Service) snipThreshold() int {
	window := s.contextWindow()
	return int(float64(window) * s.Config.SnipThresholdRatio)
}

func (s *Service) pruneThreshold() int {
	window := s.contextWindow()
	return int(float64(window) * s.Config.PruneThresholdRatio)
}

func (s *Service) contextWindow() int {
	window := s.ContextWindow
	if window <= 0 {
		window = DefaultContextWindow
	}
	return window
}

func (s *Service) reserveOutputTokens() int {
	if s == nil {
		return 0
	}
	reserve := s.Config.ReserveOutputTokens
	if reserve < 0 {
		return 0
	}
	return reserve
}

func (s *Service) overflowLimit() int {
	window := s.contextWindow()
	if window <= 0 {
		return 0
	}
	limit := window - s.reserveOutputTokens()
	if limit <= 0 {
		return window
	}
	return limit
}

func (s *Service) TotalTokens(u *llm.Usage) int {
	if u == nil {
		return 0
	}
	t := u.TotalTokens
	if u.PromptCachedTokens != nil {
		t += *u.PromptCachedTokens
	}
	if u.PromptCacheCreationTokens != nil {
		t += *u.PromptCacheCreationTokens
	}
	return t
}

func (s *Service) PromptTokens(u *llm.Usage) int {
	if u == nil {
		return 0
	}
	return u.PromptTokens
}

// IsOverflow reports whether prompt usage is at/over the hard context limit
// after reserving output tokens.
func (s *Service) IsOverflow(u *llm.Usage) bool {
	if !s.Config.Enabled {
		return false
	}
	limit := s.overflowLimit()
	if limit <= 0 {
		return false
	}
	return s.PromptTokens(u) >= limit
}

func (s *Service) ShouldCompact(u *llm.Usage) bool {
	if !s.Config.Enabled {
		return false
	}
	total := s.TotalTokens(u)
	return total >= s.snipThreshold() || total >= s.pruneThreshold() || total >= s.threshold()
}

func (s *Service) WatermarkForUsage(u *llm.Usage) string {
	if s == nil || !s.Config.Enabled {
		return ""
	}
	if s.IsOverflow(u) {
		return "overflow"
	}
	total := s.TotalTokens(u)
	if total >= s.threshold() {
		return "summarize"
	}
	if total >= s.pruneThreshold() {
		return "prune"
	}
	if total >= s.snipThreshold() {
		return "snip"
	}
	return ""
}

func (s *Service) CompactAuto(ctx context.Context, model llm.ChatModel, messages []llm.Message, usage *llm.Usage, watermark string) ([]llm.Message, Result, error) {
	watermark = strings.TrimSpace(watermark)
	if watermark == "" {
		watermark = s.WatermarkForUsage(usage)
	}
	if watermark == "snip" || watermark == "prune" {
		return s.CompactLocal(ctx, messages, usage)
	}
	if usage != nil && (watermark == "summarize" || watermark == "overflow") {
		localMsgs, localRes, err := s.CompactLocal(ctx, messages, usage)
		if err != nil {
			return messages, localRes, err
		}
		if localRes.Compacted {
			limit := s.threshold()
			if watermark == "overflow" {
				limit = s.overflowLimit()
			}
			if limit <= 0 || localRes.NewTokens < limit {
				return localMsgs, localRes, nil
			}
			summaryMsgs, summaryRes, err := s.Compact(ctx, model, localMsgs)
			summaryRes = mergeLocalSummaryResult(localRes, summaryRes)
			return summaryMsgs, summaryRes, err
		}
		summaryMsgs, summaryRes, err := s.Compact(ctx, model, messages)
		summaryRes.Warnings = append(localRes.Warnings, summaryRes.Warnings...)
		return summaryMsgs, summaryRes, err
	}
	return s.Compact(ctx, model, messages)
}

func mergeLocalSummaryResult(localRes Result, summaryRes Result) Result {
	if !localRes.Compacted {
		summaryRes.Warnings = append(localRes.Warnings, summaryRes.Warnings...)
		return summaryRes
	}
	summaryRes.OriginalTokens = localRes.OriginalTokens
	if strings.TrimSpace(summaryRes.LedgerPath) == "" {
		summaryRes.LedgerPath = strings.TrimSpace(localRes.LedgerPath)
	}
	summaryRes.Warnings = append(localRes.Warnings, summaryRes.Warnings...)
	summaryRes.TiersApplied = append([]string{"snip"}, withoutTier(summaryRes.TiersApplied, "snip")...)
	return summaryRes
}

func withoutTier(tiers []string, tier string) []string {
	out := make([]string, 0, len(tiers))
	for _, t := range tiers {
		if strings.TrimSpace(t) == "" || strings.TrimSpace(t) == tier {
			continue
		}
		out = append(out, strings.TrimSpace(t))
	}
	return out
}

func (s *Service) Compact(ctx context.Context, model llm.ChatModel, messages []llm.Message) (newMessages []llm.Message, res Result, err error) {
	if model == nil {
		return messages, Result{Compacted: false}, nil
	}

	originalTokens := approximateMessageTokens(messages)
	sessionID := strings.TrimSpace(s.Config.SessionID)
	ledger, ledgerWarnings, err := s.loadLedger(ctx, sessionID)
	if err != nil {
		return messages, Result{Compacted: false}, err
	}
	modelID := stringsTrim(model.Model())
	summaryPrompt := DefaultSummaryPrompt
	if s.summaryPromptFn != nil {
		summaryPrompt = s.summaryPromptFn(modelID)
	}
	keepCount := s.Config.KeepRecentUserMessages
	if keepCount <= 0 {
		keepCount = DefaultKeepRecentUserMessages
	}
	prepared := s.buildCompactionRequest(messages, ledger, keepCount, summaryPrompt)
	invokeCtx := ctx
	if s.Config.CompactionTimeout > 0 {
		var cancel context.CancelFunc
		invokeCtx, cancel = context.WithTimeout(ctx, s.Config.CompactionTimeout)
		defer cancel()
	}

	comp, err := model.Invoke(invokeCtx, llm.InvokeRequest{Messages: prepared})
	if err != nil {
		return messages, Result{Compacted: false}, err
	}
	sum := ExtractSummary(comp.PlainText())
	if sum == "" {
		return messages, Result{Compacted: false}, fmt.Errorf("compaction: summary extraction failed")
	}

	if summaryCharCount(sum) >= s.Config.MinSummaryCharsForToolContext {
		// Append recent tool context so the model knows what tools were used.
		toolCtx := toolContextSnapshot(messages, s.protectedTools, s.Config.ToolSnapshotMaxEntries, s.Config.ToolSnapshotMaxChars)
		if toolCtx != "" {
			sum += "\n\n" + toolCtx
		}
	}

	// Prefix and mark summary so future compaction rounds can reliably identify it.
	prefixed := WithSummaryPrefix(sum)

	// Keep recent user messages for immediate context.
	recent := SelectRecentUserMessages(messages, keepCount)

	newMessages = make([]llm.Message, 0, 1+len(recent))
	newMessages = append(newMessages, newCompactionSummaryMessage(prefixed))
	newMessages = append(newMessages, recent...)

	if ledger != nil {
		ledger.Summary = nextLedgerSummary(ledger.Summary, messages, sum)
		ledger.UpdatedAt = time.Now().UTC()
		ledger.ContextWindow = s.contextWindow()
		if err := ledger.Validate(sessionID); err != nil {
			return messages, Result{Compacted: false}, err
		}
		if err := s.saveLedger(ctx, sessionID, ledger); err != nil {
			return messages, Result{Compacted: false}, err
		}
	}

	res = Result{
		Compacted:      true,
		Trigger:        "manual",
		Watermark:      "summarize",
		Usage:          cloneUsage(comp.Usage),
		OriginalTokens: originalTokens,
		NewTokens:      approximateMessageTokens(newMessages),
		TiersApplied:   []string{"summarize"},
		LedgerPath:     strings.TrimSpace(s.Config.LedgerPath),
		Warnings:       ledgerWarnings,
		Summary:        sum,
	}
	return newMessages, res, nil
}

func approximateMessageTokens(messages []llm.Message) int {
	total := 0
	for _, msg := range messages {
		total += approximateTextTokens(string(msg.Role))
		total += approximateTextTokens(msg.Name)
		total += approximateTextTokens(msg.ToolCallID)
		total += approximateTextTokens(msg.ToolName)
		total += approximateTextTokens(msg.Content.PlainText())
		for _, call := range msg.ToolCalls {
			total += approximateTextTokens(call.ID)
			total += approximateTextTokens(call.Type)
			total += approximateTextTokens(call.Function.Name)
			total += approximateTextTokens(call.Function.Arguments)
		}
		total += 4
	}
	return total
}

func (s *Service) buildCompactionRequest(messages []llm.Message, ledger *Ledger, keepCount int, summaryPrompt string) []llm.Message {
	input := s.buildCompactionInput(messages, ledger, keepCount, summaryPrompt)
	if strings.TrimSpace(input) == "" {
		input = fallbackSummaryContext + "\n\n" + strings.TrimSpace(summaryPrompt)
	}
	return []llm.Message{llm.NewUserMessage(input)}
}

func (s *Service) buildCompactionInput(messages []llm.Message, ledger *Ledger, keepCount int, summaryPrompt string) string {
	var b strings.Builder
	b.WriteString("You are running Goode's internal context compaction pipeline. This is not a user conversation turn. Summarize only the compacted material below and return one updated summary in <summary></summary> tags.\n")
	if prompt := strings.TrimSpace(summaryPrompt); prompt != "" {
		b.WriteString("\n## Summary instructions\n")
		b.WriteString(prompt)
		b.WriteByte('\n')
	}
	if inc := incrementalSummaryContext(messages, ledger, keepCount); inc != "" {
		b.WriteString("\n## Compact material\n")
		b.WriteString(inc)
		return b.String()
	}
	material := selectedCompactionMaterial(messages, keepCount, s.protectedTools, s.Config.ToolSnapshotMaxEntries, s.Config.ToolSnapshotMaxChars)
	if strings.TrimSpace(material) == "" {
		material = fallbackSummaryContext
	}
	b.WriteString("\n## Compact material\n")
	b.WriteString(material)
	return b.String()
}

func selectedCompactionMaterial(messages []llm.Message, keepCount int, protectedTools map[string]struct{}, maxToolEntries int, maxToolChars int) string {
	var b strings.Builder
	if system := currentSystemContext(messages); system != "" {
		b.WriteString("## Current System / Developer Context\n")
		b.WriteString(system)
		b.WriteString("\n\n")
	}
	if summary := latestCompactionSummaryText(messages); summary != "" {
		b.WriteString("## Previous Summary\n")
		b.WriteString(summary)
		b.WriteString("\n\n")
	}
	if users := SelectRecentUserMessages(messages, keepCount); len(users) > 0 {
		b.WriteString("## Recent User Turns\n")
		for _, msg := range users {
			text := truncateCompactionMaterialText(msg.Content.PlainText(), 1600)
			if text == "" {
				continue
			}
			b.WriteString("- ")
			b.WriteString(text)
			b.WriteByte('\n')
		}
		b.WriteByte('\n')
	}
	if delta := selectedKeyEvents(messages, keepCount); strings.TrimSpace(delta) != "" {
		b.WriteString("## Key Non-Retained Events\n")
		b.WriteString(delta)
		b.WriteString("\n\n")
	}
	if toolCtx := toolContextSnapshot(messages, protectedTools, maxToolEntries, maxToolChars); toolCtx != "" {
		b.WriteString(toolCtx)
		b.WriteByte('\n')
	}
	return strings.TrimSpace(b.String())
}

func selectedKeyEvents(messages []llm.Message, keepCount int) string {
	if len(messages) == 0 {
		return ""
	}
	protectedUsers := recentUserIndexes(messages, 0, keepCount)
	const maxEvents = 24
	start := 0
	if len(messages) > 96 {
		start = len(messages) - 96
	}
	events := make([]string, 0, maxEvents)
	for i := start; i < len(messages); i++ {
		if len(events) >= maxEvents {
			break
		}
		msg := messages[i]
		if msg.Destroyed || isCompactionSummaryMessage(msg) {
			continue
		}
		if _, ok := protectedUsers[i]; ok {
			continue
		}
		line := keyEventLine(msg)
		if line == "" {
			continue
		}
		events = append(events, line)
	}
	return strings.Join(events, "\n")
}

func keyEventLine(msg llm.Message) string {
	text := strings.TrimSpace(msg.Content.PlainText())
	switch msg.Role {
	case llm.RoleUser:
		if text == "" || !isImportantCompactionText(text) {
			return ""
		}
		return "- user: " + truncateCompactionMaterialText(text, 800)
	case llm.RoleTool:
		if text == "" && strings.TrimSpace(msg.ToolName) == "" {
			return ""
		}
		label := strings.TrimSpace(msg.ToolName)
		if label == "" {
			label = "tool"
		}
		if msg.IsError {
			return "- tool error " + label + ": " + truncateCompactionMaterialText(text, 600)
		}
	case llm.RoleAssistant:
		if len(msg.ToolCalls) > 0 {
			return "- assistant tool calls: " + compactToolCallList(msg.ToolCalls)
		}
		if isImportantCompactionText(text) {
			return "- assistant: " + truncateCompactionMaterialText(text, 600)
		}
	}
	return ""
}

func compactToolCallList(calls []llm.ToolCall) string {
	parts := make([]string, 0, len(calls))
	for _, call := range calls {
		name := strings.TrimSpace(call.Function.Name)
		if name == "" {
			name = "tool"
		}
		id := strings.TrimSpace(call.ID)
		if id != "" {
			name += "/" + id
		}
		parts = append(parts, name)
	}
	return strings.Join(parts, ", ")
}

func isImportantCompactionText(text string) bool {
	low := strings.ToLower(strings.TrimSpace(text))
	if low == "" {
		return false
	}
	markers := []string{"error", "failed", "blocked", "todo", "remaining", "commit", "test", "verify", "path", "goal", "request", "/mnt/", "/root/", "/repo/", "c:\\", "http://", "https://"}
	for _, marker := range markers {
		if strings.Contains(low, marker) {
			return true
		}
	}
	return false
}

func currentSystemContext(messages []llm.Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		msg := messages[i]
		if msg.Destroyed || msg.Role != llm.RoleSystem {
			continue
		}
		if text := truncateCompactionMaterialText(msg.Content.PlainText(), 2000); text != "" {
			return text
		}
	}
	return ""
}

func latestCompactionSummaryText(messages []llm.Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		msg := messages[i]
		if !isCompactionSummaryMessage(msg) {
			continue
		}
		return truncateCompactionMaterialText(stripSummaryPrefix(msg.Content.PlainText()), 4000)
	}
	return ""
}

func truncateCompactionMaterialText(text string, max int) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}
	if max <= 0 || len(text) <= max {
		return text
	}
	return strings.TrimSpace(text[:max]) + "..."
}

func approximateTextTokens(text string) int {
	text = strings.TrimSpace(text)
	if text == "" {
		return 0
	}
	return (len(text) + 3) / 4
}

func cloneUsage(u *llm.Usage) *llm.Usage {
	if u == nil {
		return nil
	}
	cpy := *u
	if u.PromptCachedTokens != nil {
		v := *u.PromptCachedTokens
		cpy.PromptCachedTokens = &v
	}
	if u.PromptCacheCreationTokens != nil {
		v := *u.PromptCacheCreationTokens
		cpy.PromptCacheCreationTokens = &v
	}
	if u.PromptImageTokens != nil {
		v := *u.PromptImageTokens
		cpy.PromptImageTokens = &v
	}
	return &cpy
}

func summaryCharCount(summary string) int {
	return utf8.RuneCountInString(summary)
}

func prepareForSummary(messages []llm.Message) []llm.Message {
	if len(messages) == 0 {
		return nil
	}
	out := make([]llm.Message, 0, len(messages))
	for i, m := range messages {
		// Skip destroyed ephemeral messages — their content has been replaced
		// with placeholder text and adds no value to the summary.
		if m.Destroyed {
			continue
		}
		isLast := i == len(messages)-1
		if isLast && m.Role == llm.RoleAssistant && len(m.ToolCalls) > 0 {
			// Remove tool_calls from last assistant message to avoid provider errors.
			m.ToolCalls = nil
			if m.Content.IsEmpty() {
				continue
			}
		}
		out = append(out, m)
	}
	out = repairSummaryToolCallPairs(out)
	if len(out) == 0 {
		return []llm.Message{llm.NewUserMessage(fallbackSummaryContext)}
	}
	return out
}

func repairSummaryToolCallPairs(messages []llm.Message) []llm.Message {
	if len(messages) == 0 {
		return messages
	}
	out := make([]llm.Message, 0, len(messages))
	for i := 0; i < len(messages); i++ {
		m := messages[i]
		if m.Role == llm.RoleTool {
			continue
		}
		if m.Role != llm.RoleAssistant || len(m.ToolCalls) == 0 {
			out = append(out, m)
			continue
		}

		expected, validCalls := summaryToolCallIDs(m.ToolCalls)
		j := i + 1
		for j < len(messages) && messages[j].Role == llm.RoleTool {
			j++
		}
		if validCalls && summaryToolResultBlockCompletes(messages[i+1:j], expected) {
			out = append(out, m)
			out = append(out, messages[i+1:j]...)
		} else {
			m.ToolCalls = nil
			out = append(out, m)
		}
		i = j - 1
	}
	return out
}

func summaryToolCallIDs(calls []llm.ToolCall) (map[string]bool, bool) {
	ids := make(map[string]bool, len(calls))
	for _, call := range calls {
		id := strings.TrimSpace(call.ID)
		if id == "" {
			return nil, false
		}
		if _, ok := ids[id]; ok {
			return nil, false
		}
		ids[id] = false
	}
	return ids, len(ids) > 0
}

func summaryToolResultBlockCompletes(results []llm.Message, expected map[string]bool) bool {
	if len(expected) == 0 {
		return false
	}
	for _, m := range results {
		id := strings.TrimSpace(m.ToolCallID)
		seen, ok := expected[id]
		if !ok || seen {
			return false
		}
		expected[id] = true
	}
	for _, seen := range expected {
		if !seen {
			return false
		}
	}
	return true
}

// WithSummaryPrefix prepends DefaultSummaryPrefix to the summary text.
// If the summary already starts with the prefix, it is returned unchanged.
func WithSummaryPrefix(summary string) string {
	if strings.HasPrefix(summary, DefaultSummaryPrefix) {
		return summary
	}
	return DefaultSummaryPrefix + "\n\n" + summary
}

func newCompactionSummaryMessage(summary string) llm.Message {
	return llm.Message{
		Role:    llm.RoleUser,
		Name:    compactionSummaryMessageName,
		Content: llm.TextContent(summary),
	}
}

func isCompactionSummaryMessage(m llm.Message) bool {
	return m.Role == llm.RoleUser && m.Name == compactionSummaryMessageName
}

// SelectRecentUserMessages returns the most recent keepCount user messages,
// skipping compaction-authored summary messages.
func SelectRecentUserMessages(messages []llm.Message, keepCount int) []llm.Message {
	if keepCount <= 0 {
		return nil
	}
	var recent []llm.Message
	for i := len(messages) - 1; i >= 0 && len(recent) < keepCount; i-- {
		m := messages[i]
		if m.Role != llm.RoleUser {
			continue
		}
		if isCompactionSummaryMessage(m) {
			continue
		}
		recent = append(recent, m)
	}
	// Reverse to chronological order.
	for i, j := 0, len(recent)-1; i < j; i, j = i+1, j-1 {
		recent[i], recent[j] = recent[j], recent[i]
	}
	return recent
}

func toolContextSnapshot(messages []llm.Message, protectedTools map[string]struct{}, maxEntries int, maxChars int) string {
	const (
		maxEntryChars = 300
	)
	if maxEntries <= 0 {
		maxEntries = DefaultToolSnapshotMaxEntries
	}
	if maxChars <= 0 {
		maxChars = DefaultToolSnapshotMaxChars
	}
	if len(messages) == 0 {
		return ""
	}
	protected := make([]llm.Message, 0, maxEntries)
	others := make([]llm.Message, 0, maxEntries)
	for i := len(messages) - 1; i >= 0; i-- {
		m := messages[i]
		if m.Role != llm.RoleTool {
			continue
		}
		if isProtectedTool(m.ToolName, protectedTools) {
			protected = append(protected, m)
			continue
		}
		others = append(others, m)
	}
	selected := make([]llm.Message, 0, maxEntries)
	for _, m := range protected {
		if len(selected) >= maxEntries {
			break
		}
		selected = append(selected, m)
	}
	for _, m := range others {
		if len(selected) >= maxEntries {
			break
		}
		selected = append(selected, m)
	}
	if len(selected) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("## Recent Tool Results\n")
	total := b.Len()
	count := 0
	for _, m := range selected {
		text := m.Content.PlainText()
		if len(text) > maxEntryChars {
			text = text[:maxEntryChars] + "..."
		}
		line := fmt.Sprintf("- **%s**: %s\n", m.ToolName, text)
		if total+len(line) > maxChars {
			break
		}
		b.WriteString(line)
		total += len(line)
		count++
	}
	if count == 0 {
		return ""
	}
	return b.String()
}

func normalizeToolSet(names []string) map[string]struct{} {
	if len(names) == 0 {
		return nil
	}
	set := make(map[string]struct{}, len(names))
	for _, n := range names {
		key := strings.ToLower(stringsTrim(n))
		if key == "" {
			continue
		}
		set[key] = struct{}{}
	}
	if len(set) == 0 {
		return nil
	}
	return set
}

func isProtectedTool(toolName string, protectedTools map[string]struct{}) bool {
	if len(protectedTools) == 0 {
		return false
	}
	_, ok := protectedTools[strings.ToLower(stringsTrim(toolName))]
	return ok
}

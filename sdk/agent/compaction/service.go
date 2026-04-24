package compaction

import (
	"context"
	"fmt"
	"strings"
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
	return s.TotalTokens(u) >= s.threshold()
}

func (s *Service) Compact(ctx context.Context, model llm.ChatModel, messages []llm.Message) (newMessages []llm.Message, res Result, err error) {
	if model == nil {
		return messages, Result{Compacted: false}, nil
	}

	prepared := prepareForSummary(messages)
	modelID := stringsTrim(model.Model())
	summaryPrompt := DefaultSummaryPrompt
	if s.summaryPromptFn != nil {
		summaryPrompt = s.summaryPromptFn(modelID)
	}
	prepared = append(prepared, llm.NewUserMessage(summaryPrompt))
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
	keepCount := s.Config.KeepRecentUserMessages
	if keepCount <= 0 {
		keepCount = DefaultKeepRecentUserMessages
	}
	recent := SelectRecentUserMessages(messages, keepCount)

	newMessages = make([]llm.Message, 0, 1+len(recent))
	newMessages = append(newMessages, newCompactionSummaryMessage(prefixed))
	newMessages = append(newMessages, recent...)

	res = Result{Compacted: true, Summary: sum, NewTokens: 0}
	if comp.Usage != nil {
		res.NewTokens = comp.Usage.CompletionTokens
	}
	return newMessages, res, nil
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
	if len(out) == 0 {
		return []llm.Message{llm.NewUserMessage(fallbackSummaryContext)}
	}
	return out
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

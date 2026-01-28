package compaction

import (
	"context"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

type Service struct {
	Config Config

	// ContextWindow optionally overrides default window.
	ContextWindow int
}

func NewService(cfg *Config) *Service {
	c := DefaultConfig()
	ctxWindow := DefaultContextWindow
	if cfg != nil {
		c = *cfg
		if cfg.ContextWindow > 0 {
			ctxWindow = cfg.ContextWindow
		}
		if c.ThresholdRatio <= 0 {
			c.ThresholdRatio = DefaultThresholdRatio
		}
		if c.SummaryPrompt == "" {
			c.SummaryPrompt = DefaultSummaryPrompt
		}
	}
	if ctxWindow <= 0 {
		ctxWindow = DefaultContextWindow
	}
	return &Service{Config: c, ContextWindow: ctxWindow}
}

func (s *Service) threshold() int {
	window := s.ContextWindow
	if window <= 0 {
		window = DefaultContextWindow
	}
	return int(float64(window) * s.Config.ThresholdRatio)
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
	prepared = append(prepared, llm.NewUserMessage(s.Config.SummaryPrompt))

	comp, err := model.Invoke(ctx, llm.InvokeRequest{Messages: prepared})
	if err != nil {
		return messages, Result{Compacted: false}, err
	}
	sum := ExtractSummary(comp.PlainText())

	// Replace entire history with summary as a user message (matches Python behavior).
	newMessages = []llm.Message{llm.NewUserMessage(sum)}
	res = Result{Compacted: true, Summary: sum, NewTokens: 0}
	if comp.Usage != nil {
		res.NewTokens = comp.Usage.CompletionTokens
	}
	return newMessages, res, nil
}

func prepareForSummary(messages []llm.Message) []llm.Message {
	if len(messages) == 0 {
		return nil
	}
	out := make([]llm.Message, 0, len(messages))
	for i, m := range messages {
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
	return out
}

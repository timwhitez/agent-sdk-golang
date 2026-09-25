package agent

import (
	"strings"

	"github.com/timwhitez/agent-sdk-golang/sdk/llm"
)

const (
	thinkingOnlyObservedKind = "thinking_only_observed"
	// The message is fixed: it never carries reasoning text, lengths or
	// provider data.
	thinkingOnlyObservedMessage = "model response completed with reasoning activity but no visible output or tool calls; observed only, nothing was changed"
)

// completionIsThinkingOnly reports whether a successfully returned completion
// is a normally terminated response with reasoning activity and neither
// visible content nor tool calls. The caller separately excludes errors,
// cancellation and continuation episodes.
//
// Normal termination is an allowlist: max_tokens/length (continuation),
// content_filter, refusal, pause_turn, an empty or any other stop reason are
// not observed. Reasoning evidence is non-empty Thinking or a thinking /
// redacted_thinking content block. Opaque provider state is preserved but is
// never interpreted, so on its own it is not evidence.
func completionIsThinkingOnly(comp *llm.Completion) bool {
	if comp == nil || len(comp.ToolCalls) > 0 {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(comp.StopReason)) {
	case "end_turn", "stop", "stop_sequence":
	default:
		return false
	}
	if strings.TrimSpace(comp.Content.Text) != "" {
		return false
	}
	reasoning := strings.TrimSpace(comp.Thinking) != ""
	for _, block := range comp.Content.Blocks {
		switch {
		case llm.IsProviderStateBlock(block):
		case block.Type == "thinking" || block.Type == "redacted_thinking":
			reasoning = true
		case block.Type == "text" && strings.TrimSpace(block.Text) == "" && block.ImageURL == nil && block.Source == nil:
		default:
			// Any other block (text, image, document, unknown) is visible.
			return false
		}
	}
	return reasoning
}
